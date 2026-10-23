"""
Image Detective API - FastAPI + YOLOv8 + Gemini Vision
=======================================================
POST /          → Analyse one image (upload, base64, or URL)
GET  /health    → Liveness check

Architecture:
  1. YOLOv8n runs locally for fast, grounded bounding-box detection.
  2. Gemini Vision (gemini-2.5-flash / gemini-1.5-flash) verifies and
     enriches the result with zero-hallucination cross-validation.
  3. If Gemini is unavailable the local result is returned as-is.

Required env var:
  GEMINI_API_KEY   your Google AI Studio key
Optional:
  PYTHON_API_PORT  port to listen on (default 5050)
"""

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ─── Load .env ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "").strip() or \
                  os.environ.get("GOOGLE_API_KEY", "").strip()
PYTHON_API_PORT = int(os.environ.get("PYTHON_API_PORT", 5050))
MAX_IMAGE_DIM   = 1600
MAX_BYTES       = 30 * 1024 * 1024   # 30 MB

GEMINI_MODELS   = ["gemini-2.5-flash", "gemini-1.5-flash"]  # tried in order

# YOLO category → display group
CATEGORY_MAP = {
    "person": "people",
    "bicycle": "vehicles", "car": "vehicles", "motorcycle": "vehicles",
    "airplane": "vehicles", "bus": "vehicles", "train": "vehicles",
    "truck": "vehicles", "boat": "vehicles",
    "bird": "animals", "cat": "animals", "dog": "animals",
    "horse": "animals", "sheep": "animals", "cow": "animals",
    "elephant": "animals", "bear": "animals", "zebra": "animals",
    "giraffe": "animals",
    "potted plant": "plants",
    "tv": "electronic device", "laptop": "electronic device",
    "mouse": "electronic device", "remote": "electronic device",
    "keyboard": "electronic device", "cell phone": "electronic device",
}

# ─── Startup: load YOLO and Gemini ─────────────────────────────────────────
print("▶  Loading YOLOv8n model…")
YOLO_MODEL = YOLO("yolov8n.pt")
print("✔  YOLOv8n ready")

gemini_client = None
if GEMINI_API_KEY:
    try:
        import google.genai as genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✔  Gemini Vision client ready (key: …{GEMINI_API_KEY[-6:]})")
    except Exception as e:
        print(f"⚠  Gemini SDK init failed: {e}  — running YOLO-only mode")
else:
    print("⚠  GEMINI_API_KEY not set — running in YOLO-only mode")
    print("   Add your key to .env:  GEMINI_API_KEY=AIza…")

# ─── Pydantic schemas ───────────────────────────────────────────────────────

class GroundedItem(BaseModel):
    name: str     = Field(description="Object name, e.g. 'person in red jacket'")
    category: str = Field(description="'people','animals','plants','vehicles','electronic device','other'")
    count: int    = Field(description="Exact count visible in frame")
    details: str  = Field(description="Colour, posture, position")

class GeminiScene(BaseModel):
    one_line_summary:    str              = Field(description="One accurate sentence describing the image.")
    detailed_description:str             = Field(description="Rich paragraph: foreground, midground, background.")
    setting:             str             = Field(description="e.g. 'Outdoor park', 'Indoor office'")
    time_of_day:         str             = Field(description="e.g. 'Daytime (bright)', 'Night'")
    weather:             str             = Field(description="e.g. 'Sunny', 'Overcast', 'Clear indoors'")
    season:              str             = Field(description="e.g. 'Summer', 'Winter', 'N/A'")
    dominant_colours:    list[str]       = Field(description="Top 4 colours seen")
    notable_features:    list[str]       = Field(description="Key visual highlights")
    verified_items:      list[GroundedItem] = Field(description="All confirmed objects in the image")
    caveats:             str             = Field(description="Occlusions, uncertainty notes")

class AnalyzeRequest(BaseModel):
    image:      str = Field(default="", description="Base64-encoded image bytes (JPEG/PNG/WEBP)")
    image_link: str = Field(default="", description="Public URL of an image to download")

class DetectedObject(BaseModel):
    name:     str
    category: str
    count:    int
    details:  str
    certainty: str = "exact"

class AnalyzeResponse(BaseModel):
    one_line_summary:     str
    detailed_description: str
    setting:              str
    time_of_day:          str
    weather:              str
    season:               str
    dominant_colours:     list[str]
    notable_features:     list[str]
    caveats:              str
    subjects:             list[DetectedObject]
    counts:               dict[str, str]
    report:               str
    engine:               str   # "YOLO+Gemini" or "YOLO-only"

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Detective",
    description="Hybrid YOLOv8 + Gemini Vision image analysis API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

COLOR_NAMES = [
    ((0,0,0),"black"), ((35,35,45),"dark slate gray"), ((90,95,105),"cool gray"),
    ((160,165,175),"light gray"), ((245,245,250),"white"), ((160,35,35),"deep red"),
    ((210,85,35),"warm orange"), ((220,180,50),"golden yellow"), ((35,120,50),"emerald green"),
    ((30,85,150),"navy blue"), ((55,140,215),"azure blue"), ((110,60,30),"warm brown"),
    ((215,185,155),"warm beige"),
]

def _closest_color(rgb):
    r, g, b = rgb
    best_d, best_n = float("inf"), "neutral"
    for (cr, cg, cb), name in COLOR_NAMES:
        d = 0.3*(r-cr)**2 + 0.59*(g-cg)**2 + 0.11*(b-cb)**2
        if d < best_d:
            best_d, best_n = d, name
    return best_n

def _image_properties(img: Image.Image):
    rgb = img.convert("RGB")
    stat = ImageStat.Stat(rgb)
    r, g, b = stat.mean[:3]
    brightness = (r + g + b) / 3
    total = (r + g + b) or 1

    thumb = rgb.resize((48, 48))
    colors = thumb.getcolors(maxcolors=2304) or []
    colors.sort(key=lambda x: x[0], reverse=True)
    dom_colors = []
    for cnt, px in colors[:5]:
        hex_val = f"#{px[0]:02X}{px[1]:02X}{px[2]:02X}"
        dom_colors.append(f"{_closest_color(px)} ({hex_val})")

    green_ratio = g / total
    blue_ratio  = b / total
    if green_ratio > 0.38:
        setting = "Outdoor Nature & Foliage"
    elif blue_ratio > 0.40 and brightness > 135:
        setting = "Outdoor Open Daylight"
    elif brightness < 60:
        setting = "Low-Light / Night Scene"
    elif brightness > 165:
        setting = "Bright Well-Lit Interior"
    else:
        setting = "Indoor Ambient Environment"

    if brightness > 175:
        time_of_day = "Daytime (Bright)"
    elif brightness > 115:
        time_of_day = "Daytime (Moderate)"
    elif brightness > 70:
        time_of_day = "Evening / Low-Light"
    else:
        time_of_day = "Night / Shadowed"

    return dom_colors, setting, time_of_day

def _run_yolo(img_path: str) -> tuple[list[dict], dict]:
    """Run YOLOv8 and return detections + per-category counts."""
    results = YOLO_MODEL(img_path, verbose=False, conf=0.25)[0]
    detections = []
    counts: dict[str, Any] = {
        "people": 0, "animals": {}, "plants": {},
        "vehicles": {}, "electronic device": {}, "other": {}
    }
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = results.names[cls_id]
        conf   = float(box.conf[0])
        coords = [float(x) for x in box.xyxy[0]]
        group  = CATEGORY_MAP.get(name, "other")
        detections.append({"name": name, "conf": conf, "box": coords, "group": group})
        if group == "people":
            counts["people"] += 1
        else:
            grp = counts.get(group) if isinstance(counts.get(group), dict) else {}
            grp[name] = grp.get(name, 0) + 1
            counts[group] = grp
    return detections, counts

def _local_narrative(detections, setting, time_of_day, dom_colors):
    if not detections:
        return (
            f"A {setting.lower()} with no prominent focal subjects.",
            f"The scene shows a {setting.lower()} under {time_of_day.lower()} conditions, "
            f"dominated by {dom_colors[0] if dom_colors else 'neutral tones'}."
        )
    name_cnt: dict[str, int] = {}
    for d in detections:
        name_cnt[d["name"]] = name_cnt.get(d["name"], 0) + 1
    parts = [f"{c} {n}" for n, c in name_cnt.items()]
    summary = f"A {setting.lower()} featuring {', '.join(parts)}."
    desc = (
        f"This capture depicts a {setting.lower()} under {time_of_day.lower()} illumination. "
        f"Detected objects include: {', '.join(parts)}. "
        f"Dominant palette: {', '.join(dom_colors[:3])}."
    )
    return summary, desc

def _gemini_analysis(img_bytes: bytes, detections: list[dict],
                     dom_colors: list[str], setting: str, time_of_day: str) -> dict | None:
    if not gemini_client:
        return None

    yolo_text = "\n".join(
        f"- {d['name']} (conf {round(d['conf']*100)}%, bbox {[round(c,1) for c in d['box']]})"
        for d in detections
    ) or "None detected by local model"

    prompt = f"""You are a professional computer vision analyst. Analyze this image with ZERO HALLUCINATION.
Do not invent objects, people, or details that are not visibly present in the frame.

LOCAL YOLO DETECTIONS (bounding-box evidence — use as physical anchors):
{yolo_text}

PHYSICAL MEASUREMENTS:
- Detected setting: {setting}
- Estimated time of day: {time_of_day}
- Measured dominant colours: {', '.join(dom_colors)}

YOUR TASK:
1. Write a precise one-line summary.
2. Write a rich descriptive paragraph (foreground → midground → background).
3. Verify/correct object counts using the YOLO detections as ground truth.
4. List ONLY objects actually visible — filter out any YOLO false positives.
5. Note setting, time of day, weather, season, colours, notable features.
6. Record any caveats (occlusions, blur, distant objects).
"""
    import google.genai as genai
    for model in GEMINI_MODELS:
        try:
            resp = gemini_client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiScene,
                    temperature=0.15,
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            print(f"  Gemini {model} failed: {e}")
            continue
    return None

def _build_report(summary, description, subjects, setting, time_of_day,
                  weather, season, colours, notable, caveats, engine) -> str:
    lines = [
        f"## {summary}", "",
        description, "",
        "### What is in the image",
    ]
    if subjects:
        for s in subjects:
            est = " (estimated)" if s["certainty"] == "approximate" else ""
            det = f" — {s['details']}" if s.get("details") else ""
            lines.append(f"- **{s['count']} x {s['name']}{est}**{det}")
    else:
        lines.append("- No distinct objects identified.")
    lines += [
        "", "### Context",
        f"- **Setting:** {setting}",
        f"- **Time of day:** {time_of_day}",
        f"- **Weather:** {weather}",
        f"- **Season:** {season}",
        f"- **Dominant colours:** {', '.join(colours)}",
    ]
    if notable:
        lines += ["", "### Notable features"] + [f"- {n}" for n in notable]
    if caveats:
        lines += ["", "### Notes & limits", f"- {caveats}"]
    lines += ["", f"---", f"*Analysed by {engine}*"]
    return "\n".join(lines)

def _build_counts(subjects: list[dict]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for s in subjects:
        cat = s.get("category", "other")
        groups.setdefault(cat, []).append(f"{s['count']} x {s['name']}")
    return {cat: ", ".join(items) for cat, items in groups.items()}

def _prepare_image_bytes(raw: bytes) -> bytes:
    """Normalize to JPEG, resize if needed."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(img.size)
        img = img.resize((round(img.width*scale), round(img.height*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()

# ─── Core pipeline ────────────────────────────────────────────────────────────

async def _analyse_bytes(raw: bytes) -> AnalyzeResponse:
    # 1. Prepare image
    jpeg_bytes = await asyncio.to_thread(_prepare_image_bytes, raw)

    # 2. Image properties (colour, brightness heuristics)
    pil_img = Image.open(io.BytesIO(jpeg_bytes))
    dom_colors, setting, time_of_day = await asyncio.to_thread(_image_properties, pil_img)

    # 3. YOLO — run in thread to avoid blocking event loop
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(jpeg_bytes)
        tmp_path = tf.name

    try:
        detections, yolo_counts = await asyncio.to_thread(_run_yolo, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 4. Gemini cross-validation
    gemini_result = await asyncio.to_thread(
        _gemini_analysis, jpeg_bytes, detections, dom_colors, setting, time_of_day
    )

    # 5. Build response
    if gemini_result:
        engine  = "YOLOv8 + Gemini Vision"
        summary = gemini_result.get("one_line_summary", "")
        desc    = gemini_result.get("detailed_description", "")
        setting = gemini_result.get("setting", setting)
        tod     = gemini_result.get("time_of_day", time_of_day)
        weather = gemini_result.get("weather", "N/A")
        season  = gemini_result.get("season", "N/A")
        colours = gemini_result.get("dominant_colours", dom_colors)
        notable = gemini_result.get("notable_features", [])
        caveats = gemini_result.get("caveats", "")
        subjects = [
            {
                "name":     item.get("name", "object"),
                "category": item.get("category", "other"),
                "count":    int(item.get("count", 1)),
                "details":  item.get("details", ""),
                "certainty": "exact",
            }
            for item in gemini_result.get("verified_items", [])
        ]
    else:
        engine = "YOLOv8-only (Gemini unavailable)"
        summary, desc = _local_narrative(detections, setting, time_of_day, dom_colors)
        tod     = time_of_day
        weather = "N/A"
        season  = "N/A"
        colours = dom_colors
        notable = [f"{len(detections)} objects detected by YOLO"]
        caveats = "Gemini verification not available — local YOLO result only."
        subjects = []
        for d in detections:
            existing = next((s for s in subjects if s["name"] == d["name"]), None)
            if existing:
                existing["count"] += 1
            else:
                subjects.append({
                    "name": d["name"], "category": d["group"],
                    "count": 1, "details": f"conf {round(d['conf']*100)}%",
                    "certainty": "exact",
                })

    report = _build_report(summary, desc, subjects, setting, tod,
                           weather, season, colours, notable, caveats, engine)
    counts = _build_counts(subjects)

    return AnalyzeResponse(
        one_line_summary=summary,
        detailed_description=desc,
        setting=setting,
        time_of_day=tod,
        weather=weather,
        season=season,
        dominant_colours=colours,
        notable_features=notable,
        caveats=caveats,
        subjects=[DetectedObject(**s) for s in subjects],
        counts=counts,
        report=report,
        engine=engine,
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/", response_model=AnalyzeResponse, summary="Analyse an image")
async def analyse_endpoint(req: AnalyzeRequest):
    """
    Analyse one image using the hybrid YOLOv8 + Gemini Vision engine.
    Pass either:
      - `image`: base64-encoded JPEG/PNG/WEBP bytes
      - `image_link`: public URL to download the image from
    """
    raw: bytes | None = None

    if req.image.strip():
        try:
            raw = base64.b64decode(req.image.strip())
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    elif req.image_link.strip():
        url = req.image_link.strip()
        if not url.lower().startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="image_link must start with http(s)://")
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "ImageDetective/2.0"})
                resp.raise_for_status()
                raw = resp.content
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not download image: {exc}")
        if len(raw) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 30 MB limit.")

    else:
        raise HTTPException(status_code=400, detail="Provide 'image' (base64) or 'image_link' (URL).")

    return await _analyse_bytes(raw)


@app.get("/health", summary="Liveness check")
async def health():
    return {
        "status": "ok",
        "engine": "YOLOv8n + Gemini Vision",
        "gemini_ready": gemini_client is not None,
        "gemini_model": GEMINI_MODELS[0],
        "port": PYTHON_API_PORT,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Image Detective API")
    print(f"  Engine : YOLOv8n  +  {'Gemini ' + GEMINI_MODELS[0] if gemini_client else 'YOLO-only mode'}")
    print(f"  URL    : http://0.0.0.0:{PYTHON_API_PORT}")
    print(f"  Docs   : http://localhost:{PYTHON_API_PORT}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PYTHON_API_PORT,
        log_level="info",
    )
