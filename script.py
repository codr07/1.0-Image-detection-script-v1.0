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
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
import re
import csv
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageStat
from pydantic import BaseModel, Field
from ultralytics import YOLO

try:
    import pandas as pd
except ImportError:
    pd = None


try:
    import gdown
except ImportError:
    gdown = None


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

GEMINI_MODELS   = [
    "gemini-2.5-flash",        # Latest flagship — highest quality, best quota (2025)
    "gemini-2.0-flash",        # Previous flagship — reliable & fast
    "gemini-1.5-flash",        # High-throughput stable fallback
    "gemini-1.5-flash-8b",     # Lightest fastest fallback
]  # tried in order; highest accessible model attempted first for maximum analysis quality

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

class PersonProfile(BaseModel):
    person_id:       str  = Field(description="Short label, e.g. 'Person 1 (foreground-left)'")
    apparent_gender: str  = Field(description="'Male', 'Female', 'Indeterminate', or 'Unknown'")
    apparent_age:    str  = Field(description="Age estimate range, e.g. '60–70 years', 'Child (~8)', 'Young adult (~25)'")
    religion:        str  = Field(description="Inferred religion or spiritual tradition based on attire, symbols, and context. e.g. 'Hindu — Vaishnava', 'Jain monk (Digambara)', 'Sikh', 'Muslim', 'Christian priest', 'Buddhist monk', 'Secular / Not determinable'. Always qualify uncertainty.")
    role_or_status:  str  = Field(description="Social/occupational role: e.g. 'Swami / monk', 'Priest conducting puja', 'Devotee / layperson', 'Ascetic', 'Pilgrim', 'Attendant', 'Guard / security', 'Tourist', 'Child / student'")
    attire:          str  = Field(description="Detailed clothing description: colour, garment type, fabric texture, headwear, footwear, jewelry or sacred marks (tilak, sacred thread, turban colour, etc.). e.g. 'White cotton dhoti, bare upper body, saffron uttariya shawl, vibhuti (ash) markings on forehead'")
    activity:        str  = Field(description="Exactly what this person is doing at the moment the photo was taken. e.g. 'Performing aarti — holding a lit diya and waving it in a clockwise arc before the deity', 'Seated in padmasana posture, eyes closed, meditating', 'Distributing prasad from a silver thali', 'Listening to a discourse with hands folded in namaskar'")
    posture:         str  = Field(description="Body posture and orientation: e.g. 'Standing upright facing the shrine', 'Seated cross-legged on the floor', 'Bowing with head lowered', 'Walking toward camera', 'Kneeling in prostration'")
    emotion_or_expression: str = Field(description="Observable facial/body expression: e.g. 'Serene, eyes half-closed in devotion', 'Smiling broadly', 'Concentrated and focused', 'Reverent, head bowed', 'Joyful, arms raised', 'Neutral / not clearly visible'")
    cultural_markers: str = Field(description="Any visible cultural, religious, or ceremonial markers: e.g. 'Red sindoor in hair parting (indicates married Hindu woman)', 'Janeo / sacred thread visible across chest', 'Tulsi mala around neck', 'Saffron dhwaja flag in hand', 'Mala beads in right hand — counting with thumb'. Use 'None visible' if absent.")
    position_in_frame: str = Field(description="Where in the image: e.g. 'Foreground-centre', 'Background-left', 'Mid-frame right', 'Partially occluded at right edge'")

class GroundedItem(BaseModel):
    name: str     = Field(description="Object name, e.g. 'lit silver diya', 'ornate marble idol of Mahavir'")
    category: str = Field(description="'people','animals','plants','vehicles','electronic device','other'")
    count: int    = Field(description="Exact count visible in frame")
    details: str  = Field(description="Colour, material, position, condition, and cultural significance if applicable")

class GeminiScene(BaseModel):
    one_line_summary:    str  = Field(description="One precise, information-dense sentence capturing WHO is present, WHAT they are doing, and WHERE. e.g. 'A Jain Digambara monk (foreground) conducts a shastipuja ceremony before a marble idol of Mahavir inside a temple, attended by a group of approximately 20 white-clad female devotees.'")
    detailed_description: str = Field(description="""
A richly detailed, multi-paragraph description covering ALL of the following, in order:

PARA 1 — PEOPLE & ACTIONS: For every visible person, describe in sequence: their role/identity, religion or sect, attire (colours, garments, sacred marks), exact activity at the moment of capture, posture, and emotional expression. Describe inter-person relationships and interactions (e.g. 'The monk addresses the congregation while a younger assistant holds the incense burner'). Quote visible text on clothing/banners.

PARA 2 — SCENE & SETTING: Architecture, layout, furniture/altars/shrines. Materials (marble, sandstone, wood). Decorations (floral garlands, rangoli, flags, banners). Lighting quality and source (artificial, natural, flame). Overall atmosphere and mood.

PARA 3 — CULTURAL & RELIGIOUS CONTEXT: What ceremony or occasion is likely taking place. Specific rituals visible. Sacred objects present and their significance. Any iconography, deities, or symbols identifiable. Cultural tradition and denomination best supported by evidence.

PARA 4 — SPATIAL LAYOUT: Foreground, midground, background contents. Depth and scale. Camera angle and perspective.

PARA 5 — TECHNICAL OBSERVATIONS: Image quality, any motion blur, occlusions, or areas where detail is unclear.""")
    person_profiles:     list[PersonProfile] = Field(description="One entry PER individually distinguishable person in the image. If a crowd is present, profile the most clearly visible individuals and add a group summary entry.")
    religion_culture_summary: str = Field(description="Overall cultural/religious context of the entire scene: tradition, sect, occasion, and confidence level. e.g. 'Jain (Digambara sect) puja ceremony — high confidence based on white attire, absence of stitched clothing on monk, and specific ritual implements present.'")
    ceremony_or_event:   str  = Field(description="Name and description of the event/ceremony if identifiable: e.g. 'Aarti ceremony (Hindu evening prayer ritual)', 'Satsang / devotional assembly', 'Abhishek (ritual bathing of the deity)', 'Paryushana festival gathering', 'Procession / Yatra', 'Meditation session', 'Unknown / general gathering'")
    animals:             str  = Field(description="Visible animals: 'count x name' or 'None detected'")
    plants_and_trees:    str  = Field(description="Floral offerings, garlands, foliage: 'count x name' or 'None detected'")
    people:              str  = Field(description="Concise count summary with religion/role breakdown. e.g. '1 x Digambara Jain monk (foreground), 20+ x white-clad female Jain devotees (seated, background)'")
    vehicles:            str  = Field(description="Visible vehicles: 'count x name' or 'None detected'")
    electronic_devices:  str  = Field(description="Visible phones, cameras, screens: 'count x name' or 'None detected'")
    other_objects:       str  = Field(description="Key religious/sacred/structural items with material and significance: e.g. '1 x brass incense burner (dhoop), 1 x silver thali with prasad, 1 x marble Jain tirthankara idol (approx. 1.5m height), multiple orange flower garlands' or 'None detected'")
    setting:             str  = Field(description="Specific venue/setting with confidence: e.g. 'Jain temple interior (upashraya / prayer hall) — high confidence', 'Outdoor religious procession', 'Domestic puja room'")
    time_of_day:         str  = Field(description="e.g. 'Daytime', 'Evening — artificial lighting suggests evening ceremony', 'Artificial indoor lighting'")
    weather:             str  = Field(description="e.g. 'Indoors', 'Sunny outdoors', 'Not visible'")
    season:              str  = Field(description="e.g. 'Unknown', 'Summer', 'Winter festival season'")
    dominant_colours:    list[str] = Field(description="Primary colours visible across the image")
    notable_features:    list[str] = Field(description="Key focal points, significant details, any visible text/signage (quote exactly), unique cultural artifacts")
    verified_items:      list[GroundedItem] = Field(description="All confirmed physical objects in frame")
    caveats:             str  = Field(description="Professional limits: occlusions, distance-limited identification, motion blur, areas of uncertainty in person identification or religious attribution.")


class AnalyzeRequest(BaseModel):
    image:       str = Field(default="", description="Base64-encoded image bytes (JPEG/PNG/WEBP)")
    image_link:  str = Field(default="", description="Public URL of an image to download")
    folder_link: str = Field(default="", description="Google Drive folder link or local directory path")
    max_images:  int = Field(default=50, description="Max images to analyze from folder")

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
    csv_file:             str | None = None
    excel_file:           str | None = None
    records:              list[dict[str, Any]] = Field(default_factory=list)
    animals:              str = "None detected"
    plants_and_trees:     str = "None detected"
    people:               str = "None detected"
    vehicles:             str = "None detected"
    electronic_devices:   str = "None detected"
    other_objects:        str = "None detected"


class ExportRequest(BaseModel):
    format: str = Field(default="csv", description="'csv' or 'xlsx'")
    records: list[dict[str, Any]] = Field(default_factory=list)


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

    prompt = f"""You are a world-class forensic image analyst and cultural anthropologist specialising in South Asian religious traditions, world religions, and human behaviour. Produce the most detailed, precise, and professionally rigorous analysis possible.

═══════════════════════════════════════════════════════
MISSION 1 — DEEP PERSON ANALYSIS (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════
For EVERY individually visible person, document ALL of the following:

  1. ROLE & IDENTITY: What is their role? (monk, swami, priest, pujari, devotee, layperson, attendant, guard, tourist, child, elder)

  2. RELIGION & SECT: Infer from attire, symbols, marks. Look for:
     • Colours: saffron/ochre=Hindu renunciant/Buddhist, white=Jain monk/Christian, green=Islamic, navy=Sikh Nihang, plain white=general Hindu
     • Head covering: turban colour+style (Sikh: navy/blue/pink/orange/white), skull cap (Muslim), tilak (Hindu sect identifier)
     • Sacred marks: tilak type (Vaishnav=U-shape, Shaiva=horizontal lines/trident, Varkari=vertical), sindoor, bindi, vibhuti/ash
     • Sacred threads: yajnopavita across chest, kanthi mala, rudraksha, tulsi mala
     • Specific garments: dhoti (style+drape), kurta, cassock, cassock+cincture, hijab+abaya, shalwar kameez
     • Objects held: mala beads+material (rudraksha=Shiva, tulsi=Vishnu, crystal=Jain), diya, conch, bell, scripture
     State CONFIDENCE LEVEL for every religious attribution.

  3. EXACT ACTIVITY (be highly specific):
     NOT "praying" — instead: "Standing before the shrine in a slight forward lean with palms pressed together at chest height in anjali mudra, eyes closed, lips moving in silent recitation"
     NOT "sitting" — instead: "Seated in sukhasana (cross-legged) on the marble floor, holding a string of 108 tulsi beads in the right hand, thumb moving bead by bead"

  4. ATTIRE (every detail): Garment names, colours (exact shade), fabric, headwear, footwear or bare feet, jewelry, sacred accessories.

  5. APPARENT AGE RANGE & GENDER.

  6. POSTURE: Standing/seated/kneeling/prostrate. Body orientation. Arm/hand position.

  7. EMOTIONAL EXPRESSION: Serene/devotional/joyful/concentrated/reverent/distressed — be specific about what visible cues indicate this.

  8. CULTURAL MARKERS: Any specific symbol, mark, object held/worn that indicates community, caste, sect, marital status, or rank.

  9. POSITION IN FRAME: Foreground/midground/background + left/centre/right.

  10. INTERACTIONS: Are they engaging with another person, an object, or the deity? How?

═══════════════════════════════════════════════════════
MISSION 2 — SCENE, CEREMONY & CULTURAL CONTEXT
═══════════════════════════════════════════════════════
  • CEREMONY/OCCASION: Identify as specifically as possible (not just "puja" — specify: abhishek, aarti, pravachan/discourse, diksha initiation, yagna, satsang, procession/yatra, shobha yatra, annaprasad distribution, etc.)
  • SACRED OBJECTS: Name them in the tradition's own language (dhoop burner, panch-amrit thali, chaur sahib / chowri, mangalsutra, etc.) with their ritual function.
  • ARCHITECTURE: Temple style (Nagara/Dravidian/Hemadpanthi), material (marble, sandstone, granite), shrine type.
  • Quote ALL VISIBLE TEXT exactly, phonetically if Devanagari/Gujarati.
  • Identify DEITIES / TIRTHANKARAS / GURUS depicted — describe their attributes, mudra, vehicle/wahana.

═══════════════════════════════════════════════════════
GROUND TRUTH — YOLO DETECTIONS (position guide only)
═══════════════════════════════════════════════════════
{yolo_text}
IMPORTANT: YOLO frequently misclassifies religious/cultural items. Override all YOLO labels with your own direct visual analysis.

═══════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════
  ✓ Only describe what is DIRECTLY and CLEARLY visible.
  ✓ State confidence: "confirmed", "likely", "possibly", "cannot determine".
  ✓ For crowds: profile the 5 most visible individuals in full detail, then: "+ approximately N additional [role] in [attire], engaged in [activity]".
  ✓ The detailed_description field MUST be your richest, most comprehensive output — minimum 5 detailed paragraphs.
  ✗ Do NOT hallucinate, invent, or assume beyond visible evidence, calculate object depth in the mage to remove halucination, it is a spiritual image frolder, work accordingly to religious and cultural sentements also and they are Gunnvardhak Murtipujak Jain Sangha of Ahmedabad, Gujarat, Bangalore.
"""


    import google.genai as genai

    GEMINI_CALL_TIMEOUT = 85  # seconds per model attempt

    def _call_model(model: str):
        return gemini_client.models.generate_content(
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

    for model in GEMINI_MODELS:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_model, model)
                try:
                    resp = future.result(timeout=GEMINI_CALL_TIMEOUT)
                except FuturesTimeoutError:
                    print(f"  Gemini {model} timed out after {GEMINI_CALL_TIMEOUT}s — trying next model")
                    continue
            return json.loads(resp.text)
        except Exception as e:
            err_str = str(e)
            # Skip quickly on 404 (model not available) or known bad status
            if "404" in err_str or "NOT_FOUND" in err_str:
                print(f"  Gemini {model} not available (404) — skipping")
                continue
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  Gemini {model} quota exhausted (429) — instantly falling back to next model")
                continue
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

    # Build spreadsheet record for single image
    single_counts: dict[str, dict[str, int]] = {
        "animals": {}, "plants": {}, "people": 0, "vehicles": {}, "electronic device": {}, "other": {}
    }
    for s in subjects:
        cat = s["category"].lower()
        cnt = s["count"]
        name = s["name"]
        if cat in ("animal", "animals"):
            single_counts["animals"][name] = single_counts["animals"].get(name, 0) + cnt
        elif cat in ("people", "person"):
            single_counts["people"] += cnt
        elif cat in ("vehicles", "vehicle"):
            single_counts["vehicles"][name] = single_counts["vehicles"].get(name, 0) + cnt
        elif cat == "electronic device":
            single_counts["electronic device"][name] = single_counts["electronic device"].get(name, 0) + cnt
        elif "plant" in cat:
            single_counts["plants"][name] = single_counts["plants"].get(name, 0) + cnt
        else:
            single_counts["other"][name] = single_counts["other"].get(name, 0) + cnt

    animals_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["animals"].items())
    plants_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["plants"].items())
    people_str = f"{single_counts['people']} person(s)" if single_counts["people"] > 0 else ""
    # Formulate professional category strings matching Jyot standards
    if gemini_result:
        animals_str = gemini_result.get("animals", "").strip() or "None detected"
        plants_str = gemini_result.get("plants_and_trees", "").strip() or "None detected"
        people_str = gemini_result.get("people", "").strip() or "None detected"
        vehicles_str = gemini_result.get("vehicles", "").strip() or "None detected"
        devices_str = gemini_result.get("electronic_devices", "").strip() or "None detected"
        other_str = gemini_result.get("other_objects", "").strip() or "None detected"
    else:
        # Fallback local YOLO counts
        animals_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["animals"].items()) if single_counts["animals"] else "None detected"
        plants_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["plants"].items()) if single_counts["plants"] else "None detected"
        people_str = f"{single_counts['people']} person(s)" if single_counts["people"] > 0 else "None detected"
        vehicles_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["vehicles"].items()) if single_counts["vehicles"] else "None detected"
        devices_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["electronic device"].items()) if single_counts["electronic device"] else "None detected"
        other_str = ", ".join(f"{cnt} {k}" for k, cnt in single_counts["other"].items()) if single_counts["other"] else "None detected"

    single_record = {
        "Analysed At": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "File Name": "Single_Upload.jpg",
        "Image Link": "Local / Direct Stream",
        "Summary": summary,
        "Detailed Description": desc,
        "Animals": animals_str,
        "Plants & Trees": plants_str,
        "People": people_str,
        "Vehicles": vehicles_str,
        "electronic device": devices_str,
        "Other Objects": other_str,
        "Setting": setting,
        "Time of Day": tod,
        "Weather": weather,
        "Season": season,
        "Dominant Colours": ", ".join(colours),
        "Notable Features": "\n".join(notable) if isinstance(notable, list) else str(notable),
        "Notes & Limits": caveats
    }

    # Save single analysis to default report files
    csv_filename = "image_detection_analysis_report.csv"
    excel_filename = "image_detection_analysis_report.xlsx"
    if pd is not None:
        try:
            df = pd.DataFrame([single_record])
            try:
                df.to_csv(csv_filename, index=False)
            except Exception:
                pass
            try:
                df.to_excel(excel_filename, index=False)
            except Exception:
                pass
        except Exception:
            pass

    return AnalyzeResponse(
        one_line_summary=summary,
        detailed_description=desc,
        setting=setting,
        time_of_day=tod,
        weather=weather,
        season=season,
        dominant_colours=colours,
        notable_features=notable if isinstance(notable, list) else [str(notable)],
        caveats=caveats,
        subjects=[DetectedObject(**s) for s in subjects],
        counts=counts,
        report=report,
        engine=engine,
        csv_file=csv_filename,
        excel_file=excel_filename,
        records=[single_record],
        animals=animals_str,
        plants_and_trees=plants_str,
        people=people_str,
        vehicles=vehicles_str,
        electronic_devices=devices_str,
        other_objects=other_str,
    )




def _download_drive_folder(link: str, target_dir: Path, max_images: int = 50) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Match various Drive URL formats (folders/<id>, id=<id>, u/<n>/folders/<id>, etc.)
    folder_match = re.search(r"folders/([a-zA-Z0-9_-]+)", link)
    file_match = re.search(r"(?:id=|d/|open\?id=)([a-zA-Z0-9_-]+)", link)

    print(f"📥 Downloading Drive folder: {link}")
    if gdown is not None:
        try:
            if folder_match:
                folder_id = folder_match.group(1)
                print(f"   Detected Folder ID: {folder_id}")
                try:
                    gdown.download_folder(id=folder_id, output=str(target_dir), quiet=False, use_cookies=False)
                except Exception as e1:
                    print(f"   gdown folder download retry without cookies: {e1}")
                    gdown.download_folder(url=link, output=str(target_dir), quiet=False)
            elif file_match:
                file_id = file_match.group(1)
                print(f"   Detected Single File ID: {file_id}")
                dest = str(target_dir / f"file_{file_id}.jpg")
                try:
                    gdown.download(id=file_id, output=dest, quiet=False)
                except Exception:
                    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                        r = client.get(f"https://drive.google.com/uc?export=download&id={file_id}")
                        if r.status_code == 200 and len(r.content) > 1000:
                            Path(dest).write_bytes(r.content)
        except Exception as e:
            print(f"⚠ gdown extraction warning: {e}")

    # Collect images recursively from target_dir
    image_files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(target_dir.rglob(ext))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in image_files:
        if f.resolve() not in seen and f.is_file():
            seen.add(f.resolve())
            unique_files.append(f)

    print(f"   ✔ Found {len(unique_files)} image(s) from Drive folder (capped at {max_images})")
    return unique_files[:max_images]


async def _analyse_folder(folder_link: str, max_images: int = 50) -> AnalyzeResponse:
    target_path = Path(folder_link.strip().strip('"').strip("'"))
    temp_dir_obj = None
    image_paths: list[Path] = []

    if target_path.exists() and target_path.is_dir():
        # Local folder path provided
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(target_path.glob(ext))
        seen = set()
        image_paths = [p for p in image_paths if p.resolve() not in seen and not seen.add(p.resolve())]
        image_paths = image_paths[:max_images]
    else:
        # Google Drive or remote link
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gdrive_batch_")
        temp_dir = Path(temp_dir_obj.name)
        image_paths = await asyncio.to_thread(_download_drive_folder, folder_link, temp_dir, max_images)

    if not image_paths:
        if temp_dir_obj:
            temp_dir_obj.cleanup()
        raise HTTPException(
            status_code=400,
            detail="No images found in the specified folder link. Ensure the Google Drive folder link is public (Anyone with the link can view) or contains supported images (.png, .jpg, .webp)."
        )

    # Process all images
    folder_results: list[dict] = []
    agg_counts: dict[str, dict[str, int]] = {
        "animals": {}, "plants": {}, "people": {"person": 0},
        "vehicles": {}, "electronic device": {}, "objects": {}, "other": {}
    }

    try:
        for idx, img_p in enumerate(image_paths, 1):
            try:
                raw_bytes = img_p.read_bytes()
                res = await _analyse_bytes(raw_bytes)
                
                # Extract counts for photo list table
                a_count = 0
                p_count = 0
                v_count = 0
                o_count = 0
                
                for s in res.subjects:
                    cat = s.category.lower()
                    cnt = s.count
                    if cat in ("animal", "animals"):
                        a_count += cnt
                        agg_counts["animals"][s.name] = agg_counts["animals"].get(s.name, 0) + cnt
                    elif cat in ("people", "person"):
                        p_count += cnt
                        agg_counts["people"]["person"] = agg_counts["people"].get("person", 0) + cnt
                    elif cat in ("vehicles", "vehicle"):
                        v_count += cnt
                        agg_counts["vehicles"][s.name] = agg_counts["vehicles"].get(s.name, 0) + cnt
                    else:
                        o_count += cnt
                        grp = "electronic device" if cat == "electronic device" else ("plants" if "plant" in cat else "other")
                        agg_counts[grp][s.name] = agg_counts[grp].get(s.name, 0) + cnt

                folder_results.append({
                    "filename": img_p.name,
                    "summary": res.one_line_summary,
                    "animals": f"{a_count} animals" if a_count else "0",
                    "people": f"{p_count} people" if p_count else "0",
                    "vehicles": f"{v_count} vehicles" if v_count else "0",
                    "objects": f"{o_count} objects" if o_count else "0",
                    "response": res
                })
            except Exception as e:
                print(f"Error processing image {img_p.name}: {e}")
    finally:
        if temp_dir_obj:
            temp_dir_obj.cleanup()

    if not folder_results:
        raise HTTPException(status_code=500, detail="Failed to analyze any images in the folder.")

    total_images = len(folder_results)
    total_people = agg_counts["people"]["person"]
    total_animals = sum(agg_counts["animals"].values())
    total_vehicles = sum(agg_counts["vehicles"].values())
    total_devices = sum(agg_counts["electronic device"].values())
    total_plants = sum(agg_counts["plants"].values())
    total_other = sum(agg_counts["other"].values())

    summary_headline = f"Batch folder analysis complete: {total_images} image(s) processed."
    detailed_desc = (
        f"Analyzed {total_images} image(s) from folder. Aggregate detections: "
        f"{total_people} person(s), {total_animals} animal(s), {total_vehicles} vehicle(s), "
        f"{total_devices} electronic device(s), {total_plants} plant(s), and {total_other} other object(s)."
    )

    # Build subjects list from aggregates
    combined_subjects: list[DetectedObject] = []
    if total_people > 0:
        combined_subjects.append(DetectedObject(name="people", category="people", count=total_people, details="Across all images", certainty="exact"))
    for name, cnt in agg_counts["animals"].items():
        combined_subjects.append(DetectedObject(name=name, category="animals", count=cnt, details="Batch detected", certainty="exact"))
    for name, cnt in agg_counts["vehicles"].items():
        combined_subjects.append(DetectedObject(name=name, category="vehicles", count=cnt, details="Batch detected", certainty="exact"))
    for name, cnt in agg_counts["electronic device"].items():
        combined_subjects.append(DetectedObject(name=name, category="electronic device", count=cnt, details="Batch detected", certainty="exact"))
    for name, cnt in agg_counts["plants"].items():
        combined_subjects.append(DetectedObject(name=name, category="plants", count=cnt, details="Batch detected", certainty="exact"))
    for name, cnt in agg_counts["other"].items():
        combined_subjects.append(DetectedObject(name=name, category="other", count=cnt, details="Batch detected", certainty="exact"))

    # Build markdown report with photo-by-photo lines recognized by Image Detector.html parser
    report_lines = [
        f"## {summary_headline}",
        "",
        detailed_desc,
        "",
        "### What is in the image",
    ]
    for s in combined_subjects[:20]:
        report_lines.append(f"- **{s.count} x {s.name}** — {s.details}")

    report_lines.extend([
        "",
        "### Context",
        f"- **Setting:** Multi-image Batch Folder",
        f"- **Time of day:** Varied",
        f"- **Weather:** Varied",
        f"- **Season:** Varied",
        f"- **Dominant colours:** Mixed",
        "",
        "### Photo by photo analysis",
    ])
    for fr in folder_results:
        report_lines.append(
            f"- **{fr['filename']}** - {fr['summary']} (Animals: {fr['animals']}; People: {fr['people']}; Vehicles: {fr['vehicles']}; Objects: {fr['objects']})"
        )

    counts_dict = {
        "people": f"{total_people} person(s)" if total_people else "",
        "animals": ", ".join(f"{cnt} {k}" for k, cnt in agg_counts["animals"].items()) if agg_counts["animals"] else "",
        "vehicles": ", ".join(f"{cnt} {k}" for k, cnt in agg_counts["vehicles"].items()) if agg_counts["vehicles"] else "",
        "electronic device": ", ".join(f"{cnt} {k}" for k, cnt in agg_counts["electronic device"].items()) if agg_counts["electronic device"] else "",
        "plants": ", ".join(f"{cnt} {k}" for k, cnt in agg_counts["plants"].items()) if agg_counts["plants"] else "",
        "other": ", ".join(f"{cnt} {k}" for k, cnt in agg_counts["other"].items()) if agg_counts["other"] else "",
    }

    # Build spreadsheet records matching Jyot schema
    spreadsheet_records = []
    for fr in folder_results:
        sub_res = fr["response"]
        rec = {
            "Analysed At": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "File Name": fr["filename"],
            "Image Link": f"{folder_link}/{fr['filename']}" if not folder_link.endswith("/") else f"{folder_link}{fr['filename']}",
            "Summary": sub_res.one_line_summary,
            "Detailed Description": sub_res.detailed_description,
            "Animals": getattr(sub_res, "animals", "None detected") or "None detected",
            "Plants & Trees": getattr(sub_res, "plants_and_trees", "None detected") or "None detected",
            "People": getattr(sub_res, "people", "None detected") or "None detected",
            "Vehicles": getattr(sub_res, "vehicles", "None detected") or "None detected",
            "electronic device": getattr(sub_res, "electronic_devices", "None detected") or "None detected",
            "Other Objects": getattr(sub_res, "other_objects", "None detected") or "None detected",
            "Religion & Culture": getattr(sub_res, "religion_culture_summary", "") or "",
            "Ceremony / Event": getattr(sub_res, "ceremony_or_event", "") or "",
            "Person Profiles": json.dumps(getattr(sub_res, "person_profiles", []), ensure_ascii=False) if getattr(sub_res, "person_profiles", []) else "",
            "Setting": sub_res.setting,
            "Time of Day": sub_res.time_of_day,
            "Weather": sub_res.weather,
            "Season": sub_res.season,
            "Dominant Colours": ", ".join(sub_res.dominant_colours) if isinstance(sub_res.dominant_colours, list) else str(sub_res.dominant_colours),
            "Notable Features": "\n".join(sub_res.notable_features) if isinstance(sub_res.notable_features, list) else str(sub_res.notable_features),
            "Notes & Limits": sub_res.caveats
        }
        spreadsheet_records.append(rec)

    # Save to disk as CSV & Excel

    csv_filename = "image_detection_analysis_report.csv"
    excel_filename = "image_detection_analysis_report.xlsx"
    if pd is not None and spreadsheet_records:
        try:
            df = pd.DataFrame(spreadsheet_records)
            try:
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"image_detection_analysis_report_{ts}.csv"
                df.to_csv(csv_filename, index=False)
            try:
                df.to_excel(excel_filename, index=False)
            except Exception as e:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_filename = f"image_detection_analysis_report_{ts}.xlsx"
                df.to_excel(excel_filename, index=False)
        except Exception as e:
            print(f"Failed to write excel/csv files: {e}")

    return AnalyzeResponse(
        one_line_summary=summary_headline,
        detailed_description=detailed_desc,
        setting="Multi-image Batch Folder",
        time_of_day="Varied",
        weather="Varied",
        season="Varied",
        dominant_colours=["Mixed"],
        notable_features=[f"Batch processed {total_images} images successfully"],
        caveats="Batch folder aggregated analysis from Google Drive / directory source.",
        subjects=combined_subjects,
        counts={k: v for k, v in counts_dict.items() if v},
        report="\n".join(report_lines),
        engine="YOLOv8 + Gemini Vision (Batch Folder Engine)",
        csv_file=csv_filename,
        excel_file=excel_filename,
        records=spreadsheet_records
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/export/csv", summary="Download report as CSV")
async def export_csv(filename: str = "image_detection_analysis_report.csv"):
    p = Path(filename)
    if not p.exists() or not p.is_file():
        p = Path("image_detection_analysis_report.csv")
    if not p.exists():
        raise HTTPException(status_code=404, detail="No report generated yet.")
    return FileResponse(p, media_type="text/csv", filename=p.name)


@app.get("/export/xlsx", summary="Download report as Excel XLSX")
async def export_xlsx(filename: str = "image_detection_analysis_report.xlsx"):
    p = Path(filename)
    if not p.exists() or not p.is_file():
        p = Path("image_detection_analysis_report.xlsx")
    if not p.exists():
        raise HTTPException(status_code=404, detail="No Excel report generated yet.")
    return FileResponse(p, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=p.name)


@app.post("/export/custom", summary="Export custom records dynamically")
async def export_custom(req: ExportRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided for export.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if req.format.lower() in ("xlsx", "excel", "xls"):
        fname = f"detection_report_{ts}.xlsx"
        if pd is not None:
            df = pd.DataFrame(req.records)
            df.to_excel(fname, index=False)
            return FileResponse(fname, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=fname)
    # Default to CSV
    fname = f"detection_report_{ts}.csv"
    keys = list(req.records[0].keys())
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(req.records)
    return FileResponse(fname, media_type="text/csv", filename=fname)


@app.post("/", response_model=AnalyzeResponse, summary="Analyse an image or folder")
async def analyse_endpoint(req: AnalyzeRequest):
    """
    Analyse one image or a folder using the hybrid YOLOv8 + Gemini Vision engine.
    Pass:
      - `folder_link`: Google Drive folder link or local path to scan & analyse all images
      - `image`: base64-encoded JPEG/PNG/WEBP bytes
      - `image_link`: public URL to download the image from
    """
    if req.folder_link.strip():
        return await _analyse_folder(req.folder_link.strip(), req.max_images)

    raw: bytes | None = None

    if req.image.strip():
        try:
            raw = base64.b64decode(req.image.strip())
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    elif req.image_link.strip():
        url = req.image_link.strip()
        # If user pasted a Google Drive folder link into image_link, handle it gracefully
        if "drive.google.com/drive/folders/" in url or "folders/" in url:
            return await _analyse_folder(url, req.max_images)

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
        raise HTTPException(status_code=400, detail="Provide 'folder_link' (Folder URL/path), 'image' (base64) or 'image_link' (URL).")

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
