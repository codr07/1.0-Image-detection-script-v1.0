import os
import io
import glob
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any, Optional
from PIL import Image, ImageStat
import pandas as pd
from pydantic import BaseModel, Field
from ultralytics import YOLO

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------- Setup & Models
print("Initializing Local YOLOv8 Object Detection...")
yolo_model = YOLO("yolov8n.pt")

# Check for Gemini API key (supports GEMINI_API_KEY or GOOGLE_API_KEY)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()

# Gemini model priority chain — verified working model IDs (2025)
GEMINI_MODELS = [
    "gemini-2.5-flash",        # Latest flagship — highest quality, best quota
    "gemini-2.0-flash",        # Previous flagship — reliable & fast
    "gemini-1.5-flash",        # High-throughput stable fallback
    "gemini-1.5-flash-8b",     # Lightest fastest fallback
]
MAX_IMAGE_DIM = 1600

# ---- Concurrency / rate-limit settings ----
MAX_PARALLEL_WORKERS = 4      # images analyzed in parallel
API_RATE_LIMIT_DELAY  = 1.5   # seconds between API calls across all workers
_api_rate_lock = threading.Lock()  # ensures only 1 Gemini call fires at a time per delay window
_last_api_call_time: list = [0.0]  # shared mutable state for rate limiting

gemini_client = None
if GEMINI_API_KEY:
    try:
        import google.genai as genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✔  Gemini Vision AI Client connected (Key: ...{GEMINI_API_KEY[-6:]})")
        print(f"   Model Priority: {' -> '.join(GEMINI_MODELS)}")
    except Exception as e:
        print(f"Warning: Could not initialize Google GenAI SDK: {e}")
else:
    print("No GEMINI_API_KEY found in environment. Running in offline YOLO fallback mode.")
    print("To enable highest-tier Gemini analysis, add GEMINI_API_KEY=... in .env or your environment.")

CATEGORY_MAPPING = {
    "person": "people",
    "bicycle": "vehicles", "car": "vehicles", "motorcycle": "vehicles", "airplane": "vehicles",
    "bus": "vehicles", "train": "vehicles", "truck": "vehicles", "boat": "vehicles",
    "bird": "animals", "cat": "animals", "dog": "animals", "horse": "animals", "sheep": "animals",
    "cow": "animals", "elephant": "animals", "bear": "animals", "zebra": "animals", "giraffe": "animals",
    "potted plant": "plants",
    "tv": "electronic device", "laptop": "electronic device", "mouse": "electronic device",
    "remote": "electronic device", "keyboard": "electronic device", "cell phone": "electronic device",
}

# ------------------------------------------------------------------- Pydantic Schemas (Identical to Web App)
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

# ------------------------------------------------------------------- Local Heuristics
COLOR_NAMES = [
    ((0, 0, 0), "black"),
    ((35, 35, 45), "dark slate gray"),
    ((90, 95, 105), "cool gray"),
    ((160, 165, 175), "light gray"),
    ((245, 245, 250), "white"),
    ((160, 35, 35), "deep red"),
    ((210, 85, 35), "warm orange"),
    ((220, 180, 50), "golden yellow"),
    ((35, 120, 50), "emerald green"),
    ((30, 85, 150), "navy blue"),
    ((55, 140, 215), "azure blue"),
    ((110, 60, 30), "warm brown"),
    ((215, 185, 155), "warm beige"),
]

def closest_color_name(rgb):
    r, g, b = rgb
    best_dist = float('inf')
    best_name = "neutral"
    for (cr, cg, cb), name in COLOR_NAMES:
        dist = 0.3 * (r - cr)**2 + 0.59 * (g - cb)**2 + 0.11 * (b - cb)**2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name

def analyze_image_properties(img_path):
    with Image.open(img_path) as im:
        rgb_im = im.convert("RGB")
        w, h = rgb_im.size
        stat = ImageStat.Stat(rgb_im)
        r, g, b = stat.mean[:3]
        brightness = (r + g + b) / 3
        total = (r + g + b) or 1
        green_ratio = g / total
        blue_ratio = b / total

        thumb = rgb_im.resize((48, 48))
        colors = thumb.getcolors(maxcolors=2304) or []
        colors.sort(key=lambda x: x[0], reverse=True)
        dominant_colors = []
        for c in colors[:4]:
            rgb = c[1]
            hex_val = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            color_name = closest_color_name(rgb)
            dominant_colors.append(f"{color_name} ({hex_val})")

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
            lighting = "Direct bright lighting with high ambient visibility"
            time_of_day = "Daytime (Bright)"
        elif brightness > 115:
            lighting = "Balanced soft daylight with natural diffuse illumination"
            time_of_day = "Daytime (Moderate)"
        elif brightness > 70:
            lighting = "Dim interior artificial lighting with warm shadows"
            time_of_day = "Evening / Low-Light"
        else:
            lighting = "Low-key nocturnal illumination with dark background tones"
            time_of_day = "Night / Shadowed"

        return w, h, dominant_colors, setting, time_of_day, lighting

def _prepare_image_bytes(img_path: str) -> bytes:
    """Normalize image to JPEG, resize to max dimension if needed."""
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        if max(rgb.size) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(rgb.size)
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()

# ------------------------------------------------------------------- Local Fallback Narrative
def generate_local_narrative(detections, width, height, setting, lighting, dom_colors):
    if not detections:
        color_lead = dom_colors[0] if dom_colors else "neutral hues"
        summary = f"A serene {setting.lower()} without any prominent focal subjects."
        description = (
            f"The image presents a static wide-angle view framed within a {setting.lower()}. "
            f"The scene exhibits {lighting.lower()}, characterized by smooth ambient textures "
            f"and a dominant palette of {color_lead}. Across the entire field of view, no distinct "
            f"foreground people, fauna, or movable objects are identified, leaving the background architecture "
            f"and environmental surface as the central visual component."
        )
        return summary, description

    fg_items, bg_items, mid_items = [], [], []
    object_names_counted = {}

    for d in detections:
        name = d["name"]
        x1, y1, x2, y2 = d["box"]
        area_pct = ((x2 - x1) * (y2 - y1)) / (width * height)
        y_center = (y1 + y2) / 2

        pos = "in the center"
        if (x1 + x2) / 2 < width * 0.35:
            pos = "on the left"
        elif (x1 + x2) / 2 > width * 0.65:
            pos = "on the right"

        object_names_counted[name] = object_names_counted.get(name, 0) + 1
        label = f"{name} ({pos})"
        if area_pct > 0.15 or y_center > height * 0.6:
            fg_items.append(label)
        elif area_pct < 0.04 or y_center < height * 0.4:
            bg_items.append(label)
        else:
            mid_items.append(label)

    subject_parts = [f"{cnt} {name if cnt == 1 else name + 's'}" for name, cnt in object_names_counted.items()]
    subject_summary_str = ", ".join(subject_parts)

    summary = f"A {setting.lower()} featuring {subject_summary_str} under {lighting.lower()}."
    fg_text = f"In the foreground, the frame highlights {', '.join(fg_items[:3])}." if fg_items else "The immediate foreground remains open."
    mid_text = f"The midground contains {', '.join(mid_items[:3])}." if mid_items else "The subjects are positioned across the central depth of field."
    bg_text = f"In the background, {', '.join(bg_items[:2])} can be discerned." if bg_items else f"The backdrop recedes into the {setting.lower()}."

    description = (
        f"This high-resolution capture depicts an authentic {setting.lower()} illuminated by {lighting.lower()}. "
        f"{fg_text} {mid_text} {bg_text} "
        f"The visual composition is anchored by a curated palette featuring {', '.join(dom_colors[:3])}, "
        f"establishing a cohesive environmental mood throughout the entire frame."
    )
    return summary, description

# ------------------------------------------------------------------- Hybrid Vision Engine (Highest-Tier Model)
def run_hybrid_analysis(img_path: str, local_detections: List[Dict[str, Any]], w: int, h: int, dom_colors: List[str], setting: str, time_of_day: str, lighting: str):
    """
    Fuses local YOLOv8 spatial detections with highest available Gemini Vision LLM.
    Uses the exact full prompt, multi-paragraph schema, and model fallback chain as the web app.
    """
    if not gemini_client:
        return None

    yolo_text = "\n".join(
        f"- {d['name']} (conf {round(d['conf']*100)}%, bbox {[round(c, 1) for c in d['box']]})"
        for d in local_detections
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

    jpeg_bytes = _prepare_image_bytes(img_path)
    GEMINI_CALL_TIMEOUT = 85  # seconds per model attempt

    def _call_model(model: str):
        return gemini_client.models.generate_content(
            model=model,
            contents=[
                prompt,
                genai.types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            ],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiScene,
                temperature=0.15,
            ),
        )

    for model_name in GEMINI_MODELS:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_model, model_name)
                try:
                    resp = future.result(timeout=GEMINI_CALL_TIMEOUT)
                except FuturesTimeoutError:
                    print(f"  Gemini {model_name} timed out after {GEMINI_CALL_TIMEOUT}s — trying next model...")
                    continue
            parsed = json.loads(resp.text)
            print(f"  ✔ Analyzed with {model_name}")
            return parsed
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str:
                print(f"  Gemini {model_name} not available (404) — skipping")
                continue
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  Gemini {model_name} quota exhausted (429) — instantly falling back to next model")
                continue
            print(f"  Gemini {model_name} failed: {e} — trying next model...")
            continue

    return None

# ------------------------------------------------------------------- Main Pipeline

def _process_single_image(idx: int, total: int, img_path: str) -> Dict[str, Any]:
    """Process one image (YOLO + Gemini). Thread-safe; uses global rate-limit lock."""
    filename = Path(img_path).name
    print(f"\n[{idx}/{total}] Processing {filename}...")

    # 1. Local Image Properties
    w, h, dom_colors, setting, time_of_day, lighting = analyze_image_properties(img_path)

    # 2. Local YOLOv8 Detection
    results = yolo_model(img_path, verbose=False, conf=0.25)[0]
    local_detections = []
    counts = {
        "animals": {}, "plants": {}, "people": 0,
        "vehicles": {}, "electronic device": {}, "other": {}
    }
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = results.names[cls_id]
        conf   = float(box.conf[0])
        coords = [float(x) for x in box.xyxy[0]]
        group  = CATEGORY_MAPPING.get(name, "other")
        local_detections.append({"name": name, "conf": conf, "box": coords, "group": group})
        if group == "people":
            counts["people"] += 1
        else:
            counts[group][name] = counts[group].get(name, 0) + 1

    # 3. Rate-limited Gemini call
    gemini_result = None
    if gemini_client:
        with _api_rate_lock:
            # Enforce minimum gap between API calls
            elapsed = time.time() - _last_api_call_time[0]
            if elapsed < API_RATE_LIMIT_DELAY:
                time.sleep(API_RATE_LIMIT_DELAY - elapsed)
            _last_api_call_time[0] = time.time()

        gemini_result = run_hybrid_analysis(
            img_path, local_detections, w, h, dom_colors, setting, time_of_day, lighting
        )

    # 4. Build record
    if gemini_result:
        summary         = gemini_result.get("one_line_summary", "")
        description     = gemini_result.get("detailed_description", "")
        setting_res     = gemini_result.get("setting", setting)
        time_of_day_res = gemini_result.get("time_of_day", time_of_day)
        weather_res     = gemini_result.get("weather", "N/A")
        season_res      = gemini_result.get("season", "N/A")
        dom_colors_res  = ", ".join(gemini_result.get("dominant_colours", dom_colors))
        notable_list    = gemini_result.get("notable_features", [])
        notable         = "\n".join(notable_list) if isinstance(notable_list, list) else str(notable_list)
        caveats         = gemini_result.get("caveats", "Verified with Gemini Vision + Local YOLO")
        animals_str     = gemini_result.get("animals", "").strip()          or "None detected"
        plants_str      = gemini_result.get("plants_and_trees", "").strip() or "None detected"
        people_str      = gemini_result.get("people", "").strip()           or "None detected"
        vehicles_str    = gemini_result.get("vehicles", "").strip()         or "None detected"
        devices_str     = gemini_result.get("electronic_devices", "").strip() or "None detected"
        other_str       = gemini_result.get("other_objects", "").strip()    or "None detected"
        religion_summary = gemini_result.get("religion_culture_summary", "").strip()
        ceremony_event  = gemini_result.get("ceremony_or_event", "").strip()
        person_profiles = gemini_result.get("person_profiles", [])
        profiles_json   = json.dumps(person_profiles, ensure_ascii=False) if person_profiles else ""
    else:
        summary, description = generate_local_narrative(local_detections, w, h, setting, lighting, dom_colors)
        setting_res     = setting
        time_of_day_res = time_of_day
        weather_res     = "Interior / Ambient" if "Indoor" in setting else "Clear Ambient"
        season_res      = "N/A"
        dom_colors_res  = ", ".join(dom_colors)
        notable         = f"{len(local_detections)} detected objects, Resolution {w}x{h}px"
        caveats         = "Analyzed with local YOLOv8 (Gemini unavailable or all models exhausted)"
        animals_str     = ", ".join(f"{cnt} {k}" for k, cnt in counts["animals"].items())          if counts["animals"]           else "None detected"
        plants_str      = ", ".join(f"{cnt} {k}" for k, cnt in counts["plants"].items())           if counts["plants"]            else "None detected"
        people_str      = f"{counts['people']} person(s)"                                           if counts["people"] > 0        else "None detected"
        vehicles_str    = ", ".join(f"{cnt} {k}" for k, cnt in counts["vehicles"].items())         if counts["vehicles"]          else "None detected"
        devices_str     = ", ".join(f"{cnt} {k}" for k, cnt in counts["electronic device"].items()) if counts["electronic device"] else "None detected"
        other_str       = ", ".join(f"{cnt} {k}" for k, cnt in counts["other"].items())             if counts["other"]             else "None detected"
        religion_summary = ""
        ceremony_event  = ""
        profiles_json   = ""

    return {
        "Analysed At":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "File Name":          filename,
        "Image Link":         f"drive_download_test/{filename}",
        "Summary":            summary,
        "Detailed Description": description,
        "Animals":            animals_str,
        "Plants & Trees":     plants_str,
        "People":             people_str,
        "Vehicles":           vehicles_str,
        "electronic device":  devices_str,
        "Other Objects":      other_str,
        "Religion & Culture": religion_summary,
        "Ceremony / Event":   ceremony_event,
        "Person Profiles":    profiles_json,
        "Setting":            setting_res,
        "Time of Day":        time_of_day_res,
        "Weather":            weather_res,
        "Season":             season_res,
        "Dominant Colours":   dom_colors_res,
        "Notable Features":   notable,
        "Notes & Limits":     caveats,
    }


def main():
    # Collect all images from drive_download_test/ (recursive)
    base = Path("drive_download_test")
    images: List[str] = []
    if base.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG"):
            images.extend(str(p) for p in base.rglob(ext))
    images = sorted(list(set(images)))
    total  = len(images)
    print(f"Found {total} images to process — using {MAX_PARALLEL_WORKERS} parallel workers.")
    print(f"Model chain: {' → '.join(GEMINI_MODELS)}")
    if not images:
        print("No images found in drive_download_test/. Add images there and re-run.")
        return

    records: List[Dict[str, Any]] = [None] * total  # pre-allocate to preserve order
    records_lock = threading.Lock()
    csv_path   = "image_detection_analysis_report.csv"
    excel_path = "image_detection_analysis_report.xlsx"
    done_count = [0]

    def worker(args):
        idx_0based, img_path = args
        result = _process_single_image(idx_0based + 1, total, img_path)
        with records_lock:
            records[idx_0based] = result
            done_count[0] += 1
            # Incremental CSV save every 5 images or at the end
            if done_count[0] % 5 == 0 or done_count[0] == total:
                valid = [r for r in records if r is not None]
                try:
                    pd.DataFrame(valid).to_csv(csv_path, index=False)
                    print(f"  💾 Auto-saved {len(valid)}/{total} records to {csv_path}")
                except Exception as e:
                    print(f"  ⚠ Auto-save failed: {e}")
        return result

    # Run with thread pool — each thread handles one image concurrently
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(worker, (i, p)): i for i, p in enumerate(images)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                i = futures[future]
                print(f"  ✗ Error on image {images[i]}: {e}")

    # Final save — both CSV and Excel
    valid_records = [r for r in records if r is not None]
    df = pd.DataFrame(valid_records)

    try:
        df.to_excel(excel_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"image_detection_analysis_report_{ts}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Excel was locked — saved to: {excel_path}")

    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"image_detection_analysis_report_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV was locked — saved to: {csv_path}")

    print(f"\n✅ SUCCESS! Analyzed {len(valid_records)}/{total} images.")
    print(f"   Reports saved:\n   - {excel_path}\n   - {csv_path}")


if __name__ == "__main__":
    main()
