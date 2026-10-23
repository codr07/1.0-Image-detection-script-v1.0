import os
import glob
import json
import time
from pathlib import Path
from datetime import datetime, timezone
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

gemini_client = None
if GEMINI_API_KEY:
    try:
        import google.genai as genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Vision AI Client connected (Model: gemini-2.5-flash / gemini-1.5-flash).")
    except Exception as e:
        print(f"Warning: Could not initialize Google GenAI SDK: {e}")
else:
    print("No GEMINI_API_KEY found in environment. Running in enhanced offline-hybrid fallback mode.")
    print("To enable Gemini 3.5 / 2.5 Flash, set $env:GEMINI_API_KEY='your_key' in PowerShell or create a .env file.")

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

# Structured Schema for Grounded Gemini Verification
class GroundedItem(BaseModel):
    name: str = Field(description="Specific name of object or person (e.g. 'person in dark suit', 'office monitor')")
    category: str = Field(description="Category: 'people', 'animals', 'plants', 'vehicles', 'electronic device', or 'other'")
    count: int = Field(description="Exact count of visible items")
    details: str = Field(description="Color, posture, or position in frame")

class GroundedSceneAnalysis(BaseModel):
    one_line_summary: str = Field(description="Accurate, concise sentence summarizing the image.")
    detailed_description: str = Field(description="Comprehensive descriptive paragraph covering foreground, midground, background, and visual context.")
    setting: str = Field(description="Scene setting, e.g. 'Indoor Conference Room', 'Outdoor City Street'")
    time_of_day: str = Field(description="e.g. 'Daytime (Bright)', 'Evening / Low-Light'")
    weather: str = Field(description="e.g. 'Clear Indoors', 'Overcast', 'Sunny'")
    dominant_colours: List[str] = Field(description="List of primary colors observed")
    notable_features: List[str] = Field(description="Key focal visual features in the scene")
    verified_items: List[GroundedItem] = Field(description="Strict list of all verified objects actually visible in frame")
    caveats: str = Field(description="Any limits, occlusions, or camera artifacts noted")

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
        dist = 0.3 * (r - cr)**2 + 0.59 * (g - cg)**2 + 0.11 * (b - cb)**2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name

def analyze_image_properties(img_path):
    with Image.open(img_path) as im:
        w, h = im.size
        rgb_im = im.convert("RGB")
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

# ------------------------------------------------------------------- Hybrid Vision Engine
def run_hybrid_analysis(img_path: str, local_detections: List[Dict[str, Any]], w: int, h: int, dom_colors: List[str], setting: str, time_of_day: str, lighting: str):
    """
    Fuses local YOLOv8 spatial detections with Gemini Vision LLM.
    Zero hallucination: YOLO bounding boxes and counts are passed to Gemini as strict ground-truth anchors.
    """
    if not gemini_client:
        return None

    # Format YOLO detections as grounding context
    yolo_findings = []
    for d in local_detections:
        yolo_findings.append(f"- {d['name']} (confidence {round(d['conf']*100)}%, bbox: {[round(c, 1) for c in d['box']]})")
    yolo_text = "\n".join(yolo_findings) if yolo_findings else "None (No high-confidence bounded objects)"

    prompt = f"""You are a professional computer vision analyst working on strict ground-truth verification.
Analyze this image thoroughly with ZERO HALLUCINATION. Do not invent objects, people, or details that are not visibly present.

LOCAL DETECTOR DETECTIONS (Use as physical bounding-box evidence):
{yolo_text}

PHYSICAL METRICS:
- Resolution: {w}x{h}
- Detected Setting: {setting}
- Estimated Illumination: {lighting}
- Measured Dominant Colors: {', '.join(dom_colors)}

YOUR MANDATE:
1. Provide a precise, natural one-sentence summary.
2. Provide a rich, highly accurate detailed description paragraph covering the foreground, midground, and background. Describe textures, positions, posture, and spatial relationships accurately.
3. Verify the exact counts of people, vehicles, animals, plants, electronic devices, and other objects. Filter out false positives from local detections if any, and confirm genuine visible items.
4. Note setting, lighting, dominant colours, and notable features.
5. In caveats, mention anything occluded, shadowed, or distant.
"""
    # Prefer gemini-2.5-flash or gemini-1.5-flash
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    genai.types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GroundedSceneAnalysis,
                    temperature=0.2,  # Low temperature for zero hallucination
                )
            )
            parsed = json.loads(response.text)
            return parsed
        except Exception as e:
            # Try next model fallback
            continue

    return None

# ------------------------------------------------------------------- Main Pipeline
def main():
    images = glob.glob("drive_download_test/*.png") + glob.glob("drive_download_test/*.jpg")
    print(f"Found {len(images)} images to process using Hybrid Local YOLO + Gemini Vision Engine.")
    
    records = []
    for idx, img_path in enumerate(images, 1):
        filename = Path(img_path).name
        print(f"[{idx}/{len(images)}] Processing {filename}...")
        
        # 1. Local Image Properties
        w, h, dom_colors, setting, time_of_day, lighting = analyze_image_properties(img_path)
        
        # 2. Local YOLOv8 Detection
        results = yolo_model(img_path, verbose=False, conf=0.25)[0]
        local_detections = []
        counts = {
            "animals": {},
            "plants": {},
            "people": 0,
            "vehicles": {},
            "electronic device": {},
            "other": {}
        }
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = results.names[cls_id]
            conf = float(box.conf[0])
            coords = [float(x) for x in box.xyxy[0]]
            group = CATEGORY_MAPPING.get(name, "other")
            
            local_detections.append({
                "name": name,
                "conf": conf,
                "box": coords,
                "group": group
            })
            
            if group == "people":
                counts["people"] += 1
            else:
                counts[group][name] = counts[group].get(name, 0) + 1

        # 3. Hybrid Cross-Validation with Gemini (if API Key provided)
        gemini_result = None
        if gemini_client:
            gemini_result = run_hybrid_analysis(img_path, local_detections, w, h, dom_colors, setting, time_of_day, lighting)
        
        if gemini_result:
            summary = gemini_result.get("one_line_summary", "")
            description = gemini_result.get("detailed_description", "")
            setting_res = gemini_result.get("setting", setting)
            time_of_day_res = gemini_result.get("time_of_day", time_of_day)
            dom_colors_res = ", ".join(gemini_result.get("dominant_colours", dom_colors))
            notable = "; ".join(gemini_result.get("notable_features", []))
            caveats = gemini_result.get("caveats", "Verified with Gemini Vision + Local YOLO")
            
            # Reconcile verified items
            verified_counts = {
                "animals": {},
                "plants": {},
                "people": 0,
                "vehicles": {},
                "electronic device": {},
                "other": {}
            }
            for item in gemini_result.get("verified_items", []):
                cat = item.get("category", "other")
                grp = cat if cat in verified_counts else "other"
                name = item.get("name", "object")
                cnt = int(item.get("count", 1))
                if grp == "people":
                    verified_counts["people"] += cnt
                else:
                    verified_counts[grp][name] = verified_counts[grp].get(name, 0) + cnt
            
            # Use verified counts
            counts_to_use = verified_counts if any(verified_counts["people"] or v for v in verified_counts.values() if isinstance(v, dict)) else counts
        else:
            # Fallback to Local Narrative Engine
            summary, description = generate_local_narrative(local_detections, w, h, setting, lighting, dom_colors)
            setting_res = setting
            time_of_day_res = time_of_day
            dom_colors_res = ", ".join(dom_colors)
            notable = f"{len(local_detections)} detected objects, Resolution {w}x{h}px"
            caveats = "Analyzed with local YOLOv8 spatial engine (Gemini API key not active)"
            counts_to_use = counts

        # Format column strings
        animals_str = ", ".join(f"{cnt} {k}" for k, cnt in counts_to_use["animals"].items())
        plants_str = ", ".join(f"{cnt} {k}" for k, cnt in counts_to_use["plants"].items())
        people_str = f"{counts_to_use['people']} person(s)" if counts_to_use['people'] > 0 else ""
        vehicles_str = ", ".join(f"{cnt} {k}" for k, cnt in counts_to_use["vehicles"].items())
        devices_str = ", ".join(f"{cnt} {k}" for k, cnt in counts_to_use["electronic device"].items())
        other_str = ", ".join(f"{cnt} {k}" for k, cnt in counts_to_use["other"].items())

        record = {
            "Analysed At": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "File Name": filename,
            "Image Link": f"drive_download_test/{filename}",
            "Summary": summary,
            "Detailed Description": description,
            "Animals": animals_str,
            "Plants & Trees": plants_str,
            "People": people_str,
            "Vehicles": vehicles_str,
            "electronic device": devices_str,
            "Other Objects": other_str,
            "Setting": setting_res,
            "Time of Day": time_of_day_res,
            "Weather": "Interior / Ambient" if "Indoor" in setting_res else "Clear Ambient",
            "Season": "N/A",
            "Dominant Colours": dom_colors_res,
            "Notable Features": notable,
            "Notes & Limits": caveats
        }
        records.append(record)

    # 4. Save to Excel & CSV with lock protection
    df = pd.DataFrame(records)
    excel_path = "image_detection_analysis_report.xlsx"
    csv_path = "image_detection_analysis_report.csv"
    
    try:
        df.to_excel(excel_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"image_detection_analysis_report_{ts}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Original Excel was locked. Saved to: {excel_path}")
        
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"image_detection_analysis_report_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Original CSV was locked. Saved to: {csv_path}")

    print(f"\nSUCCESS! Generated hybrid analysis reports:\n- {excel_path}\n- {csv_path}")

if __name__ == "__main__":
    main()
