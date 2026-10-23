"""Image Detective - describes an image in detail and counts everything in it.

Two entry points:
  POST /         manual use: upload a photo or paste an image link, read the report.
  POST /webhook  Google Drive 'file created' event: analyse the image and append a
                 row to the Image Reports spreadsheet.
"""

import asyncio
import base64
import io
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urljoin

import httpx
from codewords_client import AsyncCodewordsClient, logger, redis_client, run_service
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- configuration
WATCH_FOLDER_ID = "1iUZzGVeELjcq_4vudAVZt5DDMJMtoDG8"
SPREADSHEET_ID = "1nHSi2L_2Q2guOm0PpYUBV35cQpayciOcUDA7I6ZoBCo"
SHEET_RANGE = "Sheet1!A1"

USER_TZ = timezone(timedelta(hours=5, minutes=30))  # Asia/Calcutta
VISION_MODEL = "gemini-3.5-flash"
MAX_IMAGE_DIM = 1600
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
SEEN_TTL_SECONDS = 60 * 60 * 24 * 30

CATEGORY_LABELS = {
    "animal": "animals",
    "plant": "plants",
    "person": "people",
    "vehicle": "vehicles",
    "object": "objects",
    "building": "buildings",
    "other": "other",
}

BATCH_CONCURRENCY = 4
SUMMARY_TAB = "Folder Summary"
SUMMARY_HEADERS = [
    "Run At",
    "Folder",
    "Images Analysed",
    "Already Done",
    "Failed",
    "Animals",
    "Plants & Trees",
    "People",
    "Vehicles",
    "Other Objects",
    "Most Common",
    "Notes",
]
FOLDER_ID_PATTERN = re.compile(r"/folders/([A-Za-z0-9_-]{10,})")
FILE_ID_IN_LINK = re.compile(r"/d/([A-Za-z0-9_-]{10,})")

# ---------------------------------------------------------------------- models

class Subject(BaseModel):
    """One distinct type of thing found in the image."""

    name: str = Field(description="Specific name, e.g. 'oak tree', 'white-tailed deer', 'person in a red jacket'.")
    category: Literal["animal", "plant", "person", "vehicle", "building", "object", "other"] = Field(
        description="Which family this belongs to."
    )
    count: int = Field(description="How many of this type are visible.")
    count_certainty: Literal["exact", "approximate"] = Field(
        description="'exact' only when reliably countable, otherwise 'approximate'."
    )
    details: list[str] = Field(description="Distinguishing features: size, colour, condition, position.")


class Scene(BaseModel):
    """The full structured reading of one image."""

    one_line_summary: str = Field(description="One sentence describing what the image shows.")
    detailed_description: str = Field(description="A full paragraph covering the whole frame, foreground to background.")
    setting: str
    time_of_day: str
    weather: str
    season: str
    subjects: list[Subject]
    dominant_colours: list[str]
    notable_features: list[str]
    text_visible: list[str]
    caveats: list[str] = Field(description="What could not be judged reliably, e.g. occluded or distant objects.")


class AnalyzeRequest(BaseModel):
    image: str = Field(
        default="",
        description="Upload a photo to analyse (JPEG, PNG or WEBP).",
        json_schema_extra={"contentMediaType": "image/*"},
    )
    image_link: str = Field(
        default="",
        description="Or paste a link to an image instead of uploading one.",
        json_schema_extra={"example": "https://example.com/photo.jpg"},
    )
    folder_link: str = Field(
        default="",
        description="Or paste a Google Drive folder link to report on every photo inside it.",
        json_schema_extra={"example": "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUv"},
    )
    max_images: int = Field(
        default=100,
        ge=1,
        le=300,
        description="How many photos to take from that folder in one run.",
    )


class AnalyzeResponse(BaseModel):
    report: str = Field(
        description="The full image report, in markdown.",
        json_schema_extra={"contentMediaType": "text/markdown"},
    )
    one_line_summary: str = Field(description="One sentence describing the image.")
    counts: dict[str, str] = Field(description="How many of each type of thing were found, grouped by family.")


# ------------------------------------------------------------------- app setup
app = FastAPI(
    title="Image Detective",
    description="Describe an image in detail and count everything visible in it.",
    version="1.0.0",
)

_composio_client = None


TRANSIENT_ERROR_MARKERS = (
    "quota exceeded",
    "rate limit",
    "too many requests",
    "429",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


def looks_transient(message: str) -> bool:
    """True for errors worth retrying (Google throttling and network blips)."""
    lowered = message.lower()
    return any(marker in lowered for marker in TRANSIENT_ERROR_MARKERS)


async def _call_google(tool_slug: str, arguments: dict) -> dict:
    """One call to the platform's Composio service."""
    async with AsyncCodewordsClient() as client:
        response = await client.run(
            service_id="composio",
            inputs={"tool_slug": tool_slug, "arguments": arguments},
        )
        response.raise_for_status()
        payload = response.json()

    if isinstance(payload, dict) and isinstance(payload.get("output"), dict):
        payload = payload["output"]
    return payload if isinstance(payload, dict) else {}


async def execute_tool(tool_slug: str, arguments: dict) -> dict:
    """Run one Google action, backing off when Google throttles us."""
    for attempt in range(4):
        last = attempt == 3
        try:
            payload = await _call_google(tool_slug, arguments)
        except HTTPException:
            raise
        except Exception as exc:
            if last or not looks_transient(str(exc)):
                logger.error("Google call failed", tool=tool_slug, error=str(exc))
                raise HTTPException(status_code=502, detail="Google could not be reached, please try again.") from exc
            wait = 5 * (2 ** attempt)
            logger.warning("Google call failed, retrying", tool=tool_slug, wait_seconds=wait)
            await asyncio.sleep(wait)
            continue

        if payload.get("successful") is False:
            error_text = str(payload.get("error") or "unknown error")
            if not last and looks_transient(error_text):
                wait = 5 * (2 ** attempt)
                logger.warning("Google is throttling, retrying", tool=tool_slug, wait_seconds=wait)
                await asyncio.sleep(wait)
                continue
            logger.error("Google action failed", tool=tool_slug, error=error_text)
            raise HTTPException(status_code=502, detail=f"Google request failed: {error_text}")

        return payload.get("data") or payload

    raise HTTPException(status_code=502, detail="Google is rate limiting this request, please try again shortly.")


# ------------------------------------------------------------------ image input

async def fetch_bytes(url: str, label: str) -> bytes:
    """Download an image into memory."""
    if not url or not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Please upload an image or paste a link to one.")

    try:
        async with httpx.AsyncClient(
            timeout=90,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CodeWords-ImageDetective/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Image download failed", label=label, error=str(exc))
        raise HTTPException(status_code=502, detail="That image could not be downloaded.") from exc

    content = response.content
    if not content:
        raise HTTPException(status_code=400, detail="The image file was empty.")
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That image is too large (30 MB limit).")
    return content


def prepare_image(raw: bytes) -> tuple[str, str]:
    """Normalise any image to a base64 JPEG the vision model accepts."""
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:
        logger.warning("Unreadable image", error=str(exc))
        raise HTTPException(
            status_code=415,
            detail="That file is not an image this tool can read (JPEG, PNG and WEBP work best).",
        ) from exc

    width, height = image.size
    if max(width, height) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"


async def fetch_drive_image(file_id: str) -> tuple[bytes, str]:
    """Download a Drive file by id; returns bytes and file name."""
    data = await execute_tool("GOOGLEDRIVE_DOWNLOAD_FILE", {"file_id": file_id})
    content = data.get("downloaded_file_content") or {}
    s3_url = content.get("s3url") or ""
    name = data.get("name") or content.get("name") or file_id
    if not s3_url:
        raise HTTPException(status_code=502, detail="Google Drive did not return the image.")
    raw = await fetch_bytes(s3_url, label=f"drive:{file_id}")
    return raw, name


# ------------------------------------------------------------------- analysis

async def analyse_image(raw: bytes, label: str) -> Scene:
    """Ask the vision model for a full, counted reading of the image."""
    encoded, mime = await asyncio.to_thread(prepare_image, raw)

    client = AsyncOpenAI(base_url=urljoin(os.environ["CODEWORDS_RUNTIME_URI"], "run/gemini/v1"))
    prompt = (
        "Analyse this image and produce a thorough report.\n"
        "1. Describe the scene in one sentence, then in a detailed paragraph covering foreground, "
        "middle ground and background.\n"
        "2. List every distinct type of animal, plant or tree, person, vehicle, building and object "
        "you can see. For each type give a count - exact when you can count them reliably, "
        "otherwise your best estimate marked approximate. Never omit a type because it is numerous.\n"
        "3. Name species, materials and object types wherever you can.\n"
        "4. Note the setting, time of day, weather, season, dominant colours, anything unusual, "
        "and any text visible in the frame.\n"
        "5. Say in caveats whatever you could not judge reliably."
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous image analyst. Nothing in the frame is ignored, "
                        "but you never invent detail you cannot actually see."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
            response_format=Scene,
        )
    except Exception as exc:
        logger.error("Vision analysis failed", label=label, error=str(exc))
        raise HTTPException(status_code=502, detail="The image could not be analysed, please try again.") from exc

    scene = response.choices[0].message.parsed
    if scene is None:
        raise HTTPException(status_code=502, detail="The image could not be analysed, please try again.")

    usage = getattr(response, "usage", None)
    logger.info(
        "Image analysed",
        label=label,
        subjects=len(scene.subjects),
        tokens=getattr(usage, "total_tokens", None),
    )
    return scene


# ------------------------------------------------------------------ reporting

def format_count(subject: Subject) -> str:
    """e.g. '2 x white-tailed deer (estimated)'."""
    marker = " (estimated)" if subject.count_certainty == "approximate" else ""
    return f"{subject.count} x {subject.name}{marker}"


def build_counts(scene: Scene) -> dict[str, str]:
    """Group the findings by family into one readable line each."""
    grouped: dict[str, list[str]] = {}
    for subject in scene.subjects:
        label = CATEGORY_LABELS.get(subject.category, "objects")
        grouped.setdefault(label, []).append(format_count(subject))
    return {label: ", ".join(items) for label, items in grouped.items()}


def build_report(scene: Scene) -> str:
    """Render the report as markdown for the manual page."""
    lines = [f"## {scene.one_line_summary}", "", scene.detailed_description, "", "### What is in the image"]
    if scene.subjects:
        for subject in scene.subjects:
            detail = "; ".join(subject.details)
            lines.append(f"- **{format_count(subject)}** - {detail}" if detail else f"- **{format_count(subject)}**")
    else:
        lines.append("- No distinct objects were identified.")

    lines += [
        "",
        "### Context",
        f"- **Setting:** {scene.setting}",
        f"- **Time of day:** {scene.time_of_day}",
        f"- **Weather:** {scene.weather}",
        f"- **Season:** {scene.season}",
        f"- **Dominant colours:** {', '.join(scene.dominant_colours) or 'not clear'}",
    ]

    if scene.notable_features:
        lines += ["", "### Notable features"] + [f"- {item}" for item in scene.notable_features]
    if scene.text_visible:
        lines += ["", "### Text in the image"] + [f"- {item}" for item in scene.text_visible]
    if scene.caveats:
        lines += ["", "### Notes and limits"] + [f"- {item}" for item in scene.caveats]
    return "\n".join(lines)


def build_row(scene: Scene, file_name: str, image_link: str) -> list[str]:
    """One spreadsheet row, matching the header order."""
    counts = build_counts(scene)
    other = ", ".join(
        part for part in (counts.get("objects", ""), counts.get("buildings", ""), counts.get("other", "")) if part
    )
    features = list(scene.notable_features)
    if scene.text_visible:
        features.append("Text: " + "; ".join(scene.text_visible))

    return [
        datetime.now(USER_TZ).strftime("%Y-%m-%d %H:%M"),
        file_name,
        image_link,
        scene.one_line_summary,
        scene.detailed_description,
        counts.get("animals", ""),
        counts.get("plants", ""),
        counts.get("people", ""),
        counts.get("vehicles", ""),
        other,
        scene.setting,
        scene.time_of_day,
        scene.weather,
        scene.season,
        ", ".join(scene.dominant_colours),
        "\n".join(features),
        "\n".join(scene.caveats),
    ]


async def append_row(values: list[str]) -> None:
    """Append one row to the Image Reports spreadsheet."""
    await execute_tool(
        "GOOGLESHEETS_SPREADSHEETS_VALUES_APPEND",
        {
            "spreadsheetId": SPREADSHEET_ID,
            "range": SHEET_RANGE,
            "valueInputOption": "USER_ENTERED",
            "values": [values],
        },
    )


# --------------------------------------------------------------------- dedupe

async def already_processed(file_id: str) -> bool:
    """True when this Drive file has already been analysed successfully."""
    try:
        async with redis_client() as (redis, ns):
            return bool(await redis.get(f"{ns}:seen:{file_id}"))
    except Exception as exc:  # dedupe is a convenience, never a hard failure
        logger.warning("Duplicate check unavailable", error=str(exc))
        return False


async def mark_processed(file_id: str) -> None:
    """Record a successful analysis so the same photo is not repeated."""
    try:
        async with redis_client() as (redis, ns):
            await redis.set(f"{ns}:seen:{file_id}", "1", ex=SEEN_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Could not record processed file", error=str(exc))


async def processed_file_ids() -> set[str]:
    """File ids that already have a row in the spreadsheet."""
    try:
        data = await execute_tool(
            "GOOGLESHEETS_BATCH_GET",
            {"spreadsheet_id": SPREADSHEET_ID, "ranges": ["Sheet1!C2:C10000"]},
        )
        value_ranges = data.get("valueRanges") or []
        rows = (value_ranges[0].get("values") if value_ranges else None) or []
    except Exception as exc:
        logger.warning("Could not read existing rows, continuing", error=str(exc))
        return set()

    ids: set[str] = set()
    for row in rows:
        match = FILE_ID_IN_LINK.search(str(row[0])) if row else None
        if match:
            ids.add(match.group(1))
    return ids


def extract_file(payload: dict) -> dict:
    """Find the Drive file record inside any webhook wrapper shape."""
    node = payload
    for _ in range(5):
        if not isinstance(node, dict):
            break
        if isinstance(node.get("file"), dict):
            return node["file"]
        if node.get("id") and node.get("mimeType"):
            return node
        nested = None
        for key in ("payload", "data", "body", "event"):
            if isinstance(node.get(key), dict):
                nested = node[key]
                break
        if nested is None:
            break
        node = nested
    return {}


# ------------------------------------------------------------- folder reports

def parse_folder_id(link: str) -> str:
    """Pull the folder id out of a Drive link, or accept a bare id."""
    match = FOLDER_ID_PATTERN.search(link)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", link):
        return link
    raise HTTPException(
        status_code=400,
        detail="That does not look like a Drive folder link. Copy the link from the folder in Google Drive.",
    )


async def list_folder_images(folder_id: str, limit: int) -> list[dict]:
    """Every image directly inside the folder, newest first."""
    found: list[dict] = []
    page_token = None

    while len(found) < limit:
        arguments = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,createdTime)",
            "pageSize": min(1000, max(limit, 100)),
            "orderBy": "createdTime desc",
        }
        if page_token:
            arguments["pageToken"] = page_token
        data = await execute_tool("GOOGLEDRIVE_FIND_FILE", arguments)
        batch = data.get("files") or []
        found.extend(
            item for item in batch if str(item.get("mimeType", "")).lower().startswith("image/")
        )
        page_token = data.get("nextPageToken")
        if not page_token or not batch:
            break

    return found[:limit]


async def append_row_to(values: list[str], target_range: str) -> None:
    """Append one row to a given tab."""
    await execute_tool(
        "GOOGLESHEETS_SPREADSHEETS_VALUES_APPEND",
        {
            "spreadsheetId": SPREADSHEET_ID,
            "range": target_range,
            "valueInputOption": "USER_ENTERED",
            "values": [values],
        },
    )


async def write_folder_summary(values: list[str]) -> None:
    """Write the run summary into its own tab, creating the tab and header on first use."""
    target = f"'{SUMMARY_TAB}'!A1"
    headers_present = False

    try:
        data = await execute_tool(
            "GOOGLESHEETS_BATCH_GET",
            {"spreadsheet_id": SPREADSHEET_ID, "ranges": [target]},
        )
        value_ranges = data.get("valueRanges") or []
        rows = (value_ranges[0].get("values") if value_ranges else None) or []
        headers_present = bool(rows)
    except HTTPException:
        logger.info("Creating the folder summary tab")
        await execute_tool(
            "GOOGLESHEETS_ADD_SHEET",
            {
                "spreadsheet_id": SPREADSHEET_ID,
                "title": SUMMARY_TAB,
                "properties": {"title": SUMMARY_TAB},
            },
        )

    if not headers_present:
        await append_row_to(SUMMARY_HEADERS, target)
    await append_row_to(values, target)


def summarise_findings(results: list[dict]) -> dict:
    """Total each family of finding across every analysed photo."""
    totals: dict[str, int] = {}
    images_with: dict[str, int] = {}

    for result in results:
        labels_seen: set[str] = set()
        for finding in result.get("findings") or []:
            label = CATEGORY_LABELS.get(str(finding.get("category") or "other"), "other")
            totals[label] = totals.get(label, 0) + int(finding.get("count") or 0)
            labels_seen.add(label)
        for label in labels_seen:
            images_with[label] = images_with.get(label, 0) + 1

    return {"totals": totals, "images_with": images_with}


def most_common_findings(results: list[dict], limit: int = 12) -> list[tuple[str, int]]:
    """The named things that appeared most often across the folder."""
    counts: dict[str, int] = {}
    for result in results:
        for finding in result.get("findings") or []:
            name = str(finding.get("name") or "unidentified").strip()
            counts[name] = counts.get(name, 0) + int(finding.get("count") or 0)
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


async def run_folder_batch(folder_link: str, max_images: int) -> AnalyzeResponse:
    """Analyse every photo in a Drive folder and report on the folder as a whole."""
    folder_id = parse_folder_id(folder_link)
    logger.info("STEPLOG START step1")

    images = await list_folder_images(folder_id, max_images)
    if not images:
        raise HTTPException(status_code=404, detail="No photos were found in that folder.")

    already_ids = await processed_file_ids()
    already_done = sum(1 for item in images if str(item.get("id")) in already_ids)
    images = [item for item in images if str(item.get("id")) not in already_ids]

    logger.info("Folder run started", photos=len(images), skipped=already_done, folder_id=folder_id)
    started = datetime.now(USER_TZ)
    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def process_one(item: dict) -> dict:
        async with semaphore:
            try:
                return await process_drive_file(item, use_cache=False)
            except Exception as exc:
                logger.error("Folder photo failed", file_name=item.get("name"), error=str(exc))
                return {
                    "status": "error",
                    "file_name": item.get("name") or item.get("id") or "unknown",
                    "reason": str(exc),
                    "findings": [],
                }

    logger.info("STEPLOG START step2")
    logger.info("STEPLOG START step3")
    results = await asyncio.gather(*(process_one(item) for item in images))
    logger.info("STEPLOG START step4")

    analysed = [item for item in results if item.get("status") == "analysed"]
    skipped = [item for item in results if item.get("status") == "skipped"]
    failed = [item for item in results if item.get("status") == "error"]
    already_total = already_done + len(skipped)

    totals = summarise_findings(analysed)
    common = most_common_findings(analysed)
    common_text = ", ".join(f"{count} x {name}" for name, count in common) or "none"

    try:
        await write_folder_summary(
            [
                started.strftime("%Y-%m-%d %H:%M"),
                f"https://drive.google.com/drive/folders/{folder_id}",
                str(len(analysed)),
                str(already_total),
                str(len(failed)),
                *[str(totals["totals"].get(label, 0)) for label in
                  ("animals", "plants", "people", "vehicles", "objects")],
                common_text,
                f"{already_total} already analysed, {len(failed)} failed",
            ]
        )
    except Exception as exc:
        logger.warning("Folder summary not written", error=str(exc))

    lines = [
        f"## Folder report - {len(analysed)} of {len(images) + already_done} photos analysed",
        "",
        f"Folder: https://drive.google.com/drive/folders/{folder_id}",
        f"Already analysed earlier: {already_total}. Failed: {len(failed)}.",
        "",
        "### Totals across the folder",
    ]
    if totals["totals"]:
        for label, count in totals["totals"].items():
            lines.append(
                f"- **{label.title()}:** {count} across {totals['images_with'].get(label, 0)} photos"
            )
    else:
        lines.append("- Nothing countable was identified.")

    if common:
        lines += ["", "### Most common findings"]
        lines += [f"- {count} x {name}" for name, count in common]

    if analysed:
        lines += ["", "### Photo by photo"]
        for result in analysed:
            counts = result.get("counts") or {}
            detail = "; ".join(f"{label}: {value}" for label, value in counts.items()) or "no distinct objects"
            lines.append(f"- **{result.get('file_name')}** - {result.get('summary', '')} ({detail})")

    if failed:
        lines += ["", "### Could not be processed"]
        lines += [f"- {item.get('file_name')}: {str(item.get('reason', ''))[:160]}" for item in failed]

    return AnalyzeResponse(
        report="\n".join(lines),
        one_line_summary=f"{len(analysed)} photos analysed in one folder run",
        counts={
            label: f"{count} across {totals['images_with'].get(label, 0)} photos"
            for label, count in totals["totals"].items()
        },
    )


# ------------------------------------------------------------------- endpoints

@app.post("/", response_model=AnalyzeResponse)
async def analyse_endpoint(request: AnalyzeRequest, http_request: Request):
    """
    Analyse one image and return a detailed report.

    Upload a photo, or paste a link to an image. The report names everything in the frame
    and counts each type, marking counts that are estimates.
    """
    folder_link = (request.folder_link or "").strip()
    if folder_link:
        return await run_folder_batch(folder_link, request.max_images)

    source = (request.image or "").strip() or (request.image_link or "").strip()

    if not source:
        # The same URL also receives Google Drive file events, so handle them here too.
        try:
            body = await http_request.json()
        except Exception:
            body = {}
        drive_file = extract_file(body if isinstance(body, dict) else {})
        if drive_file.get("id"):
            outcome = await process_drive_file(drive_file)
            summary = outcome.get("summary") or outcome.get("reason") or "No report produced."
            return AnalyzeResponse(
                report=f"**{outcome.get('status', 'done')}** - {outcome.get('file_name', '')}\n\n{summary}",
                one_line_summary=summary,
                counts=outcome.get("counts") or {},
            )
        raise HTTPException(status_code=400, detail="Please upload an image or paste a link to one.")

    logger.info("Manual analysis requested", has_upload=bool((request.image or "").strip()))
    logger.info("STEPLOG START step2")
    raw = await fetch_bytes(source, label="manual")
    logger.info("STEPLOG START step3")
    scene = await analyse_image(raw, "manual upload")

    return AnalyzeResponse(
        report=build_report(scene),
        one_line_summary=scene.one_line_summary,
        counts=build_counts(scene),
    )


@app.post("/webhook")
async def drive_webhook(request: Request):
    """
    Google Drive 'file created' event handler.

    Images are analysed and written to the Image Reports spreadsheet; anything else is skipped.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    return await process_drive_file(extract_file(payload if isinstance(payload, dict) else {}))


async def process_drive_file(file: dict, use_cache: bool = True) -> dict:
    """Analyse one Drive image and file the report row."""
    logger.info("STEPLOG START step1")
    file_id = str(file.get("id") or "").strip()
    if not file_id:
        logger.error("Webhook received without a Drive file id")
        raise HTTPException(status_code=400, detail="No Drive file in the payload.")

    file_name = file.get("name") or file_id
    mime_type = str(file.get("mimeType") or "").lower()

    if mime_type and (not mime_type.startswith("image/") or mime_type == "image/svg+xml"):
        logger.info("Skipped non-image file", file_name=file_name, mime_type=mime_type)
        return {"status": "skipped", "reason": "not a photo", "file_name": file_name, "findings": []}

    if use_cache and await already_processed(file_id):
        logger.info("Skipped duplicate file", file_name=file_name)
        return {"status": "skipped", "reason": "already analysed", "file_name": file_name, "findings": []}

    try:
        logger.info("STEPLOG START step2")
        raw, resolved_name = await fetch_drive_image(file_id)
        logger.info("STEPLOG START step3")
        scene = await analyse_image(raw, resolved_name)
        logger.info("STEPLOG START step4")
        await append_row(build_row(scene, resolved_name, f"https://drive.google.com/file/d/{file_id}/view"))
        await mark_processed(file_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Drive image processing failed", file_name=file_name, error=str(exc))
        raise HTTPException(status_code=500, detail="The image could not be processed.") from exc

    logger.info("Drive image filed", file_name=resolved_name, subjects=len(scene.subjects))
    return {
        "status": "analysed",
        "file_name": resolved_name,
        "summary": scene.one_line_summary,
        "counts": build_counts(scene),
        "findings": [
            {
                "name": subject.name,
                "category": subject.category,
                "count": subject.count,
                "certainty": subject.count_certainty,
            }
            for subject in scene.subjects
        ],
    }


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "watching_folder": WATCH_FOLDER_ID}


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    port = int(os.environ.get("PORT", 2026))
    loop = "asyncio" if sys.platform.startswith("win") else "uvloop"
    run_service(app, port=port, loop=loop)

