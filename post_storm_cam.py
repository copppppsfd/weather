"""
Builds a zoomed-in looping GIF centered on the strongest actively-tracked
storm/typhoon candidate, using the position/size data analyze_himawari.py
leaves behind in storm_history.json -- instead of animating the whole
sector, this crops down to just the storm.

Depends on the storm-scan workflow having run recently and having a
candidate tracked for at least STORM_CAM_MIN_TRACKED_SCANS scans (default
2), so this doesn't build a zoomed loop around a one-off noise blob.

Note: the crop box is centered on the candidate's *most recent known*
position and stays fixed across the whole loop (it doesn't re-center per
frame). Over a ~2 hour window with generous crop padding this is normally
fine, but a fast-moving storm could drift toward the edge of frame by the
end of the loop.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone

import requests
from PIL import Image

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")
BAND = os.environ.get("HIMAWARI_BAND", "b13")
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Optional full URL override, same as the other scripts (use {HHMM}).
URL_TEMPLATE = os.environ.get("HIMAWARI_URL_TEMPLATE")

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"
HISTORY_FILE = "storm_history.json"

FRAME_COUNT = int(os.environ.get("STORM_CAM_FRAME_COUNT", "12"))            # ~2 hours at 10min/frame
FRAME_DURATION_MS = int(os.environ.get("STORM_CAM_FRAME_DURATION_MS", "150"))
MIN_TRACKED_SCANS = int(os.environ.get("STORM_CAM_MIN_TRACKED_SCANS", "2"))  # ignore one-off blips
CROP_RADIUS_MULTIPLIER = float(os.environ.get("STORM_CAM_CROP_RADIUS_MULTIPLIER", "6"))
MIN_CROP_HALF_PX = int(os.environ.get("STORM_CAM_MIN_CROP_HALF_PX", "80"))
MAX_CROP_FRAC = float(os.environ.get("STORM_CAM_MAX_CROP_FRAC", "0.4"))      # cap crop size vs. frame
OUTPUT_SIZE = int(os.environ.get("STORM_CAM_OUTPUT_SIZE", "400"))

MAX_FILE_BYTES = 9_500_000  # stay safely under Discord's free-tier 10MB cap


def round_down_to_10min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def url_for_slot(slot_time: datetime) -> str:
    hhmm = slot_time.strftime("%H%M")
    if URL_TEMPLATE:
        return URL_TEMPLATE.format(HHMM=hhmm)
    return f"{BASE_URL}/{SECTOR}/{SECTOR}_{BAND}_{hhmm}.jpg"


def product_id() -> str:
    if URL_TEMPLATE:
        return URL_TEMPLATE.rstrip("/").split("/")[-1].replace("{HHMM}", "HHMM")
    return f"{SECTOR}_{BAND}"


def load_best_candidate():
    """Pick the strongest actively-tracked candidate from the storm
    scanner's last saved history (candidates are saved strongest-first)."""
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    tracked = [c for c in history.get("candidates", []) if c.get("tracked_scans", 1) >= MIN_TRACKED_SCANS]
    return tracked[0] if tracked else None


def fetch_frame(slot_time: datetime):
    url = url_for_slot(slot_time)
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None
    print(f"GET {url} -> {resp.status_code}")
    return resp.content if resp.status_code == 200 else None


def collect_frames():
    """Fetch up to FRAME_COUNT most recent available frames, oldest first."""
    now = round_down_to_10min(datetime.now(timezone.utc))
    frames = []
    lookback = FRAME_COUNT + 6
    for i in range(lookback):
        slot_time = now - timedelta(minutes=10 * i)
        data = fetch_frame(slot_time)
        if data:
            frames.append((slot_time, data))
        if len(frames) >= FRAME_COUNT:
            break
    frames.sort(key=lambda f: f[0])
    return frames


def crop_box_for(candidate, img_w, img_h):
    radius = candidate.get("radius", MIN_CROP_HALF_PX / CROP_RADIUS_MULTIPLIER)
    half = max(radius * CROP_RADIUS_MULTIPLIER, MIN_CROP_HALF_PX)
    half = min(half, MAX_CROP_FRAC * min(img_w, img_h) / 2)

    cx, cy = candidate["cx"], candidate["cy"]
    left, right = cx - half, cx + half
    top, bottom = cy - half, cy + half

    # Shift the box back inside bounds instead of shrinking it, so every
    # frame in the loop uses the exact same crop size.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= (right - img_w)
        right = img_w
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h
    left, top = max(left, 0), max(top, 0)

    return int(left), int(top), int(right), int(bottom)


def build_gif(frames, crop_box):
    images = []
    for _, data in frames:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        cropped = img.crop(crop_box)
        resized = cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
        images.append(resized)

    buf = io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True, append_images=images[1:],
        duration=FRAME_DURATION_MS, loop=0, optimize=True,
    )
    return buf.getvalue()


def post_gif(gif_bytes, start_label, end_label, frame_count, candidate):
    pid = product_id()
    filename = f"{pid}_storm_cam.gif"
    files = {"file": (filename, gif_bytes, "image/gif")}
    payload = {
        "content": (
            f"**Storm-cam loop** \u2014 product `{pid}` \u00b7 {start_label} \u2192 {end_label} UTC "
            f"({frame_count} frames)\nTracking a candidate tracked for "
            f"{candidate.get('tracked_scans', 1)} scans (size {candidate.get('area_frac', 0)*100:.2f}% of frame)."
        )
    }
    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=60)
    resp.raise_for_status()


def main():
    candidate = load_best_candidate()
    if candidate is None:
        print(f"No candidate tracked for >= {MIN_TRACKED_SCANS} scans yet, skipping storm-cam.")
        return

    print(
        f"Building storm-cam around candidate at ({candidate['cx']:.0f},{candidate['cy']:.0f}), "
        f"tracked {candidate.get('tracked_scans', 1)} scans"
    )

    frames = collect_frames()
    if len(frames) < 2:
        print(f"Only found {len(frames)} frame(s), need at least 2 for an animation.")
        return

    first_img = Image.open(io.BytesIO(frames[0][1]))
    img_w, img_h = first_img.size
    crop_box = crop_box_for(candidate, img_w, img_h)

    gif_bytes = build_gif(frames, crop_box)
    if len(gif_bytes) > MAX_FILE_BYTES and len(frames) > 4:
        print(f"GIF too large ({len(gif_bytes)} bytes), dropping every other frame and retrying.")
        frames = frames[::2]
        gif_bytes = build_gif(frames, crop_box)

    if len(gif_bytes) > MAX_FILE_BYTES:
        print(f"Still too large ({len(gif_bytes)} bytes) after reducing frames, skipping post.")
        return

    start_label = frames[0][0].strftime("%H:%M")
    end_label = frames[-1][0].strftime("%H:%M")
    post_gif(gif_bytes, start_label, end_label, len(frames), candidate)
    print(f"Posted storm-cam loop with {len(frames)} frame(s), {len(gif_bytes)} bytes.")


if __name__ == "__main__":
    main()
