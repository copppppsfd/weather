"""
Builds a short looping GIF from the last several Himawari frames and posts
it to Discord via webhook (same webhook as post_himawari.py — no bot needed).

Unlike post_himawari.py (posts one new still frame whenever available), this
grabs a recent window of frames and stitches them into an animation, so it's
meant to run less often (e.g. every few hours) on its own schedule.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import io
import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
from PIL import Image

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")   # r2w = Southeast Asia (extended, large)
BAND = os.environ.get("HIMAWARI_BAND", "hrp")        # b13 = infrared (day & night)
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Optional: fully override the URL pattern instead of using SECTOR/BAND.
# Use {HHMM} where the 4-digit UTC time slot (e.g. 2330) should go.
# Example (ASWind product, a different directory/naming scheme entirely):
#   HIMAWARI_URL_TEMPLATE=https://www.data.jma.go.jp/mscweb/en/product/data/fs/aswind_fsir_{HHMM}.png
URL_TEMPLATE = os.environ.get("HIMAWARI_URL_TEMPLATE")

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"

FRAME_COUNT = int(os.environ.get("ANIMATION_FRAME_COUNT", "12"))         # 12 * 10min = ~2 hours
FRAME_DURATION_MS = int(os.environ.get("ANIMATION_FRAME_DURATION_MS", "150"))
MAX_WIDTH = int(os.environ.get("ANIMATION_MAX_WIDTH", "600"))            # downscale to control file size

# Discord's free-tier per-file cap is 10MB (decimal). Stay safely under it;
# bump this up if your server has boosts or everyone posting has Nitro.
MAX_FILE_BYTES = 9_500_000


def round_down_to_10min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def url_for_slot(slot_time: datetime) -> str:
    hhmm = slot_time.strftime("%H%M")
    if URL_TEMPLATE:
        return URL_TEMPLATE.format(HHMM=hhmm)
    return f"{BASE_URL}/{SECTOR}/{SECTOR}_{BAND}_{hhmm}.jpg"


def product_id() -> str:
    if URL_TEMPLATE:
        name = URL_TEMPLATE.rstrip("/").split("/")[-1]
        return name.replace("{HHMM}", "HHMM")
    return f"{SECTOR}_{BAND}"


def fetch_frame(slot_time: datetime):
    url = url_for_slot(slot_time)
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None
        
    print(f"GET {url} -> {resp.status_code}")
    
    if resp.status_code == 200:
        # Check the server's timestamp for this file to avoid yesterday's data
        last_mod_header = resp.headers.get("Last-Modified")
        if last_mod_header:
            try:
                # Convert the HTTP header timestamp to a timezone-aware datetime object
                last_mod_dt = parsedate_to_datetime(last_mod_header)
                
                # If the image on the server is more than 6 hours older than right now,
                # it is yesterday's file still waiting to be overwritten.
                age = datetime.now(timezone.utc) - last_mod_dt
                if age > timedelta(hours=6):
                    print(f"   -> Skipping: Stale image from {age.total_seconds()/3600:.1f} hours ago.")
                    return None
            except Exception as e:
                print(f"   -> Could not parse Last-Modified header: {e}")
                
        return resp.content
        
    return None


def collect_frames():
    """Fetch up to FRAME_COUNT most recent available frames, oldest first."""
    now = round_down_to_10min(datetime.now(timezone.utc))
    frames = []
    lookback = FRAME_COUNT + 6  # search a bit further back in case some slots are missing
    for i in range(lookback):
        slot_time = now - timedelta(minutes=10 * i)
        data = fetch_frame(slot_time)
        if data:
            frames.append((slot_time, data))
        if len(frames) >= FRAME_COUNT:
            break
    frames.sort(key=lambda f: f[0])  # chronological order, oldest first
    return frames


def build_gif(frames) -> bytes:
    images = []
    for _, data in frames:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width > MAX_WIDTH:
            ratio = MAX_WIDTH / img.width
            img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
        images.append(img)

    buf = io.BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


def post_gif(gif_bytes: bytes, start_label: str, end_label: str, frame_count: int):
    filename = f"{SECTOR}_{BAND}_animation.gif"
    files = {"file": (filename, gif_bytes, "image/gif")}
    payload = {
        "content": (
            f"**Himawari animation** \u2014 sector `{SECTOR}` \u00b7 band `{BAND}` \u00b7 "
            f"{start_label} \u2192 {end_label} UTC ({frame_count} frames)"
        )
    }
    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=60)
    resp.raise_for_status()


def main():
    frames = collect_frames()
    if len(frames) < 2:
        print(f"Only found {len(frames)} frame(s), need at least 2 for an animation.")
        return

    gif_bytes = build_gif(frames)

    # If it's too big for Discord, progressively drop frames and rebuild.
    attempts = 0
    while len(gif_bytes) > MAX_FILE_BYTES and len(frames) > 4 and attempts < 4:
        frames = frames[::2]  # keep every other frame
        gif_bytes = build_gif(frames)
        attempts += 1

    if len(gif_bytes) > MAX_FILE_BYTES:
        print(f"GIF still too large ({len(gif_bytes)} bytes) after reducing frames, skipping post.")
        return

    start_label = frames[0][0].strftime("%H:%M")
    end_label = frames[0][0].strftime("%H:%M")
    post_gif(gif_bytes, start_label, end_label, len(frames))
    print(f"Posted animation with {len(frames)} frames ({start_label} -> {end_label} UTC), {len(gif_bytes)} bytes")


if __name__ == "__main__":
    main()
