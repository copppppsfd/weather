"""
Builds a short looping mp4 from the last several Himawari frames and posts
it to Discord via webhook (same webhook as post_himawari.py — no bot needed).

Unlike post_himawari.py (posts one new still frame whenever available), this
grabs a recent window of frames and stitches them into an animation, so it's
meant to run less often (e.g. every few hours) on its own schedule.

Uses mp4 (h.264 via ffmpeg) instead of GIF. GIF is capped at 256 colors per
frame and compresses poorly frame-to-frame; for the same file-size budget,
h.264 gets you meaningfully higher resolution and/or more frames because it
actually exploits color depth and inter-frame redundancy. Discord plays mp4
attachments inline the same way it does GIFs.

Requires the `ffmpeg` binary to be available on PATH (e.g. `apt install
ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import io
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
from PIL import Image

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")   # r2w = Southeast Asia (extended, large)
BAND = os.environ.get("HIMAWARI_BAND", "hrp")        # b13 = infrared (day & night)
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Optional: fully override the URL pattern instead of using SECTOR/BAND.
# Use {HHMM} where the 4-digit UTC time slot (e.g. 2330) should go.
URL_TEMPLATE = os.environ.get("HIMAWARI_URL_TEMPLATE")

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"

FRAME_COUNT = int(os.environ.get("ANIMATION_FRAME_COUNT", "12"))         # 12 * 10min = ~2 hours
FRAME_DURATION_MS = int(os.environ.get("ANIMATION_FRAME_DURATION_MS", "150"))

# Target output width. Resizes both up and down to hit this. Upscaling
# doesn't add real detail -- the source frame only has whatever resolution
# JMA published at -- it just interpolates for a bigger/smoother result.
MAX_WIDTH = int(os.environ.get("ANIMATION_MAX_WIDTH", "900"))

# CRF ("constant rate factor") is h.264's quality knob: lower = better
# quality/bigger file, higher = more compressed/smaller file. ~18 is
# visually near-lossless, ~23 is a solid default, ~28+ starts showing
# compression artifacts on detailed content. When a render is too big for
# Discord, we raise CRF first (keeps resolution + frame count intact)
# before resorting to dropping frames or shrinking width.
CRF = int(os.environ.get("ANIMATION_CRF", "23"))
CRF_MAX = int(os.environ.get("ANIMATION_CRF_MAX", "34"))
CRF_STEP = int(os.environ.get("ANIMATION_CRF_STEP", "3"))

# Discord's free-tier per-file cap is 10MB (decimal); this has crept in
# third-party reports up to 25MB and boosted servers raise it further --
# check what actually works for your server and adjust. Stay safely under
# whatever the real cap is.
MAX_FILE_BYTES = int(os.environ.get("ANIMATION_MAX_FILE_BYTES", str(9_500_000)))

# Floors for the last-resort fallback, once CRF alone can't hit the cap.
MIN_FRAMES_FLOOR = 4
MIN_WIDTH_FLOOR = 300


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
                last_mod_dt = parsedate_to_datetime(last_mod_header)
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


def _even(n: int) -> int:
    """h.264 (yuv420p) requires even width/height."""
    return n if n % 2 == 0 else n - 1


def decode_and_resize(frames, target_width: int):
    """Decode raw bytes to PIL images and resize (up or down) to
    target_width, preserving aspect ratio. Dimensions rounded down to even
    numbers since h.264 requires that. Returns (images, width, height)."""
    images = []
    width = height = None
    for _, data in frames:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if width is None:
            ratio = target_width / img.width
            width = _even(target_width)
            height = _even(int(img.height * ratio))
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        images.append(img)
    return images, width, height


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first, e.g. "
            "`apt install ffmpeg` (Debian/Ubuntu) or `brew install ffmpeg` (macOS)."
        )


def encode_mp4(images, width: int, height: int, fps: float, crf: int) -> bytes:
    """Pipe raw RGB frames into ffmpeg over stdin and encode to h.264 mp4.
    Writes to a temp file rather than stdout, since +faststart needs a
    seekable output to relocate the moov atom for streaming playback."""
    raw = b"".join(img.tobytes() for img in images)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {proc.stderr.decode(errors='replace')[-2000:]}")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def post_mp4(mp4_bytes: bytes, start_label: str, end_label: str, frame_count: int):
    filename = f"{SECTOR}_{BAND}_animation.mp4"
    files = {"file": (filename, mp4_bytes, "video/mp4")}
    payload = {
        "content": (
            f"**Himawari animation** \u2014 sector `{SECTOR}` \u00b7 band `{BAND}` \u00b7 "
            f"{start_label} \u2192 {end_label} UTC ({frame_count} frames)"
        )
    }
    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=60)
    resp.raise_for_status()


def main():
    check_ffmpeg()

    frames = collect_frames()
    if len(frames) < 2:
        print(f"Only found {len(frames)} frame(s), need at least 2 for an animation.")
        return

    fps = 1000 / FRAME_DURATION_MS
    target_width = MAX_WIDTH
    crf = CRF

    images, width, height = decode_and_resize(frames, target_width)
    mp4_bytes = encode_mp4(images, width, height, fps, crf)

    # 1st fallback: raise CRF (more compression, same resolution/frames).
    while len(mp4_bytes) > MAX_FILE_BYTES and crf < CRF_MAX:
        crf = min(crf + CRF_STEP, CRF_MAX)
        print(f"mp4 too large ({len(mp4_bytes)} bytes), raising CRF to {crf}.")
        mp4_bytes = encode_mp4(images, width, height, fps, crf)

    # 2nd fallback: drop frames, then shrink width, re-trying CRF from the
    # top each time since a smaller render can usually afford better quality.
    attempts = 0
    max_attempts = 8
    while len(mp4_bytes) > MAX_FILE_BYTES and attempts < max_attempts:
        if len(frames) > MIN_FRAMES_FLOOR:
            frames = frames[::2]
            print(f"mp4 still too large, dropping every other frame ({len(frames)} left).")
        elif target_width > MIN_WIDTH_FLOOR:
            target_width = max(int(target_width * 0.8), MIN_WIDTH_FLOOR)
            print(f"mp4 still too large, shrinking width to {target_width}px.")
        else:
            break
        crf = CRF
        images, width, height = decode_and_resize(frames, target_width)
        mp4_bytes = encode_mp4(images, width, height, fps, crf)
        while len(mp4_bytes) > MAX_FILE_BYTES and crf < CRF_MAX:
            crf = min(crf + CRF_STEP, CRF_MAX)
            mp4_bytes = encode_mp4(images, width, height, fps, crf)
        attempts += 1

    if len(mp4_bytes) > MAX_FILE_BYTES:
        print(f"mp4 still too large ({len(mp4_bytes)} bytes) after all fallbacks, skipping post.")
        return

    start_label = frames[0][0].strftime("%H:%M")
    end_label = frames[-1][0].strftime("%H:%M")
    post_mp4(mp4_bytes, start_label, end_label, len(frames))
    print(
        f"Posted animation with {len(frames)} frames ({start_label} -> {end_label} UTC), "
        f"{width}x{height} @ CRF {crf}, {len(mp4_bytes)} bytes"
    )


if __name__ == "__main__":
    main()
