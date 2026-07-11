"""
Builds a zoomed-in looping GIF centered on the strongest actively-tracked
storm/typhoon candidate, using the position/size/trail data
analyze_himawari.py leaves behind in storm_history.json -- instead of
animating the whole sector, this crops down to just the storm.

Depends on the storm-scan workflow having run recently and having a
candidate tracked for at least STORM_CAM_MIN_TRACKED_SCANS scans (default
2), so this doesn't build a zoomed loop around a one-off noise blob.

The crop box follows the storm's drift across the loop: each frame's crop
center is interpolated along the candidate's saved trail (assumed to be
spaced ~10 minutes apart, matching both the Himawari refresh cadence and
the storm-scanner's own polling interval) rather than staying fixed on the
candidate's single latest position. This is still an approximation -- if
the storm-scanner's actual polling interval drifts from 10 minutes, or the
storm changes speed/direction between scans, the interpolated position
will be off by some amount -- but it tracks real drift far better than a
static box over a ~2 hour window.

Note on staleness: the JMA image filenames only encode time-of-day (HHMM),
not the date, so a 200 response alone doesn't guarantee a frame is from
today. fetch_frame() cross-checks the Last-Modified header's date against
the requested slot, and collect_frames() additionally drops exact
byte-for-byte duplicate frames (which can slip through if a Last-Modified
header is missing) so a stale/cached repeat doesn't produce a frozen frame
in the loop.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import hashlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

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
SCAN_STEP_MINUTES = int(os.environ.get("STORM_CAM_SCAN_STEP_MINUTES", "10"))  # assumed trail spacing

MAX_FILE_BYTES = 9_500_000  # stay safely under Discord's free-tier 10MB cap
MIN_FRAMES_FLOOR = 4        # don't shrink the loop below this many frames
MIN_OUTPUT_SIZE = 150       # don't shrink resolution below this many pixels

_session = requests.Session()


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
    scanner's last saved history (candidates are saved strongest-first).
    Returns (candidate, history_slot_time) or (None, None)."""
    if not os.path.exists(HISTORY_FILE):
        return None, None
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    tracked = [c for c in history.get("candidates", []) if c.get("tracked_scans", 1) >= MIN_TRACKED_SCANS]
    if not tracked:
        return None, None

    history_slot_time = None
    slot_label = history.get("slot")
    if slot_label:
        try:
            history_slot_time = datetime.strptime(slot_label, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Could not parse history slot label {slot_label!r}, drift interpolation will be disabled.")

    return tracked[0], history_slot_time


def fetch_frame(slot_time: datetime):
    """Fetch one frame, verifying (where possible) that it's actually from
    the requested day rather than a stale cached copy at the same HHMM
    filename. Returns raw bytes, or None if unavailable/stale."""
    url = url_for_slot(slot_time)
    try:
        resp = _session.get(
            url,
            timeout=15,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        )
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

    print(f"GET {url} -> {resp.status_code}")
    if resp.status_code != 200:
        return None

    lm_header = resp.headers.get("Last-Modified")
    if lm_header:
        try:
            lm_dt = parsedate_to_datetime(lm_header)
            if lm_dt.tzinfo is None:
                lm_dt = lm_dt.replace(tzinfo=timezone.utc)
            if lm_dt.date() != slot_time.date():
                print(
                    f"  -> stale: Last-Modified {lm_dt.isoformat()} doesn't match "
                    f"requested slot date {slot_time.date()}, skipping"
                )
                return None
        except (TypeError, ValueError):
            pass
    else:
        print("  -> no Last-Modified header returned, can't verify freshness by date")

    return resp.content


def collect_frames():
    """Fetch up to FRAME_COUNT most recent available, non-duplicate frames,
    oldest first. Frames whose content exactly matches one already
    collected in this run are dropped -- a stale/cached repeat served
    without a useful Last-Modified header would otherwise show up as a
    frozen frame in the loop."""
    now = round_down_to_10min(datetime.now(timezone.utc))
    frames = []
    seen_hashes = set()
    lookback = FRAME_COUNT + 6
    for i in range(lookback):
        slot_time = now - timedelta(minutes=SCAN_STEP_MINUTES * i)
        data = fetch_frame(slot_time)
        if not data:
            continue
        digest = hashlib.md5(data).hexdigest()
        if digest in seen_hashes:
            print(f"  -> duplicate content of an already-collected frame, skipping")
            continue
        seen_hashes.add(digest)
        frames.append((slot_time, data))
        if len(frames) >= FRAME_COUNT:
            break
    frames.sort(key=lambda f: f[0])
    return frames


def decode_frames(frames):
    """Decode raw bytes into PIL images, skipping any that fail to decode
    (e.g. truncated/corrupt download) instead of crashing the whole run."""
    decoded = []
    for slot_time, data in frames:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.load()  # force full decode now so failures surface here
        except Exception as e:
            print(f"Could not decode frame for {slot_time}: {e}, skipping")
            continue
        decoded.append((slot_time, img))
    return decoded


def build_path(candidate):
    """Trail is oldest-first and stops just short of the candidate's
    current position; append the current position to get the full
    recent-position path, oldest to newest."""
    path = [(p["cx"], p["cy"]) for p in candidate.get("trail", [])]
    path.append((candidate["cx"], candidate["cy"]))
    return path


def interpolate_center(path, history_slot_time, slot_time):
    """Estimate the candidate's position at an earlier frame's slot_time by
    walking back along the trail in SCAN_STEP_MINUTES increments from the
    history's last known (most recent) position. Falls back to the latest
    known position if we can't compute an offset or run off the end of the
    trail."""
    if history_slot_time is None or not path:
        return path[-1] if path else None

    steps_back = round((history_slot_time - slot_time).total_seconds() / (SCAN_STEP_MINUTES * 60))
    idx = len(path) - 1 - steps_back
    idx = max(0, min(idx, len(path) - 1))
    return path[idx]


def crop_box_at(cx, cy, radius, img_w, img_h):
    half = max(radius * CROP_RADIUS_MULTIPLIER, MIN_CROP_HALF_PX)
    half = min(half, MAX_CROP_FRAC * min(img_w, img_h) / 2)

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


def build_gif(decoded_frames, candidate, history_slot_time, output_size):
    img_w, img_h = decoded_frames[0][1].size
    path = build_path(candidate)
    radius = candidate.get("radius", MIN_CROP_HALF_PX / CROP_RADIUS_MULTIPLIER)

    images = []
    for slot_time, img in decoded_frames:
        cx, cy = interpolate_center(path, history_slot_time, slot_time)
        crop_box = crop_box_at(cx, cy, radius, img_w, img_h)
        cropped = img.crop(crop_box)
        resized = cropped.resize((output_size, output_size), Image.LANCZOS)
        images.append(resized)

    buf = io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True, append_images=images[1:],
        duration=FRAME_DURATION_MS, loop=0, optimize=True,
    )
    return buf.getvalue()


def build_gif_under_limit(decoded_frames, candidate, history_slot_time):
    """Try to produce a GIF under MAX_FILE_BYTES, first by dropping frames
    and then, if still too big, by shrinking the output resolution -- each
    tried a few times before giving up."""
    frames = decoded_frames
    output_size = OUTPUT_SIZE

    while True:
        gif_bytes = build_gif(frames, candidate, history_slot_time, output_size)
        if len(gif_bytes) <= MAX_FILE_BYTES:
            return gif_bytes, frames

        shrunk = False
        if len(frames) > MIN_FRAMES_FLOOR:
            print(f"GIF too large ({len(gif_bytes)} bytes), dropping every other frame.")
            frames = frames[::2]
            shrunk = True
        elif output_size > MIN_OUTPUT_SIZE:
            new_size = max(int(output_size * 0.75), MIN_OUTPUT_SIZE)
            print(f"GIF too large ({len(gif_bytes)} bytes), shrinking {output_size}px -> {new_size}px.")
            output_size = new_size
            shrunk = True

        if not shrunk:
            print(f"Still too large ({len(gif_bytes)} bytes) after minimum frames/resolution, giving up.")
            return None, frames


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
    candidate, history_slot_time = load_best_candidate()
    if candidate is None:
        print(f"No candidate tracked for >= {MIN_TRACKED_SCANS} scans yet, skipping storm-cam.")
        return

    print(
        f"Building storm-cam around candidate at ({candidate['cx']:.0f},{candidate['cy']:.0f}), "
        f"tracked {candidate.get('tracked_scans', 1)} scans, "
        f"trail points={len(candidate.get('trail', []))}, "
        f"drift interpolation {'enabled' if history_slot_time else 'disabled (static box)'}"
    )

    raw_frames = collect_frames()
    if len(raw_frames) < 2:
        print(f"Only found {len(raw_frames)} usable frame(s), need at least 2 for an animation.")
        return

    decoded_frames = decode_frames(raw_frames)
    if len(decoded_frames) < 2:
        print(f"Only {len(decoded_frames)} frame(s) decoded successfully, need at least 2.")
        return

    gif_bytes, used_frames = build_gif_under_limit(decoded_frames, candidate, history_slot_time)
    if gif_bytes is None:
        return

    start_label = used_frames[0][0].strftime("%H:%M")
    end_label = used_frames[-1][0].strftime("%H:%M")
    post_gif(gif_bytes, start_label, end_label, len(used_frames), candidate)
    print(f"Posted storm-cam loop with {len(used_frames)} frame(s), {len(gif_bytes)} bytes.")


if __name__ == "__main__":
    main()
