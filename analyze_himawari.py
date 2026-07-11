"""
Fetches the latest Himawari IR tile and runs a simple heuristic scan for
tight, circular, very-bright (very cold) cloud clusters -- a rough visual
proxy for deep convective activity often seen in tropical cyclones. Tracks
candidates across scans (nearest-neighbor matching) and draws their recent
drift as a trail.

IMPORTANT: this is a hobby-project heuristic, not an official storm
detection or tracking algorithm.
- No calibrated brightness temperature (JPEGs aren't raw radiance data),
  just relative brightness within each frame.
- No georeferencing -- positions are reported as "% across / % down the
  frame", not lat/lon, and drift is "% of frame per scan", not km/h.
- Matching between scans is a naive nearest-neighbor on pixel position; it
  has no concept of storm identity beyond "closest blob last time", so a
  new storm appearing where an old one dissipated could be mismatched as
  the same one.
- Single-frame shape/brightness only -- can't distinguish a real cyclone
  eyewall from any other sufficiently round, cold, compact convective
  cluster (e.g. an ordinary thunderstorm complex). Treat any "candidate"
  as "worth a look", not a detection.

Posts an annotated image + a short text summary to Discord via webhook
whenever it finds at least one candidate cluster.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import json
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")
BAND = os.environ.get("HIMAWARI_BAND", "b13")   # IR band -- needed for the brightness heuristic to be meaningful
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Optional full URL override, same as the other scripts (use {HHMM}).
URL_TEMPLATE = os.environ.get("HIMAWARI_URL_TEMPLATE")

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"
MAX_LOOKBACK_STEPS = int(os.environ.get("MAX_LOOKBACK_STEPS", "6"))

# Detection tuning -- all overridable via env vars without touching code.
BRIGHTNESS_PERCENTILE = float(os.environ.get("STORM_BRIGHTNESS_PERCENTILE", "97"))
MIN_AREA_FRAC = float(os.environ.get("STORM_MIN_AREA_FRAC", "0.0004"))   # ignore tiny specks/noise
MAX_AREA_FRAC = float(os.environ.get("STORM_MAX_AREA_FRAC", "0.06"))     # ignore huge fronts/cloud bands
MIN_CIRCULARITY = float(os.environ.get("STORM_MIN_CIRCULARITY", "0.55"))  # 1.0 = perfect circle

# Tracking tuning
MAX_MATCH_DIST_FRAC = float(os.environ.get("STORM_MAX_MATCH_DIST_FRAC", "0.08"))  # fraction of image diagonal
TRAIL_LENGTH = int(os.environ.get("STORM_TRAIL_LENGTH", "6"))  # how many past points to remember/draw

STATE_FILE = "last_storm_slot.txt"
HISTORY_FILE = "storm_history.json"


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


def fetch_latest():
    now = round_down_to_10min(datetime.now(timezone.utc))
    for i in range(MAX_LOOKBACK_STEPS):
        slot_time = now - timedelta(minutes=10 * i)
        url = url_for_slot(slot_time)
        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            continue
        print(f"GET {url} -> {resp.status_code}")
        if resp.status_code == 200:
            slot_label = slot_time.strftime("%Y-%m-%d %H:%M UTC")
            return resp.content, slot_label
    return None, None


def detect_candidates(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    total_area = h * w

    # Isolate the brightest (coldest / highest cloud-top) pixels in this frame
    thresh_val = float(np.percentile(gray, BRIGHTNESS_PERCENTILE))
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean up speckle noise, then close small gaps within clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        area_frac = area / total_area
        if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        candidates.append({
            "cx": cx, "cy": cy, "radius": radius,
            "area_frac": area_frac, "circularity": circularity,
            "pct_across": 100 * cx / w, "pct_down": 100 * cy / h,
        })

    # Strongest (largest + most circular) candidates first
    candidates.sort(key=lambda d: d["area_frac"] * d["circularity"], reverse=True)
    return img, candidates, w, h


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_history(slot_label: str, candidates):
    # Store enough to keep tracking on the next run and let other scripts
    # (e.g. a storm-cam animator) size a crop box around a candidate.
    slim = [
        {
            "cx": c["cx"], "cy": c["cy"], "radius": c["radius"],
            "area_frac": c["area_frac"],
            "trail": c.get("trail", []),
            "tracked_scans": c.get("tracked_scans", 1),
        }
        for c in candidates
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump({"slot": slot_label, "candidates": slim}, f)


def match_and_track(candidates, history, w, h):
    """Greedy nearest-neighbor match against the previous scan's candidates
    (within MAX_MATCH_DIST_FRAC of the image diagonal), attaching a trail
    of recent positions and simple drift stats to each match."""
    diag = (w ** 2 + h ** 2) ** 0.5
    max_dist = MAX_MATCH_DIST_FRAC * diag
    prev_candidates = history["candidates"] if history else []

    used_prev = set()
    for c in candidates:
        best_idx, best_dist = None, None
        for idx, p in enumerate(prev_candidates):
            if idx in used_prev:
                continue
            dist = ((c["cx"] - p["cx"]) ** 2 + (c["cy"] - p["cy"]) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx is not None and best_dist <= max_dist:
            used_prev.add(best_idx)
            prev = prev_candidates[best_idx]
            used_prev_trail = prev.get("trail", [])
            c["trail"] = (used_prev_trail + [{"cx": prev["cx"], "cy": prev["cy"]}])[-TRAIL_LENGTH:]
            c["drift_pct_x"] = 100 * (c["cx"] - prev["cx"]) / w
            c["drift_pct_y"] = 100 * (c["cy"] - prev["cy"]) / h
            c["tracked_scans"] = prev.get("tracked_scans", 1) + 1
        else:
            c["trail"] = []
            c["tracked_scans"] = 1
    return candidates


def annotate(img, candidates):
    out = img.copy()
    for i, c in enumerate(candidates, start=1):
        trail = c.get("trail", [])
        pts = [(int(p["cx"]), int(p["cy"])) for p in trail] + [(int(c["cx"]), int(c["cy"]))]
        for j in range(len(pts) - 1):
            cv2.line(out, pts[j], pts[j + 1], (0, 165, 255), 2, cv2.LINE_AA)
        center = (int(c["cx"]), int(c["cy"]))
        radius = max(int(c["radius"]), 5)
        cv2.circle(out, center, radius, (0, 0, 255), 2)
        label = f"#{i}" + (f" ({c['tracked_scans']}x)" if c.get("tracked_scans", 1) > 1 else "")
        cv2.putText(
            out, label, (center[0] - 10, max(center[1] - radius - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    return out


def post_to_discord(image_bytes: bytes, slot_label: str, candidates):
    pid = product_id()
    filename = f"{pid}_storm_scan_{slot_label.replace(' ', '_').replace(':', '')}.jpg"
    files = {"file": (filename, image_bytes, "image/jpeg")}

    lines = [f"**Storm/typhoon candidate scan** \u2014 product `{pid}` \u00b7 {slot_label}"]
    lines.append(f"Found {len(candidates)} candidate cluster(s):")
    for i, c in enumerate(candidates, start=1):
        line = (
            f"#{i}: ~{c['pct_across']:.0f}% across, {c['pct_down']:.0f}% down \u00b7 "
            f"size {c['area_frac']*100:.2f}% of frame \u00b7 circularity {c['circularity']:.2f}"
        )
        if c.get("tracked_scans", 1) > 1:
            line += (
                f" \u00b7 tracked {c['tracked_scans']} scans, drifted "
                f"{c.get('drift_pct_x', 0):+.1f}%x / {c.get('drift_pct_y', 0):+.1f}%y since last scan"
            )
        lines.append(line)
    lines.append("_Heuristic brightness/shape scan only \u2014 not an official detection._")
    payload = {"content": "\n".join(lines)}

    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=30)
    resp.raise_for_status()


def read_last_slot():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return f.read().strip()
    return None


def write_last_slot(key: str):
    with open(STATE_FILE, "w") as f:
        f.write(key)


def main():
    pid = product_id()
    print(
        f"Config: product={pid} brightness_pct={BRIGHTNESS_PERCENTILE} "
        f"area_frac=[{MIN_AREA_FRAC},{MAX_AREA_FRAC}] min_circularity={MIN_CIRCULARITY} "
        f"max_match_dist_frac={MAX_MATCH_DIST_FRAC} trail_length={TRAIL_LENGTH}"
    )

    image_bytes, slot_label = fetch_latest()
    if image_bytes is None:
        print(f"No fresh image found in the last {10 * MAX_LOOKBACK_STEPS} minutes.")
        return

    state_key = f"{pid}:{slot_label}"
    if state_key == read_last_slot():
        print(f"{pid} slot {slot_label} already scanned, skipping.")
        return
    write_last_slot(state_key)

    img, candidates, w, h = detect_candidates(image_bytes)

    history = load_history()
    candidates = match_and_track(candidates, history, w, h)
    save_history(slot_label, candidates)  # always update, even if empty, so stale tracks don't linger

    if not candidates:
        print(f"No candidate clusters found for {slot_label}.")
        return

    annotated = annotate(img, candidates)
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        print("Failed to encode annotated image.")
        return

    post_to_discord(buf.tobytes(), slot_label, candidates)
    print(f"Posted storm scan for {slot_label}: {len(candidates)} candidate(s).")


if __name__ == "__main__":
    main()
