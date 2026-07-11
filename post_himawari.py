"""
Checks JMA's Himawari archive for a new image tile and posts it to a Discord
channel via a webhook (no bot process, no token, no server required).

Designed to run as a scheduled GitHub Actions job. State (the last slot we
posted) is kept in last_slot.txt and committed back to the repo by the
workflow, so re-runs don't duplicate posts.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import mimetypes
import os
from datetime import datetime, timedelta, timezone

import requests

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")   # r2w = Southeast Asia (extended, large)
BAND = os.environ.get("HIMAWARI_BAND", "b13")        # b13 = infrared (day & night)
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Optional: fully override the URL pattern instead of using SECTOR/BAND.
# Use {HHMM} where the 4-digit UTC time slot (e.g. 2330) should go.
# Example (ASWind product, a different directory/naming scheme entirely):
#   HIMAWARI_URL_TEMPLATE=https://www.data.jma.go.jp/mscweb/en/product/data/fs/aswind_fsir_{HHMM}.png
URL_TEMPLATE = os.environ.get("HIMAWARI_URL_TEMPLATE")

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"
STATE_FILE = "last_slot.txt"
MAX_LOOKBACK_STEPS = int(os.environ.get("MAX_LOOKBACK_STEPS", "6"))  # 6 * 10min = up to 1 hour back


def round_down_to_10min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def read_last_slot():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return None


def write_last_slot(state_key: str):
    with open(STATE_FILE, "w") as f:
        f.write(state_key)


def url_for_slot(slot_time: datetime) -> str:
    hhmm = slot_time.strftime("%H%M")
    if URL_TEMPLATE:
        return URL_TEMPLATE.format(HHMM=hhmm)
    return f"{BASE_URL}/{SECTOR}/{SECTOR}_{BAND}_{hhmm}.jpg"


def product_id() -> str:
    """A short identifier for whatever product we're fetching, used in
    filenames, captions, and the dedup state key."""
    if URL_TEMPLATE:
        name = URL_TEMPLATE.rstrip("/").split("/")[-1]
        return name.replace("{HHMM}", "HHMM")
    return f"{SECTOR}_{BAND}"


def fetch_latest_image():
    """Walk backwards in 10-min UTC steps until an image tile is found.
    Returns (image_bytes, slot_label, resolved_url) or (None, None, None)."""
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
            return resp.content, slot_label, url
    return None, None, None


def post_to_discord(image_bytes: bytes, slot_label: str, resolved_url: str):
    ext = os.path.splitext(resolved_url)[1] or ".jpg"
    content_type = mimetypes.guess_type(resolved_url)[0] or "image/jpeg"
    pid = product_id()
    filename = f"{pid}_{slot_label.replace(' ', '_').replace(':', '')}{ext}"
    files = {"file": (filename, image_bytes, content_type)}
    payload = {
        "content": f"**Himawari image** \u2014 product `{pid}` \u00b7 {slot_label}"
    }
    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=30)
    resp.raise_for_status()


def main():
    pid = product_id()
    print(f"Config: {'URL_TEMPLATE=' + URL_TEMPLATE if URL_TEMPLATE else f'SECTOR={SECTOR} BAND={BAND}'}")
    image_bytes, slot_label, resolved_url = fetch_latest_image()
    if image_bytes is None:
        print(f"No fresh image found in the last {10 * MAX_LOOKBACK_STEPS} minutes.")
        return

    # Key state on product+timestamp, not just the timestamp, so switching
    # products/sectors/bands never gets mistaken for a duplicate post.
    state_key = f"{pid}:{slot_label}"
    if state_key == read_last_slot():
        print(f"{pid} slot {slot_label} already posted, skipping.")
        return

    post_to_discord(image_bytes, slot_label, resolved_url)
    write_last_slot(state_key)
    print(f"Posted {pid} image for slot {slot_label}")


if __name__ == "__main__":
    main()
