"""
Checks JMA's Himawari archive for a new image tile and posts it to a Discord
channel via a webhook (no bot process, no token, no server required).

Designed to run as a scheduled GitHub Actions job. State (the last slot we
posted) is kept in last_slot.txt and committed back to the repo by the
workflow, so re-runs don't duplicate posts.

Source: https://www.data.jma.go.jp/mscweb/data/himawari/
"""

import os
from datetime import datetime, timedelta, timezone

import requests

SECTOR = os.environ.get("HIMAWARI_SECTOR", "r2w")   # r2w = Southeast Asia (extended, large)
BAND = os.environ.get("HIMAWARI_BAND", "b13")        # b13 = infrared (day & night)
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/img"
STATE_FILE = "last_slot.txt"
MAX_LOOKBACK_STEPS = 6  # 6 * 10min = up to 1 hour back, to cover JMA's processing lag


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


def fetch_latest_image():
    """Walk backwards in 10-min UTC steps until an image tile is found."""
    now = round_down_to_10min(datetime.now(timezone.utc))
    for i in range(MAX_LOOKBACK_STEPS):
        slot_time = now - timedelta(minutes=10 * i)
        hhmm = slot_time.strftime("%H%M")
        url = f"{BASE_URL}/{SECTOR}/{SECTOR}_{BAND}_{hhmm}.jpg"
        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            continue
        if resp.status_code == 200:
            slot_label = slot_time.strftime("%Y-%m-%d %H:%M UTC")
            return resp.content, slot_label
    return None, None


def post_to_discord(image_bytes: bytes, slot_label: str):
    filename = f"{SECTOR}_{BAND}_{slot_label.replace(' ', '_').replace(':', '')}.jpg"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    payload = {
        "content": f"**Himawari satellite image** \u2014 sector `{SECTOR}` \u00b7 band `{BAND}` \u00b7 {slot_label}"
    }
    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=30)
    resp.raise_for_status()


def main():
    image_bytes, slot_label = fetch_latest_image()
    if image_bytes is None:
        print(f"No fresh image found in the last {10 * MAX_LOOKBACK_STEPS} minutes.")
        return

    # Key state on sector+band+timestamp, not just the timestamp, so switching
    # HIMAWARI_SECTOR/HIMAWARI_BAND never gets mistaken for a duplicate post.
    state_key = f"{SECTOR}:{BAND}:{slot_label}"
    if state_key == read_last_slot():
        print(f"{SECTOR}/{BAND} slot {slot_label} already posted, skipping.")
        return

    post_to_discord(image_bytes, slot_label)
    write_last_slot(state_key)
    print(f"Posted {SECTOR}/{BAND} image for slot {slot_label}")


if __name__ == "__main__":
    main()
