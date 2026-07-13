"""
Fetches JTWC's Significant Tropical Weather Advisory and NOAA CPC's Global
Tropical Hazards Outlook and posts them to Discord via webhook, on a
periodic (every couple of days) schedule.

Both source URLs are "latest snapshot" images with no date/time in the
filename -- the source simply overwrites them whenever a new outlook is
issued. This script just fetches whatever's currently there and posts it;
there's no timestamp to compare against, so no freshness/dedup check like
the Himawari scripts do.

Sources:
  - JTWC Significant Tropical Weather Advisory (Western & South Pacific):
    https://www.metoc.navy.mil/jtwc/products/abpwsair.jpg
  - NOAA CPC Global Tropical Hazards Outlook:
    https://www.cpc.ncep.noaa.gov/products/precip/CWlink/ghaz/gth_full.png
"""

import mimetypes
import os

import requests

WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

SOURCES = [
    {
        "name": "JTWC Significant Tropical Weather Advisory",
        "url": "https://www.metoc.navy.mil/jtwc/products/abpwsair.jpg",
        "filename": "jtwc_tropical_weather_advisory.jpg",
    },
    {
        "name": "NOAA CPC Global Tropical Hazards Outlook",
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/ghaz/gth_full.png",
        "filename": "cpc_global_tropical_hazards.png",
    },
]


def fetch(url: str) -> bytes:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content


def post_to_discord(fetched: list):
    """Post everything fetched successfully in a single message (Discord
    webhooks accept multiple file attachments in one multipart POST)."""
    files = {}
    for i, (source, content) in enumerate(fetched):
        content_type = mimetypes.guess_type(source["filename"])[0] or "application/octet-stream"
        files[f"file{i}"] = (source["filename"], content, content_type)

    lines = ["**Tropical outlook update**"]
    for source, _ in fetched:
        lines.append(f"\u2022 {source['name']}")
    payload = {"content": "\n".join(lines)}

    resp = requests.post(WEBHOOK_URL, data=payload, files=files, timeout=30)
    resp.raise_for_status()


def main():
    fetched = []
    for source in SOURCES:
        try:
            content = fetch(source["url"])
            print(f"Fetched {source['name']} ({len(content)} bytes)")
            fetched.append((source, content))
        except requests.RequestException as e:
            print(f"Failed to fetch {source['name']}: {e}")

    if not fetched:
        print("No sources fetched successfully, nothing to post.")
        return

    post_to_discord(fetched)
    print(f"Posted {len(fetched)} image(s).")


if __name__ == "__main__":
    main()
