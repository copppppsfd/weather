"""
Posts an actual precipitation forecast for a specific location to Discord,
using Open-Meteo (free, no API key, blends multiple national weather
models -- including JMA's own -- rather than guessing from satellite pixel
brightness).

This is the accurate counterpart to rain_overlay.py: that script shows
"what does the sky look like right now" from satellite imagery; this one
answers "will it actually rain here" using a real numerical forecast.

You MUST set LOCATION_LAT / LOCATION_LON below (as env vars / workflow
secrets) -- there's no sensible default location to guess. Find your
coordinates at https://www.openstreetmap.org (right-click a spot -> "Show
address") or https://open-meteo.com/en/docs (has a place search box).

API docs: https://open-meteo.com/en/docs
"""

import os

import requests

WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

LOCATION_LAT = os.environ.get("LOCATION_LAT")
LOCATION_LON = os.environ.get("LOCATION_LON")
LOCATION_NAME = os.environ.get("LOCATION_NAME", "your location")

FORECAST_HOURS = int(os.environ.get("FORECAST_HOURS", "12"))
RAIN_ALERT_THRESHOLD = int(os.environ.get("RAIN_PROBABILITY_ALERT_THRESHOLD", "50"))  # percent

if not LOCATION_LAT or not LOCATION_LON:
    raise SystemExit(
        "LOCATION_LAT and LOCATION_LON must be set (see the docstring at the top of this file)."
    )

API_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast():
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "hourly": "precipitation_probability,precipitation,weather_code",
        "current": "precipitation,weather_code",
        "timezone": "auto",
        "forecast_days": 2,
    }
    resp = requests.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def summarize(data: dict) -> str:
    hourly = data["hourly"]
    times = hourly["time"]
    probs = hourly["precipitation_probability"]
    precip = hourly["precipitation"]

    # Open-Meteo's hourly arrays start at 00:00 today; find the index for
    # "now" using the current block's time (nearest hour).
    now_time = data["current"]["time"]
    try:
        start_idx = times.index(now_time)
    except ValueError:
        start_idx = 0

    window = list(zip(times[start_idx:start_idx + FORECAST_HOURS],
                       probs[start_idx:start_idx + FORECAST_HOURS],
                       precip[start_idx:start_idx + FORECAST_HOURS]))

    if not window:
        return f"No forecast data available for {LOCATION_NAME}."

    max_prob = max(p for _, p, _ in window)
    total_precip = sum(p for _, _, p in window)
    rain_hours = [(t, p) for t, p, _ in window if p >= RAIN_ALERT_THRESHOLD]

    verdict = (
        f"\U0001F327\uFE0F Rain likely in the next {FORECAST_HOURS}h (peak {max_prob}% chance)"
        if rain_hours else
        f"\u2600\uFE0F Rain unlikely in the next {FORECAST_HOURS}h (peak {max_prob}% chance)"
    )

    lines = [f"**Forecast for {LOCATION_NAME}**", verdict]
    if rain_hours:
        hour_strs = [t.split("T")[1] for t, _ in rain_hours[:6]]  # cap listing to 6 hours
        lines.append(f"Hours \u2265{RAIN_ALERT_THRESHOLD}% chance: {', '.join(hour_strs)}")
    lines.append(f"Expected total precipitation in window: {total_precip:.1f} mm")
    lines.append("_Source: Open-Meteo (blended national weather models), not a heuristic._")

    return "\n".join(lines)


def post_to_discord(message: str):
    resp = requests.post(WEBHOOK_URL, json={"content": message}, timeout=30)
    resp.raise_for_status()


def main():
    print(f"Fetching forecast for {LOCATION_NAME} ({LOCATION_LAT}, {LOCATION_LON})")
    data = fetch_forecast()
    message = summarize(data)
    post_to_discord(message)
    print("Posted forecast.")
    print(message)


if __name__ == "__main__":
    main()
