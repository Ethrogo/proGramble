from __future__ import annotations

import requests
from .config import ODDS_API_BASE, ODDS_API_KEY, ODDS_SPORT, ODDS_MARKET, BOOKMAKERS


def fetch_event_odds(
    sport: str = ODDS_SPORT,
    market: str = ODDS_MARKET,
    bookmakers: list[str] | None = None,
    odds_format: str = "american",
    date_format: str = "iso",
) -> list[dict]:
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is missing")

    if bookmakers is None:
        bookmakers = BOOKMAKERS

    url = f"{ODDS_API_BASE}/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market,
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()