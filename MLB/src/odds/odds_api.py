# MLB/src/odds/odds_api.py

from __future__ import annotations

import requests

from .config import (
    ODDS_API_BASE,
    ODDS_API_KEY,
    ODDS_SPORT,
    EVENT_DISCOVERY_MARKET,
    BOOKMAKERS,
)


def fetch_mlb_events(
    sport: str = ODDS_SPORT,
    market: str = EVENT_DISCOVERY_MARKET,
    bookmakers: list[str] | None = None,
    odds_format: str = "american",
    date_format: str = "iso",
) -> list[dict]:
    """
    Fetch MLB events using a sport-level market like h2h.
    This is used only to get event IDs.
    """
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


def fetch_event_player_props(
    event_id: str,
    market: str,
    sport: str = ODDS_SPORT,
    bookmakers: list[str] | None = None,
    odds_format: str = "american",
    date_format: str = "iso",
) -> dict:
    """
    Fetch player prop odds for a single MLB event.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is missing")

    if bookmakers is None:
        bookmakers = BOOKMAKERS

    url = f"{ODDS_API_BASE}/{sport}/events/{event_id}/odds"
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


def fetch_all_player_props(
    market: str,
    sport: str = ODDS_SPORT,
    bookmakers: list[str] | None = None,
) -> list[dict]:
    """
    Fetch player prop odds for all today's MLB events for the given market.
    Returns a list of event-level prop payloads.
    """
    events = fetch_mlb_events(sport=sport)
    prop_events: list[dict] = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        try:
            prop_data = fetch_event_player_props(
                event_id=event_id,
                sport=sport,
                market=market,
                bookmakers=bookmakers,
            )

            if prop_data and prop_data.get("bookmakers"):
                prop_events.append(prop_data)

        except requests.HTTPError:
            continue

    return prop_events