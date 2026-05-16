# MLB/src/odds/odds_api.py

from __future__ import annotations

import requests

from .config import (
    ODDS_API_BASE,
    ODDS_API_KEY,
    ODDS_SPORT,
    EVENT_DISCOVERY_MARKET,
    BOOKMAKERS,
    BOOK_DISPLAY_NAMES,
    BOOKMAKER_NOTES,
)


def summarize_event_bookmaker_coverage(
    prop_events: list[dict],
    *,
    requested_bookmakers: list[str] | None = None,
) -> dict[str, object]:
    requested = requested_bookmakers or []
    upstream_bookmaker_keys = sorted(
        {
            bookmaker.get("key", "")
            for event in prop_events
            for bookmaker in event.get("bookmakers", [])
            if bookmaker.get("key")
        }
    )
    missing_bookmakers = [book for book in requested if book not in upstream_bookmaker_keys]
    return {
        "requested_bookmakers": requested,
        "upstream_bookmaker_keys": upstream_bookmaker_keys,
        "upstream_bookmaker_names": [
            BOOK_DISPLAY_NAMES.get(bookmaker_key, bookmaker_key)
            for bookmaker_key in upstream_bookmaker_keys
        ],
        "missing_requested_bookmakers": missing_bookmakers,
        "missing_requested_notes": {
            bookmaker_key: BOOKMAKER_NOTES.get(bookmaker_key, "")
            for bookmaker_key in missing_bookmakers
            if BOOKMAKER_NOTES.get(bookmaker_key)
        },
    }


def fetch_mlb_events(
    sport: str = ODDS_SPORT,
    market: str = EVENT_DISCOVERY_MARKET,
    bookmakers: list[str] | None = None,
    *,
    use_configured_bookmakers: bool = True,
    odds_format: str = "american",
    date_format: str = "iso",
) -> list[dict]:
    """
    Fetch MLB events using a sport-level market like h2h.
    This is used only to get event IDs.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is missing")

    if bookmakers is None and use_configured_bookmakers:
        bookmakers = BOOKMAKERS

    url = f"{ODDS_API_BASE}/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_event_player_props(
    event_id: str,
    market: str,
    sport: str = ODDS_SPORT,
    bookmakers: list[str] | None = None,
    *,
    use_configured_bookmakers: bool = True,
    odds_format: str = "american",
    date_format: str = "iso",
) -> dict:
    """
    Fetch player prop odds for a single MLB event.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is missing")

    if bookmakers is None and use_configured_bookmakers:
        bookmakers = BOOKMAKERS

    url = f"{ODDS_API_BASE}/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_player_props(
    market: str,
    sport: str = ODDS_SPORT,
    bookmakers: list[str] | None = None,
    *,
    use_configured_bookmakers: bool = True,
) -> list[dict]:
    """
    Fetch player prop odds for all today's MLB events for the given market.
    Returns a list of event-level prop payloads.
    """
    if bookmakers is None and use_configured_bookmakers:
        bookmakers = BOOKMAKERS

    events = fetch_mlb_events(
        sport=sport,
        bookmakers=bookmakers,
        use_configured_bookmakers=use_configured_bookmakers,
    )
    prop_events: list[dict] = []
    failed_event_ids: list[str] = []

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
                use_configured_bookmakers=use_configured_bookmakers,
            )

            if prop_data and prop_data.get("bookmakers"):
                prop_events.append(prop_data)

        except requests.HTTPError:
            failed_event_ids.append(str(event_id))
            continue

    coverage = summarize_event_bookmaker_coverage(
        prop_events,
        requested_bookmakers=bookmakers,
    )
    print(
        "Live odds coverage:"
        f" requested={coverage['requested_bookmakers']}"
        f" upstream={coverage['upstream_bookmaker_keys']}"
        f" missing={coverage['missing_requested_bookmakers']}"
    )
    if coverage["missing_requested_notes"]:
        print(f"Live odds coverage notes: {coverage['missing_requested_notes']}")
    if failed_event_ids:
        print(
            "WARNING: Live odds event fetches failed for event ids: "
            f"{failed_event_ids}"
        )

    return prop_events
