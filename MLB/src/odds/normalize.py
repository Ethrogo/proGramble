from __future__ import annotations

import pandas as pd
from .config import BOOK_DISPLAY_NAMES


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return (
        name.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def odds_json_to_dataframe(events: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for event in events:
        event_id = event.get("id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key")
            book_name = BOOK_DISPLAY_NAMES.get(book_key, book_key)
            last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_name,
                            "bookmaker_key": book_key,
                            "book_last_update": last_update,
                            "market_key": market_key,
                            "player_name": outcome.get("description"),
                            "player_name_norm": normalize_player_name(outcome.get("description", "")),
                            "side": outcome.get("name"),
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                        }
                    )

    return pd.DataFrame(rows)