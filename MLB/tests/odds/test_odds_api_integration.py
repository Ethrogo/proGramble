# MLB/tests/odds/test_odds_api_integration.py

import os

import pandas as pd
import pytest

from odds.odds_api import fetch_all_pitcher_strikeout_props
from odds.normalize import odds_json_to_dataframe


pytestmark = pytest.mark.integration


def test_live_pitcher_strikeout_payload_normalizes():
    """
    Live integration test against The Odds API.

    Requirements:
    - ODDS_API_KEY must be set
    - at least one MLB event must currently expose pitcher_strikeouts
    """
    if not os.getenv("ODDS_API_KEY"):
        pytest.skip("ODDS_API_KEY is not set")

    prop_events = fetch_all_pitcher_strikeout_props()

    if not prop_events:
        pytest.skip("No live pitcher strikeout props available right now")

    df = odds_json_to_dataframe(prop_events)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    required_cols = {
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker",
        "bookmaker_key",
        "book_last_update",
        "market_key",
        "player_name",
        "player_name_norm",
        "side",
        "line",
        "price",
    }
    assert required_cols.issubset(df.columns)

    assert (df["market_key"] == "pitcher_strikeouts").any()
    assert df["player_name"].notna().any()
    assert df["line"].notna().any()
    assert df["price"].notna().any()

    valid_sides = {"Over", "Under"}
    assert set(df["side"].dropna().unique()).issubset(valid_sides)