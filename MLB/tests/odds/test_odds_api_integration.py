# MLB/tests/odds/test_odds_api_integration.py

import os

import pandas as pd
import pytest

from odds.odds_api import fetch_all_player_props
from odds.normalize import odds_json_to_dataframe
from pitcher_bb.config import PITCHER_BB_PROP_MARKET
from pitcher_k.config import PITCHER_K_PROP_MARKET

pytestmark = pytest.mark.integration


def test_live_pitcher_strikeout_payload_normalizes():
    """
    Live integration test against The Odds API.

    Requirements:
    - ODDS_API_KEY must be set
    - at least one MLB event must currently expose pitcher_strikeouts
    """
    if not os.getenv("ODDS_API_KEY", "").strip():
        pytest.skip("ODDS_API_KEY is not set")

    prop_events = fetch_all_player_props(PITCHER_K_PROP_MARKET)

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


def test_live_pitcher_walks_payload_normalizes():
    """
    Live integration test against The Odds API.

    Requirements:
    - ODDS_API_KEY must be set
    - at least one MLB event must currently expose pitcher_walks
    """
    if not os.getenv("ODDS_API_KEY", "").strip():
        pytest.skip("ODDS_API_KEY is not set")

    prop_events = fetch_all_player_props(PITCHER_BB_PROP_MARKET)

    if not prop_events:
        pytest.skip("No live pitcher walks props available right now")

    df = odds_json_to_dataframe(prop_events)

    if df.empty:
        pytest.skip("No live pitcher walks rows normalized from current payload")

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

    walks_df = df[df["market_key"] == PITCHER_BB_PROP_MARKET]
    if walks_df.empty:
        pytest.skip("Current live MLB odds payload did not include pitcher_walks rows")

    assert walks_df["player_name"].notna().any()
    assert walks_df["line"].notna().any()
    assert walks_df["price"].notna().any()

    valid_sides = {"Over", "Under"}
    assert set(walks_df["side"].dropna().unique()).issubset(valid_sides)
