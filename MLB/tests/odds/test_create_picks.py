# MLB/tests/odds/test_create_picks.py

import pandas as pd

from odds.create_picks import build_daily_picks, filter_postable_picks


def test_build_daily_picks_selects_best_over_market():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
            },
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": -140,
            },
        ]
    )

    picks = build_daily_picks(joined_df)

    assert len(picks) == 1
    assert picks.loc[0, "player_name"] == "Jacob deGrom"
    assert picks.loc[0, "book"] == "FanDuel"
    assert picks.loc[0, "pick_side"] == "over"
    assert picks.loc[0, "line"] == 5.5
    assert picks.loc[0, "edge"] == 6.8 - 5.5


def test_build_daily_picks_selects_best_under_market_when_under_edge_is_stronger():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Kyle Leahy",
                "team": "STL",
                "opponent": "HOU",
                "predicted_strikeouts": 1.2,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 2.5,
                "price": 100,
            },
            {
                "player_name_proj": "Kyle Leahy",
                "team": "STL",
                "opponent": "HOU",
                "predicted_strikeouts": 1.2,
                "bookmaker": "Caesars",
                "side": "Under",
                "line": 3.5,
                "price": -125,
            },
        ]
    )

    picks = build_daily_picks(joined_df)

    assert len(picks) == 1
    assert picks.loc[0, "player_name"] == "Kyle Leahy"
    assert picks.loc[0, "pick_side"] == "under"
    assert picks.loc[0, "book"] == "Caesars"
    assert picks.loc[0, "line"] == 3.5
    assert picks.loc[0, "edge"] == 3.5 - 1.2


def test_build_daily_picks_assigns_official_lean_and_pass():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Official Guy",
                "team": "AAA",
                "opponent": "BBB",
                "predicted_strikeouts": 7.0,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.0,
                "price": -110,
            },
            {
                "player_name_proj": "Lean Guy",
                "team": "CCC",
                "opponent": "DDD",
                "predicted_strikeouts": 5.0,
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 4.5,
                "price": -115,
            },
            {
                "player_name_proj": "Pass Guy",
                "team": "EEE",
                "opponent": "FFF",
                "predicted_strikeouts": 4.8,
                "bookmaker": "BetMGM",
                "side": "Over",
                "line": 4.5,
                "price": -110,
            },
        ]
    )

    picks = build_daily_picks(joined_df)

    pick_types = dict(zip(picks["player_name"], picks["pick_type"]))

    assert pick_types["Official Guy"] == "official"
    assert pick_types["Lean Guy"] == "lean"
    assert pick_types["Pass Guy"] == "pass"


def test_build_daily_picks_prefers_better_price_when_lines_match():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Joe Ryan",
                "team": "MIN",
                "opponent": "CIN",
                "predicted_strikeouts": 5.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            },
            {
                "player_name_proj": "Joe Ryan",
                "team": "MIN",
                "opponent": "CIN",
                "predicted_strikeouts": 5.8,
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": 100,
            },
        ]
    )

    picks = build_daily_picks(joined_df)

    assert len(picks) == 1
    assert picks.loc[0, "book"] == "FanDuel"


def test_build_daily_picks_raises_when_required_columns_missing():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
            }
        ]
    )

    import pytest

    with pytest.raises(ValueError, match="Missing required columns for pick creation"):
        build_daily_picks(joined_df)


def test_filter_postable_picks_limits_officials_and_leans():
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "A",
                "pick_type": "official",
                "edge": 1.2,
            },
            {
                "player_name": "B",
                "pick_type": "official",
                "edge": 1.1,
            },
            {
                "player_name": "C",
                "pick_type": "official",
                "edge": 1.0,
            },
            {
                "player_name": "D",
                "pick_type": "lean",
                "edge": 0.6,
            },
            {
                "player_name": "E",
                "pick_type": "lean",
                "edge": 0.5,
            },
            {
                "player_name": "F",
                "pick_type": "pass",
                "edge": 0.1,
            },
        ]
    )

    postable = filter_postable_picks(picks_df, max_official=2, max_leans=1)

    assert len(postable) == 3
    assert list(postable["player_name"]) == ["A", "B", "D"]


def test_build_daily_picks_returns_empty_dataframe_for_empty_input():
    joined_df = pd.DataFrame()

    picks = build_daily_picks(joined_df)

    assert isinstance(picks, pd.DataFrame)
    assert picks.empty