# MLB/tests/odds/test_create_picks.py

import pandas as pd
import pytest

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

    with pytest.raises(ValueError, match="joined_odds_df is missing required columns"):
        build_daily_picks(joined_df)

def test_filter_postable_picks_limits_officials_and_leans():
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "A",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -110,
                "pick_type": "official",
                "edge": 1.2,
                "implied_probability": 110 / 210,
                "value_score": 1.2 * (1 - (110 / 210)),
                "confidence_tier": "medium",
            },
            {
                "player_name": "B",
                "book": "FanDuel",
                "pick_side": "over",
                "line": 5.5,
                "price": -105,
                "pick_type": "official",
                "edge": 1.1,
                "implied_probability": 105 / 205,
                "value_score": 1.1 * (1 - (105 / 205)),
                "confidence_tier": "medium",
            },
            {
                "player_name": "C",
                "book": "BetMGM",
                "pick_side": "over",
                "line": 5.5,
                "price": 100,
                "pick_type": "official",
                "edge": 1.0,
                "implied_probability": 0.5,
                "value_score": 0.5,
                "confidence_tier": "medium",
            },
            {
                "player_name": "D",
                "book": "Caesars",
                "pick_side": "under",
                "line": 6.5,
                "price": -115,
                "pick_type": "lean",
                "edge": 0.6,
                "implied_probability": 115 / 215,
                "value_score": 0.6 * (1 - (115 / 215)),
                "confidence_tier": "low",
            },
            {
                "player_name": "E",
                "book": "DraftKings",
                "pick_side": "under",
                "line": 6.5,
                "price": -110,
                "pick_type": "lean",
                "edge": 0.5,
                "implied_probability": 110 / 210,
                "value_score": 0.5 * (1 - (110 / 210)),
                "confidence_tier": "low",
            },
            {
                "player_name": "F",
                "book": "FanDuel",
                "pick_side": "over",
                "line": 4.5,
                "price": -110,
                "pick_type": "pass",
                "edge": 0.1,
                "implied_probability": 110 / 210,
                "value_score": 0.1 * (1 - (110 / 210)),
                "confidence_tier": "thin",
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

def test_build_daily_picks_drops_rows_with_null_side_and_returns_empty_if_nothing_left():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": None,
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    with pytest.raises(ValueError, match="has null values in required columns"):
        build_daily_picks(joined_df)

def test_build_daily_picks_uses_player_name_fallback_when_player_name_proj_missing():
    joined_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    picks = build_daily_picks(joined_df)

    assert len(picks) == 1
    assert picks.loc[0, "player_name"] == "Jacob deGrom"
    assert picks.loc[0, "book"] == "DraftKings"

def test_filter_postable_picks_raises_when_pick_type_missing():
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "A",
                "edge": 1.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="picks_df is missing required columns"):
        filter_postable_picks(picks_df)

def test_build_daily_picks_adds_implied_probability_value_score_and_confidence_tier():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.6,
                "upper_bound": 8.0,
                "std_dev": 1.2,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    picks = build_daily_picks(joined_df)

    assert "implied_probability" in picks.columns
    assert "value_score" in picks.columns
    assert "confidence_tier" in picks.columns

    expected_edge = 6.8 - 5.5
    expected_implied_probability = 120 / 220
    expected_value_score = abs(expected_edge) * (1 - expected_implied_probability)

    assert picks.loc[0, "edge"] == pytest.approx(expected_edge)
    assert picks.loc[0, "implied_probability"] == pytest.approx(expected_implied_probability)
    assert picks.loc[0, "value_score"] == pytest.approx(expected_value_score)
    assert picks.loc[0, "confidence_tier"] in {"high", "medium", "low", "thin"}


def test_build_daily_picks_preserves_projection_uncertainty_fields():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.6,
                "upper_bound": 8.0,
                "std_dev": 1.2,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    picks = build_daily_picks(joined_df)

    assert picks.loc[0, "lower_bound"] == pytest.approx(5.6)
    assert picks.loc[0, "upper_bound"] == pytest.approx(8.0)
    assert picks.loc[0, "std_dev"] == pytest.approx(1.2)

def test_build_daily_picks_ranks_same_pick_type_by_value_score_not_raw_edge():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Expensive Bigger Edge",
                "team": "AAA",
                "opponent": "BBB",
                "predicted_strikeouts": 6.3,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -300,
            },
            {
                "player_name_proj": "Plus Money Smaller Edge",
                "team": "CCC",
                "opponent": "DDD",
                "predicted_strikeouts": 6.26,
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": 150,
            },
        ]
    )

    picks = build_daily_picks(joined_df)

    assert len(picks) == 2
    assert list(picks["pick_type"]) == ["official", "official"]
    assert picks.loc[0, "player_name"] == "Plus Money Smaller Edge"
    assert picks.loc[0, "value_score"] > picks.loc[1, "value_score"]
    assert picks.loc[0, "edge"] < picks.loc[1, "edge"]

def test_filter_postable_picks_preserves_value_fields():
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "A",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -110,
                "edge": 1.0,
                "implied_probability": 110 / 210,
                "value_score": 1.0 * (1 - (110 / 210)),
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )

    postable = filter_postable_picks(picks_df, max_official=1, max_leans=0)

    assert len(postable) == 1
    assert "implied_probability" in postable.columns
    assert "value_score" in postable.columns
    assert "confidence_tier" in postable.columns
