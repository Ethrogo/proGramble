# MLB/tests/odds/test_compare.py

import pandas as pd
import pytest

from odds.compare import (
    prepare_projection_df,
    join_projections_to_odds,
    join_projections_to_historical_lines,
    best_over_edges,
)


def test_prepare_projection_df_adds_normalized_name():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    result = prepare_projection_df(projections)

    assert "player_name_norm" in result.columns
    assert result.loc[0, "player_name_norm"] == "jacob degrom"


def test_prepare_projection_df_respects_precomputed_join_key():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "participant_norm": "jacob degrom",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    result = prepare_projection_df(
        projections,
        participant_key="player_name",
        projection_join_key="participant_norm",
    )

    assert result.loc[0, "participant_norm"] == "jacob degrom"


def test_join_projections_to_odds_computes_edge():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
                "lower_bound": 5.9,
                "upper_bound": 7.6,
                "std_dev": 0.9,
            }
        ]
    )

    odds_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
            },
            {
                "player_name": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": -140,
            },
        ]
    )

    joined = join_projections_to_odds(projections, odds_df)

    assert len(joined) == 2
    assert "edge" in joined.columns

    dk_edge = joined.loc[joined["bookmaker"] == "DraftKings", "edge"].iloc[0]
    fd_edge = joined.loc[joined["bookmaker"] == "FanDuel", "edge"].iloc[0]

    assert dk_edge == 6.78 - 6.5
    assert fd_edge == 6.78 - 5.5
    assert joined.loc[0, "lower_bound"] == pytest.approx(5.9)
    assert joined.loc[0, "upper_bound"] == pytest.approx(7.6)
    assert joined.loc[0, "std_dev"] == pytest.approx(0.9)


def test_join_projections_to_odds_matches_on_normalized_name():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    odds_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob Degrom",
                "player_name_norm": "jacob degrom",
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
            }
        ]
    )

    joined = join_projections_to_odds(projections, odds_df)

    assert len(joined) == 1
    assert joined.iloc[0]["player_name_proj"] == "Jacob deGrom"


def test_join_projections_to_odds_accepts_configured_join_keys():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "participant_norm": "jacob degrom",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    odds_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob Degrom",
                "odds_participant_norm": "jacob degrom",
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
            }
        ]
    )

    joined = join_projections_to_odds(
        projections,
        odds_df,
        participant_key="player_name",
        projection_join_key="participant_norm",
        odds_join_key="odds_participant_norm",
    )

    assert len(joined) == 1
    assert joined.iloc[0]["player_name_proj"] == "Jacob deGrom"


def test_best_over_edges_filters_out_unders_and_picks_best_book():
    joined = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
                "edge": 0.28,
            },
            {
                "player_name_proj": "Jacob deGrom",
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": -145,
                "edge": 1.28,
            },
            {
                "player_name_proj": "Jacob deGrom",
                "bookmaker": "BetMGM",
                "side": "Under",
                "line": 6.5,
                "price": 100,
                "edge": 0.28,
            },
            {
                "player_name_proj": "Joe Ryan",
                "bookmaker": "Caesars",
                "side": "Over",
                "line": 5.5,
                "price": -110,
                "edge": 0.00,
            },
        ]
    )

    best = best_over_edges(joined)

    assert len(best) == 2
    assert set(best["player_name_proj"]) == {"Jacob deGrom", "Joe Ryan"}

    degrom_row = best.loc[best["player_name_proj"] == "Jacob deGrom"].iloc[0]
    assert degrom_row["bookmaker"] == "FanDuel"
    assert degrom_row["line"] == 5.5
    assert degrom_row["edge"] == 1.28


def test_join_projections_to_odds_returns_empty_when_no_match():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    odds_df = pd.DataFrame(
        [
            {
                "player_name": "Joe Ryan",
                "player_name_norm": "joe ryan",
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    joined = join_projections_to_odds(projections, odds_df)

    assert isinstance(joined, pd.DataFrame)
    assert joined.empty

def test_join_projections_to_odds_returns_empty_when_odds_df_is_empty():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    odds_df = pd.DataFrame()

    joined = join_projections_to_odds(projections, odds_df)

    assert isinstance(joined, pd.DataFrame)
    assert joined.empty

def test_best_over_edges_returns_empty_when_joined_is_empty():
    joined = pd.DataFrame()

    best = best_over_edges(joined)

    assert isinstance(best, pd.DataFrame)
    assert best.empty

def test_best_over_edges_handles_null_side_values():
    joined = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "bookmaker": "DraftKings",
                "side": None,
                "line": 6.5,
                "price": -120,
                "edge": 0.28,
            },
            {
                "player_name_proj": "Joe Ryan",
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 5.5,
                "price": -110,
                "edge": 0.75,
            },
        ]
    )

    best = best_over_edges(joined)

    assert len(best) == 1
    assert best.iloc[0]["player_name_proj"] == "Joe Ryan"

def test_prepare_projection_df_raises_when_required_columns_missing():
    projections = pd.DataFrame(
        [
            {
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    with pytest.raises(ValueError, match="projections_df is missing required columns"):
        prepare_projection_df(projections)


def test_join_projections_to_historical_lines_matches_on_name_and_game_date():
    projections = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob deGrom",
                "predicted_strikeouts": 6.8,
                "actual_strikeouts": 7,
            }
        ]
    )
    historical_lines = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob Degrom",
                "player_name_norm": "jacob degrom",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "selection_rule": "latest_pregame_snapshot_per_game_player_book_side",
                "source": "fixture",
                "pulled_at": "2025-08-02T22:50:00Z",
                "snapshot_type": "selected",
                "is_closing_line": True,
                "snapshot_rank": 1,
            }
        ]
    )

    joined = join_projections_to_historical_lines(projections, historical_lines)

    assert len(joined) == 1
    assert joined.loc[0, "player_name_proj"] == "Jacob deGrom"
    assert joined.loc[0, "bookmaker"] == "DraftKings"
    assert joined.loc[0, "edge"] == pytest.approx(0.3)
