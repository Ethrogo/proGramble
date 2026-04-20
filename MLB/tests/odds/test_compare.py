# MLB/tests/odds/test_compare.py

import pandas as pd

from odds.compare import (
    prepare_projection_df,
    join_projections_to_odds,
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


def test_join_projections_to_odds_computes_edge():
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