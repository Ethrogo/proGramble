import pandas as pd
import pytest

from pitcher_bb.feature_engineering import (
    build_pitcher_walk_feature_table,
    build_pitcher_walk_game_table,
)


def _statcast_df() -> pd.DataFrame:
    rows = []
    walk_pattern = {
        "2026-04-10": {1},
        "2026-04-11": {1, 2},
        "2026-04-12": {1},
        "2026-04-13": {1, 2, 3},
    }
    opp_walk_pattern = {
        "2026-04-10": {11},
        "2026-04-11": {11, 12},
        "2026-04-12": {11},
        "2026-04-13": {11, 12, 13},
    }

    for game_idx, game_date in enumerate(
        ["2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13"],
        start=1,
    ):
        game_pk = 3000 + game_idx

        for batter_id in range(1, 6):
            event = "walk" if batter_id in walk_pattern[game_date] else "field_out"
            rows.append(
                {
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "pitcher": 111,
                    "player_name": "Jacob deGrom",
                    "batter": batter_id,
                    "pitch_type": "FF",
                    "release_speed": 97.0 + game_idx,
                    "release_spin_rate": 2450 + game_idx,
                    "description": "ball" if event == "walk" else "called_strike",
                    "events": event,
                    "inning": 1,
                    "outs_when_up": 0,
                    "home_team": "TEX",
                    "away_team": "SEA",
                    "stand": "R",
                    "p_throws": "R",
                    "inning_topbot": "Top",
                }
            )

        for batter_id in range(11, 16):
            event = "walk" if batter_id in opp_walk_pattern[game_date] else "single"
            rows.append(
                {
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "pitcher": 222,
                    "player_name": "Logan Gilbert",
                    "batter": batter_id,
                    "pitch_type": "FF",
                    "release_speed": 95.0 + game_idx,
                    "release_spin_rate": 2350 + game_idx,
                    "description": "ball" if event == "walk" else "in_play",
                    "events": event,
                    "inning": 1,
                    "outs_when_up": 0,
                    "home_team": "TEX",
                    "away_team": "SEA",
                    "stand": "L",
                    "p_throws": "R",
                    "inning_topbot": "Bottom",
                }
            )

    return pd.DataFrame(rows)


def test_build_pitcher_walk_game_table_adds_walk_target():
    pitcher_games = build_pitcher_walk_game_table(_statcast_df())

    assert "walks" in pitcher_games.columns
    degrom_row = pitcher_games.loc[pitcher_games["game_pk"] == 3001].iloc[0]
    assert degrom_row["walks"] == 1


def test_build_pitcher_walk_feature_table_adds_walk_specific_features():
    pitcher_games = build_pitcher_walk_feature_table(_statcast_df())

    expected_cols = {
        "walks",
        "walks_last3",
        "walks_last10",
        "walks_stddev_last10",
        "walks_p25_last10",
        "walks_p75_last10",
        "bb_per_pitch_last10",
        "bb_rate_last10",
        "opp_walks_per_game_last10",
        "opp_bb_rate_last10",
    }
    assert expected_cols.issubset(pitcher_games.columns)

    degrom_rows = pitcher_games[pitcher_games["pitcher"] == 111].sort_values("game_pk").reset_index(drop=True)
    assert pd.isna(degrom_rows.loc[2, "walks_stddev_last10"])
    assert degrom_rows.loc[3, "walks_last10"] == pytest.approx(pd.Series([1.0, 2.0, 1.0]).mean())
    assert degrom_rows.loc[3, "walks_stddev_last10"] == pytest.approx(pd.Series([1.0, 2.0, 1.0]).std(ddof=0))
