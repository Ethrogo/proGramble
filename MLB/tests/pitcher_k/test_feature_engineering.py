# MLB/tests/pitcher_k/test_feature_engineering.py

import pandas as pd
import pytest

from pitcher_k.feature_engineering import (
    build_pitcher_game_table,
    build_pitcher_team_lookup,
    add_pitcher_team_info,
    add_opponent_k_features,
    add_rolling_pitcher_features,
    add_rate_features,
    build_team_context,
)


def _statcast_df() -> pd.DataFrame:
    rows = []
    for game_idx, game_date in enumerate(
        ["2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13"],
        start=1,
    ):
        game_pk = 1000 + game_idx

        # pitcher 111 on TEX vs SEA
        for batter_id in range(1, 6):
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
                    "description": "swinging_strike" if batter_id % 2 == 0 else "called_strike",
                    "events": "strikeout" if batter_id in (2, 4) else "field_out",
                    "inning": 1,
                    "outs_when_up": 0,
                    "home_team": "TEX",
                    "away_team": "SEA",
                    "stand": "R",
                    "p_throws": "R",
                    "inning_topbot": "Top",
                    "is_k": 1 if batter_id in (2, 4) else 0,
                    "is_whiff": 1 if batter_id % 2 == 0 else 0,
                }
            )

        # opposing pitcher 222 on SEA vs TEX
        for batter_id in range(11, 16):
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
                    "description": "swinging_strike" if batter_id % 2 == 1 else "ball",
                    "events": "strikeout" if batter_id in (11, 13) else "single",
                    "inning": 1,
                    "outs_when_up": 0,
                    "home_team": "TEX",
                    "away_team": "SEA",
                    "stand": "L",
                    "p_throws": "R",
                    "inning_topbot": "Bottom",
                    "is_k": 1 if batter_id in (11, 13) else 0,
                    "is_whiff": 1 if batter_id % 2 == 1 else 0,
                }
            )

    return pd.DataFrame(rows)


def test_build_pitcher_game_table_creates_expected_columns():
    sc = _statcast_df()

    pitcher_games = build_pitcher_game_table(sc)

    expected_cols = {
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "pitches",
        "strikeouts",
        "whiffs",
        "avg_velo",
        "avg_spin",
        "batters_faced",
        "home_team",
        "away_team",
        "p_throws",
        "whiff_per_pitch",
    }
    assert expected_cols.issubset(pitcher_games.columns)
    assert not pitcher_games.empty


def test_build_pitcher_team_lookup_creates_pitching_and_opponent_team():
    sc = _statcast_df()

    team_lookup = build_pitcher_team_lookup(sc)

    assert {"pitching_team", "opponent_team"}.issubset(team_lookup.columns)

    degrom_row = team_lookup.loc[team_lookup["pitcher"] == 111].iloc[0]
    gilbert_row = team_lookup.loc[team_lookup["pitcher"] == 222].iloc[0]

    assert degrom_row["pitching_team"] == "TEX"
    assert degrom_row["opponent_team"] == "SEA"
    assert gilbert_row["pitching_team"] == "SEA"
    assert gilbert_row["opponent_team"] == "TEX"


def test_add_pitcher_team_info_adds_required_columns():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc)

    enriched = add_pitcher_team_info(pitcher_games, sc)

    assert {"pitching_team", "opponent_team"}.issubset(enriched.columns)
    assert enriched["pitching_team"].notna().all()
    assert enriched["opponent_team"].notna().all()


def test_add_opponent_k_features_adds_opponent_context_columns():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)

    enriched = add_opponent_k_features(pitcher_games, sc)

    assert {
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
    }.issubset(enriched.columns)


def test_add_rolling_pitcher_features_creates_critical_rolling_columns():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_k_features(pitcher_games, sc)

    enriched = add_rolling_pitcher_features(pitcher_games)

    expected_cols = {
        "pitches_last3",
        "pitches_last10",
        "whiff_per_pitch_last3",
        "avg_velo_last3",
        "avg_spin_last3",
        "strikeouts_last10",
        "batters_faced_last10",
    }
    assert expected_cols.issubset(enriched.columns)


def test_add_rate_features_creates_required_rate_columns():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_k_features(pitcher_games, sc)
    pitcher_games = add_rolling_pitcher_features(pitcher_games)

    enriched = add_rate_features(pitcher_games)

    assert {"k_per_pitch_last10", "k_rate_last10"}.issubset(enriched.columns)


def test_build_team_context_returns_latest_prior_team_context():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_k_features(pitcher_games, sc)
    pitcher_games = add_rolling_pitcher_features(pitcher_games)
    pitcher_games = add_rate_features(pitcher_games)

    team_context = build_team_context(pitcher_games, as_of_date="2026-04-20")

    assert {
        "opponent_team",
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
    }.issubset(team_context.columns)
    assert not team_context.empty


def test_add_pitcher_team_info_raises_when_required_columns_missing():
    sc = _statcast_df()
    pitcher_games = build_pitcher_game_table(sc).drop(columns=["pitcher"])

    with pytest.raises(ValueError, match="pitcher_games is missing required columns"):
        add_pitcher_team_info(pitcher_games, sc)


def test_build_team_context_raises_when_required_columns_missing():
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2026-04-10",
                "game_pk": 1001,
                "opponent_team": "SEA",
            }
        ]
    )

    with pytest.raises(ValueError, match="pitcher_games is missing required columns"):
        build_team_context(pitcher_games, as_of_date="2026-04-20")