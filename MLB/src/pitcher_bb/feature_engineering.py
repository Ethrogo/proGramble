from __future__ import annotations

import pandas as pd

from common.contracts import require_columns, validate_pitcher_games_contract
from pitcher_k.feature_engineering import (
    add_opponent_bb_features,
    add_pitcher_team_info,
    add_rate_features,
    add_rolling_pitcher_features,
    build_pitcher_game_table,
    filter_starter_like_appearances,
)


def build_pitcher_walk_game_table(sc: pd.DataFrame) -> pd.DataFrame:
    pitcher_games = build_pitcher_game_table(sc)
    require_columns(pitcher_games, ["walks"], "pitcher_games")
    return pitcher_games


def build_pitcher_walk_feature_table(sc: pd.DataFrame) -> pd.DataFrame:
    pitcher_games = build_pitcher_walk_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_bb_features(pitcher_games, sc)
    pitcher_games = add_rolling_pitcher_features(pitcher_games)
    pitcher_games = add_rate_features(pitcher_games)
    validate_pitcher_games_contract(pitcher_games)
    require_columns(
        pitcher_games,
        [
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
        ],
        "pitcher_games",
    )
    return pitcher_games


__all__ = [
    "build_pitcher_walk_game_table",
    "build_pitcher_walk_feature_table",
    "filter_starter_like_appearances",
]
