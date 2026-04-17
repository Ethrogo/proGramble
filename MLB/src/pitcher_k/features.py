# src/pitcher_k/features.py

import numpy as np
import pandas as pd


def build_pitcher_game_table(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Statcast data into pitcher-game level stats.
    """
    pitcher_games = (
        sc.groupby(["game_date", "game_pk", "pitcher", "player_name"])
          .agg(
              pitches=("pitch_type", "count"),
              strikeouts=("is_k", "sum"),
              whiffs=("is_whiff", "sum"),
              avg_velo=("release_speed", "mean"),
              avg_spin=("release_spin_rate", "mean"),
              batters_faced=("batter", "nunique")
          )
          .reset_index()
    )

    pitcher_games["whiff_per_pitch"] = (
        pitcher_games["whiffs"] / pitcher_games["pitches"]
    )

    return pitcher_games


def add_rolling_pitcher_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing 3-game and 10-game rolling averages using prior games only.
    """
    pitcher_games = pitcher_games.sort_values(["pitcher", "game_date"]).copy()

    rolling_cols = [
        "strikeouts",
        "pitches",
        "batters_faced",
        "whiff_per_pitch",
        "avg_velo",
        "avg_spin",
    ]

    for col in rolling_cols:
        pitcher_games[f"{col}_last3"] = (
            pitcher_games.groupby("pitcher")[col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

        pitcher_games[f"{col}_last10"] = (
            pitcher_games.groupby("pitcher")[col]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        )

    return pitcher_games


def add_rate_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived strikeout rate features from rolling windows.
    """
    pitcher_games = pitcher_games.copy()

    pitcher_games["k_per_pitch_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["pitches_last10"]
    )

    pitcher_games["k_rate_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["batters_faced_last10"]
    )

    pitcher_games["k_per_pitch_last10"] = pitcher_games["k_per_pitch_last10"].replace(
        [np.inf, -np.inf], np.nan
    )
    pitcher_games["k_rate_last10"] = pitcher_games["k_rate_last10"].replace(
        [np.inf, -np.inf], np.nan
    )

    return pitcher_games


def get_feature_columns() -> list[str]:
    """
    Return the current baseline model feature list.
    """
    return [
        "pitches_last3",
        "pitches_last10",
        "whiff_per_pitch_last3",
        "avg_velo_last3",
        "avg_spin_last3",
        "k_per_pitch_last10",
        "k_rate_last10",
    ]


def build_model_df(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final model-ready dataframe from engineered pitcher-game features.
    """
    features = get_feature_columns()

    model_df = pitcher_games[["game_date", "pitcher", "strikeouts"] + features].copy()
    model_df["game_date"] = pd.to_datetime(model_df["game_date"])

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=["strikeouts"] + features)
    model_df = model_df.sort_values("game_date")

    return model_df