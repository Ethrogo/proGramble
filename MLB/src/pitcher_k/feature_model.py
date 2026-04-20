# src/pitcher_k/features.py

import numpy as np
import pandas as pd



def get_feature_columns() -> list[str]:
    """
    Return the current model feature list.
    """
    return [
        "pitches_last3",
        "pitches_last10",
        "whiff_per_pitch_last3",
        "avg_velo_last3",
        "avg_spin_last3",
        "k_per_pitch_last10",
        "k_rate_last10",
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
    ]



def build_model_df(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final model-ready dataframe from engineered pitcher-game features.
    """
    features = get_feature_columns()

    keep_cols = [
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "strikeouts",
    ] + features

    model_df = pitcher_games[keep_cols].copy()
    model_df["game_date"] = pd.to_datetime(model_df["game_date"])

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=["strikeouts"] + features)
    model_df = model_df.sort_values("game_date")

    return model_df

