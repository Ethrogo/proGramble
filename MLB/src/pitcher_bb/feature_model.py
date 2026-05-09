import numpy as np
import pandas as pd

from .config import BASE_FEATURES, TARGET_COL

UNCERTAINTY_FEATURE_COLUMNS = [
    "walks_stddev_last10",
    "walks_p25_last10",
    "walks_p75_last10",
]


def get_feature_columns() -> list[str]:
    return list(BASE_FEATURES)


def build_model_df(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    features = get_feature_columns()

    keep_cols = [
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        TARGET_COL,
    ] + features

    extra_cols = [col for col in UNCERTAINTY_FEATURE_COLUMNS if col in pitcher_games.columns]
    keep_cols = keep_cols + [col for col in extra_cols if col not in keep_cols]

    model_df = pitcher_games[keep_cols].copy()
    model_df["game_date"] = pd.to_datetime(model_df["game_date"])
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[TARGET_COL] + features)
    model_df = model_df.sort_values("game_date")
    return model_df
