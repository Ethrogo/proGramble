from __future__ import annotations

import pandas as pd
from .normalize import normalize_player_name


def prepare_projection_df(projections: pd.DataFrame) -> pd.DataFrame:
    df = projections.copy()
    df["player_name_norm"] = df["player_name"].apply(normalize_player_name)
    return df


def join_projections_to_odds(
    projections: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    proj = prepare_projection_df(projections)

    merged = proj.merge(
        odds_df,
        on="player_name_norm",
        how="inner",
        suffixes=("_proj", "_odds"),
    )

    merged["edge"] = merged["predicted_strikeouts"] - merged["line"]

    return merged


def best_over_edges(joined: pd.DataFrame) -> pd.DataFrame:
    over_df = joined[joined["side"].str.lower() == "over"].copy()
    over_df = over_df.sort_values("edge", ascending=False)

    best = (
        over_df.groupby("player_name_proj", as_index=False)
        .first()
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )

    return best