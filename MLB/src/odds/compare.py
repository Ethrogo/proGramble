# MLB/src/odds/compare.py

from __future__ import annotations

from common.contracts import require_columns, assert_non_empty
import pandas as pd
from .normalize import normalize_player_name


def prepare_projection_df(projections: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        projections,
        ["player_name", "predicted_strikeouts"],
        "projections_df",
    )
    assert_non_empty(projections, "projections_df")

    df = projections.copy()
    df["player_name_norm"] = df["player_name"].apply(normalize_player_name)
    return df


def join_projections_to_odds(
    projections: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    proj = prepare_projection_df(projections)
    
    if odds_df.empty:
        return pd.DataFrame()
    require_columns(
        odds_df,
        ["player_name_norm", "bookmaker", "side", "line", "price"],
        "odds_df",)

    merged = proj.merge(
        odds_df,
        on="player_name_norm",
        how="inner",
        suffixes=("_proj", "_odds"),)
    
    if merged.empty:
        return merged

    require_columns(
        merged,
        ["predicted_strikeouts", "line"],
        "joined_odds_df",
        )

    merged["edge"] = merged["predicted_strikeouts"] - merged["line"]

    return merged


def best_over_edges(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()

    require_columns(joined, ["side", "edge", "player_name_proj"], "joined_odds_df")

    over_df = joined[joined["side"].fillna("").str.lower() == "over"].copy()
    if over_df.empty:
        return pd.DataFrame()

    over_df = over_df.sort_values("edge", ascending=False)

    best = (
        over_df.groupby("player_name_proj", as_index=False)
        .first()
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )

    return best