# MLB/src/odds/compare.py

from __future__ import annotations

from common.contracts import require_columns, assert_non_empty
import pandas as pd
from .normalize import normalize_player_name


def prepare_projection_df(
    projections: pd.DataFrame,
    *,
    participant_key: str = "player_name",
    projection_join_key: str = "player_name_norm",
) -> pd.DataFrame:
    require_columns(
        projections,
        [participant_key, "predicted_strikeouts"],
        "projections_df",
    )
    assert_non_empty(projections, "projections_df")

    df = projections.copy()

    if projection_join_key not in df.columns:
        if participant_key == "player_name" and projection_join_key == "player_name_norm":
            df[projection_join_key] = df[participant_key].apply(normalize_player_name)
        else:
            raise ValueError(
                "projections_df is missing the configured join key "
                f"'{projection_join_key}'."
            )

    return df


def join_projections_to_odds(
    projections: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    participant_key: str = "player_name",
    projection_join_key: str = "player_name_norm",
    odds_join_key: str = "player_name_norm",
) -> pd.DataFrame:
    proj = prepare_projection_df(
        projections,
        participant_key=participant_key,
        projection_join_key=projection_join_key,
    )

    if odds_df.empty:
        return pd.DataFrame()
    require_columns(
        odds_df,
        [odds_join_key, "bookmaker", "side", "line", "price"],
        "odds_df",)

    merged = proj.merge(
        odds_df,
        left_on=projection_join_key,
        right_on=odds_join_key,
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


def best_over_edges(
    joined: pd.DataFrame,
    *,
    group_key: str = "player_name_proj",
) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()

    require_columns(joined, ["side", "edge", group_key], "joined_odds_df")

    over_df = joined[joined["side"].fillna("").str.lower() == "over"].copy()
    if over_df.empty:
        return pd.DataFrame()

    over_df = over_df.sort_values("edge", ascending=False)

    best = (
        over_df.groupby(group_key, as_index=False)
        .first()
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )

    return best
