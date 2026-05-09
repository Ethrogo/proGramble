# MLB/src/odds/compare.py

from __future__ import annotations

import pandas as pd

from common.contracts import assert_non_empty, require_columns
from common.identity import (
    MARKET_FAMILY_COLUMN,
    MARKET_OFFER_KEY_COLUMN,
    MARKET_SELECTION_KEY_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_JOIN_KEY_COLUMN,
    PARTICIPANT_NAME_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    ensure_market_identity,
    ensure_participant_identity,
)

RESOLVED_IDENTITY_COLUMNS = [
    PARTICIPANT_NAME_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    PARTICIPANT_JOIN_KEY_COLUMN,
    "sport",
    "market_key",
    MARKET_FAMILY_COLUMN,
    "bookmaker_key",
    "event_id",
    "side_norm",
    MARKET_SELECTION_KEY_COLUMN,
    MARKET_OFFER_KEY_COLUMN,
]


def _resolve_merged_identity_columns(merged: pd.DataFrame, right_suffix: str) -> pd.DataFrame:
    resolved = merged.copy()
    for column in RESOLVED_IDENTITY_COLUMNS:
        if column in resolved.columns:
            continue

        left_column = f"{column}_proj"
        right_column = f"{column}_{right_suffix}"
        if left_column in resolved.columns and right_column in resolved.columns:
            resolved[column] = resolved[left_column].combine_first(resolved[right_column])
        elif left_column in resolved.columns:
            resolved[column] = resolved[left_column]
        elif right_column in resolved.columns:
            resolved[column] = resolved[right_column]
    return resolved


def _shared_market_join_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [column for column in ["sport", "market_key"] if column in left.columns and column in right.columns]


def _resolve_prediction_column(
    df: pd.DataFrame,
    *,
    requested: str = "predicted_value",
    context_name: str,
) -> str:
    if requested in df.columns:
        return requested

    candidates = [column for column in df.columns if column.startswith("predicted_")]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"{context_name} is missing the configured prediction column '{requested}'."
    )


def prepare_projection_df(
    projections: pd.DataFrame,
    *,
    participant_key: str = "player_name",
    prediction_column: str = "predicted_value",
    projection_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    sport: str | None = None,
    market_key: str | None = None,
) -> pd.DataFrame:
    require_columns(projections, [participant_key], "projections_df")
    assert_non_empty(projections, "projections_df")

    df = projections.copy()
    resolved_prediction_column = _resolve_prediction_column(
        df,
        requested=prediction_column,
        context_name="projections_df",
    )
    if prediction_column not in df.columns:
        df[prediction_column] = df[resolved_prediction_column]

    df = ensure_participant_identity(
        df,
        display_name_col=participant_key,
        normalized_name_col="player_name_norm" if participant_key == "player_name" else None,
        source_id_col="pitcher" if "pitcher" in df.columns else None,
        source_id_type="mlbam_player",
    )
    df = ensure_market_identity(df, sport=sport, market_key=market_key)

    if projection_join_key == "player_name_norm":
        df[projection_join_key] = df[PARTICIPANT_NAME_NORM_COLUMN]
    elif projection_join_key == PARTICIPANT_JOIN_KEY_COLUMN:
        df[projection_join_key] = df[PARTICIPANT_JOIN_KEY_COLUMN]
    elif projection_join_key not in df.columns:
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
    prediction_column: str = "predicted_value",
    projection_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    odds_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    sport: str | None = None,
    market_key: str | None = None,
) -> pd.DataFrame:
    proj = prepare_projection_df(
        projections,
        participant_key=participant_key,
        prediction_column=prediction_column,
        projection_join_key=projection_join_key,
        sport=sport,
        market_key=market_key,
    )

    if odds_df.empty:
        return pd.DataFrame()
    odds = ensure_participant_identity(
        odds_df,
        display_name_col="player_name",
        normalized_name_col="player_name_norm",
    )
    odds = ensure_market_identity(odds, sport=sport, market_key=market_key)
    require_columns(
        odds,
        [odds_join_key, "bookmaker", "side", "line", "price"],
        "odds_df",
    )

    optional_join_columns = _shared_market_join_columns(proj, odds)
    left_join_keys = [projection_join_key, *optional_join_columns]
    right_join_keys = [odds_join_key, *optional_join_columns]

    merged = proj.merge(
        odds,
        left_on=left_join_keys,
        right_on=right_join_keys,
        how="inner",
        suffixes=("_proj", "_odds"),
    )

    if merged.empty:
        return merged

    merged = _resolve_merged_identity_columns(merged, "odds")

    require_columns(
        merged,
        [prediction_column, "line"],
        "joined_odds_df",
    )
    if "predicted_value" not in merged.columns:
        merged["predicted_value"] = merged[prediction_column]
    merged["edge"] = merged[prediction_column] - merged["line"]

    return merged


def join_projections_to_historical_lines(
    projections: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
    *,
    participant_key: str = "player_name",
    prediction_column: str = "predicted_value",
    projection_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    lines_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    projection_date_key: str = "game_date",
    lines_date_key: str = "game_date",
    sport: str | None = None,
    market_key: str | None = None,
) -> pd.DataFrame:
    proj = prepare_projection_df(
        projections,
        participant_key=participant_key,
        prediction_column=prediction_column,
        projection_join_key=projection_join_key,
        sport=sport,
        market_key=market_key,
    )

    if historical_lines_df.empty:
        return pd.DataFrame()

    lines = ensure_participant_identity(
        historical_lines_df,
        display_name_col="player_name",
        normalized_name_col="player_name_norm",
    )
    lines = ensure_market_identity(lines, sport=sport, market_key=market_key)

    require_columns(
        lines,
        [lines_join_key, lines_date_key, "bookmaker", "side", "line", "price"],
        "historical_lines_df",
    )
    require_columns(proj, [projection_date_key], "projections_df")

    proj = proj.copy()
    proj[projection_date_key] = pd.to_datetime(proj[projection_date_key]).dt.strftime("%Y-%m-%d")
    lines[lines_date_key] = pd.to_datetime(lines[lines_date_key]).dt.strftime("%Y-%m-%d")

    optional_join_columns = _shared_market_join_columns(proj, lines)
    left_join_keys = [projection_join_key, projection_date_key, *optional_join_columns]
    right_join_keys = [lines_join_key, lines_date_key, *optional_join_columns]

    merged = proj.merge(
        lines,
        left_on=left_join_keys,
        right_on=right_join_keys,
        how="inner",
        suffixes=("_proj", "_line"),
    )

    if merged.empty:
        return merged

    merged = _resolve_merged_identity_columns(merged, "line")

    require_columns(
        merged,
        [prediction_column, "line"],
        "joined_historical_lines_df",
    )
    if "predicted_value" not in merged.columns:
        merged["predicted_value"] = merged[prediction_column]
    merged["edge"] = merged[prediction_column] - merged["line"]
    return merged


def best_over_edges(
    joined: pd.DataFrame,
    *,
    group_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()

    if group_key not in joined.columns:
        group_key = "player_name_proj"

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
