# MLB/src/odds/create_picks.py

from __future__ import annotations

import pandas as pd

from common.contracts import (
    require_columns,
    validate_joined_odds_contract,
    validate_final_picks_contract,
)


OFFICIAL_EDGE_THRESHOLD = 0.75
LEAN_EDGE_THRESHOLD = 0.40


def _normalize_side(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()



def _classify_pick_type(edge: float) -> str:
    abs_edge = abs(edge)

    if abs_edge >= OFFICIAL_EDGE_THRESHOLD:
        return "official"
    if abs_edge >= LEAN_EDGE_THRESHOLD:
        return "lean"
    return "pass"


def _american_odds_sort_key(price: float | int | None) -> float:
    """
    Higher is better for ranking value.
    Example:
      +120 > +100 > -105 > -110 > -120
    """
    if pd.isna(price):
        return float("-inf")
    return float(price)


def _select_best_over_market(player_df: pd.DataFrame) -> pd.Series:
    """
    For overs:
    - prefer the lowest line
    - then prefer the best price
    """
    over_df = player_df[player_df["side_norm"] == "over"].copy()
    if over_df.empty:
        return pd.Series(dtype="object")

    over_df = over_df.sort_values(
        by=["line", "price_sort_key"],
        ascending=[True, False],
    )
    return over_df.iloc[0]


def _select_best_under_market(player_df: pd.DataFrame) -> pd.Series:
    """
    For unders:
    - prefer the highest line
    - then prefer the best price
    """
    under_df = player_df[player_df["side_norm"] == "under"].copy()
    if under_df.empty:
        return pd.Series(dtype="object")

    under_df = under_df.sort_values(
        by=["line", "price_sort_key"],
        ascending=[False, False],
    )
    return under_df.iloc[0]


def _choose_best_market_for_player(player_df: pd.DataFrame) -> pd.Series:
    """
    Choose the best market based on the player's strongest edge direction.
    Expects all rows in player_df to belong to the same player.
    """
    require_columns(
        player_df,
        ["predicted_strikeouts", "side_norm", "line"],
        "player_df",
    )

    predicted = float(player_df["predicted_strikeouts"].iloc[0])

    best_over = _select_best_over_market(player_df)
    best_under = _select_best_under_market(player_df)

    over_edge = None
    under_edge = None

    if not best_over.empty:
        over_edge = predicted - float(best_over["line"])

    if not best_under.empty:
        under_edge = float(best_under["line"]) - predicted

    if over_edge is None and under_edge is None:
        return pd.Series(dtype="object")

    if over_edge is None:
        chosen = best_under.copy()
        chosen["edge"] = under_edge
        chosen["pick_side"] = "under"
        return chosen

    if under_edge is None:
        chosen = best_over.copy()
        chosen["edge"] = over_edge
        chosen["pick_side"] = "over"
        return chosen

    if over_edge >= under_edge:
        chosen = best_over.copy()
        chosen["edge"] = over_edge
        chosen["pick_side"] = "over"
        return chosen

    chosen = best_under.copy()
    chosen["edge"] = under_edge
    chosen["pick_side"] = "under"
    return chosen


def build_daily_picks(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final daily picks from joined projections + odds rows.

    Expected joined_df columns include:
    - player_name_proj (or player_name)
    - team
    - opponent
    - predicted_strikeouts
    - bookmaker
    - side
    - line
    - price

    Returns one best market row per player with:
    - pick_side
    - edge
    - pick_type
    """
    if joined_df.empty:
        return pd.DataFrame()

    df = joined_df.copy()

    if "player_name_proj" not in df.columns:
        if "player_name" in df.columns:
            df["player_name_proj"] = df["player_name"]
        else:
            raise ValueError("joined_df must include 'player_name_proj' or 'player_name'.")

    validate_joined_odds_contract(df)

    required_cols = [
        "player_name_proj",
        "predicted_strikeouts",
        "bookmaker",
        "side",
        "line",
        "price",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for pick creation: {missing}")

    df = df.dropna(subset=["player_name_proj", "predicted_strikeouts", "side", "line"])
    if df.empty:
        return pd.DataFrame()

    df["side_norm"] = df["side"].apply(_normalize_side)
    df["price_sort_key"] = df["price"].apply(_american_odds_sort_key)

    best_rows: list[pd.Series] = []

    for _, player_df in df.groupby("player_name_proj", sort=False):
        best_row = _choose_best_market_for_player(player_df)
        if not best_row.empty:
            best_rows.append(best_row)

    if not best_rows:
        return pd.DataFrame()

    picks = pd.DataFrame(best_rows).reset_index(drop=True)
    picks["pick_type"] = picks["edge"].apply(_classify_pick_type)

    if "player_name" in picks.columns:
        picks["player_name"] = picks["player_name_proj"].combine_first(picks["player_name"])
    else:
        picks["player_name"] = picks["player_name_proj"]

    # Drop the proj column after merge
    picks = picks.drop(columns=["player_name_proj"])

    # Rename bookmaker → book
    picks = picks.rename(columns={"bookmaker": "book"})

    preferred_cols = [
        "player_name",
        "team",
        "opponent",
        "predicted_strikeouts",
        "book",
        "pick_side",
        "line",
        "price",
        "edge",
        "pick_type",
    ]

    existing_cols = [col for col in preferred_cols if col in picks.columns]
    other_cols = [col for col in picks.columns if col not in existing_cols]

    picks = picks[existing_cols + other_cols]

    pick_type_order = {"official": 0, "lean": 1, "pass": 2}
    picks["pick_type_order"] = picks["pick_type"].map(pick_type_order).fillna(99)

    picks = (
        picks.sort_values(
            by=["pick_type_order", "edge"],
            ascending=[True, False],
        )
        .drop(columns=["pick_type_order"])
        .reset_index(drop=True)
    )

    validate_final_picks_contract(picks)
    return picks


def filter_postable_picks(
    picks_df: pd.DataFrame,
    max_official: int = 4,
    max_leans: int = 2,
) -> pd.DataFrame:
    """
    Return a smaller set of picks to post publicly.
    """
    if picks_df.empty:
        return picks_df.copy()

    require_columns(picks_df, ["pick_type"], "picks_df")

    officials = picks_df[picks_df["pick_type"] == "official"].head(max_official)
    leans = picks_df[picks_df["pick_type"] == "lean"].head(max_leans)

    result = pd.concat([officials, leans], ignore_index=True)

    if not result.empty:
        validate_final_picks_contract(result)

    return result