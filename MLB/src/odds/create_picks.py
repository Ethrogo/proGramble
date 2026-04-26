# MLB/src/odds/create_picks.py

from __future__ import annotations

import pandas as pd
from odds.value import american_to_implied_probability
from odds.policy import DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY, PickRankingPolicy

from common.contracts import (
    require_columns,
    validate_joined_odds_contract,
    validate_final_picks_contract,
)


def _normalize_side(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _american_odds_sort_key(price: float | int | None) -> float:
    """
    Higher is better for ranking value.
    Example:
      +120 > +100 > -105 > -110 > -120
    """
    if pd.isna(price):
        return float("-inf")
    return float(price)


def _choose_best_market_for_player(
    player_df: pd.DataFrame,
    policy: PickRankingPolicy,
) -> pd.Series:
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

    best_over = policy.select_best_market(player_df, "over")
    best_under = policy.select_best_market(player_df, "under")

    return policy.choose_pick_side(
        best_over=best_over,
        best_under=best_under,
        predicted=predicted,
    )


def build_daily_picks(
    joined_df: pd.DataFrame,
    policy: PickRankingPolicy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
) -> pd.DataFrame:
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
    df["implied_probability"] = df["price"].apply(american_to_implied_probability)

    best_rows: list[pd.Series] = []

    for _, player_df in df.groupby("player_name_proj", sort=False):
        best_row = _choose_best_market_for_player(player_df, policy)
        if not best_row.empty:
            best_rows.append(best_row)

    if not best_rows:
        return pd.DataFrame()

    picks = pd.DataFrame(best_rows).reset_index(drop=True)
    picks["pick_type"] = picks["edge"].apply(policy.classify_pick_type)
    picks["value_score"] = picks["edge"].abs() * (1 - picks["implied_probability"])
    picks["confidence_tier"] = picks["value_score"].apply(policy.classify_confidence_tier)

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
        "implied_probability",
        "value_score",
        "confidence_tier",
        "pick_type",
    ]

    existing_cols = [col for col in preferred_cols if col in picks.columns]
    other_cols = [col for col in picks.columns if col not in existing_cols]

    picks = picks[existing_cols + other_cols]

    picks = policy.sort_picks(picks)

    validate_final_picks_contract(picks)
    return picks


def filter_postable_picks(
    picks_df: pd.DataFrame,
    max_official: int | None = None,
    max_leans: int | None = None,
    policy: PickRankingPolicy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
) -> pd.DataFrame:
    """
    Return a smaller set of picks to post publicly.
    """
    if picks_df.empty:
        return picks_df.copy()

    require_columns(picks_df, ["pick_type"], "picks_df")

    limits = policy.resolved_postable_limits(
        max_official=max_official,
        max_leans=max_leans,
    )

    officials = picks_df[picks_df["pick_type"] == "official"].head(limits.max_official)
    leans = picks_df[picks_df["pick_type"] == "lean"].head(limits.max_leans)

    result = pd.concat([officials, leans], ignore_index=True)

    if not result.empty:
        validate_final_picks_contract(result)

    return result
