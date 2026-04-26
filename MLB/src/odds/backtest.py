from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.contracts import require_columns, validate_joined_odds_contract
from odds.create_picks import build_daily_picks
from odds.policy import DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY, PickRankingPolicy

BACKTEST_REQUIRED_COLUMNS = ["actual_strikeouts"]


def _line_band(line: float) -> str:
    if pd.isna(line):
        return "unknown"
    if line <= 4.5:
        return "<=4.5"
    if line <= 5.5:
        return "4.5-5.5"
    if line <= 6.5:
        return "5.5-6.5"
    if line <= 7.5:
        return "6.5-7.5"
    return "7.5+"


def _grade_pick_outcome(actual: float, line: float, pick_side: str) -> str:
    if pick_side == "over":
        if actual > line:
            return "win"
        if actual < line:
            return "loss"
        return "push"

    if actual < line:
        return "win"
    if actual > line:
        return "loss"
    return "push"


def _american_odds_profit_units(price: float, outcome: str) -> float:
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -1.0

    price = float(price)
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def grade_pick_backtest(
    picks_df: pd.DataFrame,
    *,
    actual_column: str = "actual_strikeouts",
) -> pd.DataFrame:
    require_columns(
        picks_df,
        [
            "player_name",
            "book",
            "pick_side",
            "line",
            "price",
            "pick_type",
            "confidence_tier",
            actual_column,
        ],
        "picks_backtest_df",
    )

    graded = picks_df.copy()
    graded["outcome"] = graded.apply(
        lambda row: _grade_pick_outcome(
            actual=float(row[actual_column]),
            line=float(row["line"]),
            pick_side=str(row["pick_side"]).lower(),
        ),
        axis=1,
    )
    graded["is_win"] = graded["outcome"] == "win"
    graded["is_loss"] = graded["outcome"] == "loss"
    graded["is_push"] = graded["outcome"] == "push"
    graded["decision"] = ~graded["is_push"]
    graded["profit_units"] = graded.apply(
        lambda row: _american_odds_profit_units(float(row["price"]), str(row["outcome"])),
        axis=1,
    )
    graded["line_band"] = graded["line"].apply(_line_band)
    return graded


def _summarize_groups(graded_df: pd.DataFrame, group_column: str | None = None) -> list[dict]:
    if graded_df.empty:
        return []

    working = graded_df.copy()
    if group_column is None:
        working = working.assign(_overall="overall")
        group_cols = ["_overall"]
    else:
        group_cols = [group_column]

    summary = (
        working.groupby(group_cols, dropna=False)
        .agg(
            picks=("player_name", "size"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            decisions=("decision", "sum"),
            avg_edge=("edge", "mean"),
            avg_value_score=("value_score", "mean"),
            profit_units=("profit_units", "sum"),
        )
        .reset_index()
    )
    summary["win_rate"] = np.where(
        summary["decisions"] > 0,
        summary["wins"] / summary["decisions"],
        np.nan,
    )
    summary["roi_per_pick"] = np.where(
        summary["picks"] > 0,
        summary["profit_units"] / summary["picks"],
        np.nan,
    )

    records = summary.to_dict(orient="records")
    for record in records:
        if "_overall" in record:
            record.pop("_overall", None)
        for key, value in list(record.items()):
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def run_pick_backtest(
    joined_df: pd.DataFrame,
    *,
    policy: PickRankingPolicy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    actual_column: str = "actual_strikeouts",
) -> dict:
    """
    Backtest the same market-selection and pick-tier workflow used for live cards.
    """
    validate_joined_odds_contract(joined_df)
    require_columns(joined_df, BACKTEST_REQUIRED_COLUMNS, "joined_odds_backtest_df")

    scored_input = joined_df.dropna(subset=[actual_column]).copy()
    if scored_input.empty:
        return {
            "available": False,
            "reason": "no_rows_with_realized_outcomes",
            "overall": [],
            "by_pick_type": [],
            "by_confidence_tier": [],
            "by_book": [],
            "by_line_band": [],
            "by_pick_side": [],
            "graded_picks": pd.DataFrame(),
        }

    picks = build_daily_picks(scored_input, policy=policy)
    if picks.empty:
        return {
            "available": True,
            "overall": [],
            "by_pick_type": [],
            "by_confidence_tier": [],
            "by_book": [],
            "by_line_band": [],
            "by_pick_side": [],
            "graded_picks": picks,
        }

    graded = grade_pick_backtest(picks, actual_column=actual_column)
    return {
        "available": True,
        "overall": _summarize_groups(graded),
        "by_pick_type": _summarize_groups(graded, "pick_type"),
        "by_confidence_tier": _summarize_groups(graded, "confidence_tier"),
        "by_book": _summarize_groups(graded, "book"),
        "by_line_band": _summarize_groups(graded, "line_band"),
        "by_pick_side": _summarize_groups(graded, "pick_side"),
        "graded_picks": graded,
    }
