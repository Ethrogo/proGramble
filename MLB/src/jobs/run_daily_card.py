# MLB/src/jobs/run_daily_card.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from starters.today_starters import get_today_starters_df, save_today_starters_csv

from odds.run_edges import run_edge_pipeline
from odds.create_picks import build_daily_picks, filter_postable_picks
from common.contracts import (
    FINAL_PICKS_REQUIRED_COLUMNS,
    JOINED_ODDS_REQUIRED_COLUMNS,
    validate_starters_contract,
    validate_joined_odds_contract,
    validate_final_picks_contract,
    assert_non_empty,
    require_columns,
)
from common.workflows import ModelingWorkflowSpec
from pitcher_k.workflow import MLB_PITCHER_STRIKEOUT_WORKFLOW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
LATEST_ARTIFACTS_DIR = ARTIFACTS_DIR / "latest"
PREVIOUS_ARTIFACTS_DIR = ARTIFACTS_DIR / "previous"

OUTPUT_DIR = DATA_DIR / "outputs"
TRACKING_DIR = DATA_DIR / "tracking"

PROJECTIONS_DIR = OUTPUT_DIR / "projections"
EDGES_DIR = OUTPUT_DIR / "edges"
PICKS_DIR = OUTPUT_DIR / "picks"
RUN_STATUS_PATH = OUTPUT_DIR / "run_daily_card_status.json"
OFFICIAL_PICKS_HISTORY_PATH = TRACKING_DIR / "official_picks_history.csv"
OFFICIAL_PICKS_GRADES_PATH = TRACKING_DIR / "official_picks_profit_report.csv"
OFFICIAL_PICKS_BOOK_SUMMARY_PATH = TRACKING_DIR / "official_picks_profit_by_book.csv"
OFFICIAL_PICKS_OVERALL_SUMMARY_PATH = TRACKING_DIR / "official_picks_profit_summary.json"
OFFICIAL_PICKS_SKIPPED_PATH = TRACKING_DIR / "official_picks_profit_skipped.csv"

OFFICIAL_PICKS_HISTORY_COLUMNS = [
    "pick_key",
    "game_date",
    "player_name",
    "team",
    "opponent",
    "book",
    "odds",
    "price",
    "pick_side",
    "line",
    "predicted_strikeouts",
    "edge",
    "confidence_tier",
    "pick_type",
    "result",
    "actual_strikeouts",
    "record_source",
]

OFFICIAL_PICKS_PROFIT_REPORT_COLUMNS = OFFICIAL_PICKS_HISTORY_COLUMNS + [
    "result_normalized",
    "odds_numeric",
    "units_risked",
    "units_result",
]

OFFICIAL_PICKS_PROFIT_SUMMARY_COLUMNS = [
    "book",
    "picks",
    "wins",
    "losses",
    "pushes",
    "decisions",
    "units_risked",
    "units_profit",
    "win_rate",
    "roi",
]

BuildPicksFn = Callable[[pd.DataFrame], pd.DataFrame]
FilterPostablePicksFn = Callable[[pd.DataFrame], pd.DataFrame]


def ensure_output_dirs() -> None:
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)


def empty_official_picks_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_HISTORY_COLUMNS)


def empty_joined_odds_df() -> pd.DataFrame:
    return pd.DataFrame(columns=JOINED_ODDS_REQUIRED_COLUMNS)


def empty_final_picks_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_PICKS_REQUIRED_COLUMNS)


def _format_american_odds(price: float | int | str | None) -> str:
    if pd.isna(price):
        return ""

    numeric_price = int(float(price))
    if numeric_price > 0:
        return f"+{numeric_price}"
    return str(numeric_price)


def _normalize_pick_key_name(player_name: str) -> str:
    return " ".join(str(player_name).strip().lower().split())


def _build_pick_key(game_date: str, player_name: str) -> str:
    return f"{game_date}|{_normalize_pick_key_name(player_name)}"


def _normalize_pick_result(result: str | None) -> str:
    normalized = str(result or "").strip().lower()
    if normalized in {"w", "win"}:
        return "W"
    if normalized in {"l", "loss"}:
        return "L"
    if normalized == "push":
        return "Push"
    return ""


def _parse_american_odds(odds: float | int | str | None) -> float | None:
    if pd.isna(odds):
        return None

    text = str(odds).strip()
    if not text:
        return None

    try:
        numeric_odds = float(text)
    except ValueError:
        return None

    if numeric_odds == 0:
        return None

    return numeric_odds


def _resolve_history_row_odds(row: pd.Series) -> float | None:
    odds_value = _parse_american_odds(row.get("odds"))
    if odds_value is not None:
        return odds_value
    return _parse_american_odds(row.get("price"))


def _profit_units_for_result(odds: float | None, result: str) -> float | None:
    if result == "Push":
        return 0.0
    if result == "L":
        return -1.0
    if result != "W" or odds is None:
        return None
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def empty_official_picks_profit_report_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_PROFIT_REPORT_COLUMNS)


def empty_official_picks_profit_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_PROFIT_SUMMARY_COLUMNS)


def summarize_official_picks_profit_by_book(graded_df: pd.DataFrame) -> pd.DataFrame:
    if graded_df.empty:
        return empty_official_picks_profit_summary_df()

    summary = (
        graded_df.groupby("book", dropna=False)
        .agg(
            picks=("pick_key", "size"),
            wins=("result_normalized", lambda s: int((s == "W").sum())),
            losses=("result_normalized", lambda s: int((s == "L").sum())),
            pushes=("result_normalized", lambda s: int((s == "Push").sum())),
            decisions=("units_risked", "sum"),
            units_risked=("units_risked", "sum"),
            units_profit=("units_result", "sum"),
        )
        .reset_index()
    )
    summary["picks"] = summary["picks"].astype(int)
    summary["wins"] = summary["wins"].astype(int)
    summary["losses"] = summary["losses"].astype(int)
    summary["pushes"] = summary["pushes"].astype(int)
    summary["decisions"] = summary["decisions"].astype(int)
    summary["win_rate"] = summary.apply(
        lambda row: row["wins"] / row["decisions"] if row["decisions"] else None,
        axis=1,
    )
    summary["roi"] = summary.apply(
        lambda row: row["units_profit"] / row["units_risked"] if row["units_risked"] else None,
        axis=1,
    )
    return summary[OFFICIAL_PICKS_PROFIT_SUMMARY_COLUMNS].sort_values(
        by=["units_profit", "book"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_official_picks_profit_report(history_df: pd.DataFrame) -> dict[str, object]:
    if history_df.empty:
        return {
            "graded_df": empty_official_picks_profit_report_df(),
            "summary_by_book_df": empty_official_picks_profit_summary_df(),
            "overall_summary": {
                "books": 0,
                "picks": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "decisions": 0,
                "units_risked": 0.0,
                "units_profit": 0.0,
                "win_rate": None,
                "roi": None,
                "skipped_rows": 0,
            },
            "skipped_df": empty_official_picks_profit_report_df(),
        }

    official_df = history_df[history_df["pick_type"] == "official"].copy()
    if official_df.empty:
        return {
            "graded_df": empty_official_picks_profit_report_df(),
            "summary_by_book_df": empty_official_picks_profit_summary_df(),
            "overall_summary": {
                "books": 0,
                "picks": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "decisions": 0,
                "units_risked": 0.0,
                "units_profit": 0.0,
                "win_rate": None,
                "roi": None,
                "skipped_rows": 0,
            },
            "skipped_df": empty_official_picks_profit_report_df(),
        }

    working = official_df.copy()
    working["result_normalized"] = working["result"].apply(_normalize_pick_result)
    working["odds_numeric"] = working.apply(_resolve_history_row_odds, axis=1)
    working["units_risked"] = 0.0
    working["units_result"] = pd.NA

    resolved_mask = working["result_normalized"].isin(["W", "L", "Push"])
    valid_odds_mask = working["odds_numeric"].notna()
    push_mask = working["result_normalized"] == "Push"
    gradeable_mask = resolved_mask & (valid_odds_mask | push_mask)

    if gradeable_mask.any():
        working.loc[gradeable_mask, "units_risked"] = working.loc[
            gradeable_mask, "result_normalized"
        ].apply(lambda result: 0.0 if result == "Push" else 1.0)
        working.loc[gradeable_mask, "units_result"] = working.loc[gradeable_mask].apply(
            lambda row: _profit_units_for_result(row["odds_numeric"], row["result_normalized"]),
            axis=1,
        )

    graded_df = working.loc[gradeable_mask, OFFICIAL_PICKS_PROFIT_REPORT_COLUMNS].copy()
    skipped_df = working.loc[~gradeable_mask, OFFICIAL_PICKS_PROFIT_REPORT_COLUMNS].copy()
    summary_by_book_df = summarize_official_picks_profit_by_book(graded_df)

    wins = int((graded_df["result_normalized"] == "W").sum())
    losses = int((graded_df["result_normalized"] == "L").sum())
    pushes = int((graded_df["result_normalized"] == "Push").sum())
    decisions = wins + losses
    units_risked = float(graded_df["units_risked"].sum()) if not graded_df.empty else 0.0
    units_profit = float(pd.to_numeric(graded_df["units_result"], errors="coerce").fillna(0.0).sum())

    overall_summary = {
        "books": int(summary_by_book_df["book"].nunique()) if not summary_by_book_df.empty else 0,
        "picks": int(len(graded_df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "decisions": decisions,
        "units_risked": units_risked,
        "units_profit": units_profit,
        "win_rate": (wins / decisions) if decisions else None,
        "roi": (units_profit / units_risked) if units_risked else None,
        "skipped_rows": int(len(skipped_df)),
    }

    return {
        "graded_df": graded_df,
        "summary_by_book_df": summary_by_book_df,
        "overall_summary": overall_summary,
        "skipped_df": skipped_df,
    }


def persist_official_picks_profit_reports() -> dict[str, object]:
    history_df = load_official_picks_history()
    report = build_official_picks_profit_report(history_df)
    report["graded_df"].to_csv(OFFICIAL_PICKS_GRADES_PATH, index=False)
    report["summary_by_book_df"].to_csv(OFFICIAL_PICKS_BOOK_SUMMARY_PATH, index=False)
    report["skipped_df"].to_csv(OFFICIAL_PICKS_SKIPPED_PATH, index=False)
    OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.write_text(
        json.dumps(report["overall_summary"], indent=2),
        encoding="utf-8",
    )
    return report


def load_official_picks_history() -> pd.DataFrame:
    if not OFFICIAL_PICKS_HISTORY_PATH.exists():
        return empty_official_picks_history_df()

    history_df = pd.read_csv(OFFICIAL_PICKS_HISTORY_PATH, keep_default_na=False)
    missing = [col for col in OFFICIAL_PICKS_HISTORY_COLUMNS if col not in history_df.columns]
    if missing:
        raise ValueError(
            "official_picks_history.csv is missing required columns: "
            f"{missing}"
        )

    return history_df[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()


def build_official_picks_history_rows(
    starters_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> pd.DataFrame:
    official_df = post_df[post_df["pick_type"] == "official"].copy()
    if official_df.empty:
        return empty_official_picks_history_df()

    starter_lookup = starters_df[
        ["player_name", "team", "opponent", "game_date"]
    ].copy()
    starter_lookup["game_date"] = pd.to_datetime(starter_lookup["game_date"]).dt.strftime("%Y-%m-%d")
    merge_keys = ["player_name"]
    if {"team", "opponent"}.issubset(official_df.columns):
        merge_keys = ["player_name", "team", "opponent"]

    starter_lookup = starter_lookup.drop_duplicates(subset=merge_keys, keep="last")
    history_rows = official_df.merge(
        starter_lookup,
        on=merge_keys,
        how="left",
        suffixes=("", "_starter"),
    )

    if "game_date" not in history_rows.columns:
        candidate_columns = [
            "game_date_starter",
            "game_date_x",
            "game_date_y",
        ]
        available_candidates = [col for col in candidate_columns if col in history_rows.columns]
        if available_candidates:
            resolved_game_date = history_rows[available_candidates[0]].copy()
            for candidate in available_candidates[1:]:
                resolved_game_date = resolved_game_date.combine_first(history_rows[candidate])
            history_rows["game_date"] = resolved_game_date
        else:
            history_rows["game_date"] = pd.Series(pd.NA, index=history_rows.index, dtype="object")

    if history_rows["game_date"].isna().any():
        unique_game_dates = pd.to_datetime(starters_df["game_date"]).dt.strftime("%Y-%m-%d").dropna().unique()
        if len(unique_game_dates) == 1:
            history_rows["game_date"] = history_rows["game_date"].fillna(unique_game_dates[0])

    history_rows["game_date"] = history_rows["game_date"].fillna("").astype(str)

    if "game_date" not in history_rows.columns and "game_date_starter" in history_rows.columns:
        history_rows["game_date"] = history_rows["game_date_starter"]

    if history_rows["game_date"].isna().any():
        unique_game_dates = pd.to_datetime(starters_df["game_date"]).dt.strftime("%Y-%m-%d").dropna().unique()
        if len(unique_game_dates) == 1:
            history_rows["game_date"] = history_rows["game_date"].fillna(unique_game_dates[0])

    history_rows["game_date"] = history_rows["game_date"].fillna("").astype(str)
    history_rows["pick_key"] = history_rows.apply(
        lambda row: _build_pick_key(row["game_date"], row["player_name"]),
        axis=1,
    )
    history_rows["odds"] = history_rows["price"].apply(_format_american_odds)
    history_rows["result"] = ""
    history_rows["actual_strikeouts"] = ""
    history_rows["record_source"] = "run_daily_card"

    for column in OFFICIAL_PICKS_HISTORY_COLUMNS:
        if column not in history_rows.columns:
            history_rows[column] = ""

    return history_rows[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()


def persist_official_picks_history(
    starters_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> Path:
    existing_df = load_official_picks_history()
    new_rows = build_official_picks_history_rows(starters_df, post_df)

    if new_rows.empty:
        if not OFFICIAL_PICKS_HISTORY_PATH.exists():
            existing_df.to_csv(OFFICIAL_PICKS_HISTORY_PATH, index=False)
        return OFFICIAL_PICKS_HISTORY_PATH

    existing_by_key = existing_df.set_index("pick_key", drop=False)
    merged_rows: list[dict] = []

    for _, new_row in new_rows.iterrows():
        new_record = new_row.to_dict()
        pick_key = new_record["pick_key"]

        if pick_key in existing_by_key.index:
            existing_record = existing_by_key.loc[pick_key].to_dict()
            if existing_record.get("result") and not new_record.get("result"):
                new_record["result"] = existing_record["result"]
            if existing_record.get("actual_strikeouts") and not new_record.get("actual_strikeouts"):
                new_record["actual_strikeouts"] = existing_record["actual_strikeouts"]
            if existing_record.get("record_source") and not new_record.get("record_source"):
                new_record["record_source"] = existing_record["record_source"]

        merged_rows.append(new_record)

    merged_df = pd.DataFrame(merged_rows, columns=OFFICIAL_PICKS_HISTORY_COLUMNS)
    untouched_existing = existing_df[~existing_df["pick_key"].isin(merged_df["pick_key"])]
    history_df = pd.concat([untouched_existing, merged_df], ignore_index=True)
    history_df = history_df[OFFICIAL_PICKS_HISTORY_COLUMNS]
    history_df.to_csv(OFFICIAL_PICKS_HISTORY_PATH, index=False)
    return OFFICIAL_PICKS_HISTORY_PATH


def resolve_artifact_path(filename: str) -> Path:
    candidate_paths = [
        LATEST_ARTIFACTS_DIR / filename,
        PREVIOUS_ARTIFACTS_DIR / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing {filename} artifact in both latest/ and previous/."
    )


def load_workflow_history_artifact(workflow: ModelingWorkflowSpec) -> pd.DataFrame:
    path = resolve_artifact_path(workflow.artifacts.history_filename)
    history_df = workflow.artifacts.history_loader(path)
    print(f"Loaded workflow history artifact from: {path}")
    return history_df


def load_workflow_model_artifact(workflow: ModelingWorkflowSpec):
    path = resolve_artifact_path(workflow.artifacts.model_filename)
    model = workflow.artifacts.model_loader(path)
    print(f"Loaded model artifact from: {path}")
    return model


def load_model_metadata(workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW) -> dict:
    model_path = resolve_artifact_path(workflow.artifacts.model_filename)
    metadata_path = model_path.with_name(workflow.artifacts.metadata_filename)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata artifact paired with model: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Loaded model metadata from: {metadata_path}")
    print("Model metadata:")
    print(json.dumps(metadata, indent=2))
    return metadata


def build_today_predictions(starters_df: pd.DataFrame, pitcher_games: pd.DataFrame, model):
    return build_today_predictions_for_workflow(
        starters_df=starters_df,
        pitcher_games=pitcher_games,
        model=model,
        workflow=MLB_PITCHER_STRIKEOUT_WORKFLOW,
    )


def build_today_predictions_for_workflow(
    *,
    starters_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    model,
    workflow: ModelingWorkflowSpec,
):
    validate_starters_contract(starters_df)

    today_features = workflow.feature_builder(starters_df, pitcher_games)

    if today_features.empty:
        return today_features

    today_preds = workflow.predictor(model, today_features)
    assert_non_empty(today_preds, "today_preds")
    require_columns(
        today_preds,
        list(workflow.prediction_columns),
        "today_preds",
    )
    return today_preds


def apply_metadata_uncertainty(
    today_preds: pd.DataFrame,
    metadata: dict | None,
    workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW,
) -> pd.DataFrame:
    adjuster = workflow.prediction_metadata_adjuster
    if adjuster is None:
        return today_preds

    return adjuster(today_preds, metadata)


def save_run_status(*, status: str, message: str | None = None) -> None:
    payload = {
        "status": status,
        "message": message or "",
    }
    RUN_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_outputs(
    starters_df: pd.DataFrame,
    today_preds: pd.DataFrame,
    joined_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    run_status: str = "success",
    run_message: str | None = None,
) -> None:
    save_today_starters_csv(starters_df)

    today_preds.to_csv(PROJECTIONS_DIR / "today_projections.csv", index=False)
    joined_df.to_csv(EDGES_DIR / "today_joined_edges.csv", index=False)
    picks_df.to_csv(PICKS_DIR / "today_all_picks.csv", index=False)
    post_df.to_csv(PICKS_DIR / "today_postable_picks.csv", index=False)
    persist_official_picks_history(starters_df, post_df)
    persist_official_picks_profit_reports()
    save_run_status(status=run_status, message=run_message)


def run_daily_card(
    *,
    workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW,
    market: str | None = None,
    build_picks_fn: BuildPicksFn | None = None,
    filter_postable_picks_fn: FilterPostablePicksFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()

    if build_picks_fn is None:
        def build_picks_fn(joined_df: pd.DataFrame) -> pd.DataFrame:
            return build_daily_picks(
                joined_df,
                policy=workflow.pick_ranking_policy,
            )

    if filter_postable_picks_fn is None:
        postable_limits = workflow.resolved_postable_limits()

        def filter_postable_picks_fn(picks_df: pd.DataFrame) -> pd.DataFrame:
            return filter_postable_picks(
                picks_df,
                max_official=postable_limits.max_official,
                max_leans=postable_limits.max_leans,
                policy=workflow.pick_ranking_policy,
            )

    starters_df = get_today_starters_df()
    validate_starters_contract(starters_df)
    history_df = load_workflow_history_artifact(workflow)
    model = load_workflow_model_artifact(workflow)
    metadata = load_model_metadata(workflow)

    today_preds = build_today_predictions_for_workflow(
        starters_df=starters_df,
        pitcher_games=history_df,
        model=model,
        workflow=workflow,
    )
    today_preds = apply_metadata_uncertainty(today_preds, metadata, workflow)

    if today_preds.empty:
        raise ValueError("No today predictions were generated.")

    selected_market = market or workflow.market_key
    run_status = "success"
    run_message: str | None = None

    try:
        joined_df, _ = run_edge_pipeline(
            today_preds,
            selected_market,
            participant_key=workflow.participant_key,
            projection_join_key=workflow.projection_odds_join_keys.projection,
            odds_join_key=workflow.projection_odds_join_keys.odds,
        )
        validate_joined_odds_contract(joined_df)

        picks_df = build_picks_fn(joined_df)
        validate_final_picks_contract(picks_df)

        post_df = filter_postable_picks_fn(picks_df)
        validate_final_picks_contract(post_df)
    except requests.RequestException as exc:
        run_status = "degraded"
        run_message = (
            "Live odds fetch failed; projections were saved but no edges or picks were generated. "
            f"Reason: {exc.__class__.__name__}: {exc}"
        )
        print(f"WARNING: {run_message}")
        joined_df = empty_joined_odds_df()
        picks_df = empty_final_picks_df()
        post_df = empty_final_picks_df()

    save_outputs(
        starters_df=starters_df,
        today_preds=today_preds,
        joined_df=joined_df,
        picks_df=picks_df,
        post_df=post_df,
        run_status=run_status,
        run_message=run_message,
    )

    return starters_df, today_preds, picks_df, post_df


if __name__ == "__main__":
    _, _, picks_df, post_df = run_daily_card()

    if RUN_STATUS_PATH.exists():
        status_payload = json.loads(RUN_STATUS_PATH.read_text(encoding="utf-8"))
        if status_payload.get("status") == "degraded" and status_payload.get("message"):
            print(f"\n{status_payload['message']}")

    print("\nTop postable picks:")
    if post_df.empty:
        print("No postable picks found.")
    else:
        print(
            post_df[
                [
                    "player_name",
                    "book",
                    "pick_side",
                    "line",
                    "price",
                    "predicted_strikeouts",
                    "edge",
                    "pick_type",
                ]
            ].to_string(index=False)
        )
