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
from common.identity import (
    MARKET_OFFER_KEY_COLUMN,
    MARKET_SELECTION_KEY_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_JOIN_KEY_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    ensure_market_identity,
    ensure_participant_identity,
)
from common.workflows import ModelingWorkflowSpec
from pitcher_k.workflow import MLB_PITCHER_STRIKEOUT_WORKFLOW
from pitcher_bb.workflow import MLB_PITCHER_WALK_WORKFLOW
from pitcher_k.data_loader import load_statcast_data
from pitcher_k.feature_engineering import build_pitcher_game_table
from pitcher_k.preprocessing import add_outcome_flags

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
OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH = TRACKING_DIR / "official_picks_profit_summary_all_time.json"
OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH = TRACKING_DIR / "official_picks_profit_summary_current_regime.json"
OFFICIAL_PICKS_SKIPPED_PATH = TRACKING_DIR / "official_picks_profit_skipped.csv"
OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH = TRACKING_DIR / "official_picks_concentration_audit.json"
CURRENT_REGIME_START_DATE = "2026-05-07"
LEGACY_WORKFLOW_MODEL_VERSION = "workflow_unversioned"
LEGACY_MANUAL_MODEL_VERSION = "manual_unversioned"
LEGACY_UNKNOWN_MODEL_VERSION = "unknown_unversioned"
LEGACY_WORKFLOW_POLICY_VERSION = "workflow_unversioned"
LEGACY_MANUAL_POLICY_VERSION = "manual_unversioned"
LEGACY_UNKNOWN_POLICY_VERSION = "unknown_unversioned"
TRACKING_REGIME_MANUAL_BACKFILL = "manual_backfill"
TRACKING_REGIME_LEGACY_WORKFLOW = "legacy_workflow"
TRACKING_REGIME_CURRENT_WORKFLOW = "current_workflow"
TRACKING_REGIME_UNKNOWN = "unknown"
TRACKING_SEGMENT_COLUMNS = [
    "record_source",
    "model_version",
    "policy_version",
    "tracking_regime",
]

OFFICIAL_PICKS_HISTORY_COLUMNS = [
    "pick_key",
    "game_date",
    "player_name",
    "prop_type",
    PARTICIPANT_JOIN_KEY_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    "sport",
    "market_key",
    "market_family",
    "team",
    "opponent",
    "book",
    "bookmaker_key",
    "event_id",
    "odds",
    "price",
    "pick_side",
    "line",
    MARKET_SELECTION_KEY_COLUMN,
    MARKET_OFFER_KEY_COLUMN,
    "predicted_value",
    "predicted_strikeouts",
    "edge",
    "confidence_tier",
    "pick_type",
    "result",
    "actual_value",
    "actual_strikeouts",
    "record_source",
    "model_version",
    "policy_version",
    "tracking_regime",
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
OFFICIAL_PICKS_PROFIT_SUMMARY_SCOPE_COLUMNS = [
    "summary_scope",
    "segment_type",
    "segment_value",
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

SUPPORTED_STATCAST_GRADING_MARKETS = {
    "": "strikeouts",
    "pitcher_strikeouts": "strikeouts",
    "pitcher_walks": "walks",
}

BuildPicksFn = Callable[[pd.DataFrame], pd.DataFrame]
FilterPostablePicksFn = Callable[[pd.DataFrame], pd.DataFrame]
DEFAULT_DAILY_CARD_WORKFLOWS = [
    MLB_PITCHER_STRIKEOUT_WORKFLOW,
    MLB_PITCHER_WALK_WORKFLOW,
]


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


def _artifact_dir_candidates(workflow: ModelingWorkflowSpec) -> list[Path]:
    artifact_subdir = workflow.artifacts.artifact_subdir
    if artifact_subdir:
        workflow_root = ARTIFACTS_DIR / artifact_subdir
        return [
            workflow_root / "latest",
            workflow_root / "previous",
        ]
    return [
        LATEST_ARTIFACTS_DIR,
        PREVIOUS_ARTIFACTS_DIR,
    ]


def _tag_workflow_frame(
    df: pd.DataFrame,
    workflow: ModelingWorkflowSpec,
) -> pd.DataFrame:
    tagged = df.copy()
    tagged["prop_type"] = workflow.prop_type
    return tagged


def _adapt_predictions_for_output(
    today_preds: pd.DataFrame,
    workflow: ModelingWorkflowSpec,
) -> pd.DataFrame:
    adapted = today_preds.copy()
    prediction_column = workflow.prop_fields.prediction
    shared_prediction_column = workflow.prop_fields.shared_prediction
    if prediction_column in adapted.columns and shared_prediction_column not in adapted.columns:
        adapted[shared_prediction_column] = adapted[prediction_column]
    return _tag_workflow_frame(adapted, workflow)


def _prediction_value_from_frame(
    df: pd.DataFrame,
) -> pd.Series:
    if "predicted_value" in df.columns:
        return df["predicted_value"]
    candidates = [column for column in df.columns if column.startswith("predicted_")]
    if len(candidates) == 1:
        return df[candidates[0]]
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _format_american_odds(price: float | int | str | None) -> str:
    if pd.isna(price):
        return ""

    numeric_price = int(float(price))
    if numeric_price > 0:
        return f"+{numeric_price}"
    return str(numeric_price)


def _normalize_pick_key_name(player_name: str) -> str:
    return " ".join(str(player_name).strip().lower().split())


def _build_pick_key(
    game_date: str,
    player_name: str,
    market_offer_key: str | None = None,
) -> str:
    if market_offer_key:
        return f"{game_date}|{market_offer_key}"
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


def _normalize_record_source(value: object) -> str:
    return str(value or "").strip()


def _is_manual_record_source(record_source: object) -> bool:
    source = _normalize_record_source(record_source).lower()
    return "manual" in source or "backfill" in source


def _infer_tracking_regime(record_source: object, game_date: object) -> str:
    if _is_manual_record_source(record_source):
        return TRACKING_REGIME_MANUAL_BACKFILL

    source = _normalize_record_source(record_source).lower()
    parsed_game_date = pd.to_datetime(game_date, errors="coerce")
    if source == "run_daily_card":
        if not pd.isna(parsed_game_date) and parsed_game_date >= pd.Timestamp(CURRENT_REGIME_START_DATE):
            return TRACKING_REGIME_CURRENT_WORKFLOW
        return TRACKING_REGIME_LEGACY_WORKFLOW

    if pd.isna(parsed_game_date):
        return TRACKING_REGIME_UNKNOWN
    if parsed_game_date >= pd.Timestamp(CURRENT_REGIME_START_DATE):
        return TRACKING_REGIME_CURRENT_WORKFLOW
    return TRACKING_REGIME_LEGACY_WORKFLOW


def _default_model_version_for_record(record_source: object) -> str:
    if _is_manual_record_source(record_source):
        return LEGACY_MANUAL_MODEL_VERSION
    source = _normalize_record_source(record_source).lower()
    if source == "run_daily_card":
        return LEGACY_WORKFLOW_MODEL_VERSION
    return LEGACY_UNKNOWN_MODEL_VERSION


def _default_policy_version_for_record(record_source: object) -> str:
    if _is_manual_record_source(record_source):
        return LEGACY_MANUAL_POLICY_VERSION
    source = _normalize_record_source(record_source).lower()
    if source == "run_daily_card":
        return LEGACY_WORKFLOW_POLICY_VERSION
    return LEGACY_UNKNOWN_POLICY_VERSION


def _hydrate_history_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    hydrated = df.copy()
    for column in TRACKING_SEGMENT_COLUMNS:
        if column not in hydrated.columns:
            hydrated[column] = ""

    hydrated["record_source"] = hydrated["record_source"].apply(_normalize_record_source)
    hydrated["model_version"] = hydrated.apply(
        lambda row: str(row.get("model_version", "") or "").strip()
        or _default_model_version_for_record(row.get("record_source")),
        axis=1,
    )
    hydrated["policy_version"] = hydrated.apply(
        lambda row: str(row.get("policy_version", "") or "").strip()
        or _default_policy_version_for_record(row.get("record_source")),
        axis=1,
    )
    hydrated["tracking_regime"] = hydrated.apply(
        lambda row: str(row.get("tracking_regime", "") or "").strip()
        or _infer_tracking_regime(row.get("record_source"), row.get("game_date")),
        axis=1,
    )
    return hydrated


def _resolve_model_version(workflow: ModelingWorkflowSpec, metadata: dict | None) -> str:
    metadata = metadata or {}
    explicit_version = str(metadata.get("model_version", "") or "").strip()
    if explicit_version:
        return explicit_version

    artifact_version = metadata.get("artifact_version", 1)
    training_window = metadata.get("training_window", {})
    train_split_date = training_window.get("train_split_date") if isinstance(training_window, dict) else None
    if train_split_date:
        return f"{workflow.prop_type}_artifact_v{artifact_version}_split_{train_split_date}"
    return f"{workflow.prop_type}_artifact_v{artifact_version}"


def _annotate_workflow_provenance(
    df: pd.DataFrame,
    *,
    workflow: ModelingWorkflowSpec,
    metadata: dict | None,
    game_date: object,
) -> pd.DataFrame:
    annotated = df.copy()
    annotated["record_source"] = "run_daily_card"
    annotated["model_version"] = _resolve_model_version(workflow, metadata)
    annotated["policy_version"] = workflow.pick_ranking_policy.version
    annotated["tracking_regime"] = _infer_tracking_regime("run_daily_card", game_date)
    return annotated


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


def _yesterday_game_date() -> str:
    return (
        pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


def _supports_statcast_history_row(row: pd.Series) -> bool:
    market_key = str(row.get("market_key", "") or "").strip()
    return market_key in SUPPORTED_STATCAST_GRADING_MARKETS


def _statcast_result_column_for_row(row: pd.Series) -> str | None:
    market_key = str(row.get("market_key", "") or "").strip()
    return SUPPORTED_STATCAST_GRADING_MARKETS.get(market_key)


def _grade_pick_result_from_actual(actual: float, line: float, pick_side: str) -> str:
    pick_side_norm = str(pick_side or "").strip().lower()
    if pick_side_norm == "over":
        if actual > line:
            return "W"
        if actual < line:
            return "L"
        return "Push"

    if pick_side_norm == "under":
        if actual < line:
            return "W"
        if actual > line:
            return "L"
        return "Push"

    return ""


def _format_stat_value(value: float | int) -> str:
    return f"{float(value):g}"


def load_pitcher_results_from_statcast(game_date: str) -> pd.DataFrame:
    sc = load_statcast_data(game_date, game_date, chunk_days=1)
    if sc.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "pitcher",
                "player_name",
                "strikeouts",
                "walks",
            ]
        )

    sc = add_outcome_flags(sc)
    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"]).dt.strftime("%Y-%m-%d")
    pitcher_games = ensure_participant_identity(
        pitcher_games,
        display_name_col="player_name",
        source_id_col="pitcher",
        source_id_type="mlbam_player",
    )
    return pitcher_games


def apply_statcast_results_to_official_picks_history(
    history_df: pd.DataFrame,
    pitcher_results_df: pd.DataFrame,
    *,
    game_date: str,
) -> pd.DataFrame:
    if history_df.empty:
        return history_df.copy()

    working = history_df.copy()
    for column in OFFICIAL_PICKS_HISTORY_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    working = ensure_participant_identity(
        working,
        display_name_col="player_name",
        normalized_name_col=PARTICIPANT_NAME_NORM_COLUMN,
        source_id_col=PARTICIPANT_SOURCE_ID_COLUMN,
        source_id_type="mlbam_player",
    )

    if pitcher_results_df.empty:
        return working[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()

    pitcher_results = pitcher_results_df.copy()
    pitcher_results = ensure_participant_identity(
        pitcher_results,
        display_name_col="player_name",
        source_id_col="pitcher" if "pitcher" in pitcher_results.columns else PARTICIPANT_SOURCE_ID_COLUMN,
        source_id_type="mlbam_player",
    )
    for stat_column in ["strikeouts", "walks"]:
        if stat_column not in pitcher_results.columns:
            pitcher_results[stat_column] = pd.NA

    result_lookup = pitcher_results.drop_duplicates(
        subset=["game_date", PARTICIPANT_JOIN_KEY_COLUMN],
        keep="last",
    )[
        ["game_date", PARTICIPANT_JOIN_KEY_COLUMN, "strikeouts", "walks"]
    ].copy()

    pending_mask = (
        (working["pick_type"] == "official")
        & (working["game_date"].astype(str) == game_date)
        & working.apply(_supports_statcast_history_row, axis=1)
        & (
            working["actual_value"].astype(str).str.strip().eq("")
            | working["result"].astype(str).str.strip().eq("")
        )
    )
    if not pending_mask.any():
        return working[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()

    pending = working.loc[pending_mask].copy()
    pending["history_row_index"] = pending.index
    pending = pending.merge(
        result_lookup,
        on=["game_date", PARTICIPANT_JOIN_KEY_COLUMN],
        how="left",
    )
    pending["statcast_result_column"] = pending.apply(_statcast_result_column_for_row, axis=1)
    pending["actual_stat_value"] = pending.apply(
        lambda row: row.get(str(row["statcast_result_column"]), pd.NA)
        if row["statcast_result_column"]
        else pd.NA,
        axis=1,
    )
    resolved_mask = pd.to_numeric(pending["actual_stat_value"], errors="coerce").notna()
    if not resolved_mask.any():
        return working[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()

    pending.loc[resolved_mask, "actual_value"] = pending.loc[resolved_mask, "actual_stat_value"].apply(_format_stat_value)
    strikeout_mask = resolved_mask & pending["statcast_result_column"].eq("strikeouts")
    pending.loc[strikeout_mask, "actual_strikeouts"] = pending.loc[strikeout_mask, "actual_value"]
    pending.loc[resolved_mask, "result"] = pending.loc[resolved_mask].apply(
        lambda row: _grade_pick_result_from_actual(
            actual=float(row["actual_stat_value"]),
            line=float(row["line"]),
            pick_side=str(row["pick_side"]),
        ),
        axis=1,
    )

    working.loc[pending["history_row_index"], OFFICIAL_PICKS_HISTORY_COLUMNS] = pending[
        OFFICIAL_PICKS_HISTORY_COLUMNS
    ].values
    return working[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()


def grade_official_picks_from_statcast(
    *,
    game_date: str | None = None,
) -> dict[str, object]:
    target_date = game_date or _yesterday_game_date()
    history_df = load_official_picks_history()
    if history_df.empty:
        persist_official_picks_profit_reports()
        return {"game_date": target_date, "updated_rows": 0, "history_df": history_df}

    pending_mask = (
        (history_df["pick_type"] == "official")
        & (history_df["game_date"].astype(str) == target_date)
        & history_df.apply(_supports_statcast_history_row, axis=1)
        & (
            history_df["actual_value"].astype(str).str.strip().eq("")
            | history_df["result"].astype(str).str.strip().eq("")
        )
    )
    if not pending_mask.any():
        persist_official_picks_profit_reports()
        return {"game_date": target_date, "updated_rows": 0, "history_df": history_df}

    pitcher_results_df = load_pitcher_results_from_statcast(target_date)
    updated_history_df = apply_statcast_results_to_official_picks_history(
        history_df,
        pitcher_results_df,
        game_date=target_date,
    )
    before = history_df["actual_value"].astype(str).str.strip()
    after = updated_history_df["actual_value"].astype(str).str.strip()
    updated_rows = int(((before == "") & (after != "")).sum())
    updated_history_df.to_csv(OFFICIAL_PICKS_HISTORY_PATH, index=False)
    persist_official_picks_profit_reports()
    return {
        "game_date": target_date,
        "updated_rows": updated_rows,
        "history_df": updated_history_df,
    }


def empty_official_picks_profit_report_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_PROFIT_REPORT_COLUMNS)


def empty_official_picks_profit_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_PROFIT_SUMMARY_COLUMNS)


def empty_official_picks_profit_summary_by_scope_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_PICKS_PROFIT_SUMMARY_SCOPE_COLUMNS)


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


def _summarize_official_picks_profit_by_dimension(
    graded_df: pd.DataFrame,
    *,
    dimension: str,
) -> dict[str, dict[str, int | float | None]]:
    if graded_df.empty or dimension not in graded_df.columns:
        return {}

    segmented: dict[str, dict[str, int | float | None]] = {}
    for segment_value, segment_df in graded_df.groupby(dimension, dropna=False, sort=True):
        normalized_value = str(segment_value or "").strip() or TRACKING_REGIME_UNKNOWN
        segmented[normalized_value] = _build_profit_summary_metrics(
            graded_df=segment_df,
            summary_by_book_df=summarize_official_picks_profit_by_book(segment_df),
            skipped_df=empty_official_picks_profit_report_df(),
        )
    return segmented


def _append_segmented_summary_rows(
    rows: list[pd.DataFrame],
    *,
    graded_df: pd.DataFrame,
    segment_type: str,
) -> None:
    if graded_df.empty or segment_type not in graded_df.columns:
        return

    for segment_value, segment_df in graded_df.groupby(segment_type, dropna=False, sort=True):
        normalized_value = str(segment_value or "").strip() or TRACKING_REGIME_UNKNOWN
        segment_summary = summarize_official_picks_profit_by_book(segment_df)
        if segment_summary.empty:
            continue
        rows.append(
            segment_summary.assign(
                summary_scope="all_time",
                segment_type=segment_type,
                segment_value=normalized_value,
            )
        )


def _empty_profit_summary_metrics() -> dict[str, int | float | None]:
    return {
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
    }


def _current_regime_rule_payload() -> dict[str, str]:
    return {
        "type": "start_date",
        "start_date": CURRENT_REGIME_START_DATE,
    }


def _summary_view_rule_payload(summary_scope: str) -> dict[str, str]:
    if summary_scope == "current_regime":
        return _current_regime_rule_payload()
    return {
        "type": "all_tracked_official_picks",
        "source": "official_picks_history.csv",
    }


def _build_published_profit_summary_view(
    *,
    summary_scope: str,
    metrics: dict[str, int | float | None],
    by_book_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "artifact_type": "official_picks_profit_summary_view",
        "summary_scope": summary_scope,
        "summary_scope_rule": _summary_view_rule_payload(summary_scope),
        "summary_metrics": metrics,
        "by_book": by_book_df.to_dict(orient="records"),
        "reproducibility": {
            "source_artifact": "official_picks_history.csv",
            "rebuild_entrypoint": "jobs.run_daily_card.persist_official_picks_profit_reports",
        },
    }


def _build_profit_summary_metrics(
    graded_df: pd.DataFrame,
    summary_by_book_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> dict[str, int | float | None]:
    wins = int((graded_df["result_normalized"] == "W").sum())
    losses = int((graded_df["result_normalized"] == "L").sum())
    pushes = int((graded_df["result_normalized"] == "Push").sum())
    decisions = wins + losses
    units_risked = float(graded_df["units_risked"].sum()) if not graded_df.empty else 0.0
    units_profit = float(pd.to_numeric(graded_df["units_result"], errors="coerce").fillna(0.0).sum())

    return {
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


def _current_regime_mask(history_df: pd.DataFrame) -> pd.Series:
    if "tracking_regime" in history_df.columns:
        return history_df["tracking_regime"].astype(str).eq(TRACKING_REGIME_CURRENT_WORKFLOW)
    game_dates = pd.to_datetime(history_df.get("game_date", ""), errors="coerce")
    return game_dates >= pd.Timestamp(CURRENT_REGIME_START_DATE)


def _audit_line_bucket(value: float | int | str | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if float(numeric) < 5.5:
        return "<5.5"
    if float(numeric) <= 7.0:
        return "5.5-7.0"
    return "7.5+"


def _audit_pitcher_archetype(prop_type: str | None, line: float | int | str | None) -> str:
    prop = str(prop_type or "").strip().lower()
    bucket = _audit_line_bucket(line)

    if prop == "pitcher_k":
        if bucket == "7.5+":
            return "high-K ace"
        if bucket == "<5.5":
            return "contact-oriented low-K arm"
        return "mid-K starter"

    if prop == "pitcher_bb":
        if bucket == "<5.5":
            return "control specialist"
        return "wild/high-BB arm"

    return f"{prop or 'unknown'} {bucket}".strip()


def _json_ready_value(value):
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _build_audit_group_records(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
) -> list[dict]:
    if df.empty:
        return []

    total_picks = int(len(df))
    records: list[dict] = []

    for group_key, group_df in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        picks = int(len(group_df))
        graded_mask = group_df["is_gradeable"].fillna(False)
        graded_df = group_df.loc[graded_mask].copy()
        wins = int((group_df["result_normalized"] == "W").sum())
        losses = int((group_df["result_normalized"] == "L").sum())
        pushes = int((group_df["result_normalized"] == "Push").sum())
        units_risked = float(pd.to_numeric(group_df["units_risked"], errors="coerce").fillna(0.0).sum())
        units_profit = float(pd.to_numeric(group_df["units_result"], errors="coerce").fillna(0.0).sum())
        pitcher_counts = (
            group_df["player_name"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )
        concentration_index = (
            float(((pitcher_counts / picks) ** 2).sum())
            if picks and not pitcher_counts.empty
            else None
        )
        unique_pitchers = int(pitcher_counts.shape[0])
        unique_game_dates = int(
            group_df["game_date"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

        record = {
            column: _json_ready_value(value)
            for column, value in zip(group_columns, group_key)
        }
        record.update(
            {
                "picks": picks,
                "graded_picks": int(len(graded_df)),
                "share_of_official_picks": (picks / total_picks) if total_picks else None,
                "unique_pitchers": unique_pitchers,
                "unique_game_dates": unique_game_dates,
                "repeat_pitcher_frequency": (
                    (picks - unique_pitchers) / picks
                    if picks
                    else None
                ),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "decisions": wins + losses,
                "units_risked": units_risked,
                "units_profit": units_profit,
                "roi": (units_profit / units_risked) if units_risked else None,
                "average_edge": _json_ready_value(pd.to_numeric(group_df["edge"], errors="coerce").mean()),
                "average_line": _json_ready_value(pd.to_numeric(group_df["line"], errors="coerce").mean()),
                "concentration_index": concentration_index,
            }
        )
        records.append(record)

    return records


def _sort_audit_records(
    records: list[dict],
    *,
    key_fn,
    reverse: bool = True,
    limit: int | None = None,
) -> list[dict]:
    sorted_records = sorted(records, key=key_fn, reverse=reverse)
    if limit is not None:
        return sorted_records[:limit]
    return sorted_records


def _build_regime_comparison_records(
    *,
    all_time_records: list[dict],
    current_regime_records: list[dict],
    key_columns: list[str],
) -> list[dict]:
    all_lookup = {
        tuple(record.get(column) for column in key_columns): record
        for record in all_time_records
    }
    current_lookup = {
        tuple(record.get(column) for column in key_columns): record
        for record in current_regime_records
    }
    combined_keys = list(dict.fromkeys([*all_lookup.keys(), *current_lookup.keys()]))

    records: list[dict] = []
    for combined_key in combined_keys:
        all_record = all_lookup.get(combined_key, {})
        current_record = current_lookup.get(combined_key, {})
        record = {
            column: value
            for column, value in zip(key_columns, combined_key)
        }
        all_time_picks = int(all_record.get("picks", 0) or 0)
        current_picks = int(current_record.get("picks", 0) or 0)
        all_time_units_profit = float(all_record.get("units_profit", 0.0) or 0.0)
        current_units_profit = float(current_record.get("units_profit", 0.0) or 0.0)
        record.update(
            {
                "all_time_picks": all_time_picks,
                "current_regime_picks": current_picks,
                "all_time_units_profit": all_time_units_profit,
                "current_regime_units_profit": current_units_profit,
                "all_time_roi": all_record.get("roi"),
                "current_regime_roi": current_record.get("roi"),
                "current_regime_share_of_all_time_picks": (
                    current_picks / all_time_picks
                    if all_time_picks
                    else None
                ),
                "persists_in_current_regime": bool(
                    current_picks > 0 and current_units_profit < 0
                ),
            }
        )
        records.append(record)

    return records


def build_official_picks_concentration_audit(history_df: pd.DataFrame) -> dict[str, object]:
    working_history = history_df.copy()
    for column in OFFICIAL_PICKS_HISTORY_COLUMNS:
        if column not in working_history.columns:
            working_history[column] = ""
    working_history = _hydrate_history_provenance_columns(working_history)

    official_df = working_history.loc[working_history["pick_type"] == "official"].copy()
    official_df["line_bucket"] = official_df["line"].apply(_audit_line_bucket)
    official_df["archetype"] = official_df.apply(
        lambda row: _audit_pitcher_archetype(row.get("prop_type"), row.get("line")),
        axis=1,
    )
    official_df["result_normalized"] = official_df["result"].apply(_normalize_pick_result)
    official_df["odds_numeric"] = official_df.apply(_resolve_history_row_odds, axis=1)
    official_df["units_risked"] = 0.0
    official_df["units_result"] = pd.NA

    resolved_mask = official_df["result_normalized"].isin(["W", "L", "Push"])
    valid_odds_mask = official_df["odds_numeric"].notna()
    push_mask = official_df["result_normalized"] == "Push"
    official_df["is_gradeable"] = resolved_mask & (valid_odds_mask | push_mask)

    if official_df["is_gradeable"].any():
        official_df.loc[official_df["is_gradeable"], "units_risked"] = official_df.loc[
            official_df["is_gradeable"], "result_normalized"
        ].apply(lambda result: 0.0 if result == "Push" else 1.0)
        official_df.loc[official_df["is_gradeable"], "units_result"] = official_df.loc[
            official_df["is_gradeable"]
        ].apply(
            lambda row: _profit_units_for_result(row["odds_numeric"], row["result_normalized"]),
            axis=1,
        )

    def build_scope(scope_name: str, scoped_df: pd.DataFrame) -> dict[str, object]:
        by_pitcher = _build_audit_group_records(scoped_df, group_columns=["player_name"])
        by_archetype = _build_audit_group_records(scoped_df, group_columns=["archetype"])
        by_pitcher_side = _build_audit_group_records(scoped_df, group_columns=["player_name", "pick_side"])
        by_archetype_side = _build_audit_group_records(scoped_df, group_columns=["archetype", "pick_side"])
        by_archetype_line_bucket = _build_audit_group_records(
            scoped_df,
            group_columns=["archetype", "line_bucket"],
        )
        by_combo = _build_audit_group_records(
            scoped_df,
            group_columns=["prop_type", "pick_side", "line_bucket", "archetype"],
        )

        scope_units_risked = float(pd.to_numeric(scoped_df["units_risked"], errors="coerce").fillna(0.0).sum())
        scope_units_profit = float(pd.to_numeric(scoped_df["units_result"], errors="coerce").fillna(0.0).sum())
        scope_roi = (scope_units_profit / scope_units_risked) if scope_units_risked else None

        overselected_archetypes: list[dict] = []
        for record in by_archetype:
            record_copy = record.copy()
            roi = record_copy.get("roi")
            pick_share = record_copy.get("share_of_official_picks")
            if roi is None or scope_roi is None or pick_share is None:
                record_copy["roi_gap_vs_scope"] = None
                record_copy["overselection_score"] = None
            else:
                roi_gap_vs_scope = float(roi) - float(scope_roi)
                record_copy["roi_gap_vs_scope"] = roi_gap_vs_scope
                record_copy["overselection_score"] = float(pick_share) * max(0.0, float(scope_roi) - float(roi))
            overselected_archetypes.append(record_copy)

        return {
            "scope": scope_name,
            "summary": {
                "official_picks": int(len(scoped_df)),
                "graded_picks": int(scoped_df["is_gradeable"].fillna(False).sum()),
                "units_risked": scope_units_risked,
                "units_profit": scope_units_profit,
                "roi": scope_roi,
            },
            "questions": {
                "largest_share_of_official_picks": _sort_audit_records(
                    by_pitcher,
                    key_fn=lambda record: (
                        float(record.get("share_of_official_picks") or 0.0),
                        int(record.get("picks", 0) or 0),
                    ),
                    limit=10,
                ),
                "largest_share_of_losses_or_negative_units": {
                    "by_losses": _sort_audit_records(
                        [record for record in by_pitcher if int(record.get("losses", 0) or 0) > 0],
                        key_fn=lambda record: (
                            int(record.get("losses", 0) or 0),
                            -(float(record.get("units_profit") or 0.0)),
                        ),
                        limit=10,
                    ),
                    "by_negative_units": _sort_audit_records(
                        by_pitcher,
                        key_fn=lambda record: (
                            -(float(record.get("units_profit") or 0.0)),
                            int(record.get("picks", 0) or 0),
                        ),
                        limit=10,
                    ),
                },
                "overselected_archetypes_relative_to_performance": _sort_audit_records(
                    [
                        record
                        for record in overselected_archetypes
                        if int(record.get("picks", 0) or 0) > 0
                    ],
                    key_fn=lambda record: (
                        float(record.get("overselection_score") or 0.0),
                        float(record.get("share_of_official_picks") or 0.0),
                    ),
                    limit=10,
                ),
                "repeatedly_failing_combos": _sort_audit_records(
                    [
                        record
                        for record in by_combo
                        if int(record.get("picks", 0) or 0) >= 2
                    ],
                    key_fn=lambda record: (
                        -(float(record.get("units_profit") or 0.0)),
                        int(record.get("losses", 0) or 0),
                        int(record.get("picks", 0) or 0),
                    ),
                    limit=10,
                ),
            },
            "groupings": {
                "by_pitcher": by_pitcher,
                "by_archetype": by_archetype,
                "by_pitcher_side": by_pitcher_side,
                "by_archetype_side": by_archetype_side,
                "by_archetype_line_bucket": by_archetype_line_bucket,
                "by_combo": by_combo,
            },
        }

    all_time_scope = build_scope("all_time", official_df)
    current_regime_scope = build_scope(
        "current_regime",
        official_df.loc[_current_regime_mask(official_df)].copy(),
    )

    regime_comparison = {
        "overall": {
            "all_time_official_picks": all_time_scope["summary"]["official_picks"],
            "current_regime_official_picks": current_regime_scope["summary"]["official_picks"],
            "current_regime_share_of_all_time_picks": (
                current_regime_scope["summary"]["official_picks"] / all_time_scope["summary"]["official_picks"]
                if all_time_scope["summary"]["official_picks"]
                else None
            ),
            "all_time_units_profit": all_time_scope["summary"]["units_profit"],
            "current_regime_units_profit": current_regime_scope["summary"]["units_profit"],
            "all_time_roi": all_time_scope["summary"]["roi"],
            "current_regime_roi": current_regime_scope["summary"]["roi"],
        },
        "by_pitcher": _sort_audit_records(
            _build_regime_comparison_records(
                all_time_records=all_time_scope["groupings"]["by_pitcher"],
                current_regime_records=current_regime_scope["groupings"]["by_pitcher"],
                key_columns=["player_name"],
            ),
            key_fn=lambda record: (
                int(record.get("current_regime_picks", 0) or 0),
                -(float(record.get("current_regime_units_profit") or 0.0)),
                int(record.get("all_time_picks", 0) or 0),
            ),
            limit=20,
        ),
        "by_archetype": _sort_audit_records(
            _build_regime_comparison_records(
                all_time_records=all_time_scope["groupings"]["by_archetype"],
                current_regime_records=current_regime_scope["groupings"]["by_archetype"],
                key_columns=["archetype"],
            ),
            key_fn=lambda record: (
                int(record.get("current_regime_picks", 0) or 0),
                -(float(record.get("current_regime_units_profit") or 0.0)),
                int(record.get("all_time_picks", 0) or 0),
            ),
            limit=20,
        ),
        "by_combo": _sort_audit_records(
            _build_regime_comparison_records(
                all_time_records=all_time_scope["groupings"]["by_combo"],
                current_regime_records=current_regime_scope["groupings"]["by_combo"],
                key_columns=["prop_type", "pick_side", "line_bucket", "archetype"],
            ),
            key_fn=lambda record: (
                int(record.get("current_regime_picks", 0) or 0),
                -(float(record.get("current_regime_units_profit") or 0.0)),
                int(record.get("all_time_picks", 0) or 0),
            ),
            limit=20,
        ),
    }

    return {
        "artifact_type": "official_picks_concentration_audit",
        "artifact_version": 1,
        "current_regime_rule": {
            "type": "start_date",
            "start_date": CURRENT_REGIME_START_DATE,
        },
        "archetype_definition": {
            "version": "v1_coarse_line_based",
            "notes": [
                "Archetypes are inferred from tracked prop type and posted line only.",
                "This keeps the audit reproducible from official_picks_history.csv without external pitcher metadata.",
                "market + side + line bucket groupings are included alongside the coarse archetypes.",
            ],
            "line_buckets": {
                "<5.5": "low line bucket",
                "5.5-7.0": "mid line bucket",
                "7.5+": "high line bucket",
            },
        },
        "scopes": {
            "all_time": all_time_scope,
            "current_regime": current_regime_scope,
        },
        "provenance_groupings": {
            segment: _build_audit_group_records(official_df, group_columns=[segment])
            for segment in TRACKING_SEGMENT_COLUMNS
        },
        "regime_comparison": regime_comparison,
    }


def _build_archetype_risk_lookup_from_audit(audit_payload: dict | None) -> dict[tuple, float]:
    if not audit_payload:
        return {}

    max_penalty = 0.35
    roi_weight = 0.60
    concentration_weight = 0.25
    repeat_weight = 0.15

    def sample_weight(record: dict) -> float:
        all_time = float(record.get("all_time_picks", record.get("picks", 0)) or 0.0)
        current_regime = float(record.get("current_regime_picks", 0) or 0.0)
        blended = (0.7 * current_regime) + (0.3 * all_time)
        return min(1.0, blended / 20.0)

    def underperformance(record: dict) -> float:
        all_time_roi = record.get("all_time_roi", record.get("roi"))
        current_regime_roi = record.get("current_regime_roi")
        if current_regime_roi is None and all_time_roi is None:
            return 0.0
        blended_roi = (
            0.7 * float(current_regime_roi if current_regime_roi is not None else all_time_roi or 0.0)
            + 0.3 * float(all_time_roi or 0.0)
        )
        return max(0.0, 0.0 - blended_roi)

    lookup: dict[tuple, float] = {}

    scopes = audit_payload.get("scopes", {})
    all_time_scope = scopes.get("all_time", {})
    all_time_combo_records = all_time_scope.get("groupings", {}).get("by_combo", [])
    current_combo_records = scopes.get("current_regime", {}).get("groupings", {}).get("by_combo", [])
    all_time_combo_lookup = {
        (
            str(record.get("prop_type", "")).strip().lower(),
            str(record.get("pick_side", "")).strip().lower(),
            str(record.get("line_bucket", "")).strip(),
            str(record.get("archetype", "")).strip(),
        ): record
        for record in all_time_combo_records
    }
    current_combo_lookup = {
        (
            str(record.get("prop_type", "")).strip().lower(),
            str(record.get("pick_side", "")).strip().lower(),
            str(record.get("line_bucket", "")).strip(),
            str(record.get("archetype", "")).strip(),
        ): record
        for record in current_combo_records
    }
    combined_combo_keys = set(all_time_combo_lookup) | set(current_combo_lookup)
    line_bucket_side_rollups: dict[tuple, dict[str, float]] = {}
    for combo_key in combined_combo_keys:
        all_record = all_time_combo_lookup.get(combo_key, {})
        current_record = current_combo_lookup.get(combo_key, {})
        combined_record = {
            "all_time_picks": all_record.get("picks", 0),
            "current_regime_picks": current_record.get("picks", 0),
            "all_time_roi": all_record.get("roi"),
            "current_regime_roi": current_record.get("roi"),
            "share_of_official_picks": (
                0.7 * float(current_record.get("share_of_official_picks", 0.0) or 0.0)
                + 0.3 * float(all_record.get("share_of_official_picks", 0.0) or 0.0)
            ),
            "repeat_pitcher_frequency": (
                0.7 * float(current_record.get("repeat_pitcher_frequency", 0.0) or 0.0)
                + 0.3 * float(all_record.get("repeat_pitcher_frequency", 0.0) or 0.0)
            ),
        }
        risk_score = min(
            max_penalty,
            sample_weight(combined_record)
            * (
                roi_weight * underperformance(combined_record)
                + concentration_weight * float(combined_record["share_of_official_picks"])
                + repeat_weight * float(combined_record["repeat_pitcher_frequency"])
            ),
        )
        lookup[("combo", *combo_key)] = risk_score
        line_bucket_side_key = ("line_bucket_side", combo_key[0], combo_key[1], combo_key[2])
        rollup = line_bucket_side_rollups.setdefault(
            line_bucket_side_key,
            {
                "weighted_risk": 0.0,
                "total_picks": 0.0,
            },
        )
        blended_picks = (0.7 * float(combined_record["current_regime_picks"])) + (
            0.3 * float(combined_record["all_time_picks"])
        )
        rollup["weighted_risk"] += risk_score * blended_picks
        rollup["total_picks"] += blended_picks

    all_time_by_archetype = {
        str(record.get("archetype", "")).strip(): record
        for record in all_time_scope.get("groupings", {}).get("by_archetype", [])
    }
    current_by_archetype = {
        str(record.get("archetype", "")).strip(): record
        for record in scopes.get("current_regime", {}).get("groupings", {}).get("by_archetype", [])
    }
    combined_archetype_keys = set(all_time_by_archetype) | set(current_by_archetype)
    for archetype_key in combined_archetype_keys:
        all_record = all_time_by_archetype.get(archetype_key, {})
        current_record = current_by_archetype.get(archetype_key, {})
        combined_record = {
            "all_time_picks": all_record.get("picks", 0),
            "current_regime_picks": current_record.get("picks", 0),
            "all_time_roi": all_record.get("roi"),
            "current_regime_roi": current_record.get("roi"),
            "share_of_official_picks": (
                0.7 * float(current_record.get("share_of_official_picks", 0.0) or 0.0)
                + 0.3 * float(all_record.get("share_of_official_picks", 0.0) or 0.0)
            ),
            "repeat_pitcher_frequency": (
                0.7 * float(current_record.get("repeat_pitcher_frequency", 0.0) or 0.0)
                + 0.3 * float(all_record.get("repeat_pitcher_frequency", 0.0) or 0.0)
            ),
        }
        risk_score = min(
            max_penalty,
            sample_weight(combined_record)
            * (
                roi_weight * underperformance(combined_record)
                + concentration_weight * float(combined_record["share_of_official_picks"])
                + repeat_weight * float(combined_record["repeat_pitcher_frequency"])
            ),
        )
        lookup[("archetype", archetype_key)] = risk_score

    for line_bucket_side_key, rollup in line_bucket_side_rollups.items():
        total_picks = float(rollup["total_picks"])
        lookup[line_bucket_side_key] = (
            float(rollup["weighted_risk"]) / total_picks
            if total_picks
            else 0.0
        )

    return lookup


def load_archetype_risk_lookup() -> dict[tuple, float]:
    if not OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.exists():
        return {}
    try:
        audit_payload = json.loads(OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return _build_archetype_risk_lookup_from_audit(audit_payload)


def build_official_picks_profit_report(history_df: pd.DataFrame) -> dict[str, object]:
    if history_df.empty:
        all_time_metrics = _empty_profit_summary_metrics()
        current_regime_metrics = _empty_profit_summary_metrics()
        return {
            "graded_df": empty_official_picks_profit_report_df(),
            "summary_by_book_df": empty_official_picks_profit_summary_by_scope_df(),
            "overall_summary": {
                "summary_views": {
                    "all_time": all_time_metrics,
                    "current_regime": current_regime_metrics,
                },
                "current_regime_rule": _current_regime_rule_payload(),
                "segmented_views": {
                    segment: {}
                    for segment in TRACKING_SEGMENT_COLUMNS
                },
                "published_view_artifacts": {
                    "all_time": OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.name,
                    "current_regime": OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.name,
                },
            },
            "published_summary_views": {
                "all_time": _build_published_profit_summary_view(
                    summary_scope="all_time",
                    metrics=all_time_metrics,
                    by_book_df=empty_official_picks_profit_summary_df(),
                ),
                "current_regime": _build_published_profit_summary_view(
                    summary_scope="current_regime",
                    metrics=current_regime_metrics,
                    by_book_df=empty_official_picks_profit_summary_df(),
                ),
            },
            "skipped_df": empty_official_picks_profit_report_df(),
        }

    working_history = history_df.copy()
    for column in OFFICIAL_PICKS_HISTORY_COLUMNS:
        if column not in working_history.columns:
            working_history[column] = ""
    working_history = _hydrate_history_provenance_columns(working_history)

    official_df = working_history[working_history["pick_type"] == "official"].copy()
    if official_df.empty:
        all_time_metrics = _empty_profit_summary_metrics()
        current_regime_metrics = _empty_profit_summary_metrics()
        return {
            "graded_df": empty_official_picks_profit_report_df(),
            "summary_by_book_df": empty_official_picks_profit_summary_by_scope_df(),
            "overall_summary": {
                "summary_views": {
                    "all_time": all_time_metrics,
                    "current_regime": current_regime_metrics,
                },
                "current_regime_rule": _current_regime_rule_payload(),
                "segmented_views": {
                    segment: {}
                    for segment in TRACKING_SEGMENT_COLUMNS
                },
                "published_view_artifacts": {
                    "all_time": OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.name,
                    "current_regime": OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.name,
                },
            },
            "published_summary_views": {
                "all_time": _build_published_profit_summary_view(
                    summary_scope="all_time",
                    metrics=all_time_metrics,
                    by_book_df=empty_official_picks_profit_summary_df(),
                ),
                "current_regime": _build_published_profit_summary_view(
                    summary_scope="current_regime",
                    metrics=current_regime_metrics,
                    by_book_df=empty_official_picks_profit_summary_df(),
                ),
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
    all_time_summary_by_book_df = summarize_official_picks_profit_by_book(graded_df)

    current_regime_history_df = official_df.loc[_current_regime_mask(official_df)].copy()
    current_regime_keys = set(current_regime_history_df["pick_key"].astype(str))
    current_regime_graded_df = graded_df.loc[
        graded_df["pick_key"].astype(str).isin(current_regime_keys)
    ].copy()
    current_regime_skipped_df = skipped_df.loc[
        skipped_df["pick_key"].astype(str).isin(current_regime_keys)
    ].copy()
    current_regime_summary_by_book_df = summarize_official_picks_profit_by_book(current_regime_graded_df)

    summary_rows: list[pd.DataFrame] = [
        all_time_summary_by_book_df.assign(
            summary_scope="all_time",
            segment_type="summary_scope",
            segment_value="all_time",
        ),
        current_regime_summary_by_book_df.assign(
            summary_scope="current_regime",
            segment_type="summary_scope",
            segment_value="current_regime",
        ),
    ]
    for segment in TRACKING_SEGMENT_COLUMNS:
        _append_segmented_summary_rows(
            summary_rows,
            graded_df=graded_df,
            segment_type=segment,
        )
    summary_by_book_df = pd.concat(summary_rows, ignore_index=True)
    if summary_by_book_df.empty:
        summary_by_book_df = empty_official_picks_profit_summary_by_scope_df()
    else:
        summary_by_book_df = summary_by_book_df[OFFICIAL_PICKS_PROFIT_SUMMARY_SCOPE_COLUMNS].copy()

    all_time_metrics = _build_profit_summary_metrics(
        graded_df=graded_df,
        summary_by_book_df=all_time_summary_by_book_df,
        skipped_df=skipped_df,
    )
    current_regime_metrics = _build_profit_summary_metrics(
        graded_df=current_regime_graded_df,
        summary_by_book_df=current_regime_summary_by_book_df,
        skipped_df=current_regime_skipped_df,
    )

    overall_summary = {
        "summary_views": {
            "all_time": all_time_metrics,
            "current_regime": current_regime_metrics,
        },
        "current_regime_rule": _current_regime_rule_payload(),
        "published_view_artifacts": {
            "all_time": OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.name,
            "current_regime": OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.name,
        },
        "segmented_views": {
            segment: _summarize_official_picks_profit_by_dimension(
                graded_df,
                dimension=segment,
            )
            for segment in TRACKING_SEGMENT_COLUMNS
        },
    }

    return {
        "graded_df": graded_df,
        "summary_by_book_df": summary_by_book_df,
        "overall_summary": overall_summary,
        "published_summary_views": {
            "all_time": _build_published_profit_summary_view(
                summary_scope="all_time",
                metrics=all_time_metrics,
                by_book_df=all_time_summary_by_book_df,
            ),
            "current_regime": _build_published_profit_summary_view(
                summary_scope="current_regime",
                metrics=current_regime_metrics,
                by_book_df=current_regime_summary_by_book_df,
            ),
        },
        "skipped_df": skipped_df,
    }


def _csv_artifact_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return False
    return not df.empty


def _summary_artifact_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    summary_views = payload.get("summary_views", {})
    if not summary_views:
        summary_metrics = payload.get("summary_metrics", {})
        if summary_metrics:
            return any(
                int(summary_metrics.get(key, 0) or 0) > 0
                for key in ["books", "picks", "wins", "losses", "pushes", "decisions", "skipped_rows"]
            )
        return any(
            int(payload.get(key, 0) or 0) > 0
            for key in ["books", "picks", "wins", "losses", "pushes", "decisions", "skipped_rows"]
        )
    return any(
        int(summary.get(key, 0) or 0) > 0
        for summary in summary_views.values()
        for key in ["books", "picks", "wins", "losses", "pushes", "decisions", "skipped_rows"]
    )


def _report_has_tracking_content(report: dict[str, object]) -> bool:
    return any(
        not report[df_name].empty
        for df_name in ["graded_df", "summary_by_book_df", "skipped_df"]
    ) or any(
        int(summary.get(key, 0) or 0) > 0
        for summary in report["overall_summary"].get("summary_views", {}).values()
        for key in ["books", "picks", "wins", "losses", "pushes", "decisions", "skipped_rows"]
    )


def _existing_tracking_artifacts_have_content() -> bool:
    return any(
        [
            _csv_artifact_has_rows(OFFICIAL_PICKS_GRADES_PATH),
            _csv_artifact_has_rows(OFFICIAL_PICKS_BOOK_SUMMARY_PATH),
            _csv_artifact_has_rows(OFFICIAL_PICKS_SKIPPED_PATH),
            _summary_artifact_has_content(OFFICIAL_PICKS_OVERALL_SUMMARY_PATH),
            _summary_artifact_has_content(OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH),
            _summary_artifact_has_content(OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH),
        ]
    )


def persist_official_picks_profit_reports(*, allow_empty_replacement: bool = False) -> dict[str, object]:
    history_df = load_official_picks_history()
    report = build_official_picks_profit_report(history_df)
    concentration_audit = build_official_picks_concentration_audit(history_df)
    if (
        not allow_empty_replacement
        and not _report_has_tracking_content(report)
        and _existing_tracking_artifacts_have_content()
    ):
        raise ValueError(
            "Refusing to overwrite non-empty tracking summaries with empty artifacts. "
            "official_picks_history is empty or incomplete for rebuilding summaries."
        )
    report["graded_df"].to_csv(OFFICIAL_PICKS_GRADES_PATH, index=False)
    report["summary_by_book_df"].to_csv(OFFICIAL_PICKS_BOOK_SUMMARY_PATH, index=False)
    report["skipped_df"].to_csv(OFFICIAL_PICKS_SKIPPED_PATH, index=False)
    OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.write_text(
        json.dumps(report["overall_summary"], indent=2),
        encoding="utf-8",
    )
    OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.write_text(
        json.dumps(report["published_summary_views"]["all_time"], indent=2),
        encoding="utf-8",
    )
    OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.write_text(
        json.dumps(report["published_summary_views"]["current_regime"], indent=2),
        encoding="utf-8",
    )
    OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.write_text(
        json.dumps(concentration_audit, indent=2),
        encoding="utf-8",
    )
    return report


def load_official_picks_history() -> pd.DataFrame:
    if not OFFICIAL_PICKS_HISTORY_PATH.exists():
        return empty_official_picks_history_df()

    history_df = pd.read_csv(OFFICIAL_PICKS_HISTORY_PATH, keep_default_na=False)
    missing = [col for col in OFFICIAL_PICKS_HISTORY_COLUMNS if col not in history_df.columns]
    for column in missing:
        history_df[column] = ""
    if "predicted_value" in history_df.columns and "predicted_strikeouts" in history_df.columns:
        history_df["predicted_value"] = history_df["predicted_value"].where(
            history_df["predicted_value"].astype(str).str.strip() != "",
            history_df["predicted_strikeouts"],
        )
        history_df["predicted_strikeouts"] = history_df["predicted_strikeouts"].where(
            history_df["predicted_strikeouts"].astype(str).str.strip() != "",
            history_df["predicted_value"],
        )
    if "actual_value" in history_df.columns and "actual_strikeouts" in history_df.columns:
        history_df["actual_value"] = history_df["actual_value"].where(
            history_df["actual_value"].astype(str).str.strip() != "",
            history_df["actual_strikeouts"],
        )
        history_df["actual_strikeouts"] = history_df["actual_strikeouts"].where(
            history_df["actual_strikeouts"].astype(str).str.strip() != "",
            history_df["actual_value"],
        )
    history_df = _hydrate_history_provenance_columns(history_df)

    return history_df[OFFICIAL_PICKS_HISTORY_COLUMNS].copy()


def build_official_picks_history_rows(
    starters_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> pd.DataFrame:
    official_df = post_df[post_df["pick_type"] == "official"].copy()
    if official_df.empty:
        return empty_official_picks_history_df()

    official_df = ensure_participant_identity(
        official_df,
        display_name_col="player_name",
        normalized_name_col="player_name_norm" if "player_name_norm" in official_df.columns else None,
        source_id_col=PARTICIPANT_SOURCE_ID_COLUMN if PARTICIPANT_SOURCE_ID_COLUMN in official_df.columns else None,
        source_id_type="mlbam_player",
    )
    official_df = ensure_market_identity(official_df)

    starter_lookup = starters_df[
        ["player_name", "team", "opponent", "game_date", "pitcher"]
        if "pitcher" in starters_df.columns
        else ["player_name", "team", "opponent", "game_date"]
    ].copy()
    starter_lookup = ensure_participant_identity(
        starter_lookup,
        display_name_col="player_name",
        normalized_name_col="player_name_norm" if "player_name_norm" in starter_lookup.columns else None,
        source_id_col="pitcher" if "pitcher" in starter_lookup.columns else None,
        source_id_type="mlbam_player",
    )
    starter_lookup["game_date"] = pd.to_datetime(starter_lookup["game_date"]).dt.strftime("%Y-%m-%d")
    merge_keys = [PARTICIPANT_JOIN_KEY_COLUMN]
    if {"team", "opponent"}.issubset(official_df.columns):
        merge_keys = [PARTICIPANT_JOIN_KEY_COLUMN, "team", "opponent"]

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
        lambda row: _build_pick_key(
            row["game_date"],
            row["player_name"],
            row.get(MARKET_OFFER_KEY_COLUMN),
        ),
        axis=1,
    )
    history_rows["odds"] = history_rows["price"].apply(_format_american_odds)
    history_rows["result"] = ""
    history_rows["predicted_value"] = _prediction_value_from_frame(history_rows)
    if "predicted_strikeouts" in history_rows.columns:
        history_rows["predicted_strikeouts"] = history_rows["predicted_strikeouts"].where(
            history_rows["predicted_strikeouts"].notna(),
            history_rows["predicted_value"],
        )
    else:
        history_rows["predicted_strikeouts"] = history_rows["predicted_value"]
    history_rows["actual_value"] = ""
    history_rows["actual_strikeouts"] = ""
    if "record_source" not in history_rows.columns:
        history_rows["record_source"] = "run_daily_card"
    history_rows["record_source"] = history_rows["record_source"].where(
        history_rows["record_source"].astype(str).str.strip() != "",
        "run_daily_card",
    )
    history_rows = _hydrate_history_provenance_columns(history_rows)

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
            if existing_record.get("actual_value") and not new_record.get("actual_value"):
                new_record["actual_value"] = existing_record["actual_value"]
            if existing_record.get("actual_strikeouts") and not new_record.get("actual_strikeouts"):
                new_record["actual_strikeouts"] = existing_record["actual_strikeouts"]
            if _is_manual_record_source(existing_record.get("record_source")):
                new_record["record_source"] = existing_record["record_source"]
                if existing_record.get("model_version"):
                    new_record["model_version"] = existing_record["model_version"]
                if existing_record.get("policy_version"):
                    new_record["policy_version"] = existing_record["policy_version"]
                if existing_record.get("tracking_regime"):
                    new_record["tracking_regime"] = existing_record["tracking_regime"]
            else:
                if existing_record.get("record_source") and not new_record.get("record_source"):
                    new_record["record_source"] = existing_record["record_source"]
                if existing_record.get("model_version") and not new_record.get("model_version"):
                    new_record["model_version"] = existing_record["model_version"]
                if existing_record.get("policy_version") and not new_record.get("policy_version"):
                    new_record["policy_version"] = existing_record["policy_version"]
                if existing_record.get("tracking_regime") and not new_record.get("tracking_regime"):
                    new_record["tracking_regime"] = existing_record["tracking_regime"]

        merged_rows.append(new_record)

    merged_df = pd.DataFrame(merged_rows, columns=OFFICIAL_PICKS_HISTORY_COLUMNS)
    untouched_existing = existing_df[~existing_df["pick_key"].isin(merged_df["pick_key"])]
    history_df = pd.concat([untouched_existing, merged_df], ignore_index=True)
    history_df = _hydrate_history_provenance_columns(history_df)
    history_df = history_df[OFFICIAL_PICKS_HISTORY_COLUMNS]
    history_df.to_csv(OFFICIAL_PICKS_HISTORY_PATH, index=False)
    return OFFICIAL_PICKS_HISTORY_PATH


def resolve_artifact_path(filename: str, workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW) -> Path:
    candidate_paths = [
        candidate_dir / filename
        for candidate_dir in _artifact_dir_candidates(workflow)
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing {filename} artifact in both latest/ and previous/."
    )


def load_workflow_history_artifact(workflow: ModelingWorkflowSpec) -> pd.DataFrame:
    path = resolve_artifact_path(workflow.artifacts.history_filename, workflow)
    history_df = workflow.artifacts.history_loader(path)
    print(f"Loaded workflow history artifact from: {path}")
    return history_df


def load_workflow_model_artifact(workflow: ModelingWorkflowSpec):
    path = resolve_artifact_path(workflow.artifacts.model_filename, workflow)
    model = workflow.artifacts.model_loader(path)
    print(f"Loaded model artifact from: {path}")
    return model


def load_model_metadata(workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW) -> dict:
    model_path = resolve_artifact_path(workflow.artifacts.model_filename, workflow)
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
    today_preds = ensure_participant_identity(
        today_preds,
        display_name_col=workflow.participant_key,
        normalized_name_col="player_name_norm" if "player_name_norm" in today_preds.columns else None,
        source_id_col="pitcher" if "pitcher" in today_preds.columns else None,
        source_id_type="mlbam_player",
    )
    today_preds = ensure_market_identity(
        today_preds,
        sport=workflow.sport,
        market_key=workflow.market_key,
    )
    if workflow.prop_fields.prediction not in today_preds.columns:
        raise ValueError(
            "today_preds is missing the workflow prediction column "
            f"'{workflow.prop_fields.prediction}'."
        )
    today_preds[workflow.prop_fields.shared_prediction] = today_preds[workflow.prop_fields.prediction]
    if workflow.prop_fields.actual in today_preds.columns:
        today_preds[workflow.prop_fields.shared_actual] = today_preds[workflow.prop_fields.actual]
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
    try:
        grade_official_picks_from_statcast()
    except Exception as exc:
        print(
            "WARNING: Failed to grade official picks from Statcast for "
            f"{_yesterday_game_date()}: {exc.__class__.__name__}: {exc}"
        )
        persist_official_picks_profit_reports()
    save_run_status(status=run_status, message=run_message)


def workflow_is_ready(workflow: ModelingWorkflowSpec) -> bool:
    try:
        resolve_artifact_path(workflow.artifacts.history_filename, workflow)
        resolve_artifact_path(workflow.artifacts.model_filename, workflow)
        resolve_artifact_path(workflow.artifacts.metadata_filename, workflow)
    except FileNotFoundError:
        return False
    return True


def _build_no_edges_message(
    *,
    workflow: ModelingWorkflowSpec,
    diagnostics: dict[str, object] | None,
) -> str:
    diagnostics = diagnostics or {}
    normalized_odds_rows = int(diagnostics.get("normalized_odds_rows", 0) or 0)
    raw_event_count = int(diagnostics.get("raw_event_count", 0) or 0)
    joined_rows = int(diagnostics.get("joined_rows", 0) or 0)
    fetch_scope = str(diagnostics.get("fetch_scope", "") or "")
    initial_fetch = diagnostics.get("initial_fetch")
    initial_odds_rows = 0
    if isinstance(initial_fetch, dict):
        initial_odds_rows = int(initial_fetch.get("normalized_odds_rows", 0) or 0)

    if normalized_odds_rows == 0 and initial_odds_rows == 0:
        if fetch_scope == "all_region_books":
            return (
                "No live odds rows were returned for today's "
                f"{workflow.prop_type} market from either the configured bookmakers or the broader "
                "US bookmaker feed, so no edges or picks were generated."
            )
        return (
            "No live odds rows were returned for today's "
            f"{workflow.prop_type} market from the configured bookmakers, so no edges or picks "
            "were generated."
        )

    if joined_rows == 0:
        return (
            "Live odds were available for today's "
            f"{workflow.prop_type} market ({normalized_odds_rows} normalized rows across "
            f"{raw_event_count} event(s)), but none matched today's projections, so no edges or "
            "picks were generated."
        )

    return (
        f"No edges or picks were generated for today's {workflow.prop_type} run."
    )


def resolve_daily_card_workflows(
    workflows: list[ModelingWorkflowSpec] | None = None,
) -> list[ModelingWorkflowSpec]:
    if workflows is not None:
        return workflows

    resolved: list[ModelingWorkflowSpec] = []
    for workflow in DEFAULT_DAILY_CARD_WORKFLOWS:
        if not workflow.enabled_in_default_daily_card:
            continue
        if workflow_is_ready(workflow):
            resolved.append(workflow)

    if resolved:
        return resolved

    return [MLB_PITCHER_STRIKEOUT_WORKFLOW]


def run_workflow_daily_card(
    *,
    starters_df: pd.DataFrame,
    workflow: ModelingWorkflowSpec,
    market: str | None = None,
    build_picks_fn: BuildPicksFn | None = None,
    filter_postable_picks_fn: FilterPostablePicksFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None]:
    archetype_risk_lookup = load_archetype_risk_lookup()

    if build_picks_fn is None:
        def build_picks_fn(joined_df: pd.DataFrame) -> pd.DataFrame:
            try:
                return build_daily_picks(
                    joined_df,
                    policy=workflow.pick_ranking_policy,
                    prediction_column=workflow.prop_fields.shared_prediction,
                    archetype_risk_lookup=archetype_risk_lookup,
                )
            except TypeError as exc:
                if "archetype_risk_lookup" not in str(exc):
                    raise
                return build_daily_picks(
                    joined_df,
                    policy=workflow.pick_ranking_policy,
                    prediction_column=workflow.prop_fields.shared_prediction,
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

    today_preds = _adapt_predictions_for_output(today_preds, workflow)
    workflow_game_date = pd.to_datetime(starters_df["game_date"], errors="coerce").min()
    today_preds = _annotate_workflow_provenance(
        today_preds,
        workflow=workflow,
        metadata=metadata,
        game_date=workflow_game_date,
    )
    selected_market = market or workflow.market_key
    run_status = "success"
    run_message: str | None = None

    try:
        joined_df, _, edge_diagnostics = run_edge_pipeline(
            today_preds,
            selected_market,
            participant_key=workflow.participant_key,
            prediction_column=workflow.prop_fields.shared_prediction,
            projection_join_key=workflow.projection_odds_join_keys.projection,
            odds_join_key=workflow.projection_odds_join_keys.odds,
            sport=workflow.sport,
        )
        if not joined_df.empty and workflow.prop_fields.shared_prediction not in joined_df.columns:
            joined_df[workflow.prop_fields.shared_prediction] = _prediction_value_from_frame(joined_df)
        joined_df = _tag_workflow_frame(joined_df, workflow)
        joined_df = _annotate_workflow_provenance(
            joined_df,
            workflow=workflow,
            metadata=metadata,
            game_date=workflow_game_date,
        )
        if joined_df.empty:
            run_status = "degraded"
            run_message = _build_no_edges_message(
                workflow=workflow,
                diagnostics=edge_diagnostics,
            )
            print(f"WARNING: {run_message}")
            joined_df = _tag_workflow_frame(empty_joined_odds_df(), workflow)
            picks_df = _tag_workflow_frame(empty_final_picks_df(), workflow)
            post_df = _tag_workflow_frame(empty_final_picks_df(), workflow)
            joined_df = _annotate_workflow_provenance(
                joined_df,
                workflow=workflow,
                metadata=metadata,
                game_date=workflow_game_date,
            )
            picks_df = _annotate_workflow_provenance(
                picks_df,
                workflow=workflow,
                metadata=metadata,
                game_date=workflow_game_date,
            )
            post_df = _annotate_workflow_provenance(
                post_df,
                workflow=workflow,
                metadata=metadata,
                game_date=workflow_game_date,
            )
        else:
            validate_joined_odds_contract(
                joined_df,
                prediction_column=workflow.prop_fields.shared_prediction,
            )

            picks_df = build_picks_fn(joined_df)
            picks_df = _tag_workflow_frame(picks_df, workflow)
            picks_df = _annotate_workflow_provenance(
                picks_df,
                workflow=workflow,
                metadata=metadata,
                game_date=workflow_game_date,
            )
            validate_final_picks_contract(picks_df)

            post_df = filter_postable_picks_fn(picks_df)
            post_df = _tag_workflow_frame(post_df, workflow)
            post_df = _annotate_workflow_provenance(
                post_df,
                workflow=workflow,
                metadata=metadata,
                game_date=workflow_game_date,
            )
            validate_final_picks_contract(post_df, require_non_empty_frame=False)
    except requests.RequestException as exc:
        run_status = "degraded"
        run_message = (
            f"Live odds fetch failed for {workflow.prop_type}; projections were saved but no edges "
            f"or picks were generated. Reason: {exc.__class__.__name__}: {exc}"
        )
        print(f"WARNING: {run_message}")
        joined_df = _tag_workflow_frame(empty_joined_odds_df(), workflow)
        picks_df = _tag_workflow_frame(empty_final_picks_df(), workflow)
        post_df = _tag_workflow_frame(empty_final_picks_df(), workflow)
        joined_df = _annotate_workflow_provenance(
            joined_df,
            workflow=workflow,
            metadata=metadata,
            game_date=workflow_game_date,
        )
        picks_df = _annotate_workflow_provenance(
            picks_df,
            workflow=workflow,
            metadata=metadata,
            game_date=workflow_game_date,
        )
        post_df = _annotate_workflow_provenance(
            post_df,
            workflow=workflow,
            metadata=metadata,
            game_date=workflow_game_date,
        )

    return today_preds, joined_df, picks_df, post_df, run_status, run_message


def run_daily_card(
    *,
    workflow: ModelingWorkflowSpec = MLB_PITCHER_STRIKEOUT_WORKFLOW,
    workflows: list[ModelingWorkflowSpec] | None = None,
    market: str | None = None,
    build_picks_fn: BuildPicksFn | None = None,
    filter_postable_picks_fn: FilterPostablePicksFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()

    starters_df = get_today_starters_df()
    validate_starters_contract(starters_df)

    if workflows is not None:
        selected_workflows = resolve_daily_card_workflows(workflows)
    elif workflow is MLB_PITCHER_STRIKEOUT_WORKFLOW:
        selected_workflows = resolve_daily_card_workflows()
    else:
        selected_workflows = [workflow]

    today_preds_frames: list[pd.DataFrame] = []
    joined_frames: list[pd.DataFrame] = []
    picks_frames: list[pd.DataFrame] = []
    post_frames: list[pd.DataFrame] = []
    run_status = "success"
    run_messages: list[str] = []

    for active_workflow in selected_workflows:
        (
            workflow_today_preds,
            workflow_joined_df,
            workflow_picks_df,
            workflow_post_df,
            workflow_status,
            workflow_message,
        ) = run_workflow_daily_card(
            starters_df=starters_df,
            workflow=active_workflow,
            market=market,
            build_picks_fn=build_picks_fn,
            filter_postable_picks_fn=filter_postable_picks_fn,
        )
        today_preds_frames.append(workflow_today_preds)
        joined_frames.append(workflow_joined_df)
        picks_frames.append(workflow_picks_df)
        post_frames.append(workflow_post_df)

        if workflow_status != "success":
            run_status = "degraded"
        if workflow_message:
            run_messages.append(workflow_message)

    today_preds = pd.concat(today_preds_frames, ignore_index=True) if today_preds_frames else pd.DataFrame()
    joined_df = pd.concat(joined_frames, ignore_index=True) if joined_frames else empty_joined_odds_df()
    picks_df = pd.concat(picks_frames, ignore_index=True) if picks_frames else empty_final_picks_df()
    post_df = pd.concat(post_frames, ignore_index=True) if post_frames else empty_final_picks_df()
    run_message = " ".join(run_messages) if run_messages else None

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
                    "prop_type",
                    "player_name",
                    "book",
                    "pick_side",
                    "line",
                    "price",
                    "predicted_value",
                    "edge",
                    "pick_type",
                ]
            ].to_string(index=False)
        )
