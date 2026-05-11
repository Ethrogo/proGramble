from __future__ import annotations

from copy import deepcopy


def _listify_limitations(*values: object) -> list[str]:
    limitations: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                limitations.append(stripped)
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    limitations.append(item.strip())
    return limitations


def _build_holdout_sections(metadata: dict) -> list[dict]:
    training_window = metadata.get("training_window", {})
    evaluation_metrics = metadata.get("evaluation_metrics", {})
    uncertainty_model = metadata.get("uncertainty_model", {})
    sample_sizes = evaluation_metrics.get("sample_sizes", {})

    base_context = {
        "mode": "fixed_holdout",
        "sample_size": {
            "rows": sample_sizes.get("test_rows", training_window.get("test_rows")),
            "train_rows": sample_sizes.get("train_rows", training_window.get("train_rows")),
        },
        "date_range": training_window.get("test_game_date_range", {"start": None, "end": None}),
    }

    return [
        {
            "name": "holdout_regression",
            **deepcopy(base_context),
            "metrics": deepcopy(evaluation_metrics.get("regression", {})),
            "limitations": _listify_limitations(
                "Uses one fixed time-based holdout split defined by train_split_date.",
                "Measures prediction error only; it does not reflect pick-selection or bankroll outcomes.",
            ),
        },
        {
            "name": "holdout_uncertainty",
            **deepcopy(base_context),
            "metrics": deepcopy(evaluation_metrics.get("uncertainty", {})),
            "calibration": deepcopy(uncertainty_model),
            "limitations": _listify_limitations(
                uncertainty_model.get("documented_interpretation"),
                "Coverage is evaluated on the same fixed holdout window and depends on the recent-stddev signal being present.",
            ),
        },
        {
            "name": "holdout_bucketed_error",
            **deepcopy(base_context),
            "metrics": deepcopy(evaluation_metrics.get("bucketed_error", {})),
            "limitations": _listify_limitations(
                "Bucket summaries are descriptive slices of the fixed holdout set and can be noisy in small buckets.",
            ),
        },
    ]


def _build_workflow_backtest_section(metadata: dict) -> dict:
    training_window = metadata.get("training_window", {})
    workflow_backtest = deepcopy(metadata.get("evaluation_metrics", {}).get("workflow_backtest", {}))
    historical_lines_artifact = metadata.get("historical_lines_artifact", {})
    overall_records = workflow_backtest.get("overall") or []
    overall_summary = overall_records[0] if overall_records else {}

    return {
        "name": "workflow_backtest",
        "mode": "fixed_holdout_workflow_backtest",
        "available": bool(workflow_backtest.get("available", False)),
        "sample_size": {
            "holdout_rows": metadata.get("evaluation_metrics", {}).get("sample_sizes", {}).get(
                "test_rows",
                training_window.get("test_rows"),
            ),
            "graded_pick_rows": workflow_backtest.get("graded_pick_rows", 0),
            "picks": overall_summary.get("picks"),
            "decisions": overall_summary.get("decisions"),
        },
        "date_range": training_window.get("test_game_date_range", {"start": None, "end": None}),
        "metrics": workflow_backtest,
        "limitations": _listify_limitations(
            historical_lines_artifact.get("limitations"),
            workflow_backtest.get("reason"),
            "Runs the live pick-selection workflow on holdout-era historical lines rather than a rolling retrain backtest.",
        ),
    }


def _build_tracked_performance_section() -> dict:
    return {
        "name": "tracked_performance",
        "mode": "all_time_tracked_performance",
        "available": False,
        "included_in_reproducible_artifact": False,
        "metrics": {},
        "limitations": [
            "Excluded from this reproducible artifact build because tracked performance depends on operational pick history and grading files, not just training inputs.",
        ],
    }


def build_evaluation_summary(
    metadata: dict,
    *,
    workflow_name: str,
    artifact_family: str,
) -> dict:
    training_window = metadata.get("training_window", {})

    sections = [
        *_build_holdout_sections(metadata),
        _build_workflow_backtest_section(metadata),
        _build_tracked_performance_section(),
    ]

    return {
        "artifact_type": "evaluation_summary",
        "artifact_version": 1,
        "workflow_name": workflow_name,
        "artifact_family": artifact_family,
        "target": metadata.get("target"),
        "train_split_date": training_window.get("train_split_date"),
        "training_window": {
            "raw_statcast_start": training_window.get("raw_statcast_start"),
            "raw_statcast_end": training_window.get("raw_statcast_end"),
            "train_game_date_range": deepcopy(
                training_window.get("train_game_date_range", {"start": None, "end": None})
            ),
            "test_game_date_range": deepcopy(
                training_window.get("test_game_date_range", {"start": None, "end": None})
            ),
        },
        "sections": sections,
        "limitations": _listify_limitations(
            "This artifact is reproducible from the training-artifact build and intentionally separates model evaluation from daily card outputs.",
            "Fixed-holdout sections use the train/test split defined during model training.",
        ),
    }
