from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from common.contracts import require_columns, validate_pitcher_games_contract
from common.evaluation_artifacts import build_evaluation_summary
from odds.backtest import run_historical_workflow_backtest, summarize_backtest_for_metadata
from odds.historical_lines import build_historical_lines_artifact_df, empty_historical_lines_df
from pitcher_bb.config import (
    BASE_FEATURES,
    PITCHER_BB_PROP_MARKET,
    RAW_STATCAST_END,
    RAW_STATCAST_START,
    TARGET_COL,
    TARGET_FORMULATION,
    TRAIN_SPLIT_DATE,
    TRAIN_VALIDATION_FRACTION,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_NUM_BOOST_ROUND,
    XGB_PARAMS,
)
from pitcher_bb.feature_engineering import (
    build_pitcher_walk_feature_table,
    filter_starter_like_appearances,
)
from pitcher_bb.feature_model import build_model_df
from pitcher_bb.train import time_split, train_model
from pitcher_k.data_loader import load_statcast_data
from pitcher_k.evaluate import evaluate_predictions, fit_interval_calibration, summarize_interval_coverage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts" / "pitcher_bb"
STAGING_DIR = ARTIFACTS_DIR / "_staging"
LATEST_DIR = ARTIFACTS_DIR / "latest"
PREVIOUS_DIR = ARTIFACTS_DIR / "previous"
MODEL_FILENAME = "model.ubj"
PITCHER_GAMES_FILENAME = "pitcher_games.csv"
MODEL_DF_FILENAME = "model_df.csv"
HISTORICAL_LINES_FILENAME = "historical_lines.csv"
METADATA_FILENAME = "metadata.json"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
RAW_HISTORICAL_LINES_DIR = DATA_DIR / "raw" / "historical_lines"
UNCERTAINTY_STDDEV_COLUMN = "walks_stddev_last10"
MODEL_VERSION_LABEL = "pitcher_bb_model_v1"


def ensure_artifact_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    PREVIOUS_DIR.mkdir(parents=True, exist_ok=True)


def artifact_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "model": base_dir / MODEL_FILENAME,
        "pitcher_games": base_dir / PITCHER_GAMES_FILENAME,
        "model_df": base_dir / MODEL_DF_FILENAME,
        "historical_lines": base_dir / HISTORICAL_LINES_FILENAME,
        "metadata": base_dir / METADATA_FILENAME,
        "evaluation_summary": base_dir / EVALUATION_SUMMARY_FILENAME,
    }


def artifact_set_exists(base_dir: Path) -> bool:
    return all(path.exists() for path in artifact_paths(base_dir).values())


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_artifact_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def promote_latest_to_previous() -> None:
    if artifact_set_exists(LATEST_DIR):
        copy_artifact_dir(LATEST_DIR, PREVIOUS_DIR)


def promote_staging_to_latest() -> None:
    copy_artifact_dir(STAGING_DIR, LATEST_DIR)


def build_historical_pitcher_games() -> pd.DataFrame:
    sc = load_statcast_data(RAW_STATCAST_START, RAW_STATCAST_END)
    pitcher_games = build_pitcher_walk_feature_table(sc)

    validate_pitcher_games_contract(pitcher_games)
    require_columns(pitcher_games, BASE_FEATURES + [TARGET_COL], "pitcher_games")
    return pitcher_games


def build_native_historical_lines() -> pd.DataFrame:
    return build_historical_lines_artifact_df(
        RAW_HISTORICAL_LINES_DIR,
        market_key=PITCHER_BB_PROP_MARKET,
    )


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty:
        return {"start": None, "end": None}

    game_dates = pd.to_datetime(df["game_date"])
    return {
        "start": game_dates.min().strftime("%Y-%m-%d"),
        "end": game_dates.max().strftime("%Y-%m-%d"),
    }


def _build_workflow_backtest_summary(
    *,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test,
    historical_lines_df: pd.DataFrame | None,
) -> dict:
    if historical_lines_df is None or historical_lines_df.empty:
        return {
            "available": False,
            "reason": "historical_market_lines_not_provided",
            "reproducible_path": "odds.backtest.run_historical_workflow_backtest",
        }

    projections = test_df[["game_date", "player_name"]].copy()
    projections["predicted_walks"] = pd.Series(y_pred_test, index=test_df.index).values
    projections["actual_walks"] = pd.Series(y_test, index=test_df.index).values

    backtest_result = run_historical_workflow_backtest(
        projections,
        historical_lines_df,
        prediction_column="predicted_walks",
        actual_column="actual_walks",
        market_key=PITCHER_BB_PROP_MARKET,
    )
    backtest_summary = summarize_backtest_for_metadata(backtest_result)
    backtest_summary["reproducible_path"] = "odds.backtest.run_historical_workflow_backtest"
    return backtest_summary


def _build_prediction_results(X_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    results = X_test.copy()
    results["actual_walks"] = pd.Series(y_test, dtype="float64").reset_index(drop=True)
    results["predicted_walks"] = pd.Series(y_pred, dtype="float64").reset_index(drop=True)
    results["error"] = results["predicted_walks"] - results["actual_walks"]
    results["abs_error"] = results["error"].abs()
    return results


def _build_error_bucket_summary(results_df: pd.DataFrame) -> list[dict]:
    working = results_df.copy()
    bins = [-float("inf"), 1.5, 2.5, 3.5, 4.5, float("inf")]
    labels = ["<=1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5", "4.5+"]
    working["error_bucket"] = pd.cut(
        working["predicted_walks"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    summary = (
        working.groupby("error_bucket", observed=False)
        .agg(
            rows=("predicted_walks", "size"),
            mean_actual=("actual_walks", "mean"),
            mean_prediction=("predicted_walks", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["rows"] > 0]
    summary["error_bucket"] = summary["error_bucket"].astype(str)
    return summary.to_dict(orient="records")


def _evaluation_metrics(
    train_output: dict,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    historical_lines_df: pd.DataFrame | None = None,
) -> dict:
    y_train = train_output["y_train"]
    y_test = train_output["y_test"]
    y_pred_train = train_output["model"].predict(train_output["dtrain"])
    y_pred_test = train_output["model"].predict(train_output["dtest"])

    test_results = _build_prediction_results(test_df, y_test, y_pred_test)
    interval_config = fit_interval_calibration(
        train_df,
        y_train,
        y_pred_train,
        stddev_column=UNCERTAINTY_STDDEV_COLUMN,
    )

    return {
        "regression": evaluate_predictions(y_test, y_pred_test),
        "bucketed_error": {
            "bucket_by": "predicted_walks",
            "buckets": _build_error_bucket_summary(test_results),
        },
        "uncertainty": summarize_interval_coverage(
            test_df,
            y_test,
            y_pred_test,
            interval_config,
            stddev_column=UNCERTAINTY_STDDEV_COLUMN,
        ),
        "workflow_backtest": _build_workflow_backtest_summary(
            test_df=test_df,
            y_test=y_test,
            y_pred_test=y_pred_test,
            historical_lines_df=historical_lines_df,
        ),
        "sample_sizes": {
            "train_rows": int(len(train_output["X_train"])),
            "test_rows": int(len(train_output["X_test"])),
        },
    }


def build_training_metadata(
    model_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_output: dict,
    historical_lines_df: pd.DataFrame | None = None,
) -> dict:
    uncertainty_model = fit_interval_calibration(
        train_df,
        train_output["y_train"],
        train_output["model"].predict(train_output["dtrain"]),
        stddev_column=UNCERTAINTY_STDDEV_COLUMN,
    )
    uncertainty_model["documented_interpretation"] = (
        "lower_bound and upper_bound represent an empirical central prediction "
        "interval targeting 80% coverage on held-out data when the recent walk "
        "standard deviation signal is present."
    )
    validation_df = train_output.get("validation_df")
    validation_rows = (
        int(len(train_output["X_validation"]))
        if train_output.get("X_validation") is not None
        else 0
    )

    return {
        "artifact_version": 1,
        "model_version": MODEL_VERSION_LABEL,
        "target": TARGET_COL,
        "target_formulation": TARGET_FORMULATION,
        "features": BASE_FEATURES,
        "model_params": {
            "xgb_params": XGB_PARAMS,
            "candidate_num_boost_round": int(
                train_output.get("candidate_num_boost_round", XGB_NUM_BOOST_ROUND)
            ),
            "selected_num_boost_round": int(
                train_output.get("selected_num_boost_round", XGB_NUM_BOOST_ROUND)
            ),
            "early_stopping_rounds": train_output.get(
                "early_stopping_rounds",
                XGB_EARLY_STOPPING_ROUNDS,
            ),
            "validation_fraction": float(
                train_output.get("validation_fraction", TRAIN_VALIDATION_FRACTION)
            ),
        },
        "training_window": {
            "raw_statcast_start": RAW_STATCAST_START,
            "raw_statcast_end": RAW_STATCAST_END,
            "train_split_date": TRAIN_SPLIT_DATE,
            "model_df_rows": int(len(model_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "validation_rows": validation_rows,
            "train_game_date_range": _date_range(train_df),
            "validation_game_date_range": _date_range(validation_df)
            if validation_df is not None
            else {"start": None, "end": None},
            "test_game_date_range": _date_range(test_df),
        },
        "model_selection": {
            "method": "time_ordered_validation_early_stopping",
            "best_validation_mae": train_output.get("best_validation_mae"),
        },
        "evaluation_metrics": _evaluation_metrics(
            train_output,
            train_df=train_df,
            test_df=test_df,
            historical_lines_df=historical_lines_df,
        ),
        "uncertainty_model": uncertainty_model,
        "historical_lines_artifact": {
            "selection_rule": "latest_pregame_snapshot_per_game_player_book_side",
            "source_directory": str(RAW_HISTORICAL_LINES_DIR),
            "rows": int(len(historical_lines_df)) if historical_lines_df is not None else 0,
            "limitations": (
                "v1 stores one selected line per game_date x player x sportsbook x side. "
                "It does not persist full snapshot history or intraday replay data."
            ),
        },
    }


def train_pitcher_bb_model(
    pitcher_games: pd.DataFrame,
    historical_lines_df: pd.DataFrame | None = None,
):
    starter_like_pitcher_games = filter_starter_like_appearances(pitcher_games)
    model_df = build_model_df(starter_like_pitcher_games)
    train_df, test_df = time_split(model_df)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Starter-like filtering produced an empty train/test split. "
            f"Expected non-empty rows on both sides of {TRAIN_SPLIT_DATE}."
        )

    train_output = train_model(train_df, test_df)
    metadata = build_training_metadata(
        model_df=model_df,
        train_df=train_df,
        test_df=test_df,
        train_output=train_output,
        historical_lines_df=historical_lines_df,
    )
    return train_output["model"], model_df, metadata


def validate_saved_artifacts(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing saved artifact(s): {missing}")

    pitcher_games = pd.read_csv(paths["pitcher_games"])
    model_df = pd.read_csv(paths["model_df"])
    historical_lines = pd.read_csv(paths["historical_lines"])

    if pitcher_games.empty:
        raise ValueError("Saved pitcher_games artifact is empty.")

    if model_df.empty:
        raise ValueError("Saved model_df artifact is empty.")

    if list(historical_lines.columns) != list(empty_historical_lines_df().columns):
        raise ValueError("Saved historical_lines artifact does not match expected schema.")


def save_artifacts_to_dir(
    output_dir: Path,
    pitcher_games: pd.DataFrame,
    model_df: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
    model,
    metadata: dict,
    evaluation_summary: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    pitcher_games.to_csv(paths["pitcher_games"], index=False)
    model_df.to_csv(paths["model_df"], index=False)
    historical_lines_df.to_csv(paths["historical_lines"], index=False)
    model.save_model(str(paths["model"]))
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["evaluation_summary"].write_text(
        json.dumps(evaluation_summary, indent=2),
        encoding="utf-8",
    )
    return paths


def build_training_artifacts() -> dict[str, Path]:
    ensure_artifact_dirs()
    reset_dir(STAGING_DIR)

    pitcher_games = build_historical_pitcher_games()
    historical_lines_df = build_native_historical_lines()
    model, model_df, metadata = train_pitcher_bb_model(
        pitcher_games,
        historical_lines_df=historical_lines_df,
    )
    evaluation_summary = build_evaluation_summary(
        metadata,
        workflow_name="mlb_pitcher_walks",
        artifact_family="pitcher_bb",
    )

    staging_paths = save_artifacts_to_dir(
        output_dir=STAGING_DIR,
        pitcher_games=pitcher_games,
        model_df=model_df,
        historical_lines_df=historical_lines_df,
        model=model,
        metadata=metadata,
        evaluation_summary=evaluation_summary,
    )
    validate_saved_artifacts(staging_paths)

    promote_latest_to_previous()
    promote_staging_to_latest()
    return artifact_paths(LATEST_DIR)


if __name__ == "__main__":
    paths = build_training_artifacts()
    print("Saved pitcher walks training artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
