# MLB/src/jobs/build_training_artifacts.py

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from common.contracts import validate_pitcher_games_contract, require_columns
from odds.backtest import run_historical_workflow_backtest, summarize_backtest_for_metadata
from odds.historical_lines import build_historical_lines_artifact_df, empty_historical_lines_df
from pitcher_k.evaluate import (
    build_error_bucket_summary,
    build_prediction_results,
    evaluate_predictions,
    fit_interval_calibration,
    summarize_interval_coverage,
)
from pitcher_k.config import (
    BASE_FEATURES,
    RAW_STATCAST_START,
    RAW_STATCAST_END,
    TARGET_COL,
    TRAIN_SPLIT_DATE,
    XGB_PARAMS,
)
from pitcher_k.data_loader import load_statcast_data
from pitcher_k.preprocessing import add_outcome_flags
from pitcher_k.feature_engineering import (
    build_pitcher_game_table,
    add_pitcher_team_info,
    add_opponent_k_features,
    add_rolling_pitcher_features,
    add_rate_features,
    filter_starter_like_appearances,
)
from pitcher_k.feature_model import build_model_df
from pitcher_k.train import time_split, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

STAGING_DIR = ARTIFACTS_DIR / "_staging"
LATEST_DIR = ARTIFACTS_DIR / "latest"
PREVIOUS_DIR = ARTIFACTS_DIR / "previous"

MODEL_FILENAME = "model.ubj"
PITCHER_GAMES_FILENAME = "pitcher_games.csv"
MODEL_DF_FILENAME = "model_df.csv"
HISTORICAL_LINES_FILENAME = "historical_lines.csv"
METADATA_FILENAME = "metadata.json"
RAW_HISTORICAL_LINES_DIR = DATA_DIR / "raw" / "historical_lines"


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
    }


def artifact_set_exists(base_dir: Path) -> bool:
    paths = artifact_paths(base_dir)
    return all(path.exists() for path in paths.values())


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
    sc = add_outcome_flags(sc)

    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_k_features(pitcher_games, sc)
    pitcher_games = add_rolling_pitcher_features(pitcher_games)
    pitcher_games = add_rate_features(pitcher_games)

    validate_pitcher_games_contract(pitcher_games)
    require_columns(
        pitcher_games,
        BASE_FEATURES,
        "pitcher_games",
    )

    return pitcher_games


def build_native_historical_lines() -> pd.DataFrame:
    return build_historical_lines_artifact_df(RAW_HISTORICAL_LINES_DIR)


def train_pitcher_k_model(
    pitcher_games: pd.DataFrame,
    historical_lines_df: pd.DataFrame | None = None,
):
    starter_like_pitcher_games = filter_starter_like_appearances(pitcher_games)
    model_df = build_model_df(starter_like_pitcher_games)
    train_df, test_df = time_split(model_df)
    train_output = train_model(train_df, test_df)
    metadata = build_training_metadata(
        model_df=model_df,
        train_df=train_df,
        test_df=test_df,
        train_output=train_output,
        historical_lines_df=historical_lines_df,
    )
    return train_output["model"], model_df, metadata


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
    projections["predicted_strikeouts"] = pd.Series(y_pred_test, index=test_df.index).values
    projections["actual_strikeouts"] = pd.Series(y_test, index=test_df.index).values

    backtest_result = run_historical_workflow_backtest(
        projections,
        historical_lines_df,
        actual_column="actual_strikeouts",
    )
    backtest_summary = summarize_backtest_for_metadata(backtest_result)
    backtest_summary["reproducible_path"] = "odds.backtest.run_historical_workflow_backtest"
    return backtest_summary


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

    test_results = build_prediction_results(
        test_df,
        y_test,
        y_pred_test,
    )
    interval_config = fit_interval_calibration(
        train_df,
        y_train,
        y_pred_train,
    )

    return {
        "regression": evaluate_predictions(y_test, y_pred_test),
        "bucketed_error": {
            "bucket_by": "predicted_strikeouts",
            "buckets": build_error_bucket_summary(test_results),
        },
        "uncertainty": summarize_interval_coverage(
            test_df,
            y_test,
            y_pred_test,
            interval_config,
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
        train_output["X_train"],
        train_output["y_train"],
        train_output["model"].predict(train_output["dtrain"]),
    )
    return {
        "target": TARGET_COL,
        "features": BASE_FEATURES,
        "model_params": {
            "xgb_params": XGB_PARAMS,
            "num_boost_round": 200,
        },
        "training_window": {
            "raw_statcast_start": RAW_STATCAST_START,
            "raw_statcast_end": RAW_STATCAST_END,
            "train_split_date": TRAIN_SPLIT_DATE,
            "model_df_rows": int(len(model_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_game_date_range": _date_range(train_df),
            "test_game_date_range": _date_range(test_df),
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


def save_artifacts_to_dir(
    output_dir: Path,
    pitcher_games: pd.DataFrame,
    model_df: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
    model,
    metadata: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)

    pitcher_games.to_csv(paths["pitcher_games"], index=False)
    model_df.to_csv(paths["model_df"], index=False)
    historical_lines_df.to_csv(paths["historical_lines"], index=False)
    model.save_model(str(paths["model"]))
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return paths


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


def build_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    ensure_artifact_dirs()
    reset_dir(STAGING_DIR)

    pitcher_games = build_historical_pitcher_games()
    historical_lines_df = build_native_historical_lines()
    model, model_df, metadata = train_pitcher_k_model(
        pitcher_games,
        historical_lines_df=historical_lines_df,
    )

    staging_paths = save_artifacts_to_dir(
        output_dir=STAGING_DIR,
        pitcher_games=pitcher_games,
        model_df=model_df,
        historical_lines_df=historical_lines_df,
        model=model,
        metadata=metadata,
    )
    validate_saved_artifacts(staging_paths)

    promote_latest_to_previous()
    promote_staging_to_latest()

    latest_paths = artifact_paths(LATEST_DIR)

    return (
        pitcher_games,
        model_df,
        latest_paths["pitcher_games"],
        latest_paths["model"],
    )


if __name__ == "__main__":
    pitcher_games, model_df, pitcher_games_path, model_path = build_training_artifacts()

    print("Saved training artifacts:")
    print(f"- pitcher_games: {pitcher_games_path}")
    print(f"- model: {model_path}")
    print(f"- pitcher_games shape: {pitcher_games.shape}")
    print(f"- model_df shape: {model_df.shape}")
