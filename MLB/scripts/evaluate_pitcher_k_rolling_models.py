from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "actions_audit" / "pitcher_k_rolling_eval"
TMP_AUDIT_DIR = ROOT / "tmp" / "actions_audit"
PITCHER_GAMES_CACHE = TMP_AUDIT_DIR / "pitcher_k_pitcher_games.pkl"
STATCAST_CACHE = TMP_AUDIT_DIR / "statcast_raw.pkl"
LATEST_PITCHER_GAMES = ROOT / "data" / "artifacts" / "latest" / "pitcher_games.csv"
PREVIOUS_PITCHER_GAMES = ROOT / "data" / "artifacts" / "previous" / "pitcher_games.csv"
LATEST_HISTORICAL_LINES = ROOT / "data" / "artifacts" / "latest" / "historical_lines.csv"
PREVIOUS_HISTORICAL_LINES = ROOT / "data" / "artifacts" / "previous" / "historical_lines.csv"
RAW_HISTORICAL_LINES_DIR = ROOT / "data" / "raw" / "historical_lines"

RESULTS_CSV = OUT_DIR / "pitcher_k_rolling_model_results.csv"
SUMMARY_JSON = OUT_DIR / "pitcher_k_rolling_model_summary.json"
REGRESSION_PNG = OUT_DIR / "pitcher_k_rolling_model_regression.png"
CALIBRATION_PNG = OUT_DIR / "pitcher_k_rolling_model_calibration.png"
BACKTEST_PNG = OUT_DIR / "pitcher_k_rolling_model_backtest.png"


@dataclass
class ModelEvalResult:
    model_name: str
    train_rows: int
    validation_rows: int
    test_rows: int
    train_mae: float
    test_mae: float
    train_rmse: float
    test_rmse: float
    calibration_nominal_coverage: float
    calibration_empirical_coverage: float
    calibration_abs_gap: float
    calibration_mean_interval_width: float
    calibration_mean_half_width: float
    backtest_available: bool
    backtest_reason: str | None
    backtest_picks: int | None
    backtest_wins: int | None
    backtest_losses: int | None
    backtest_profit_units: float | None
    backtest_roi_per_pick: float | None
    tuning_value: float | None = None
    tuning_label: str | None = None

    def to_record(self, *, window_start: str, window_end: str) -> dict:
        return {
            "window_start": window_start,
            "window_end": window_end,
            "model_name": self.model_name,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_mae": self.train_mae,
            "test_mae": self.test_mae,
            "train_rmse": self.train_rmse,
            "test_rmse": self.test_rmse,
            "generalization_gap_mae": self.test_mae - self.train_mae,
            "calibration_nominal_coverage": self.calibration_nominal_coverage,
            "calibration_empirical_coverage": self.calibration_empirical_coverage,
            "calibration_abs_gap": self.calibration_abs_gap,
            "calibration_mean_interval_width": self.calibration_mean_interval_width,
            "calibration_mean_half_width": self.calibration_mean_half_width,
            "backtest_available": self.backtest_available,
            "backtest_reason": self.backtest_reason,
            "backtest_picks": self.backtest_picks,
            "backtest_wins": self.backtest_wins,
            "backtest_losses": self.backtest_losses,
            "backtest_profit_units": self.backtest_profit_units,
            "backtest_roi_per_pick": self.backtest_roi_per_pick,
            "tuning_label": self.tuning_label,
            "tuning_value": self.tuning_value,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rolling pitcher_k performance, calibration, and backtest quality for XGBoost vs linear baselines."
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Rolling test window start date in YYYY-MM-DD. Defaults to two months before TRAIN_SPLIT_DATE, aligned to month start.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Rolling evaluation end date in YYYY-MM-DD. Defaults to the end of the latest month in the model data.",
    )
    parser.add_argument(
        "--window-freq",
        default="MS",
        help="Pandas date_range frequency for rolling test windows. Default: MS (month start).",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=2000,
        help="Skip rolling windows with fewer training rows than this threshold.",
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=100,
        help="Skip rolling windows with fewer test rows than this threshold.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip historical lines workflow backtests even if local line artifacts are available.",
    )
    return parser.parse_args()


def _project_src() -> None:
    import sys

    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def load_or_build_pitcher_games() -> tuple[pd.DataFrame, dict]:
    _project_src()
    from pitcher_k import config, data_loader, feature_engineering, preprocessing

    metadata: dict[str, object] = {
        "source": None,
        "refreshed_missing_features": [],
    }

    if PITCHER_GAMES_CACHE.exists():
        print(f"[stage] loading cached pitcher_games from {PITCHER_GAMES_CACHE}")
        pitcher_games = pd.read_pickle(PITCHER_GAMES_CACHE)
        metadata["source"] = str(PITCHER_GAMES_CACHE)
    elif LATEST_PITCHER_GAMES.exists():
        print(f"[stage] loading artifact pitcher_games from {LATEST_PITCHER_GAMES}")
        pitcher_games = pd.read_csv(LATEST_PITCHER_GAMES, parse_dates=["game_date"])
        metadata["source"] = str(LATEST_PITCHER_GAMES)
    elif PREVIOUS_PITCHER_GAMES.exists():
        print(f"[stage] loading artifact pitcher_games from {PREVIOUS_PITCHER_GAMES}")
        pitcher_games = pd.read_csv(PREVIOUS_PITCHER_GAMES, parse_dates=["game_date"])
        metadata["source"] = str(PREVIOUS_PITCHER_GAMES)
    else:
        print("[stage] building pitcher_games from raw Statcast data")
        if STATCAST_CACHE.exists():
            statcast_df = pd.read_pickle(STATCAST_CACHE)
        else:
            statcast_df = data_loader.load_statcast_data(
                config.RAW_STATCAST_START,
                config.RAW_STATCAST_END,
            )
        sc_with_flags = preprocessing.add_outcome_flags(statcast_df)
        pitcher_games = feature_engineering.build_pitcher_game_table(sc_with_flags)
        pitcher_games = feature_engineering.add_pitcher_team_info(pitcher_games, sc_with_flags)
        pitcher_games = feature_engineering.add_opponent_k_features(pitcher_games, sc_with_flags)
        pitcher_games = feature_engineering.add_rolling_pitcher_features(pitcher_games)
        pitcher_games = feature_engineering.add_rate_features(pitcher_games)
        metadata["source"] = "raw_statcast"

    required_refresh_cols = [
        "pitches_trend_last3_vs_last10",
        "pitches_per_batter_last10",
    ]
    missing = [column for column in required_refresh_cols if column not in pitcher_games.columns]
    if missing:
        print(f"[stage] refreshing derived rate features for missing columns: {missing}")
        pitcher_games = feature_engineering.add_rate_features(pitcher_games)
        metadata["refreshed_missing_features"] = missing

    pitcher_games = feature_engineering.filter_starter_like_appearances(pitcher_games)
    metadata["rows"] = int(len(pitcher_games))
    return pitcher_games, metadata


def load_historical_lines(skip_backtest: bool) -> tuple[pd.DataFrame, dict]:
    _project_src()
    from odds.historical_lines import build_historical_lines_artifact_df, empty_historical_lines_df

    if skip_backtest:
        return empty_historical_lines_df(), {"source": None, "available": False, "reason": "skipped_by_flag"}

    if LATEST_HISTORICAL_LINES.exists():
        df = pd.read_csv(LATEST_HISTORICAL_LINES)
        return df, {"source": str(LATEST_HISTORICAL_LINES), "available": not df.empty}

    if PREVIOUS_HISTORICAL_LINES.exists():
        df = pd.read_csv(PREVIOUS_HISTORICAL_LINES)
        return df, {"source": str(PREVIOUS_HISTORICAL_LINES), "available": not df.empty}

    if RAW_HISTORICAL_LINES_DIR.exists():
        df = build_historical_lines_artifact_df(RAW_HISTORICAL_LINES_DIR)
        if not df.empty:
            return df, {"source": str(RAW_HISTORICAL_LINES_DIR), "available": True}

    return empty_historical_lines_df(), {
        "source": None,
        "available": False,
        "reason": "no_local_historical_lines_artifact_found",
    }


def build_model_history(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    _project_src()
    from pitcher_k.feature_model import build_model_df

    print("[stage] building model_df from starter-like pitcher_games")
    model_df = build_model_df(pitcher_games)
    model_df = model_df.sort_values(["game_date", "game_pk", "pitcher"], kind="stable").reset_index(drop=True)
    return model_df


def default_start_date(split_date: str) -> str:
    split_ts = pd.Timestamp(split_date)
    return (split_ts - pd.DateOffset(months=2)).replace(day=1).strftime("%Y-%m-%d")


def default_end_date(model_df: pd.DataFrame) -> str:
    max_date = pd.to_datetime(model_df["game_date"]).max()
    next_month = (max_date + pd.offsets.MonthBegin(1)).normalize()
    if next_month <= max_date:
        next_month = (max_date + pd.offsets.MonthBegin(2)).normalize()
    return next_month.strftime("%Y-%m-%d")


def build_windows(
    model_df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    window_freq: str,
    min_train_rows: int,
    min_test_rows: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    starts = list(pd.date_range(start=start_date, end=end_date, freq=window_freq))
    if len(starts) < 2:
        raise ValueError("Rolling evaluation produced fewer than two window boundaries.")

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
    model_dates = pd.to_datetime(model_df["game_date"])
    for idx, window_start in enumerate(starts[:-1]):
        window_end = starts[idx + 1]
        train_df = model_df[model_dates < window_start].copy()
        test_df = model_df[(model_dates >= window_start) & (model_dates < window_end)].copy()
        if len(train_df) < min_train_rows or len(test_df) < min_test_rows:
            continue
        windows.append((window_start, window_end, train_df, test_df))
    return windows


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _ridge_solver(X_design: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X_design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X_design.T @ X_design + alpha * penalty, X_design.T @ y)


def _design_matrix(df: pd.DataFrame, features: list[str], mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X = df[features].to_numpy(dtype=float)
    X_std = (X - mu) / sigma
    return np.column_stack([np.ones(len(X_std)), X_std])


def _backtest_summary(
    projections: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
    *,
    market_key: str,
) -> dict:
    _project_src()
    from odds.backtest import run_historical_workflow_backtest

    if historical_lines_df.empty:
        return {
            "available": False,
            "reason": "historical_lines_unavailable",
            "picks": None,
            "wins": None,
            "losses": None,
            "profit_units": None,
            "roi_per_pick": None,
        }

    backtest = run_historical_workflow_backtest(
        projections,
        historical_lines_df,
        prediction_column="predicted_strikeouts",
        actual_column="actual_strikeouts",
        sport="MLB",
        market_key=market_key,
    )
    if not backtest.get("available"):
        return {
            "available": False,
            "reason": backtest.get("reason"),
            "picks": None,
            "wins": None,
            "losses": None,
            "profit_units": None,
            "roi_per_pick": None,
        }

    overall = backtest.get("overall", [])
    if not overall:
        return {
            "available": True,
            "reason": None,
            "picks": 0,
            "wins": 0,
            "losses": 0,
            "profit_units": 0.0,
            "roi_per_pick": 0.0,
        }

    row = overall[0]
    return {
        "available": True,
        "reason": None,
        "picks": int(row.get("picks", 0) or 0),
        "wins": int(row.get("wins", 0) or 0),
        "losses": int(row.get("losses", 0) or 0),
        "profit_units": float(row.get("profit_units", 0.0) or 0.0),
        "roi_per_pick": float(row.get("roi_per_pick", 0.0) or 0.0),
    }


def _projection_frame(test_df: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    projections = test_df[["game_date", "player_name"]].copy()
    if "pitcher" in test_df.columns:
        projections["pitcher"] = test_df["pitcher"].values
    projections["predicted_strikeouts"] = np.asarray(y_pred, dtype=float)
    projections["actual_strikeouts"] = np.asarray(y_test, dtype=float)
    return projections


def evaluate_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
) -> ModelEvalResult:
    _project_src()
    from pitcher_k import config, train
    from pitcher_k.evaluate import fit_interval_calibration, summarize_interval_coverage

    features = list(config.BASE_FEATURES)
    subtrain_df, validation_df = train.validation_time_split(
        train_df,
        validation_fraction=config.TRAIN_VALIDATION_FRACTION,
    )
    dsub, dvalidation, _, _, y_sub, y_val = train.make_dmats(
        subtrain_df,
        validation_df,
        features=features,
        target=config.TARGET_COL,
    )
    candidate = xgb.train(
        params=config.XGB_PARAMS,
        dtrain=dsub,
        num_boost_round=config.XGB_NUM_BOOST_ROUND,
        evals=[(dsub, "train"), (dvalidation, "validation")],
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    selected_rounds = int(getattr(candidate, "best_iteration", config.XGB_NUM_BOOST_ROUND - 1)) + 1
    validation_pred = candidate.predict(dvalidation, iteration_range=(0, selected_rounds))
    interval_config = fit_interval_calibration(validation_df, y_val, validation_pred)

    dtrain, dtest, _, _, y_train, y_test = train.make_dmats(
        train_df,
        test_df,
        features=features,
        target=config.TARGET_COL,
    )
    final_model = xgb.train(
        params=config.XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=selected_rounds,
        verbose_eval=False,
    )
    train_pred = final_model.predict(dtrain)
    test_pred = final_model.predict(dtest)
    coverage = summarize_interval_coverage(test_df, y_test, test_pred, interval_config)
    backtest = _backtest_summary(
        _projection_frame(test_df, y_test, test_pred),
        historical_lines_df,
        market_key=config.PITCHER_K_PROP_MARKET,
    )

    return ModelEvalResult(
        model_name="xgboost",
        train_rows=int(len(train_df)),
        validation_rows=int(len(validation_df)),
        test_rows=int(len(test_df)),
        train_mae=mae(y_train, train_pred),
        test_mae=mae(y_test, test_pred),
        train_rmse=rmse(y_train, train_pred),
        test_rmse=rmse(y_test, test_pred),
        calibration_nominal_coverage=float(coverage["nominal_coverage"]),
        calibration_empirical_coverage=float(coverage["empirical_coverage"]),
        calibration_abs_gap=abs(float(coverage["empirical_coverage"]) - float(coverage["nominal_coverage"])),
        calibration_mean_interval_width=float(coverage["mean_interval_width"]),
        calibration_mean_half_width=float(coverage["mean_half_width"]),
        backtest_available=bool(backtest["available"]),
        backtest_reason=backtest["reason"],
        backtest_picks=backtest["picks"],
        backtest_wins=backtest["wins"],
        backtest_losses=backtest["losses"],
        backtest_profit_units=backtest["profit_units"],
        backtest_roi_per_pick=backtest["roi_per_pick"],
        tuning_value=float(selected_rounds),
        tuning_label="selected_num_boost_round",
    )


def evaluate_linear_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    historical_lines_df: pd.DataFrame,
    *,
    model_name: str,
    alpha_candidates: tuple[float, ...] | None,
) -> ModelEvalResult:
    _project_src()
    from pitcher_k import config, train
    from pitcher_k.evaluate import fit_interval_calibration, summarize_interval_coverage

    features = list(config.BASE_FEATURES)
    subtrain_df, validation_df = train.validation_time_split(
        train_df,
        validation_fraction=config.TRAIN_VALIDATION_FRACTION,
    )

    X_sub = subtrain_df[features].to_numpy(dtype=float)
    y_sub = subtrain_df[config.TARGET_COL].to_numpy(dtype=float)
    X_val = validation_df[features].to_numpy(dtype=float)
    y_val = validation_df[config.TARGET_COL].to_numpy(dtype=float)
    mu = X_sub.mean(axis=0)
    sigma = X_sub.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_sub_design = _design_matrix(subtrain_df, features, mu, sigma)
    X_val_design = _design_matrix(validation_df, features, mu, sigma)

    if alpha_candidates is None:
        selected_alpha = 0.0
        coef, *_ = np.linalg.lstsq(X_sub_design, y_sub, rcond=None)
        validation_pred = X_val_design @ coef
    else:
        best_alpha = None
        best_val_mae = None
        for alpha in alpha_candidates:
            coef = _ridge_solver(X_sub_design, y_sub, alpha)
            validation_pred = X_val_design @ coef
            val_mae = mae(y_val, validation_pred)
            if best_val_mae is None or val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha
        selected_alpha = float(best_alpha)
        coef = _ridge_solver(X_sub_design, y_sub, selected_alpha)
        validation_pred = X_val_design @ coef

    interval_config = fit_interval_calibration(validation_df, y_val, validation_pred)

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[config.TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[config.TARGET_COL].to_numpy(dtype=float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train_design = _design_matrix(train_df, features, mu, sigma)
    X_test_design = _design_matrix(test_df, features, mu, sigma)
    if alpha_candidates is None:
        full_coef, *_ = np.linalg.lstsq(X_train_design, y_train, rcond=None)
    else:
        full_coef = _ridge_solver(X_train_design, y_train, selected_alpha)
    train_pred = X_train_design @ full_coef
    test_pred = X_test_design @ full_coef
    coverage = summarize_interval_coverage(test_df, y_test, test_pred, interval_config)
    backtest = _backtest_summary(
        _projection_frame(test_df, y_test, test_pred),
        historical_lines_df,
        market_key=config.PITCHER_K_PROP_MARKET,
    )

    return ModelEvalResult(
        model_name=model_name,
        train_rows=int(len(train_df)),
        validation_rows=int(len(validation_df)),
        test_rows=int(len(test_df)),
        train_mae=mae(y_train, train_pred),
        test_mae=mae(y_test, test_pred),
        train_rmse=rmse(y_train, train_pred),
        test_rmse=rmse(y_test, test_pred),
        calibration_nominal_coverage=float(coverage["nominal_coverage"]),
        calibration_empirical_coverage=float(coverage["empirical_coverage"]),
        calibration_abs_gap=abs(float(coverage["empirical_coverage"]) - float(coverage["nominal_coverage"])),
        calibration_mean_interval_width=float(coverage["mean_interval_width"]),
        calibration_mean_half_width=float(coverage["mean_half_width"]),
        backtest_available=bool(backtest["available"]),
        backtest_reason=backtest["reason"],
        backtest_picks=backtest["picks"],
        backtest_wins=backtest["wins"],
        backtest_losses=backtest["losses"],
        backtest_profit_units=backtest["profit_units"],
        backtest_roi_per_pick=backtest["roi_per_pick"],
        tuning_value=float(selected_alpha),
        tuning_label="alpha",
    )


def summarize_results(results_df: pd.DataFrame, *, windows_evaluated: int) -> dict:
    summary: dict[str, object] = {
        "windows_evaluated": windows_evaluated,
        "models": {},
    }
    for model_name, group in results_df.groupby("model_name"):
        numeric = group.select_dtypes(include=[np.number])
        model_summary = {
            "windows": int(len(group)),
            "mean_test_mae": float(group["test_mae"].mean()),
            "mean_test_rmse": float(group["test_rmse"].mean()),
            "mean_calibration_abs_gap": float(group["calibration_abs_gap"].mean()),
            "mean_empirical_coverage": float(group["calibration_empirical_coverage"].mean()),
            "mean_interval_width": float(group["calibration_mean_interval_width"].mean()),
        }
        if group["backtest_available"].any():
            available = group[group["backtest_available"]].copy()
            model_summary.update(
                {
                    "backtest_windows_available": int(len(available)),
                    "mean_backtest_profit_units": float(available["backtest_profit_units"].mean()),
                    "mean_backtest_roi_per_pick": float(available["backtest_roi_per_pick"].mean()),
                    "total_backtest_picks": int(available["backtest_picks"].sum()),
                }
            )
        summary["models"][model_name] = model_summary
    return summary


def plot_regression(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["window_start"] = pd.to_datetime(plot_df["window_start"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for model_name, group in plot_df.groupby("model_name"):
        axes[0].plot(group["window_start"], group["test_mae"], marker="o", label=model_name)
        axes[1].plot(group["window_start"], group["test_rmse"], marker="o", label=model_name)

    axes[0].set_title("Pitcher K Rolling Regression Performance")
    axes[0].set_ylabel("Test MAE")
    axes[1].set_ylabel("Test RMSE")
    axes[1].set_xlabel("Window start")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(REGRESSION_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["window_start"] = pd.to_datetime(plot_df["window_start"])
    nominal = float(plot_df["calibration_nominal_coverage"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for model_name, group in plot_df.groupby("model_name"):
        axes[0].plot(group["window_start"], group["calibration_empirical_coverage"], marker="o", label=model_name)
        axes[1].plot(group["window_start"], group["calibration_abs_gap"], marker="o", label=model_name)

    axes[0].axhline(nominal, color="black", linestyle="--", linewidth=1, label=f"nominal {nominal:.0%}")
    axes[0].set_title("Pitcher K Rolling Calibration Quality")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Absolute coverage gap")
    axes[1].set_xlabel("Window start")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(results_df: pd.DataFrame) -> bool:
    plot_df = results_df[results_df["backtest_available"]].copy()
    if plot_df.empty:
        return False

    plot_df["window_start"] = pd.to_datetime(plot_df["window_start"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for model_name, group in plot_df.groupby("model_name"):
        axes[0].plot(group["window_start"], group["backtest_profit_units"], marker="o", label=model_name)
        axes[1].plot(group["window_start"], group["backtest_roi_per_pick"], marker="o", label=model_name)

    axes[0].set_title("Pitcher K Rolling Workflow Backtest")
    axes[0].set_ylabel("Profit units")
    axes[1].set_ylabel("ROI per pick")
    axes[1].set_xlabel("Window start")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(BACKTEST_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _project_src()
    from pitcher_k import config

    print("[stage] loading pitcher history")
    pitcher_games, pitcher_games_meta = load_or_build_pitcher_games()
    model_df = build_model_history(pitcher_games)

    start_date = args.start_date or default_start_date(config.TRAIN_SPLIT_DATE)
    end_date = args.end_date or default_end_date(model_df)
    print(f"[stage] rolling windows from {start_date} to {end_date} using freq={args.window_freq}")

    historical_lines_df, historical_lines_meta = load_historical_lines(args.skip_backtest)
    if historical_lines_meta.get("available"):
        print(f"[stage] loaded historical lines from {historical_lines_meta.get('source')}")
    else:
        print(f"[stage] historical line backtests unavailable: {historical_lines_meta.get('reason')}")

    windows = build_windows(
        model_df,
        start_date=start_date,
        end_date=end_date,
        window_freq=args.window_freq,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
    )
    if not windows:
        raise ValueError("No rolling windows met the minimum train/test row thresholds.")
    print(f"[stage] evaluating {len(windows)} rolling windows")

    records: list[dict] = []
    for window_start, window_end, train_df, test_df in windows:
        print(f"[window] {window_start.date()} -> {window_end.date()} ({len(test_df)} test rows)")
        evaluations = [
            evaluate_xgboost(train_df, test_df, historical_lines_df),
            evaluate_linear_model(
                train_df,
                test_df,
                historical_lines_df,
                model_name="ols",
                alpha_candidates=None,
            ),
            evaluate_linear_model(
                train_df,
                test_df,
                historical_lines_df,
                model_name="ridge",
                alpha_candidates=(0.01, 0.1, 1.0, 10.0, 100.0),
            ),
        ]
        for evaluation in evaluations:
            records.append(
                evaluation.to_record(
                    window_start=window_start.strftime("%Y-%m-%d"),
                    window_end=window_end.strftime("%Y-%m-%d"),
                )
            )

    results_df = pd.DataFrame(records).sort_values(["window_start", "model_name"]).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    plot_regression(results_df)
    plot_calibration(results_df)
    backtest_plot_written = plot_backtest(results_df)

    summary = {
        "analysis_type": "pitcher_k_rolling_model_comparison",
        "target_formulation": config.TARGET_FORMULATION,
        "features": list(config.BASE_FEATURES),
        "xgb_params": config.XGB_PARAMS,
        "window_config": {
            "start_date": start_date,
            "end_date": end_date,
            "window_freq": args.window_freq,
            "min_train_rows": args.min_train_rows,
            "min_test_rows": args.min_test_rows,
        },
        "pitcher_games_source": pitcher_games_meta,
        "historical_lines_source": historical_lines_meta,
        "outputs": {
            "results_csv": str(RESULTS_CSV),
            "summary_json": str(SUMMARY_JSON),
            "regression_png": str(REGRESSION_PNG),
            "calibration_png": str(CALIBRATION_PNG),
            "backtest_png": str(BACKTEST_PNG) if backtest_plot_written else None,
        },
        "aggregate_summary": summarize_results(results_df, windows_evaluated=len(windows)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {RESULTS_CSV}")
    print(f"Saved {SUMMARY_JSON}")
    print(f"Saved {REGRESSION_PNG}")
    print(f"Saved {CALIBRATION_PNG}")
    if backtest_plot_written:
        print(f"Saved {BACKTEST_PNG}")
    else:
        print("[stage] no backtest plot written because no historical line backtests were available")


if __name__ == "__main__":
    main()
