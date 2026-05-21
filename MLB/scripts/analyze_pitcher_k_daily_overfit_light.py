from __future__ import annotations

import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "actions_audit"
RUNS_CACHE = OUT_DIR / "workflow_runs.json"
STATCAST_CACHE = OUT_DIR / "statcast_raw.pkl"
PITCHER_GAMES_CACHE = OUT_DIR / "pitcher_k_pitcher_games.pkl"
MODEL_DF_CACHE = OUT_DIR / "pitcher_k_model_df.pkl"
LATEST_MODEL_DF = ROOT / "data" / "artifacts" / "latest" / "model_df.csv"
PREVIOUS_MODEL_DF = ROOT / "data" / "artifacts" / "previous" / "model_df.csv"
LATEST_PITCHER_GAMES = ROOT / "data" / "artifacts" / "latest" / "pitcher_games.csv"
PREVIOUS_PITCHER_GAMES = ROOT / "data" / "artifacts" / "previous" / "pitcher_games.csv"

SUMMARY_JSON = OUT_DIR / "pitcher_k_light_overfit_summary.json"
CURRENT_BOOSTING_PNG = OUT_DIR / "pitcher_k_light_current_boosting_curve.png"
RUNS_CALENDAR_PNG = OUT_DIR / "pitcher_k_light_daily_runs.png"
RUNS_CSV = OUT_DIR / "pitcher_k_light_daily_runs.csv"

TIMEZONE = ZoneInfo("America/New_York")


def load_workflow_runs() -> list[dict]:
    print(f"[stage] loading workflow runs from {RUNS_CACHE}")
    if not RUNS_CACHE.exists():
        raise FileNotFoundError(
            f"Missing local workflow cache: {RUNS_CACHE}. "
            "This lightweight script only uses local tmp/actions_audit inputs."
        )
    payload = json.loads(RUNS_CACHE.read_text(encoding="utf-8"))
    return payload.get("workflow_runs", [])


def select_one_run_per_day(runs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        created = pd.Timestamp(run["created_at"], tz="UTC").tz_convert(TIMEZONE)
        rows.append(
            {
                "run_id": int(run["id"]),
                "run_number": int(run["run_number"]),
                "event": run["event"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "created_at_et": created,
                "run_date_et": created.date().isoformat(),
                "head_sha": run["head_sha"],
                "html_url": run["html_url"],
            }
        )

    runs_df = pd.DataFrame(rows).sort_values("created_at_et")
    if runs_df.empty:
        raise ValueError("No workflow runs found in local workflow cache.")

    run_counts = runs_df.groupby("run_date_et").size().rename("runs_that_day").reset_index()
    successful = runs_df[runs_df["conclusion"] == "success"].copy()
    if successful.empty:
        raise ValueError("No successful workflow runs found in local workflow cache.")

    selected = (
        successful.sort_values("created_at_et")
        .groupby("run_date_et", as_index=False)
        .tail(1)
        .sort_values("created_at_et")
        .merge(run_counts, on="run_date_et", how="left")
        .reset_index(drop=True)
    )
    selected["multiple_runs_that_day"] = selected["runs_that_day"] > 1
    print(f"[stage] selected {len(selected)} successful ET dates from {len(runs_df)} workflow runs")
    return selected


def load_or_build_model_df() -> tuple[pd.DataFrame, dict]:
    import gc
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from pitcher_k import config, feature_engineering, feature_model, preprocessing

    metadata: dict[str, object] = {
        "used_pitcher_games_cache": PITCHER_GAMES_CACHE.exists(),
        "used_model_df_cache": MODEL_DF_CACHE.exists(),
        "used_artifact_model_df": False,
        "used_artifact_pitcher_games": False,
    }

    if MODEL_DF_CACHE.exists():
        print(f"[stage] loading cached model_df from {MODEL_DF_CACHE}")
        model_df = pd.read_pickle(MODEL_DF_CACHE)
        missing_model_features = [feature for feature in config.BASE_FEATURES if feature not in model_df.columns]
        if not missing_model_features:
            metadata["model_df_rows"] = int(len(model_df))
            return model_df, metadata
        print(f"[stage] cached model_df is missing features {missing_model_features}; rebuilding cache")

    for artifact_model_df in (LATEST_MODEL_DF, PREVIOUS_MODEL_DF):
        if artifact_model_df.exists():
            print(f"[stage] loading artifact model_df from {artifact_model_df}")
            model_df = pd.read_csv(artifact_model_df, parse_dates=["game_date"])
            missing_model_features = [feature for feature in config.BASE_FEATURES if feature not in model_df.columns]
            if missing_model_features:
                print(
                    f"[stage] artifact model_df is missing features {missing_model_features}; "
                    "falling back to pitcher_games/raw cache"
                )
                continue
            MODEL_DF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            model_df.to_pickle(MODEL_DF_CACHE)
            metadata["used_artifact_model_df"] = True
            metadata["artifact_model_df_source"] = str(artifact_model_df)
            metadata["model_df_rows"] = int(len(model_df))
            return model_df, metadata

    if PITCHER_GAMES_CACHE.exists():
        print(f"[stage] loading cached pitcher_games from {PITCHER_GAMES_CACHE}")
        pitcher_games = pd.read_pickle(PITCHER_GAMES_CACHE)
        missing_pitcher_game_features = [
            feature for feature in config.BASE_FEATURES if feature not in pitcher_games.columns
        ]
        if missing_pitcher_game_features:
            print(
                f"[stage] cached pitcher_games is missing features {missing_pitcher_game_features}; "
                "refreshing derived features"
            )
            pitcher_games = feature_engineering.add_rate_features(pitcher_games)
    else:
        pitcher_games = None
        for artifact_pitcher_games in (LATEST_PITCHER_GAMES, PREVIOUS_PITCHER_GAMES):
            if artifact_pitcher_games.exists():
                print(f"[stage] loading artifact pitcher_games from {artifact_pitcher_games}")
                pitcher_games = pd.read_csv(artifact_pitcher_games, parse_dates=["game_date"])
                metadata["used_artifact_pitcher_games"] = True
                metadata["artifact_pitcher_games_source"] = str(artifact_pitcher_games)
                break

        if pitcher_games is None:
            if not STATCAST_CACHE.exists():
                raise FileNotFoundError(
                    f"Missing local Statcast cache: {STATCAST_CACHE}. "
                    "Build the local tmp/actions_audit cache before running this script."
                )
            print(f"[stage] loading raw Statcast cache from {STATCAST_CACHE}")
            statcast_df = pd.read_pickle(STATCAST_CACHE)
            metadata["statcast_rows"] = int(len(statcast_df))

            print("[stage] adding outcome flags")
            sc_with_flags = preprocessing.add_outcome_flags(statcast_df)
            del statcast_df
            gc.collect()

            print("[stage] building pitcher game table")
            pitcher_games = feature_engineering.build_pitcher_game_table(sc_with_flags)
            print("[stage] adding pitcher team info")
            pitcher_games = feature_engineering.add_pitcher_team_info(pitcher_games, sc_with_flags)
            print("[stage] adding opponent k features")
            pitcher_games = feature_engineering.add_opponent_k_features(pitcher_games, sc_with_flags)
            del sc_with_flags
            gc.collect()

            print("[stage] adding rolling pitcher features")
            pitcher_games = feature_engineering.add_rolling_pitcher_features(pitcher_games)
            print("[stage] adding rate features")
            pitcher_games = feature_engineering.add_rate_features(pitcher_games)
            print("[stage] filtering starter-like appearances")
            pitcher_games = feature_engineering.filter_starter_like_appearances(pitcher_games)

        PITCHER_GAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stage] saving pitcher_games cache to {PITCHER_GAMES_CACHE}")
        pitcher_games.to_pickle(PITCHER_GAMES_CACHE)

    metadata["pitcher_games_rows"] = int(len(pitcher_games))
    print("[stage] building model_df")
    model_df = feature_model.build_model_df(pitcher_games)
    MODEL_DF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[stage] saving model_df cache to {MODEL_DF_CACHE}")
    model_df.to_pickle(MODEL_DF_CACHE)
    metadata["model_df_rows"] = int(len(model_df))
    return model_df, metadata


def compute_current_overfit_metrics(model_df: pd.DataFrame) -> dict:
    import sys

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(ROOT / "src"))
    from pitcher_k import config, train

    print("[stage] splitting train and test")
    train_df, test_df = train.time_split(model_df, config.TRAIN_SPLIT_DATE)
    print("[stage] building DMatrix objects")
    dtrain, dtest, _, _, y_train, y_test = train.make_dmats(
        train_df=train_df,
        test_df=test_df,
        features=config.BASE_FEATURES,
        target=config.TARGET_COL,
    )

    evals_result: dict[str, dict[str, list[float]]] = {}
    print("[stage] training current XGBoost model")
    model = xgb.train(
        params=config.XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=config.XGB_NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False,
        evals_result=evals_result,
    )

    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)

    train_mae = float(np.mean(np.abs(train_pred - y_train)))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

    train_curve = [float(x) for x in evals_result["train"]["mae"]]
    test_curve = [float(x) for x in evals_result["test"]["mae"]]
    best_idx = int(np.argmin(test_curve))

    return {
        "feature_count": len(config.BASE_FEATURES),
        "features": list(config.BASE_FEATURES),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "generalization_gap_mae": test_mae - train_mae,
        "generalization_gap_rmse": test_rmse - train_rmse,
        "boost_rounds": len(train_curve),
        "best_round": best_idx + 1,
        "best_test_mae": float(test_curve[best_idx]),
        "last_round_train_mae": float(train_curve[-1]),
        "last_round_test_mae": float(test_curve[-1]),
        "train_curve_mae": train_curve,
        "test_curve_mae": test_curve,
    }


def plot_current_boosting_curve(metrics: dict) -> None:
    rounds = list(range(1, len(metrics["train_curve_mae"]) + 1))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rounds, metrics["train_curve_mae"], label="Train MAE")
    ax.plot(rounds, metrics["test_curve_mae"], label="Test MAE")
    ax.axvline(
        metrics["best_round"],
        color="crimson",
        linestyle="--",
        label=f"Best round {metrics['best_round']}",
    )
    ax.set_title("Pitcher K Current Boosting Curve")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CURRENT_BOOSTING_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_daily_runs(selected_runs: pd.DataFrame) -> None:
    plot_df = selected_runs.copy()
    plot_df["run_date_et"] = pd.to_datetime(plot_df["run_date_et"])
    plot_df["y"] = plot_df["runs_that_day"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df["run_date_et"], plot_df["y"], color="steelblue", linewidth=1.5, alpha=0.7)
    ax.scatter(plot_df["run_date_et"], plot_df["y"], color="steelblue", s=40)

    multi = plot_df[plot_df["multiple_runs_that_day"]]
    if not multi.empty:
        ax.scatter(multi["run_date_et"], multi["y"], color="darkorange", s=60, zorder=5)
        for row in multi.itertuples(index=False):
            ax.text(
                row.run_date_et,
                row.y + 0.08,
                f"x{int(row.runs_that_day)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkorange",
            )

    ax.set_title("Pitcher K Daily GitHub Actions Cadence")
    ax.set_xlabel("Run date (America/New_York)")
    ax.set_ylabel("Workflow runs that day")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUNS_CALENDAR_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_summary(selected_runs: pd.DataFrame, cache_metadata: dict, current_metrics: dict) -> dict:
    first_date = selected_runs["run_date_et"].min()
    last_date = selected_runs["run_date_et"].max()
    multi_run_days = int(selected_runs["multiple_runs_that_day"].sum())

    return {
        "analysis_mode": "lightweight_current_model_plus_daily_run_calendar",
        "tradeoffs": [
            "Uses local tmp/actions_audit caches only; no network calls and no GitHub artifact downloads.",
            "Does not replay historical SHAs, so daily dates are run-calendar annotations rather than exact historical overfit metrics.",
            "Trains the current pitcher_k model once and plots the current train/test MAE boosting curve.",
            "Caches compact pitcher_k intermediates so future reruns can avoid reprocessing the 1.6 GB raw Statcast pickle.",
        ],
        "daily_run_selection_rule": "last successful GitHub Actions run per America/New_York date, with runs_that_day noting duplicates",
        "date_range_et": {
            "first_selected_run_date": first_date,
            "last_selected_run_date": last_date,
            "selected_days": int(len(selected_runs)),
            "days_with_multiple_runs": multi_run_days,
        },
        "cache_metadata": cache_metadata,
        "current_model_metrics": {
            key: value
            for key, value in current_metrics.items()
            if key not in {"train_curve_mae", "test_curve_mae", "features"}
        },
        "features": current_metrics["features"],
        "output_files": {
            "summary_json": str(SUMMARY_JSON),
            "current_boosting_png": str(CURRENT_BOOSTING_PNG),
            "daily_runs_png": str(RUNS_CALENDAR_PNG),
            "daily_runs_csv": str(RUNS_CSV),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    runs = load_workflow_runs()
    selected_runs = select_one_run_per_day(runs)
    model_df, cache_metadata = load_or_build_model_df()
    current_metrics = compute_current_overfit_metrics(model_df)

    plot_current_boosting_curve(current_metrics)
    plot_daily_runs(selected_runs)
    selected_runs.to_csv(RUNS_CSV, index=False)

    summary = build_summary(selected_runs, cache_metadata, current_metrics)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {SUMMARY_JSON}")
    print(f"Saved {CURRENT_BOOSTING_PNG}")
    print(f"Saved {RUNS_CALENDAR_PNG}")
    print(f"Saved {RUNS_CSV}")
    print(f"[done] total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
