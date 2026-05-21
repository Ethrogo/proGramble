from __future__ import annotations

import json
import importlib
import subprocess
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REPO = "Ethrogo/proGramble"
WORKFLOW_FILE = "daily-mlb-card.yml"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "actions_audit"
WORKTREE_DIR = OUT_DIR / "worktrees"
RUNS_CACHE = OUT_DIR / "workflow_runs.json"
STATCAST_CACHE = OUT_DIR / "statcast_raw.pkl"
METRICS_CACHE = OUT_DIR / "sha_metrics.json"
DAILY_CSV = OUT_DIR / "pitcher_k_daily_overfit.csv"
DAILY_JSON = OUT_DIR / "pitcher_k_daily_overfit.json"
DAILY_PNG = OUT_DIR / "pitcher_k_daily_overfit.png"
BOOSTING_PNG = OUT_DIR / "pitcher_k_latest_boosting_curve.png"
TIMEZONE = ZoneInfo("America/New_York")

IGNORED_PREFIXES = (
    ".github/",
    "MLB/data/tracking/",
    "MLB/data/outputs/",
    "MLB/data/inputs/",
    "MLB/data/raw/",
    "MLB/data/artifacts/",
    "MLB/docs/",
    "MLB/tests/",
)
IGNORED_FILES = {
    "README.md",
    "MLB/README.md",
}


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT.parent),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _github_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Codex"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def load_workflow_runs() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUNS_CACHE.exists():
        payload = json.loads(RUNS_CACHE.read_text(encoding="utf-8"))
    else:
        payload = _github_json(
            f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs?per_page=100"
        )
        RUNS_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
                "conclusion": run["conclusion"],
                "created_at_et": created,
                "run_date_et": created.date().isoformat(),
                "head_sha": run["head_sha"],
                "html_url": run["html_url"],
            }
        )

    runs_df = pd.DataFrame(rows).sort_values("created_at_et")
    if runs_df.empty:
        raise ValueError("No workflow runs were returned by the GitHub Actions API.")

    run_counts = runs_df.groupby("run_date_et").size().rename("runs_that_day").reset_index()
    successful = runs_df[runs_df["conclusion"] == "success"].copy()
    selected = (
        successful.sort_values("created_at_et")
        .groupby("run_date_et", as_index=False)
        .tail(1)
        .sort_values("created_at_et")
        .merge(run_counts, on="run_date_et", how="left")
        .reset_index(drop=True)
    )
    return selected


def commit_touches_model_behavior(sha: str) -> bool:
    changed = [
        line.strip()
        for line in _run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", sha]).splitlines()
        if line.strip()
    ]
    if not changed:
        return True

    for path in changed:
        normalized = path.replace("\\", "/")
        if normalized in IGNORED_FILES:
            continue
        if normalized.startswith(IGNORED_PREFIXES):
            continue
        return True
    return False


def attach_effective_shas(selected_runs: pd.DataFrame) -> pd.DataFrame:
    effective_sha: str | None = None
    effective_changed: list[bool] = []
    effective_shas: list[str] = []

    for row in selected_runs.itertuples(index=False):
        if effective_sha is None or commit_touches_model_behavior(row.head_sha):
            effective_sha = row.head_sha
            effective_changed.append(True)
        else:
            effective_changed.append(False)
        effective_shas.append(effective_sha)

    enriched = selected_runs.copy()
    enriched["effective_sha"] = effective_shas
    enriched["effective_sha_changed_today"] = effective_changed
    return enriched


def load_or_build_statcast_cache() -> pd.DataFrame:
    if STATCAST_CACHE.exists():
        try:
            return pd.read_pickle(STATCAST_CACHE)
        except Exception as exc:
            print(f"Discarding unreadable Statcast cache {STATCAST_CACHE}: {exc}")
            STATCAST_CACHE.unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT / "src"))
    from pitcher_k.config import RAW_STATCAST_END, RAW_STATCAST_START
    from pitcher_k.data_loader import load_statcast_data

    sc = load_statcast_data(RAW_STATCAST_START, RAW_STATCAST_END)
    temp_cache = STATCAST_CACHE.with_suffix(".tmp")
    sc.to_pickle(temp_cache)
    temp_cache.replace(STATCAST_CACHE)
    return sc


def _worktree_for_sha(sha: str) -> Path:
    path = WORKTREE_DIR / sha[:12]
    if path.exists():
        return path
    WORKTREE_DIR.mkdir(parents=True, exist_ok=True)
    _run_git(["worktree", "add", "--detach", str(path), sha])
    return path


def _purge_repo_modules() -> None:
    prefixes = ("pitcher_k", "common", "jobs")
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def compute_metrics_for_sha(sha: str, sc: pd.DataFrame) -> dict:
    existing: dict[str, dict] = {}
    if METRICS_CACHE.exists():
        existing = json.loads(METRICS_CACHE.read_text(encoding="utf-8"))
    if sha in existing:
        print(f"[cache-hit] {sha[:7]}")
        return existing[sha]

    print(f"[replay] {sha[:7]}")
    worktree = _worktree_for_sha(sha)
    src_path = worktree / "MLB" / "src"
    original_sys_path = list(sys.path)
    try:
        _purge_repo_modules()
        sys.path.insert(0, str(src_path))

        import numpy as np
        import xgboost as xgb

        config = importlib.import_module("pitcher_k.config")
        feature_engineering = importlib.import_module("pitcher_k.feature_engineering")
        feature_model = importlib.import_module("pitcher_k.feature_model")
        preprocessing = importlib.import_module("pitcher_k.preprocessing")
        train = importlib.import_module("pitcher_k.train")

        def mae(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            return float(np.mean(np.abs(y_pred - y_true)))

        def rmse(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        sc_with_flags = sc.copy()
        if hasattr(preprocessing, "add_outcome_flags"):
            sc_with_flags = preprocessing.add_outcome_flags(sc_with_flags)

        pitcher_games = feature_engineering.build_pitcher_game_table(sc_with_flags)
        pitcher_games = feature_engineering.add_pitcher_team_info(pitcher_games, sc_with_flags)
        pitcher_games = feature_engineering.add_opponent_k_features(pitcher_games, sc_with_flags)
        pitcher_games = feature_engineering.add_rolling_pitcher_features(pitcher_games)
        pitcher_games = feature_engineering.add_rate_features(pitcher_games)
        if hasattr(feature_engineering, "filter_starter_like_appearances"):
            pitcher_games = feature_engineering.filter_starter_like_appearances(pitcher_games)
        model_df = feature_model.build_model_df(pitcher_games)
        train_df, test_df = train.time_split(model_df, config.TRAIN_SPLIT_DATE)
        dtrain, dtest, _, _, y_train, y_test = train.make_dmats(
            train_df=train_df,
            test_df=test_df,
            features=config.BASE_FEATURES,
            target=config.TARGET_COL,
        )
        evals_result = {}
        model = xgb.train(
            params=config.XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=200,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=False,
            evals_result=evals_result,
        )
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)
        train_curve = evals_result["train"]["mae"]
        test_curve = evals_result["test"]["mae"]
        payload = {
            "features": list(config.BASE_FEATURES),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_mae": mae(y_train, train_pred),
            "test_mae": mae(y_test, test_pred),
            "train_rmse": rmse(y_train, train_pred),
            "test_rmse": rmse(y_test, test_pred),
            "best_round": int(np.argmin(test_curve) + 1),
            "best_test_mae": float(np.min(test_curve)),
            "last_round_train_mae": float(train_curve[-1]),
            "last_round_test_mae": float(test_curve[-1]),
            "train_curve_mae": [float(x) for x in train_curve],
            "test_curve_mae": [float(x) for x in test_curve],
        }
    finally:
        sys.path[:] = original_sys_path
        _purge_repo_modules()

    existing[sha] = payload
    METRICS_CACHE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return payload


def build_daily_metrics(selected_runs: pd.DataFrame) -> pd.DataFrame:
    unique_effective_shas = list(dict.fromkeys(selected_runs["effective_sha"].tolist()))
    print(f"Selected {len(selected_runs)} days across {len(unique_effective_shas)} unique effective snapshots.")
    sc = load_or_build_statcast_cache()
    print(f"Loaded Statcast cache with {len(sc):,} rows.")
    metrics_by_sha = {sha: compute_metrics_for_sha(sha, sc) for sha in unique_effective_shas}

    rows: list[dict] = []
    for row in selected_runs.itertuples(index=False):
        metrics = metrics_by_sha[row.effective_sha]
        rows.append(
            {
                "run_date_et": row.run_date_et,
                "created_at_et": row.created_at_et.isoformat(),
                "run_id": row.run_id,
                "run_number": row.run_number,
                "event": row.event,
                "runs_that_day": row.runs_that_day,
                "head_sha": row.head_sha,
                "effective_sha": row.effective_sha,
                "effective_sha_changed_today": bool(row.effective_sha_changed_today),
                "html_url": row.html_url,
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
                "train_mae": metrics["train_mae"],
                "test_mae": metrics["test_mae"],
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
                "generalization_gap_mae": metrics["test_mae"] - metrics["train_mae"],
                "generalization_gap_rmse": metrics["test_rmse"] - metrics["train_rmse"],
                "best_round": metrics["best_round"],
                "best_test_mae": metrics["best_test_mae"],
                "last_round_train_mae": metrics["last_round_train_mae"],
                "last_round_test_mae": metrics["last_round_test_mae"],
            }
        )

    return pd.DataFrame(rows).sort_values("run_date_et").reset_index(drop=True)


def plot_daily_overfit(daily_df: pd.DataFrame) -> None:
    plot_df = daily_df.copy()
    plot_df["run_date_et"] = pd.to_datetime(plot_df["run_date_et"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(plot_df["run_date_et"], plot_df["train_mae"], marker="o", label="Train MAE")
    axes[0].plot(plot_df["run_date_et"], plot_df["test_mae"], marker="o", label="Test MAE")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Pitcher K Daily Overfitting Reconstruction")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(plot_df["run_date_et"], plot_df["train_rmse"], marker="o", label="Train RMSE")
    axes[1].plot(plot_df["run_date_et"], plot_df["test_rmse"], marker="o", label="Test RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    multi = plot_df[plot_df["runs_that_day"] > 1]
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.06
        for row in multi.itertuples(index=False):
            value = row.test_mae if ax is axes[0] else row.test_rmse
            ax.scatter(row.run_date_et, value, color="darkorange", zorder=5)
            ax.text(
                row.run_date_et,
                value + offset,
                f"x{int(row.runs_that_day)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkorange",
            )

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xlabel("Run date (America/New_York)")
    fig.tight_layout()
    fig.savefig(DAILY_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_latest_boosting_curve(daily_df: pd.DataFrame) -> None:
    cache = json.loads(METRICS_CACHE.read_text(encoding="utf-8"))
    latest = daily_df.iloc[-1]
    metrics = cache[latest["effective_sha"]]
    rounds = list(range(1, len(metrics["train_curve_mae"]) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rounds, metrics["train_curve_mae"], label="Train MAE by round")
    ax.plot(rounds, metrics["test_curve_mae"], label="Test MAE by round")
    ax.axvline(metrics["best_round"], color="crimson", linestyle="--", label=f"Best round {metrics['best_round']}")
    ax.set_title(f"Pitcher K Boosting Curve for {latest['run_date_et']}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(BOOSTING_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = load_workflow_runs()
    selected_runs = attach_effective_shas(select_one_run_per_day(runs))
    daily_df = build_daily_metrics(selected_runs)
    plot_daily_overfit(daily_df)
    plot_latest_boosting_curve(daily_df)
    DAILY_CSV.write_text(daily_df.to_csv(index=False), encoding="utf-8")
    DAILY_JSON.write_text(daily_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Saved {DAILY_CSV}")
    print(f"Saved {DAILY_JSON}")
    print(f"Saved {DAILY_PNG}")
    print(f"Saved {BOOSTING_PNG}")


if __name__ == "__main__":
    main()
