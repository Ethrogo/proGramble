from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = ROOT / "data" / "tracking"
OUT_DIR = ROOT / "tmp" / "actions_audit" / "pitcher_k_rolling_eval"

OFFICIAL_PICKS_PROFIT_REPORT = TRACKING_DIR / "official_picks_profit_report.csv"
ROLLING_RESULTS_CSV = OUT_DIR / "pitcher_k_rolling_model_results.csv"
PLOT_PNG = OUT_DIR / "pitcher_k_official_history_backtest.png"
SUMMARY_JSON = OUT_DIR / "pitcher_k_official_history_backtest_summary.json"


def load_pitcher_k_official_picks() -> pd.DataFrame:
    df = pd.read_csv(OFFICIAL_PICKS_PROFIT_REPORT, keep_default_na=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["units_result"] = pd.to_numeric(df["units_result"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    market_key = df.get("market_key", pd.Series("", index=df.index)).astype(str).str.strip()
    prop_type = df.get("prop_type", pd.Series("", index=df.index)).astype(str).str.strip()
    tracking_regime = df.get("tracking_regime", pd.Series("", index=df.index)).astype(str).str.strip()

    pitcher_k_mask = (
        (market_key == "pitcher_strikeouts")
        | (
            (market_key == "")
            & prop_type.isin(["", "pitcher_k"])
            & tracking_regime.ne("manual_backfill")
        )
    )
    realized_mask = df["game_date"].notna() & df["units_result"].notna()
    out = df[pitcher_k_mask & realized_mask].copy()
    return out.sort_values(["game_date", "pick_key"]).reset_index(drop=True)


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("game_date", as_index=False)
        .agg(
            picks=("pick_key", "size"),
            wins=("result_normalized", lambda s: int((s == "W").sum())),
            losses=("result_normalized", lambda s: int((s == "L").sum())),
            units_result=("units_result", "sum"),
        )
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    daily["cumulative_units"] = daily["units_result"].cumsum()
    daily["daily_roi"] = daily["units_result"] / daily["picks"]
    return daily


def load_rolling_window_summary() -> dict:
    if not ROLLING_RESULTS_CSV.exists():
        return {"available": False}

    results = pd.read_csv(ROLLING_RESULTS_CSV)
    if results.empty:
        return {"available": False}

    window_start = pd.to_datetime(results["window_start"], errors="coerce")
    window_end = pd.to_datetime(results["window_end"], errors="coerce")
    return {
        "available": True,
        "window_start_min": window_start.min().strftime("%Y-%m-%d"),
        "window_end_max": window_end.max().strftime("%Y-%m-%d"),
    }


def plot_backtest(daily: pd.DataFrame, overlap_note: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(daily["game_date"], daily["cumulative_units"], marker="o", color="#0f766e")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Pitcher K Official Picks Backtest")
    axes[0].set_ylabel("Cumulative profit units")
    axes[0].text(
        0.01,
        0.02,
        overlap_note,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    axes[1].bar(daily["game_date"], daily["units_result"], width=0.8, color="#1d4ed8", alpha=0.75, label="daily units")
    axes[1].plot(
        daily["game_date"],
        daily["daily_roi"],
        marker="o",
        color="#b45309",
        label="daily ROI per pick",
    )
    axes[1].set_ylabel("Daily units / ROI")
    axes[1].set_xlabel("Game date")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].legend(loc="best")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = load_pitcher_k_official_picks()
    if picks.empty:
        raise ValueError("No realized pitcher_k official picks were available for plotting.")

    daily = build_daily_summary(picks)
    rolling_summary = load_rolling_window_summary()
    official_start = picks["game_date"].min()
    official_end = picks["game_date"].max()
    if rolling_summary.get("available"):
        rolling_start = pd.to_datetime(rolling_summary["window_start_min"])
        rolling_end = pd.to_datetime(rolling_summary["window_end_max"])
        has_overlap = not (official_end < rolling_start or official_start > rolling_end)
        overlap_note = (
            f"Rolling model windows: {rolling_start:%Y-%m-%d} to {rolling_end:%Y-%m-%d}. "
            f"Official picks: {official_start:%Y-%m-%d} to {official_end:%Y-%m-%d}. "
            f"Overlap: {'yes' if has_overlap else 'no'}."
        )
    else:
        has_overlap = False
        overlap_note = (
            f"Official picks: {official_start:%Y-%m-%d} to {official_end:%Y-%m-%d}. "
            "Rolling model results CSV was unavailable."
        )

    plot_backtest(daily, overlap_note)

    summary = {
        "analysis_type": "pitcher_k_official_picks_backtest",
        "official_picks_range": {
            "start_date": official_start.strftime("%Y-%m-%d"),
            "end_date": official_end.strftime("%Y-%m-%d"),
        },
        "rolling_model_results": rolling_summary,
        "has_overlap_with_rolling_model_results": has_overlap,
        "picks": int(len(picks)),
        "wins": int((picks["result_normalized"] == "W").sum()),
        "losses": int((picks["result_normalized"] == "L").sum()),
        "profit_units": float(picks["units_result"].sum()),
        "roi_per_pick": float(picks["units_result"].sum() / len(picks)),
        "plot_path": str(PLOT_PNG),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {PLOT_PNG}")
    print(f"Saved {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
