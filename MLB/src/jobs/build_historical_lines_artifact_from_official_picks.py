from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from jobs.build_training_artifacts import (
    DATA_DIR,
    LATEST_DIR,
    PREVIOUS_DIR,
    artifact_paths,
    ensure_artifact_dirs,
)
from odds.historical_lines import (
    OFFICIAL_PICKS_HISTORY_SOURCE,
    build_historical_lines_artifact_from_official_picks_history_df,
)


OFFICIAL_PICKS_HISTORY_PATH = DATA_DIR / "tracking" / "official_picks_history.csv"


def build_historical_lines_artifact_from_official_picks(
    history_path: Path | None = None,
) -> tuple[Path, int]:
    """
    Build a historical_lines artifact from official_picks_history.csv.

    This produces a narrower artifact than native raw sportsbook snapshots:
    it contains only lines that were published as official picks.
    """
    if history_path is None:
        history_path = OFFICIAL_PICKS_HISTORY_PATH
    ensure_artifact_dirs()
    if not history_path.exists():
        raise FileNotFoundError(f"Missing official picks history: {history_path}")

    history_df = pd.read_csv(history_path, keep_default_na=False)
    historical_lines_df = build_historical_lines_artifact_from_official_picks_history_df(history_df)
    if not historical_lines_df.empty:
        historical_lines_df["source"] = OFFICIAL_PICKS_HISTORY_SOURCE

    latest_path = artifact_paths(LATEST_DIR)["historical_lines"]
    previous_path = artifact_paths(PREVIOUS_DIR)["historical_lines"]

    if latest_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_path, previous_path)

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    historical_lines_df.to_csv(latest_path, index=False)
    return latest_path, int(len(historical_lines_df))


if __name__ == "__main__":
    output_path, rows = build_historical_lines_artifact_from_official_picks()
    print("Saved official-picks-derived historical lines artifact:")
    print(f"- path: {output_path}")
    print(f"- rows: {rows}")
