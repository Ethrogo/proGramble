from __future__ import annotations

import shutil
from pathlib import Path

from jobs.build_training_artifacts import (
    LATEST_DIR,
    PREVIOUS_DIR,
    RAW_HISTORICAL_LINES_DIR,
    artifact_paths,
    ensure_artifact_dirs,
)
from odds.historical_lines import build_historical_lines_artifact_df


def build_historical_lines_artifact() -> tuple[Path, int]:
    """
    Build the curated native historical lines artifact from raw snapshot files.
    """
    ensure_artifact_dirs()

    historical_lines_df = build_historical_lines_artifact_df(RAW_HISTORICAL_LINES_DIR)
    latest_path = artifact_paths(LATEST_DIR)["historical_lines"]
    previous_path = artifact_paths(PREVIOUS_DIR)["historical_lines"]

    if latest_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_path, previous_path)

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    historical_lines_df.to_csv(latest_path, index=False)

    return latest_path, int(len(historical_lines_df))


if __name__ == "__main__":
    output_path, rows = build_historical_lines_artifact()
    print("Saved native historical lines artifact:")
    print(f"- path: {output_path}")
    print(f"- rows: {rows}")
