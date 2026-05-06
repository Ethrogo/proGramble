from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jobs.populate_historical_lines_raw import populate_historical_lines_raw


def test_populate_historical_lines_raw_copies_fixture_tree(
    tmp_path: Path,
    historical_lines_fixture_dir: Path,
):
    target_dir = tmp_path / "data" / "raw" / "historical_lines"

    copied = populate_historical_lines_raw(historical_lines_fixture_dir, target=target_dir)
    copied_relative = sorted(path.relative_to(target_dir).as_posix() for path in copied)

    assert copied_relative == ["2025-08-02/pitcher_strikeouts_sample.csv"]
    populated = pd.read_csv(target_dir / "2025-08-02" / "pitcher_strikeouts_sample.csv")
    assert set(populated["player_name"]) == {"Jacob deGrom", "Jacob Degrom", "Tarik Skubal"}


def test_populate_historical_lines_raw_rejects_missing_required_columns(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"player_name": "Jacob deGrom"}]).to_csv(source_dir / "bad.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        populate_historical_lines_raw(source_dir, target=tmp_path / "target")
