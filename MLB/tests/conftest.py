from __future__ import annotations

import shutil
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HISTORICAL_LINES_FIXTURES_DIR = FIXTURES_DIR / "historical_lines"


def copy_historical_lines_fixtures(destination: Path) -> list[Path]:
    copied_paths: list[Path] = []
    for source_path in sorted(HISTORICAL_LINES_FIXTURES_DIR.rglob("*.csv")):
        target_path = destination / source_path.relative_to(HISTORICAL_LINES_FIXTURES_DIR)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
    return copied_paths


@pytest.fixture
def historical_lines_fixture_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "historical_lines"
    copy_historical_lines_fixtures(raw_dir)
    return raw_dir
