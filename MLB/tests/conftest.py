from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HISTORICAL_LINES_FIXTURES_DIR = FIXTURES_DIR / "historical_lines"
TRACKING_DIR = PROJECT_ROOT / "data" / "tracking"


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


def _tracking_file_digests() -> dict[Path, str]:
    digests: dict[Path, str] = {}
    if not TRACKING_DIR.exists():
        return digests

    for path in sorted(TRACKING_DIR.iterdir()):
        if path.is_file():
            digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


@pytest.fixture(scope="session", autouse=True)
def guard_committed_tracking_artifacts_unchanged():
    before = _tracking_file_digests()
    yield
    after = _tracking_file_digests()
    assert after == before, "Pytest modified committed files under MLB/data/tracking/."


@pytest.fixture(autouse=True)
def isolate_tracking_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from jobs import run_daily_card as daily_card

    tracking_dir = tmp_path / "data" / "tracking"
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tracking_dir / "official_picks_history.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_GRADES_PATH",
        tracking_dir / "official_picks_profit_report.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_BOOK_SUMMARY_PATH",
        tracking_dir / "official_picks_profit_by_book.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH",
        tracking_dir / "official_picks_profit_summary.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_SKIPPED_PATH",
        tracking_dir / "official_picks_profit_skipped.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH",
        tracking_dir / "official_picks_concentration_audit.json",
    )
    return tracking_dir
