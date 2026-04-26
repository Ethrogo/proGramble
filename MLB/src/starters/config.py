# MLB/src/starters/config.py

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN_TZ = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STARTERS_INPUT_DIR = DATA_DIR / "inputs" / "starters"

TODAY_STARTERS_FILENAME = "today_starters.csv"
