# MLB/src/starters/config.py

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN_TZ = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STARTERS_INPUT_DIR = DATA_DIR / "inputs" / "starters"

TODAY_STARTERS_FILENAME = "today_starters.csv"

REQUIRED_STARTER_COLUMNS = [
    "game_date",
    "game_pk",
    "pitcher",
    "player_name",
    "team",
    "opponent",
    "home_team",
    "away_team",
    "is_home",
    "p_throws",
]