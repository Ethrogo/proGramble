from pybaseball import statcast
import pybaseball
import pandas as pd

pybaseball.cache.enable()

STATCAST_COLUMNS = [
    "game_date", "game_pk", "pitcher", "player_name", "batter",
    "pitch_type", "release_speed", "release_spin_rate",
    "description", "events", "inning", "outs_when_up",
    "home_team", "away_team", "stand", "p_throws"
]


def load_statcast_data(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Pull raw Statcast pitch-level data and keep only needed columns.
    """
    sc = statcast(start_dt=start_dt, end_dt=end_dt)
    sc = sc[STATCAST_COLUMNS].copy()
    return sc