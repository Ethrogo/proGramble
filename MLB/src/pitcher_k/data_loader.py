from pybaseball import statcast
from pybaseball import statcast
import time
import pandas as pd



STATCAST_COLUMNS = [
    "game_date", "game_pk", "pitcher", "player_name", "batter",
    "pitch_type", "release_speed", "release_spin_rate",
    "description", "events", "inning", "outs_when_up",
    "home_team", "away_team", "stand", "p_throws",
    "inning_topbot",
]


def load_statcast_data(start_dt: str, end_dt: str, chunk_days: int = 7, max_retries: int = 3) -> pd.DataFrame:
    start = pd.to_datetime(start_dt)
    end = pd.to_datetime(end_dt)

    frames = []
    cur = start

    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Loading Statcast chunk {cur.date()} to {chunk_end.date()} (attempt {attempt})")
                df = statcast(
                    start_dt=cur.strftime("%Y-%m-%d"),
                    end_dt=chunk_end.strftime("%Y-%m-%d"),
                    parallel=False,
                )
                frames.append(df)
                success = True
                break
            except Exception as e:
                print(f"Chunk failed {cur.date()} to {chunk_end.date()}: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)

        if not success:
            raise RuntimeError(
                f"Failed Statcast chunk after retries: {cur.date()} to {chunk_end.date()}"
            )

        cur = chunk_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)