from __future__ import annotations

import time

import pandas as pd
from pybaseball import statcast


STATCAST_COLUMNS = [
    "game_date",
    "game_pk",
    "pitcher",
    "player_name",
    "batter",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "description",
    "events",
    "inning",
    "outs_when_up",
    "home_team",
    "away_team",
    "stand",
    "p_throws",
    "inning_topbot",
]


def _load_statcast_chunk(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_retries: int,
) -> list[pd.DataFrame]:
    success = False
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Loading Statcast chunk {start.date()} to {end.date()} (attempt {attempt})")
            df = statcast(
                start_dt=start.strftime("%Y-%m-%d"),
                end_dt=end.strftime("%Y-%m-%d"),
                parallel=False,
            )
            success = True
            return [df]
        except Exception as exc:
            last_error = exc
            print(f"Chunk failed {start.date()} to {end.date()}: {exc}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    if success:
        return []

    if start < end:
        midpoint = start + (end - start) / 2
        midpoint = pd.Timestamp(midpoint).normalize()
        if midpoint < start:
            midpoint = start
        if midpoint >= end:
            midpoint = end - pd.Timedelta(days=1)

        # When a broad Statcast export returns malformed CSV, narrower slices
        # often succeed even though retrying the same wide range does not.
        left_frames = _load_statcast_chunk(start, midpoint, max_retries=max_retries)
        right_frames = _load_statcast_chunk(
            midpoint + pd.Timedelta(days=1),
            end,
            max_retries=max_retries,
        )
        return [*left_frames, *right_frames]

    raise RuntimeError(
        "Failed Statcast chunk after retries: "
        f"{start.date()} to {end.date()}"
    ) from last_error


def load_statcast_data(
    start_dt: str,
    end_dt: str,
    chunk_days: int = 7,
    max_retries: int = 3,
) -> pd.DataFrame:
    start = pd.to_datetime(start_dt)
    end = pd.to_datetime(end_dt)

    frames: list[pd.DataFrame] = []
    cur = start

    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        frames.extend(
            _load_statcast_chunk(
                cur,
                chunk_end,
                max_retries=max_retries,
            )
        )
        cur = chunk_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
