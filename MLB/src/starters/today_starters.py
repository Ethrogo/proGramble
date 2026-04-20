# MLB/src/starters/today_starters.py

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import requests

from .config import EASTERN_TZ, STARTERS_INPUT_DIR, TODAY_STARTERS_FILENAME
from .normalize import finalize_starters_df
from .validate import validate_starters_df

EASTERN = ZoneInfo("America/New_York")


def get_today_date_str() -> str:
    return datetime.now(EASTERN).date().isoformat()


def fetch_today_schedule(date_str: str | None = None) -> dict:
    date_str = date_str or get_today_date_str()
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _extract_pitcher_row(
    game_date: str,
    game_pk: int,
    side_data: dict,
    team_code: str,
    opponent_code: str,
    home_team: str,
    away_team: str,
    is_home: int,
) -> dict | None:
    probable = side_data.get("probablePitcher")
    if not probable:
        return None

    pitcher_id = probable.get("id")
    player_name = probable.get("fullName")

    # throws may not always be present in schedule hydrate response
    p_throws = None
    if isinstance(probable.get("pitchHand"), dict):
        p_throws = probable["pitchHand"].get("code")

    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "pitcher": pitcher_id,
        "player_name": player_name,
        "team": team_code,
        "opponent": opponent_code,
        "home_team": home_team,
        "away_team": away_team,
        "is_home": is_home,
        "p_throws": p_throws,
    }


def schedule_json_to_raw_starters_df(schedule_json: dict) -> pd.DataFrame:
    rows: list[dict] = []

    for date_block in schedule_json.get("dates", []):
        game_date = date_block.get("date")

        for game in date_block.get("games", []):
            game_pk = game.get("gamePk")
            teams = game.get("teams", {})

            away = teams.get("away", {})
            home = teams.get("home", {})

            away_team = away.get("team", {}).get("abbreviation")
            home_team = home.get("team", {}).get("abbreviation")

            away_row = _extract_pitcher_row(
                game_date=game_date,
                game_pk=game_pk,
                side_data=away,
                team_code=away_team,
                opponent_code=home_team,
                home_team=home_team,
                away_team=away_team,
                is_home=0,
            )
            if away_row:
                rows.append(away_row)

            home_row = _extract_pitcher_row(
                game_date=game_date,
                game_pk=game_pk,
                side_data=home,
                team_code=home_team,
                opponent_code=away_team,
                home_team=home_team,
                away_team=away_team,
                is_home=1,
            )
            if home_row:
                rows.append(home_row)

    return pd.DataFrame(rows)


def build_today_starters_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw probable-starters dataframe into the clean slate format
    expected by the pitcher_k pipeline.
    """
    starters_df = finalize_starters_df(raw_df)
    validate_starters_df(starters_df)
    return starters_df


def get_today_starters_df(date_str: str | None = None) -> pd.DataFrame:
    """
    Pull today's MLB schedule, extract probable starters, normalize,
    and validate the final slate dataframe.
    """
    schedule_json = fetch_today_schedule(date_str=date_str)
    raw_df = schedule_json_to_raw_starters_df(schedule_json)

    if raw_df.empty:
        raise ValueError("No probable starters found in MLB schedule response.")

    return build_today_starters_df(raw_df)


def save_today_starters_csv(
    starters_df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = TODAY_STARTERS_FILENAME,
) -> Path:
    output_dir = output_dir or STARTERS_INPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filename
    starters_df.to_csv(out_path, index=False)
    return out_path


def pull_and_save_today_starters(date_str: str | None = None) -> Path:
    """
    End-to-end helper:
    1. Pull today's starters
    2. Validate/normalize
    3. Save to CSV
    """
    starters_df = get_today_starters_df(date_str=date_str)
    return save_today_starters_csv(starters_df)


if __name__ == "__main__":
    out_path = pull_and_save_today_starters()
    print(f"Saved starters to: {out_path}")