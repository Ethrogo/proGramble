# MLB/src/starters/normalize.py

from __future__ import annotations

import pandas as pd


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return (
        name.strip()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )


def normalize_team_code(team: str) -> str:
    if not isinstance(team, str):
        return ""
    return team.strip().upper()


def finalize_starters_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the slate structure expected by pitcher_k.
    """
    out = df.copy()

    out["player_name"] = out["player_name"].map(normalize_player_name)
    out["team"] = out["team"].map(normalize_team_code)
    out["opponent"] = out["opponent"].map(normalize_team_code)
    out["home_team"] = out["home_team"].map(normalize_team_code)
    out["away_team"] = out["away_team"].map(normalize_team_code)

    out["is_home"] = out["is_home"].astype(int)

    ordered_cols = [
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

    return out[ordered_cols].sort_values(
        ["game_date", "game_pk", "is_home", "player_name"]
    ).reset_index(drop=True)