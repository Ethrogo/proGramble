# MLB/src/starters/validate.py

from __future__ import annotations

import pandas as pd

def validate_home_away_logic(df: pd.DataFrame) -> None:
    bad_home = df[(df["is_home"] == 1) & (df["team"] != df["home_team"])]
    bad_away = df[(df["is_home"] == 0) & (df["team"] != df["away_team"])]

    if not bad_home.empty or not bad_away.empty:
        raise ValueError("Starter slate failed home/away team consistency checks.")


def validate_starters_df(df: pd.DataFrame) -> None:
    validate_home_away_logic(df)
