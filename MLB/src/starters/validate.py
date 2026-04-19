# MLB/src/starters/validate.py

from __future__ import annotations

import pandas as pd

from .config import REQUIRED_STARTER_COLUMNS


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_STARTER_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required starter columns: {missing}")


def validate_no_nulls(df: pd.DataFrame) -> None:
    required_non_null = [
        "game_date",
        "player_name",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "is_home",
    ]

    null_counts = df[required_non_null].isnull().sum()
    bad = null_counts[null_counts > 0]

    if not bad.empty:
        raise ValueError(
            f"Nulls found in starter slate required fields: {bad.to_dict()}"
        )


def validate_one_row_per_pitcher_game(df: pd.DataFrame) -> None:
    dupes = df.duplicated(subset=["game_date", "game_pk", "player_name"], keep=False)
    if dupes.any():
        bad_rows = df.loc[dupes, ["game_date", "game_pk", "player_name"]]
        raise ValueError(
            "Duplicate pitcher rows found in starter slate: "
            f"{bad_rows.to_dict(orient='records')}"
        )


def validate_home_away_logic(df: pd.DataFrame) -> None:
    bad_home = df[(df["is_home"] == 1) & (df["team"] != df["home_team"])]
    bad_away = df[(df["is_home"] == 0) & (df["team"] != df["away_team"])]

    if not bad_home.empty or not bad_away.empty:
        raise ValueError("Starter slate failed home/away team consistency checks.")


def validate_starters_df(df: pd.DataFrame) -> None:
    validate_required_columns(df)
    validate_no_nulls(df)
    validate_one_row_per_pitcher_game(df)
    validate_home_away_logic(df)