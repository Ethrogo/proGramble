# MLB/src/pitcher_k/slate.py

from __future__ import annotations

import pandas as pd

from common.contracts import (
    STARTERS_REQUIRED_COLUMNS,
    require_columns,
    validate_starters_contract,
)

SLATE_COLUMNS = STARTERS_REQUIRED_COLUMNS




def load_tomorrow_slate_from_csv(path: str) -> pd.DataFrame:
    """
    Load tomorrow's probable starter slate from a CSV.

    Expected columns:
        game_date, game_pk, pitcher, player_name, team, opponent,
        home_team, away_team, is_home, p_throws
    """
    df = pd.read_csv(path)

    require_columns(df, SLATE_COLUMNS, "starters_df")

    df = df[SLATE_COLUMNS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    df["pitcher"] = df["pitcher"].astype(int)
    df["game_pk"] = df["game_pk"].astype(int)
    df["is_home"] = df["is_home"].astype(int)

    return df


def validate_slate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic validation and cleanup for tomorrow's starter slate.
    """
    df = df.copy()

    df = df.drop_duplicates(subset=["game_date", "game_pk", "pitcher"])
    validate_starters_contract(df)

    return df


def build_prediction_base(slate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the clean base table that downstream feature builders will enrich.
    """
    slate_df = validate_slate(slate_df)

    return slate_df.sort_values(["game_date", "game_pk", "player_name"]).reset_index(drop=True)

