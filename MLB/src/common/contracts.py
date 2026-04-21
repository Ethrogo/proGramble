# MLB/src/common/contracts.py

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


STARTERS_REQUIRED_COLUMNS = [
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

PITCHER_GAMES_REQUIRED_COLUMNS = [
    "game_date",
    "game_pk",
    "pitcher",
    "player_name",
    "pitching_team",
    "opponent_team",
]

JOINED_ODDS_REQUIRED_COLUMNS = [
    "player_name_proj",
    "predicted_strikeouts",
    "bookmaker",
    "side",
    "line",
    "price",
]

FINAL_PICKS_REQUIRED_COLUMNS = [
    "player_name",
    "book",
    "pick_side",
    "line",
    "price",
    "edge",
    "pick_type",
]


def require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_non_empty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty.")


def assert_no_duplicate_keys(
    df: pd.DataFrame,
    keys: Sequence[str],
    name: str,
) -> None:
    require_columns(df, keys, name)

    dupes = df.duplicated(subset=list(keys), keep=False)
    if dupes.any():
        bad_rows = df.loc[dupes, list(keys)]
        raise ValueError(
            f"{name} has duplicate rows for key(s) {list(keys)}: "
            f"{bad_rows.to_dict(orient='records')}"
        )


def assert_non_null_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    require_columns(df, columns, name)

    null_counts = df[list(columns)].isnull().sum()
    bad = null_counts[null_counts > 0]

    if not bad.empty:
        raise ValueError(
            f"{name} has null values in required columns: {bad.to_dict()}"
        )


def assert_boolean_like_int_column(
    df: pd.DataFrame,
    column: str,
    name: str,
    allowed_values: Iterable[int] = (0, 1),
) -> None:
    require_columns(df, [column], name)

    values = set(df[column].dropna().unique())
    allowed = set(allowed_values)

    if not values.issubset(allowed):
        raise ValueError(
            f"{name}.{column} contains invalid values: {sorted(values)}. "
            f"Allowed values: {sorted(allowed)}"
        )


def validate_starters_contract(df: pd.DataFrame) -> None:
    require_columns(df, STARTERS_REQUIRED_COLUMNS, "starters_df")
    assert_non_empty(df, "starters_df")
    assert_non_null_columns(
        df,
        [
            "game_date",
            "game_pk",
            "player_name",
            "team",
            "opponent",
            "home_team",
            "away_team",
            "is_home",
        ],
        "starters_df",
    )
    assert_no_duplicate_keys(
        df,
        ["game_date", "game_pk", "player_name"],
        "starters_df",
    )
    assert_boolean_like_int_column(df, "is_home", "starters_df")


def validate_pitcher_games_contract(df: pd.DataFrame) -> None:
    require_columns(df, PITCHER_GAMES_REQUIRED_COLUMNS, "pitcher_games")
    assert_non_empty(df, "pitcher_games")
    assert_non_null_columns(
        df,
        ["game_date", "game_pk", "pitcher", "player_name"],
        "pitcher_games",
    )
    assert_no_duplicate_keys(
        df,
        ["game_date", "game_pk", "pitcher"],
        "pitcher_games",
    )


def validate_joined_odds_contract(df: pd.DataFrame) -> None:
    require_columns(df, JOINED_ODDS_REQUIRED_COLUMNS, "joined_odds_df")
    assert_non_empty(df, "joined_odds_df")
    assert_non_null_columns(
        df,
        ["player_name_proj", "predicted_strikeouts", "bookmaker", "side", "line"],
        "joined_odds_df",
    )


def validate_final_picks_contract(df: pd.DataFrame) -> None:
    require_columns(df, FINAL_PICKS_REQUIRED_COLUMNS, "final_picks_df")
    assert_non_empty(df, "final_picks_df")
    assert_non_null_columns(
        df,
        ["player_name", "book", "pick_side", "line", "edge", "pick_type"],
        "final_picks_df",
    )