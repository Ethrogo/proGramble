import pandas as pd
import pytest

from starters.validate import validate_starters_df


def _valid_starters_df():
    return pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 111,
                "player_name": "Logan Gilbert",
                "team": "SEA",
                "opponent": "TEX",
                "home_team": "TEX",
                "away_team": "SEA",
                "is_home": 0,
                "p_throws": "R",
            },
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 222,
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "home_team": "TEX",
                "away_team": "SEA",
                "is_home": 1,
                "p_throws": "R",
            },
        ]
    )


def test_validate_starters_df_passes_on_valid_input():
    df = _valid_starters_df()
    validate_starters_df(df)


def test_validate_starters_df_raises_on_missing_required_column():
    df = _valid_starters_df().drop(columns=["team"])

    with pytest.raises(ValueError, match="Missing required starter columns"):
        validate_starters_df(df)


def test_validate_starters_df_raises_on_duplicate_pitcher_rows():
    df = pd.concat([_valid_starters_df(), _valid_starters_df().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate pitcher rows found"):
        validate_starters_df(df)


def test_validate_starters_df_raises_on_bad_home_away_logic():
    df = _valid_starters_df()
    df.loc[df["player_name"] == "Jacob deGrom", "team"] = "SEA"

    with pytest.raises(ValueError, match="home/away team consistency checks"):
        validate_starters_df(df)