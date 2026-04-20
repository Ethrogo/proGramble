# MLB/tests/starters/test_today_starters.py

import pandas as pd

from starters.today_starters import schedule_json_to_raw_starters_df


def test_schedule_json_to_raw_starters_df_builds_two_pitcher_rows():
    schedule_json = {
        "dates": [
            {
                "date": "2026-04-19",
                "games": [
                    {
                        "gamePk": 123456,
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "SEA"},
                                "probablePitcher": {
                                    "id": 111,
                                    "fullName": "Logan Gilbert",
                                    "pitchHand": {"code": "R"},
                                },
                            },
                            "home": {
                                "team": {"abbreviation": "TEX"},
                                "probablePitcher": {
                                    "id": 222,
                                    "fullName": "Jacob deGrom",
                                    "pitchHand": {"code": "R"},
                                },
                            },
                        },
                    }
                ],
            }
        ]
    }

    df = schedule_json_to_raw_starters_df(schedule_json)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_cols = {
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
    }
    assert expected_cols.issubset(df.columns)

    away_row = df.loc[df["team"] == "SEA"].iloc[0]
    assert away_row["game_date"] == "2026-04-19"
    assert away_row["game_pk"] == 123456
    assert away_row["pitcher"] == 111
    assert away_row["player_name"] == "Logan Gilbert"
    assert away_row["opponent"] == "TEX"
    assert away_row["home_team"] == "TEX"
    assert away_row["away_team"] == "SEA"
    assert away_row["is_home"] == 0
    assert away_row["p_throws"] == "R"

    home_row = df.loc[df["team"] == "TEX"].iloc[0]
    assert home_row["pitcher"] == 222
    assert home_row["player_name"] == "Jacob deGrom"
    assert home_row["opponent"] == "SEA"
    assert home_row["is_home"] == 1
    assert home_row["p_throws"] == "R"

def test_schedule_json_to_raw_starters_df_skips_missing_probable_pitcher():
    schedule_json = {
        "dates": [
            {
                "date": "2026-04-19",
                "games": [
                    {
                        "gamePk": 123456,
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "SEA"},
                                "probablePitcher": {
                                    "id": 111,
                                    "fullName": "Logan Gilbert",
                                    "pitchHand": {"code": "R"},
                                },
                            },
                            "home": {
                                "team": {"abbreviation": "TEX"},
                            },
                        },
                    }
                ],
            }
        ]
    }

    df = schedule_json_to_raw_starters_df(schedule_json)

    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Logan Gilbert"
    assert df.iloc[0]["team"] == "SEA"

def test_schedule_json_to_raw_starters_df_allows_missing_pitch_hand():
    schedule_json = {
        "dates": [
            {
                "date": "2026-04-19",
                "games": [
                    {
                        "gamePk": 123456,
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "SEA"},
                                "probablePitcher": {
                                    "id": 111,
                                    "fullName": "Logan Gilbert",
                                },
                            },
                            "home": {
                                "team": {"abbreviation": "TEX"},
                                "probablePitcher": {
                                    "id": 222,
                                    "fullName": "Jacob deGrom",
                                },
                            },
                        },
                    }
                ],
            }
        ]
    }

    df = schedule_json_to_raw_starters_df(schedule_json)

    assert len(df) == 2
    assert "p_throws" in df.columns
    assert pd.isna(df.loc[df["team"] == "SEA", "p_throws"]).iloc[0]
    assert pd.isna(df.loc[df["team"] == "TEX", "p_throws"]).iloc[0]

def test_schedule_json_to_raw_starters_df_empty_schedule():
    schedule_json = {"dates": []}

    df = schedule_json_to_raw_starters_df(schedule_json)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    
        