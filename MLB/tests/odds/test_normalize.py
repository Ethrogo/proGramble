# MLB/tests/odds/test_normalize.py

import pandas as pd
import numbers


from odds.normalize import normalize_player_name, odds_json_to_dataframe


def test_normalize_player_name_basic():
    assert normalize_player_name("Jacob deGrom") == "jacob degrom"
    assert normalize_player_name("Max Scherzer.") == "max scherzer"
    assert normalize_player_name("D'Andre-Smith") == "dandre smith"


def test_normalize_player_name_non_string():
    assert normalize_player_name(None) == ""
    assert normalize_player_name(123) == ""


def test_odds_json_to_dataframe_parses_event_level_props():
    events = [
        {
            "id": "event_1",
            "commence_time": "2026-04-19T19:10:00Z",
            "home_team": "Texas Rangers",
            "away_team": "Seattle Mariners",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-04-19T14:00:00Z",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Jacob deGrom",
                                    "point": 6.5,
                                    "price": -120,
                                },
                                {
                                    "name": "Under",
                                    "description": "Jacob deGrom",
                                    "point": 6.5,
                                    "price": 100,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = odds_json_to_dataframe(events)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_cols = {
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker",
        "bookmaker_key",
        "book_last_update",
        "market_key",
        "player_name",
        "player_name_norm",
        "side",
        "line",
        "price",
    }
    assert expected_cols.issubset(df.columns)

    assert set(df["bookmaker"]) == {"DraftKings"}
    assert set(df["side"]) == {"Over", "Under"}
    assert set(df["player_name_norm"]) == {"jacob degrom"}


def test_odds_json_to_dataframe_maps_williamhill_us_to_caesars():
    events = [
        {
            "id": "event_1",
            "commence_time": "2026-04-19T19:10:00Z",
            "home_team": "Texas Rangers",
            "away_team": "Seattle Mariners",
            "bookmakers": [
                {
                    "key": "williamhill_us",
                    "last_update": "2026-04-19T14:00:00Z",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Jacob deGrom",
                                    "point": 6.5,
                                    "price": -110,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = odds_json_to_dataframe(events)

    assert len(df) == 1
    assert df.loc[0, "bookmaker"] == "Caesars"


def test_odds_json_to_dataframe_handles_positive_and_negative_american_odds():
    events = [
        {
            "id": "event_1",
            "commence_time": "2026-04-19T19:10:00Z",
            "home_team": "Texas Rangers",
            "away_team": "Seattle Mariners",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2026-04-19T14:00:00Z",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Jacob deGrom",
                                    "point": 6.5,
                                    "price": -125,
                                },
                                {
                                    "name": "Under",
                                    "description": "Jacob deGrom",
                                    "point": 6.5,
                                    "price": 105,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = odds_json_to_dataframe(events)

    assert len(df) == 2

    over_price = df.loc[df["side"] == "Over", "price"].iloc[0]
    under_price = df.loc[df["side"] == "Under", "price"].iloc[0]

    assert over_price == -125
    assert under_price == 105
    assert isinstance(over_price, numbers.Real)
    assert isinstance(under_price, numbers.Real)


def test_odds_json_to_dataframe_returns_empty_dataframe_for_empty_input():
    df = odds_json_to_dataframe([])

    assert isinstance(df, pd.DataFrame)
    assert df.empty