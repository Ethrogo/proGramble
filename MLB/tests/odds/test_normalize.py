import pandas as pd

from odds.normalize import odds_json_to_dataframe


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

    assert set(["player_name", "side", "line", "price", "bookmaker"]).issubset(df.columns)
    assert df.iloc[0]["player_name"] == "Jacob deGrom"
    assert set(df["side"]) == {"Over", "Under"}
    assert set(df["line"]) == {6.5}
    assert set(df["bookmaker"]) == {"DraftKings"}