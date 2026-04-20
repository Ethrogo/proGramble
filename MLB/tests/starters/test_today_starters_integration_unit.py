import pandas as pd
import pytest

from starters import today_starters


def test_get_today_starters_df_fetches_builds_and_validates(monkeypatch):
    fake_schedule = {
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

    monkeypatch.setattr(today_starters, "fetch_today_schedule", lambda date_str=None: fake_schedule)

    df = today_starters.get_today_starters_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["player_name"]) == {"Logan Gilbert", "Jacob deGrom"}
    assert set(df["team"]) == {"SEA", "TEX"}

def test_get_today_starters_df_raises_when_no_probable_starters_found(monkeypatch):
    fake_schedule = {"dates": []}

    monkeypatch.setattr(today_starters, "fetch_today_schedule", lambda date_str=None: fake_schedule)

    with pytest.raises(ValueError, match="No probable starters found"):
        today_starters.get_today_starters_df()    