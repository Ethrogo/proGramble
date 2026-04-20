# MLB/tests/odds/test_run_edges.py

import pandas as pd

from odds import run_edges


def test_run_edge_pipeline_returns_joined_and_best_edges(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    fake_events = [
        {
            "id": "game_1",
            "commence_time": "2026-04-18T19:10:00Z",
            "home_team": "Texas Rangers",
            "away_team": "Seattle Mariners",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-04-18T14:00:00Z",
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

    monkeypatch.setattr(run_edges, "fetch_all_pitcher_strikeout_props", lambda: fake_events)

    joined, best = run_edges.run_edge_pipeline(projections)

    assert len(joined) == 2
    assert len(best) == 1
    assert best.iloc[0]["player_name_proj"] == "Jacob deGrom"
    assert best.iloc[0]["bookmaker"] == "DraftKings"


def test_run_edge_pipeline_returns_empty_when_no_odds(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    monkeypatch.setattr(run_edges, "fetch_all_pitcher_strikeout_props", lambda: [])

    joined, best = run_edges.run_edge_pipeline(projections)

    assert isinstance(joined, pd.DataFrame)
    assert isinstance(best, pd.DataFrame)
    assert joined.empty
    assert best.empty


def test_run_edge_pipeline_returns_empty_when_odds_do_not_match_projection_names(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.78,
            }
        ]
    )

    fake_events = [
        {
            "id": "game_1",
            "commence_time": "2026-04-18T19:10:00Z",
            "home_team": "Minnesota Twins",
            "away_team": "Cincinnati Reds",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2026-04-18T14:00:00Z",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Joe Ryan",
                                    "point": 5.5,
                                    "price": -110,
                                },
                                {
                                    "name": "Under",
                                    "description": "Joe Ryan",
                                    "point": 5.5,
                                    "price": -110,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    monkeypatch.setattr(run_edges, "fetch_all_pitcher_strikeout_props", lambda: fake_events)

    joined, best = run_edges.run_edge_pipeline(projections)

    assert joined.empty
    assert best.empty