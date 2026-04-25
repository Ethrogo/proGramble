# MLB/tests/odds/test_run_edges.py

import pandas as pd

from odds import run_edges
from pitcher_k.config import PITCHER_K_PROP_MARKET


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
                            "key": PITCHER_K_PROP_MARKET,
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

    def fake_fetch_all_player_props(market):
        assert market == PITCHER_K_PROP_MARKET
        return fake_events

    monkeypatch.setattr(
        run_edges,
        "fetch_all_player_props",
        fake_fetch_all_player_props,
    )

    joined, best = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

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

    def fake_fetch_all_player_props(market):
        assert market == PITCHER_K_PROP_MARKET
        return []

    monkeypatch.setattr(
        run_edges,
        "fetch_all_player_props",
        fake_fetch_all_player_props,
    )

    joined, best = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

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
                            "key": PITCHER_K_PROP_MARKET,
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

    def fake_fetch_all_player_props(market):
        assert market == PITCHER_K_PROP_MARKET
        return fake_events

    monkeypatch.setattr(
        run_edges,
        "fetch_all_player_props",
        fake_fetch_all_player_props,
    )

    joined, best = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

    assert joined.empty
    assert best.empty