# MLB/tests/odds/test_run_edges.py

import pandas as pd

from odds import run_edges
from common.contracts import validate_joined_odds_contract
from pitcher_bb.config import PITCHER_BB_PROP_MARKET
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

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

    assert len(joined) == 2
    assert len(best) == 1
    assert diagnostics["raw_event_count"] == 1
    assert diagnostics["normalized_odds_rows"] == 2
    assert diagnostics["joined_rows"] == 2
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

    def fake_fetch_all_player_props(market, **kwargs):
        assert market == PITCHER_K_PROP_MARKET
        return []

    monkeypatch.setattr(
        run_edges,
        "fetch_all_player_props",
        fake_fetch_all_player_props,
    )

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

    assert isinstance(joined, pd.DataFrame)
    assert isinstance(best, pd.DataFrame)
    assert joined.empty
    assert best.empty
    assert diagnostics["normalized_odds_rows"] == 0


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

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_K_PROP_MARKET,
    )

    assert joined.empty
    assert best.empty
    assert diagnostics["normalized_odds_rows"] == 2
    assert diagnostics["joined_rows"] == 0


def test_run_edge_pipeline_handles_pitcher_walks_market_with_shared_odds_shape(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Tarik Skubal",
                "team": "DET",
                "opponent": "CLE",
                "predicted_walks": 2.2,
                "predicted_value": 2.2,
                "player_name_norm": "tarik skubal",
            }
        ]
    )

    fake_events = [
        {
            "id": "game_2",
            "commence_time": "2026-04-18T19:10:00Z",
            "home_team": "Detroit Tigers",
            "away_team": "Cleveland Guardians",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2026-04-18T14:00:00Z",
                    "markets": [
                        {
                            "key": PITCHER_BB_PROP_MARKET,
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Tarik Skubal",
                                    "point": 1.5,
                                    "price": -102,
                                },
                                {
                                    "name": "Under",
                                    "description": "Tarik Skubal",
                                    "point": 1.5,
                                    "price": -118,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_fetch_all_player_props(market):
        assert market == PITCHER_BB_PROP_MARKET
        return fake_events

    monkeypatch.setattr(run_edges, "fetch_all_player_props", fake_fetch_all_player_props)

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_BB_PROP_MARKET,
        prediction_column="predicted_value",
        projection_join_key="player_name_norm",
        odds_join_key="player_name_norm",
    )

    validate_joined_odds_contract(joined, prediction_column="predicted_value")
    assert len(joined) == 2
    assert set(joined["market_key"]) == {PITCHER_BB_PROP_MARKET}
    assert set(joined["player_name_norm"]) == {"tarik skubal"}
    assert joined["predicted_value"].iloc[0] == 2.2
    assert diagnostics["joined_rows"] == 2
    assert best.iloc[0]["player_name_proj"] == "Tarik Skubal"


def test_run_edge_pipeline_retries_only_missing_projected_event_without_bookmaker_filter(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Tarik Skubal",
                "predicted_walks": 2.2,
                "predicted_value": 2.2,
                "player_name_norm": "tarik skubal",
                "team": "DET",
                "opponent": "CLE",
                "home_team": "DET",
                "away_team": "CLE",
            }
        ]
    )

    retry_calls: list[dict[str, object]] = []

    def fake_fetch_all_player_props(market, **kwargs):
        assert market == PITCHER_BB_PROP_MARKET
        return [
            {
                "id": "game_2",
                "commence_time": "2026-04-18T19:10:00Z",
                "home_team": "Detroit Tigers",
                "away_team": "Cleveland Guardians",
                "bookmakers": [],
            }
        ]

    monkeypatch.setattr(run_edges, "fetch_all_player_props", fake_fetch_all_player_props)
    monkeypatch.setattr(
        run_edges,
        "fetch_event_player_props",
        lambda event_id, market, **kwargs: retry_calls.append(
            {"event_id": event_id, "kwargs": kwargs}
        ) or {
            "id": event_id,
            "commence_time": "2026-04-18T19:10:00Z",
            "home_team": "Detroit Tigers",
            "away_team": "Cleveland Guardians",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2026-04-18T14:00:00Z",
                    "markets": [
                        {
                            "key": PITCHER_BB_PROP_MARKET,
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Tarik Skubal",
                                    "point": 1.5,
                                    "price": -102,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_BB_PROP_MARKET,
        prediction_column="predicted_value",
        projection_join_key="player_name_norm",
        odds_join_key="player_name_norm",
    )

    assert len(joined) == 1
    assert len(best) == 1
    assert len(retry_calls) == 1
    assert retry_calls[0]["event_id"] == "game_2"
    assert retry_calls[0]["kwargs"]["use_configured_bookmakers"] is False
    assert retry_calls[0]["kwargs"]["bookmakers"] is None
    assert diagnostics["fetch_scope"] == "targeted_all_region_books"
    assert diagnostics["bookmaker_filter_applied"] is False
    assert diagnostics["targeted_retry_event_ids"] == ["game_2"]


def test_missing_projection_event_ids_matches_team_aliases_for_targeted_retry():
    projections = pd.DataFrame(
        [
            {
                "player_name": "Shane Baz",
                "predicted_walks": 2.1,
                "predicted_value": 2.1,
                "player_name_norm": "shane baz",
                "team": "TB",
                "opponent": "KC",
                "home_team": "TB",
                "away_team": "KC",
            }
        ]
    )

    raw_events = [
        {
            "id": "game_alias",
            "commence_time": "2026-04-18T19:10:00Z",
            "home_team": "Tampa Bay Rays",
            "away_team": "Kansas City Royals",
            "bookmakers": [],
        }
    ]

    retry_event_ids = run_edges._missing_projection_event_ids(
        projections,
        pd.DataFrame(),
        raw_events,
        projection_join_key="player_name_norm",
        sport="MLB",
        market=PITCHER_BB_PROP_MARKET,
        participant_key="player_name",
        prediction_column="predicted_value",
    )

    assert retry_event_ids == ["game_alias"]


def test_run_edge_pipeline_retries_full_pitcher_walks_market_without_bookmaker_filter_when_configured_books_miss(monkeypatch):
    projections = pd.DataFrame(
        [
            {
                "player_name": "Shane Baz",
                "predicted_walks": 2.1,
                "predicted_value": 2.1,
                "player_name_norm": "shane baz",
                "team": "TB",
                "opponent": "KC",
                "home_team": "TB",
                "away_team": "KC",
            }
        ]
    )

    fetch_calls: list[dict[str, object]] = []

    def fake_fetch_all_player_props(market, **kwargs):
        fetch_calls.append({"market": market, "kwargs": kwargs})
        assert market == PITCHER_BB_PROP_MARKET
        if len(fetch_calls) == 1:
            return [
                {
                    "id": "game_alias",
                    "commence_time": "2026-04-18T19:10:00Z",
                    "home_team": "Tampa Bay Rays",
                    "away_team": "Kansas City Royals",
                    "bookmakers": [],
                }
            ]
        return [
            {
                "id": "game_alias",
                "commence_time": "2026-04-18T19:10:00Z",
                "home_team": "Tampa Bay Rays",
                "away_team": "Kansas City Royals",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "last_update": "2026-04-18T14:00:00Z",
                        "markets": [
                            {
                                "key": PITCHER_BB_PROP_MARKET,
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "description": "Shane Baz",
                                        "point": 1.5,
                                        "price": -110,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    monkeypatch.setattr(run_edges, "fetch_all_player_props", fake_fetch_all_player_props)
    monkeypatch.setattr(run_edges, "fetch_event_player_props", lambda *args, **kwargs: {})

    joined, best, diagnostics = run_edges.run_edge_pipeline(
        projections,
        market=PITCHER_BB_PROP_MARKET,
        prediction_column="predicted_value",
        projection_join_key="player_name_norm",
        odds_join_key="player_name_norm",
    )

    assert len(joined) == 1
    assert len(best) == 1
    assert len(fetch_calls) == 2
    assert fetch_calls[1]["kwargs"]["bookmakers"] is None
    assert fetch_calls[1]["kwargs"]["use_configured_bookmakers"] is False
    assert diagnostics["fetch_scope"] == "all_region_books"
    assert diagnostics["bookmaker_filter_applied"] is False
    assert diagnostics["initial_fetch"]["normalized_odds_rows"] == 0
