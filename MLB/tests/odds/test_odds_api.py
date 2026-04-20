# MLB/tests/odds/test_odds_api.py

from unittest.mock import Mock

import requests
import pytest

from odds import odds_api


def test_fetch_mlb_events_uses_sport_level_odds_endpoint(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = [{"id": "event_1"}]
    mock_response.raise_for_status.return_value = None

    mock_get = Mock(return_value=mock_response)

    monkeypatch.setattr(odds_api.requests, "get", mock_get)
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "test_key")

    result = odds_api.fetch_mlb_events()

    assert result == [{"id": "event_1"}]

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

    params = kwargs["params"]
    assert params["markets"] == odds_api.EVENT_DISCOVERY_MARKET
    assert params["bookmakers"] == "draftkings,fanduel,betmgm,williamhill_us"


def test_fetch_event_player_props_uses_event_level_endpoint(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {"id": "event_1", "bookmakers": []}
    mock_response.raise_for_status.return_value = None

    mock_get = Mock(return_value=mock_response)

    monkeypatch.setattr(odds_api.requests, "get", mock_get)
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "test_key")

    result = odds_api.fetch_event_player_props("event_1")

    assert result == {"id": "event_1", "bookmakers": []}

    args, kwargs = mock_get.call_args
    assert (
        args[0]
        == "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/event_1/odds"
    )

    params = kwargs["params"]
    assert params["markets"] == odds_api.PLAYER_PROP_MARKET
    assert params["bookmakers"] == "draftkings,fanduel,betmgm,williamhill_us"


def test_fetch_all_pitcher_strikeout_props_skips_event_without_id(monkeypatch):
    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda: [{"id": "event_1"}, {"foo": "bar"}],
    )
    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        lambda event_id: {"id": event_id, "bookmakers": [{"key": "draftkings"}]},
    )

    result = odds_api.fetch_all_pitcher_strikeout_props()

    assert len(result) == 1
    assert result[0]["id"] == "event_1"


def test_fetch_all_pitcher_strikeout_props_skips_http_422_like_error(monkeypatch):
    def fake_fetch_event_player_props(event_id: str):
        if event_id == "bad_event":
            raise requests.HTTPError("422 Client Error")
        return {"id": event_id, "bookmakers": [{"key": "fanduel"}]}

    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda: [{"id": "good_event"}, {"id": "bad_event"}],
    )
    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        fake_fetch_event_player_props,
    )

    result = odds_api.fetch_all_pitcher_strikeout_props()

    assert len(result) == 1
    assert result[0]["id"] == "good_event"


def test_fetch_all_pitcher_strikeout_props_only_keeps_events_with_bookmakers(monkeypatch):
    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda: [{"id": "event_1"}, {"id": "event_2"}, {"id": "event_3"}],
    )

    def fake_fetch_event_player_props(event_id: str):
        if event_id == "event_1":
            return {"id": "event_1", "bookmakers": []}
        if event_id == "event_2":
            return {"id": "event_2"}
        return {"id": "event_3", "bookmakers": [{"key": "betmgm"}]}

    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        fake_fetch_event_player_props,
    )

    result = odds_api.fetch_all_pitcher_strikeout_props()

    assert len(result) == 1
    assert result[0]["id"] == "event_3"


def test_fetch_mlb_events_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "")

    with pytest.raises(ValueError, match="ODDS_API_KEY is missing"):
        odds_api.fetch_mlb_events()