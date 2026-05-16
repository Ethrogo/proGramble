# MLB/tests/odds/test_odds_api.py

from unittest.mock import Mock

import requests
import pytest

from odds import odds_api
from pitcher_k.config import PITCHER_K_PROP_MARKET


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

    result = odds_api.fetch_event_player_props(
        event_id="event_1",
        market=PITCHER_K_PROP_MARKET,
    )

    assert result == {"id": "event_1", "bookmakers": []}

    args, kwargs = mock_get.call_args
    assert (
        args[0]
        == "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/event_1/odds"
    )

    params = kwargs["params"]
    assert params["markets"] == PITCHER_K_PROP_MARKET
    assert params["bookmakers"] == "draftkings,fanduel,betmgm,williamhill_us"


def test_fetch_all_player_props_skips_event_without_id(monkeypatch):
    event_calls = {}

    def fake_fetch_mlb_events(**kwargs):
        event_calls["kwargs"] = kwargs
        return [{"id": "event_1"}, {"foo": "bar"}]

    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        fake_fetch_mlb_events,
    )

    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        lambda event_id, market, **kwargs: {
            "id": event_id,
            "bookmakers": [{"key": "draftkings"}],
        },
    )

    result = odds_api.fetch_all_player_props(market=PITCHER_K_PROP_MARKET)

    assert len(result) == 1
    assert result[0]["id"] == "event_1"
    assert event_calls["kwargs"]["bookmakers"] == odds_api.BOOKMAKERS


def test_fetch_all_player_props_skips_http_422_like_error(monkeypatch):
    def fake_fetch_event_player_props(event_id: str, market: str, **kwargs):
        assert market == PITCHER_K_PROP_MARKET

        if event_id == "bad_event":
            raise requests.HTTPError("422 Client Error")

        return {"id": event_id, "bookmakers": [{"key": "fanduel"}]}

    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda **kwargs: [{"id": "good_event"}, {"id": "bad_event"}]
    )

    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        fake_fetch_event_player_props,
    )

    result = odds_api.fetch_all_player_props(market=PITCHER_K_PROP_MARKET)

    assert len(result) == 1
    assert result[0]["id"] == "good_event"


def test_fetch_all_player_props_only_keeps_events_with_bookmakers(monkeypatch):
    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda **kwargs: [{"id": "event_1"}, {"id": "event_2"}, {"id": "event_3"}],
    )

    def fake_fetch_event_player_props(event_id: str, market: str, **kwargs):
        assert market == PITCHER_K_PROP_MARKET

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

    result = odds_api.fetch_all_player_props(market=PITCHER_K_PROP_MARKET)

    assert len(result) == 1
    assert result[0]["id"] == "event_3"


def test_fetch_all_player_props_passes_custom_bookmakers_to_event_discovery(monkeypatch):
    event_calls = {}

    def fake_fetch_mlb_events(**kwargs):
        event_calls["kwargs"] = kwargs
        return [{"id": "event_1"}]

    monkeypatch.setattr(odds_api, "fetch_mlb_events", fake_fetch_mlb_events)
    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        lambda event_id, market, **kwargs: {
            "id": event_id,
            "bookmakers": [{"key": "fanduel"}],
        },
    )

    result = odds_api.fetch_all_player_props(
        market=PITCHER_K_PROP_MARKET,
        bookmakers=["draftkings", "williamhill_us"],
    )

    assert len(result) == 1
    assert event_calls["kwargs"]["bookmakers"] == ["draftkings", "williamhill_us"]


def test_fetch_all_player_props_without_configured_bookmakers_removes_bookmaker_filter(monkeypatch):
    event_calls = {}
    prop_calls = {}

    def fake_fetch_mlb_events(**kwargs):
        event_calls["kwargs"] = kwargs
        return [{"id": "event_1"}]

    def fake_fetch_event_player_props(event_id: str, market: str, **kwargs):
        prop_calls["event_id"] = event_id
        prop_calls["market"] = market
        prop_calls["kwargs"] = kwargs
        return {"id": event_id, "bookmakers": [{"key": "espnbet"}]}

    monkeypatch.setattr(odds_api, "fetch_mlb_events", fake_fetch_mlb_events)
    monkeypatch.setattr(odds_api, "fetch_event_player_props", fake_fetch_event_player_props)

    result = odds_api.fetch_all_player_props(
        market=PITCHER_K_PROP_MARKET,
        use_configured_bookmakers=False,
    )

    assert len(result) == 1
    assert event_calls["kwargs"]["bookmakers"] is None
    assert event_calls["kwargs"]["use_configured_bookmakers"] is False
    assert prop_calls["kwargs"]["bookmakers"] is None
    assert prop_calls["kwargs"]["use_configured_bookmakers"] is False


def test_summarize_event_bookmaker_coverage_reports_missing_caesars_with_note():
    coverage = odds_api.summarize_event_bookmaker_coverage(
        [
            {
                "id": "event_1",
                "bookmakers": [
                    {"key": "draftkings"},
                    {"key": "fanduel"},
                    {"key": "betmgm"},
                ],
            }
        ],
        requested_bookmakers=["draftkings", "fanduel", "betmgm", "williamhill_us"],
    )

    assert coverage["upstream_bookmaker_keys"] == ["betmgm", "draftkings", "fanduel"]
    assert coverage["missing_requested_bookmakers"] == ["williamhill_us"]
    assert "paid subscriptions only" in coverage["missing_requested_notes"]["williamhill_us"]


def test_fetch_all_player_props_logs_coverage_and_failed_events(monkeypatch, capsys):
    def fake_fetch_event_player_props(event_id: str, market: str, **kwargs):
        if event_id == "bad_event":
            raise requests.HTTPError("422 Client Error")
        return {
            "id": event_id,
            "bookmakers": [{"key": "draftkings"}],
        }

    monkeypatch.setattr(
        odds_api,
        "fetch_mlb_events",
        lambda **kwargs: [{"id": "good_event"}, {"id": "bad_event"}],
    )
    monkeypatch.setattr(
        odds_api,
        "fetch_event_player_props",
        fake_fetch_event_player_props,
    )

    odds_api.fetch_all_player_props(
        market=PITCHER_K_PROP_MARKET,
        bookmakers=["draftkings", "williamhill_us"],
    )
    captured = capsys.readouterr()

    assert "Live odds coverage:" in captured.out
    assert "missing=['williamhill_us']" in captured.out
    assert "failed for event ids" in captured.out


def test_fetch_mlb_events_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "")

    with pytest.raises(ValueError, match="ODDS_API_KEY is missing"):
        odds_api.fetch_mlb_events()
