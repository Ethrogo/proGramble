# MLB/tests/odds/test_odds_api.py

from unittest.mock import Mock

import pytest

from odds import odds_api


def test_fetch_event_odds_raises_when_api_key_missing(monkeypatch):
    """
    The fetch function should fail fast if no API key is available.
    """
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "")

    with pytest.raises(ValueError, match="ODDS_API_KEY is missing"):
        odds_api.fetch_event_odds()


def test_fetch_event_odds_builds_expected_request(monkeypatch):
    """
    Verify request URL/params are constructed correctly and the JSON
    response is returned unchanged.
    """
    mock_response = Mock()
    mock_response.json.return_value = [{"id": "game_1"}]
    mock_response.raise_for_status.return_value = None

    mock_get = Mock(return_value=mock_response)

    monkeypatch.setattr(odds_api.requests, "get", mock_get)
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "test_key")

    result = odds_api.fetch_event_odds()

    assert result == [{"id": "game_1"}]
    mock_get.assert_called_once()

    call_args = mock_get.call_args
    args, kwargs = call_args

    assert args[0] == "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    assert kwargs["timeout"] == 30

    params = kwargs["params"]
    assert params["apiKey"] == "test_key"
    assert params["regions"] == "us"
    assert params["markets"] == "pitcher_strikeouts"
    assert params["bookmakers"] == "draftkings,fanduel,betmgm,williamhill_us"
    assert params["oddsFormat"] == "american"
    assert params["dateFormat"] == "iso"


def test_fetch_event_odds_allows_custom_arguments(monkeypatch):
    """
    Verify caller-supplied arguments override defaults.
    """
    mock_response = Mock()
    mock_response.json.return_value = [{"id": "custom_game"}]
    mock_response.raise_for_status.return_value = None

    mock_get = Mock(return_value=mock_response)

    monkeypatch.setattr(odds_api.requests, "get", mock_get)
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "test_key")

    result = odds_api.fetch_event_odds(
        sport="baseball_mlb",
        market="pitcher_strikeouts",
        bookmakers=["draftkings", "fanduel"],
        odds_format="decimal",
        date_format="unix",
    )

    assert result == [{"id": "custom_game"}]

    _, kwargs = mock_get.call_args
    params = kwargs["params"]

    assert params["bookmakers"] == "draftkings,fanduel"
    assert params["oddsFormat"] == "decimal"
    assert params["dateFormat"] == "unix"


def test_fetch_event_odds_calls_raise_for_status(monkeypatch):
    """
    Ensure HTTP errors are surfaced instead of silently ignored.
    """
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = []

    mock_get = Mock(return_value=mock_response)

    monkeypatch.setattr(odds_api.requests, "get", mock_get)
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "test_key")

    odds_api.fetch_event_odds()

    mock_response.raise_for_status.assert_called_once()