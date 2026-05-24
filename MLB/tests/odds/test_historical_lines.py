import pandas as pd
import pytest

from odds.historical_lines import (
    DEFAULT_SELECTION_RULE,
    OFFICIAL_PICKS_HISTORY_SELECTION_RULE,
    build_historical_lines_artifact_df,
    build_historical_lines_artifact_from_official_picks_history_df,
    curate_historical_lines,
    normalize_historical_line_snapshots,
)


def test_curate_historical_lines_prefers_latest_pregame_snapshot():
    raw_df = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob deGrom",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "pulled_at": "2025-08-02T22:00:00Z",
                "snapshot_type": "raw",
                "source": "fixture",
            },
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob deGrom",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 7.5,
                "price": 100,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "pulled_at": "2025-08-02T23:30:00Z",
                "snapshot_type": "raw",
                "source": "fixture",
            },
        ]
    )

    snapshots = normalize_historical_line_snapshots(raw_df)
    curated = curate_historical_lines(snapshots)

    assert len(curated) == 1
    assert curated.loc[0, "line"] == pytest.approx(6.5)
    assert curated.loc[0, "selection_rule"] == DEFAULT_SELECTION_RULE
    assert bool(curated.loc[0, "is_closing_line"]) is True


def test_build_historical_lines_artifact_df_is_deterministic_with_fixture_input(
    historical_lines_fixture_dir,
):
    curated = build_historical_lines_artifact_df(historical_lines_fixture_dir)

    assert len(curated) == 2
    assert curated.loc[0, "player_name_norm"] == "jacob degrom"
    assert curated.loc[0, "line"] == pytest.approx(7.0)
    assert curated.loc[1, "player_name_norm"] == "tarik skubal"
    assert curated.loc[1, "market_key"] == "pitcher_strikeouts"


def test_build_historical_lines_artifact_from_official_picks_history_df_filters_and_dedupes():
    history_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-11",
                "player_name": "Zack Wheeler",
                "participant_source_id": "12345",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "zack wheeler",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "book": "FanDuel",
                "bookmaker_key": "fanduel",
                "event_id": "evt_1",
                "price": -124,
                "pick_side": "over",
                "line": 5.5,
                "market_selection_key": "",
                "market_offer_key": "",
            },
            {
                "game_date": "2026-05-11",
                "player_name": "Zack Wheeler",
                "participant_source_id": "12345",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "zack wheeler",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "book": "FanDuel",
                "bookmaker_key": "fanduel",
                "event_id": "evt_1",
                "price": -124,
                "pick_side": "over",
                "line": 5.5,
                "market_selection_key": "MLB|pitcher_strikeouts|12345|over|5.5",
                "market_offer_key": "MLB|pitcher_strikeouts|12345|over|5.5|fanduel",
            },
            {
                "game_date": "2026-05-11",
                "player_name": "Peter Lambert",
                "sport": "MLB",
                "market_key": "pitcher_walks",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "price": -188,
                "pick_side": "under",
                "line": 2.5,
            },
        ]
    )

    curated = build_historical_lines_artifact_from_official_picks_history_df(history_df)

    assert len(curated) == 1
    assert curated.loc[0, "game_date"] == "2026-05-11"
    assert curated.loc[0, "player_name_norm"] == "zack wheeler"
    assert curated.loc[0, "bookmaker"] == "FanDuel"
    assert curated.loc[0, "line"] == pytest.approx(5.5)
    assert curated.loc[0, "market_key"] == "pitcher_strikeouts"
    assert curated.loc[0, "market_offer_key"] == "MLB|pitcher_strikeouts|12345|over|5.5|fanduel"
    assert curated.loc[0, "selection_rule"] == OFFICIAL_PICKS_HISTORY_SELECTION_RULE
