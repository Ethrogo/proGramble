import pandas as pd
import pytest

from odds.historical_lines import (
    DEFAULT_SELECTION_RULE,
    build_historical_lines_artifact_df,
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


def test_build_historical_lines_artifact_df_is_deterministic_with_duplicate_snapshots(tmp_path):
    raw_dir = tmp_path / "raw" / "historical_lines"
    raw_dir.mkdir(parents=True, exist_ok=True)
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
                "player_name": "Jacob Degrom",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 7.0,
                "price": -110,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "pulled_at": "2025-08-02T22:30:00Z",
                "snapshot_type": "raw",
                "source": "fixture",
            },
        ]
    )
    raw_df.to_csv(raw_dir / "pitcher_strikeouts_sample.csv", index=False)

    curated = build_historical_lines_artifact_df(raw_dir)

    assert len(curated) == 1
    assert curated.loc[0, "player_name_norm"] == "jacob degrom"
    assert curated.loc[0, "line"] == pytest.approx(7.0)
