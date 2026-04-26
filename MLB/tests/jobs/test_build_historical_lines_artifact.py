import pandas as pd

from jobs import build_historical_lines_artifact as historical_lines_job
from jobs import build_training_artifacts as training_job


def test_build_historical_lines_artifact_writes_native_curated_file(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw" / "historical_lines"
    latest_dir = tmp_path / "data" / "artifacts" / "latest"
    previous_dir = tmp_path / "data" / "artifacts" / "previous"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
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
            }
        ]
    ).to_csv(raw_dir / "sample.csv", index=False)

    monkeypatch.setattr(historical_lines_job, "RAW_HISTORICAL_LINES_DIR", raw_dir)
    monkeypatch.setattr(historical_lines_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(historical_lines_job, "PREVIOUS_DIR", previous_dir)
    monkeypatch.setattr(training_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(training_job, "PREVIOUS_DIR", previous_dir)

    output_path, rows = historical_lines_job.build_historical_lines_artifact()
    saved = pd.read_csv(output_path)

    assert output_path.exists()
    assert rows == 1
    assert saved.loc[0, "player_name_norm"] == "jacob degrom"
    assert saved.loc[0, "selection_rule"] == "latest_pregame_snapshot_per_game_player_book_side"
