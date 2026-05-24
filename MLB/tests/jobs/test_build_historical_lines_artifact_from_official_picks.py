import pandas as pd

from jobs import build_historical_lines_artifact_from_official_picks as history_job
from jobs import build_training_artifacts as training_job


def test_build_historical_lines_artifact_from_official_picks_writes_curated_file(
    tmp_path,
    monkeypatch,
):
    history_path = tmp_path / "data" / "tracking" / "official_picks_history.csv"
    latest_dir = tmp_path / "data" / "artifacts" / "latest"
    previous_dir = tmp_path / "data" / "artifacts" / "previous"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pick_key": "pick-1",
                "game_date": "2026-05-23",
                "player_name": "Roki Sasaki",
                "participant_join_key": "name:roki sasaki",
                "participant_id": "",
                "participant_source_id": "",
                "participant_source_id_type": "",
                "participant_name_norm": "roki sasaki",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_23",
                "price": -149,
                "pick_side": "under",
                "line": 5.5,
                "market_selection_key": "",
                "market_offer_key": "",
            }
        ]
    ).to_csv(history_path, index=False)

    monkeypatch.setattr(history_job, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(history_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(history_job, "PREVIOUS_DIR", previous_dir)
    monkeypatch.setattr(training_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(training_job, "PREVIOUS_DIR", previous_dir)

    output_path, rows = history_job.build_historical_lines_artifact_from_official_picks()
    saved = pd.read_csv(output_path)

    assert output_path.exists()
    assert rows == 1
    assert saved.loc[0, "player_name_norm"] == "roki sasaki"
    assert saved.loc[0, "bookmaker"] == "DraftKings"
    assert saved.loc[0, "source"] == "official_picks_history_selected_lines"
