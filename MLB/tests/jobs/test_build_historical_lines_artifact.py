import pandas as pd

from jobs import build_historical_lines_artifact as historical_lines_job
from jobs import build_training_artifacts as training_job


def test_build_historical_lines_artifact_writes_native_curated_file(
    tmp_path,
    monkeypatch,
    historical_lines_fixture_dir,
):
    raw_dir = historical_lines_fixture_dir
    latest_dir = tmp_path / "data" / "artifacts" / "latest"
    previous_dir = tmp_path / "data" / "artifacts" / "previous"

    monkeypatch.setattr(historical_lines_job, "RAW_HISTORICAL_LINES_DIR", raw_dir)
    monkeypatch.setattr(historical_lines_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(historical_lines_job, "PREVIOUS_DIR", previous_dir)
    monkeypatch.setattr(training_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(training_job, "PREVIOUS_DIR", previous_dir)

    output_path, rows = historical_lines_job.build_historical_lines_artifact()
    saved = pd.read_csv(output_path)

    assert output_path.exists()
    assert rows == 2
    assert saved.loc[0, "player_name_norm"] == "jacob degrom"
    assert saved.loc[0, "selection_rule"] == "latest_pregame_snapshot_per_game_player_book_side"
