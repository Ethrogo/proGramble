import json

import pandas as pd

from jobs import build_training_artifacts as training_job


class FakeModel:
    def save_model(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fake-model")


def test_save_artifacts_to_dir_writes_metadata_json(tmp_path):
    output_dir = tmp_path / "artifacts"
    pitcher_games = pd.DataFrame([{"game_date": "2026-04-19", "pitcher": 1}])
    model_df = pd.DataFrame([{"game_date": "2026-04-19", "strikeouts": 7}])
    metadata = {
        "target": "strikeouts",
        "features": ["pitches_last3"],
        "model_params": {"xgb_params": {"max_depth": 4}},
        "training_window": {"train_split_date": "2025-08-01"},
        "evaluation_metrics": {"mae": 0.91},
    }

    paths = training_job.save_artifacts_to_dir(
        output_dir=output_dir,
        pitcher_games=pitcher_games,
        model_df=model_df,
        model=FakeModel(),
        metadata=metadata,
    )

    saved_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    assert paths["metadata"].exists()
    assert saved_metadata == metadata


def test_promote_latest_to_previous_preserves_matching_metadata(tmp_path, monkeypatch):
    latest_dir = tmp_path / "artifacts" / "latest"
    previous_dir = tmp_path / "artifacts" / "previous"
    latest_dir.mkdir(parents=True, exist_ok=True)
    previous_dir.mkdir(parents=True, exist_ok=True)

    latest_paths = training_job.artifact_paths(latest_dir)
    latest_paths["model"].write_text("model-bytes", encoding="utf-8")
    latest_paths["pitcher_games"].write_text("game_date\n2026-04-19\n", encoding="utf-8")
    latest_paths["model_df"].write_text("game_date,strikeouts\n2026-04-19,7\n", encoding="utf-8")
    latest_paths["metadata"].write_text(
        '{"target": "strikeouts", "training_window": {"train_split_date": "2025-08-01"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(training_job, "LATEST_DIR", latest_dir)
    monkeypatch.setattr(training_job, "PREVIOUS_DIR", previous_dir)

    training_job.promote_latest_to_previous()

    previous_paths = training_job.artifact_paths(previous_dir)
    saved_metadata = json.loads(previous_paths["metadata"].read_text(encoding="utf-8"))

    assert previous_paths["model"].read_text(encoding="utf-8") == "model-bytes"
    assert saved_metadata["target"] == "strikeouts"
    assert saved_metadata["training_window"]["train_split_date"] == "2025-08-01"
