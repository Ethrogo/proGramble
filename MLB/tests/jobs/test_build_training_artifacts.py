import json

import pandas as pd

from jobs import build_training_artifacts as training_job


class FakeModel:
    def save_model(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fake-model")


class FakePredictModel:
    def predict(self, dmatrix):
        if dmatrix == "train":
            return pd.Series([4.8, 6.2], dtype="float64")
        if dmatrix == "test":
            return pd.Series([5.4, 7.1], dtype="float64")
        raise AssertionError(f"Unexpected dmatrix: {dmatrix}")


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


def test_build_training_metadata_includes_richer_evaluation_sections():
    model_df = pd.DataFrame(
        [
            {"game_date": "2025-07-30", "strikeouts": 5},
            {"game_date": "2025-08-02", "strikeouts": 7},
        ]
    )
    train_df = pd.DataFrame([{"game_date": "2025-07-30", "strikeouts": 5}])
    test_df = pd.DataFrame([{"game_date": "2025-08-02", "strikeouts": 7}])
    train_output = {
        "model": FakePredictModel(),
        "dtrain": "train",
        "dtest": "test",
        "X_train": pd.DataFrame(
            [
                {"pitches_last3": 90.0, "strikeouts_stddev_last10": 1.1},
                {"pitches_last3": 95.0, "strikeouts_stddev_last10": 1.4},
            ]
        ),
        "X_test": pd.DataFrame(
            [
                {"pitches_last3": 92.0, "strikeouts_stddev_last10": 1.2},
                {"pitches_last3": 98.0, "strikeouts_stddev_last10": 1.5},
            ]
        ),
        "y_train": pd.Series([5.0, 6.0], dtype="float64"),
        "y_test": pd.Series([5.0, 8.0], dtype="float64"),
    }

    metadata = training_job.build_training_metadata(
        model_df=model_df,
        train_df=train_df,
        test_df=test_df,
        train_output=train_output,
    )

    evaluation = metadata["evaluation_metrics"]

    assert "regression" in evaluation
    assert "bucketed_error" in evaluation
    assert "uncertainty" in evaluation
    assert "workflow_backtest" in evaluation
    assert evaluation["workflow_backtest"]["available"] is False
    assert metadata["uncertainty_model"]["interval_multiplier"] > 0
    assert "documented_interpretation" in metadata["uncertainty_model"]
