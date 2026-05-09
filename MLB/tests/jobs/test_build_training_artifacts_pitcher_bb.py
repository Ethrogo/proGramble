import json

import pandas as pd
import pytest

from jobs import build_training_artifacts_pitcher_bb as training_job


class FakeModel:
    def save_model(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fake-model")


class FakePredictModel:
    def predict(self, dmatrix):
        if dmatrix == "train":
            return pd.Series([1.8, 2.2], dtype="float64")
        if dmatrix == "test":
            return pd.Series([2.4, 2.9], dtype="float64")
        raise AssertionError(f"Unexpected dmatrix: {dmatrix}")


def test_save_artifacts_to_dir_writes_pitcher_bb_metadata(tmp_path):
    output_dir = tmp_path / "artifacts"
    pitcher_games = pd.DataFrame([{"game_date": "2026-04-19", "pitcher": 1, "walks": 2}])
    model_df = pd.DataFrame([{"game_date": "2026-04-19", "walks": 2}])
    historical_lines_df = pd.DataFrame()
    metadata = {
        "target": "walks",
        "features": ["walks_last3"],
        "model_params": {"xgb_params": {"max_depth": 4}},
        "training_window": {"train_split_date": "2025-08-01"},
        "evaluation_metrics": {"regression": {"mae": 0.5}},
    }

    paths = training_job.save_artifacts_to_dir(
        output_dir=output_dir,
        pitcher_games=pitcher_games,
        model_df=model_df,
        historical_lines_df=historical_lines_df,
        model=FakeModel(),
        metadata=metadata,
    )

    saved_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert paths["metadata"].exists()
    assert saved_metadata == metadata


def test_build_training_metadata_uses_walk_target_and_features():
    model_df = pd.DataFrame(
        [
            {"game_date": "2025-07-30", "walks": 2, "walks_stddev_last10": 0.6},
            {"game_date": "2025-07-31", "walks": 3, "walks_stddev_last10": 0.7},
            {"game_date": "2025-08-02", "walks": 2, "walks_stddev_last10": 0.5},
            {"game_date": "2025-08-03", "walks": 4, "walks_stddev_last10": 0.9},
        ]
    )
    train_df = pd.DataFrame(
        [
            {"game_date": "2025-07-30", "walks": 2, "walks_stddev_last10": 0.6},
            {"game_date": "2025-07-31", "walks": 3, "walks_stddev_last10": 0.7},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"game_date": "2025-08-02", "walks": 2, "walks_stddev_last10": 0.5},
            {"game_date": "2025-08-03", "walks": 4, "walks_stddev_last10": 0.9},
        ]
    )
    train_output = {
        "model": FakePredictModel(),
        "dtrain": "train",
        "dtest": "test",
        "X_train": pd.DataFrame([{"walks_last3": 1.5}, {"walks_last3": 2.0}]),
        "X_test": pd.DataFrame([{"walks_last3": 2.2}, {"walks_last3": 2.7}]),
        "y_train": pd.Series([2.0, 3.0], dtype="float64"),
        "y_test": pd.Series([2.0, 4.0], dtype="float64"),
    }

    metadata = training_job.build_training_metadata(
        model_df=model_df,
        train_df=train_df,
        test_df=test_df,
        train_output=train_output,
        historical_lines_df=pd.DataFrame(),
    )

    assert metadata["target"] == "walks"
    assert "walks_last3" not in metadata["features"] or isinstance(metadata["features"], list)
    assert metadata["uncertainty_model"]["base_stddev_column"] == "walks_stddev_last10"
    assert metadata["evaluation_metrics"]["bucketed_error"]["bucket_by"] == "predicted_walks"
    assert metadata["evaluation_metrics"]["workflow_backtest"]["available"] is False


def test_train_pitcher_bb_model_filters_to_starter_like_appearances(monkeypatch):
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2025-07-30",
                "game_pk": 1,
                "pitcher": 111,
                "player_name": "Starter One",
                "walks": 2,
                "pitches": 91,
                "batters_faced": 25,
                "pitches_last3": 92.0,
                "pitches_last10": 94.0,
                "batters_faced_last3": 24.0,
                "batters_faced_last10": 25.0,
                "walks_last3": 2.0,
                "walks_last10": 2.1,
                "avg_velo_last3": 97.1,
                "avg_spin_last3": 2450.0,
                "bb_per_pitch_last10": 0.02,
                "bb_rate_last10": 0.08,
                "opp_walks_per_game_last10": 3.2,
                "opp_bb_rate_last10": 0.09,
                "walks_stddev_last10": 0.6,
                "walks_p25_last10": 1.5,
                "walks_p75_last10": 2.5,
            },
            {
                "game_date": "2025-08-02",
                "game_pk": 2,
                "pitcher": 111,
                "player_name": "Starter One",
                "walks": 1,
                "pitches": 19,
                "batters_faced": 5,
                "pitches_last3": 19.0,
                "pitches_last10": 19.0,
                "batters_faced_last3": 5.0,
                "batters_faced_last10": 5.0,
                "walks_last3": 1.0,
                "walks_last10": 1.0,
                "avg_velo_last3": 96.0,
                "avg_spin_last3": 2430.0,
                "bb_per_pitch_last10": 0.05,
                "bb_rate_last10": 0.20,
                "opp_walks_per_game_last10": 2.8,
                "opp_bb_rate_last10": 0.07,
                "walks_stddev_last10": 0.2,
                "walks_p25_last10": 1.0,
                "walks_p75_last10": 1.0,
            },
        ]
    )

    captured = {}

    def fake_train_model(train_df, test_df):
        captured["train_df"] = train_df.copy()
        captured["test_df"] = test_df.copy()
        return {
            "model": FakePredictModel(),
            "dtrain": "train",
            "dtest": "test",
            "X_train": pd.DataFrame([{"walks_last3": 2.0, "walks_stddev_last10": 0.6}]),
            "X_test": pd.DataFrame([{"walks_last3": 2.1, "walks_stddev_last10": 0.7}]),
            "y_train": pd.Series([2.0], dtype="float64"),
            "y_test": pd.Series([3.0], dtype="float64"),
        }

    def fake_time_split(model_df):
        captured["model_df"] = model_df.copy()
        return model_df.iloc[:1].copy(), model_df.iloc[:1].copy()

    monkeypatch.setattr(training_job, "time_split", fake_time_split)
    monkeypatch.setattr(training_job, "train_model", fake_train_model)
    monkeypatch.setattr(training_job, "build_training_metadata", lambda **kwargs: {"ok": True})

    _, model_df, metadata = training_job.train_pitcher_bb_model(
        pitcher_games,
        historical_lines_df=pd.DataFrame(),
    )

    assert metadata == {"ok": True}
    assert len(model_df) == 1
    assert model_df["game_pk"].tolist() == [1]
    assert captured["model_df"]["game_pk"].tolist() == [1]


def test_train_pitcher_bb_model_raises_when_starter_filtering_leaves_empty_held_out_split():
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2025-08-03",
                "game_pk": 2,
                "pitcher": 111,
                "player_name": "Starter One",
                "walks": 1,
                "pitches": 19,
                "batters_faced": 5,
                "pitches_last3": 19.0,
                "pitches_last10": 19.0,
                "batters_faced_last3": 5.0,
                "batters_faced_last10": 5.0,
                "walks_last3": 1.0,
                "walks_last10": 1.0,
                "avg_velo_last3": 96.0,
                "avg_spin_last3": 2430.0,
                "bb_per_pitch_last10": 0.05,
                "bb_rate_last10": 0.20,
                "opp_walks_per_game_last10": 2.8,
                "opp_bb_rate_last10": 0.07,
                "walks_stddev_last10": 0.2,
                "walks_p25_last10": 1.0,
                "walks_p75_last10": 1.0,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="Starter-like filtering produced an empty train/test split",
    ):
        training_job.train_pitcher_bb_model(
            pitcher_games,
            historical_lines_df=pd.DataFrame(),
        )
