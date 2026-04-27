import json

import pandas as pd

from jobs import build_training_artifacts as training_job
from odds.historical_lines import empty_historical_lines_df


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


class FakeSingleTestPredictModel:
    def predict(self, dmatrix):
        if dmatrix == "train":
            return pd.Series([4.8, 6.2], dtype="float64")
        if dmatrix == "test":
            return pd.Series([7.3], dtype="float64")
        raise AssertionError(f"Unexpected dmatrix: {dmatrix}")


def test_save_artifacts_to_dir_writes_metadata_json(tmp_path):
    output_dir = tmp_path / "artifacts"
    pitcher_games = pd.DataFrame([{"game_date": "2026-04-19", "pitcher": 1}])
    model_df = pd.DataFrame([{"game_date": "2026-04-19", "strikeouts": 7}])
    historical_lines_df = empty_historical_lines_df()
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
        historical_lines_df=historical_lines_df,
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
    latest_paths["historical_lines"].write_text(
        ",".join(empty_historical_lines_df().columns) + "\n",
        encoding="utf-8",
    )
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


def test_build_training_metadata_uses_native_historical_lines_for_real_backtest():
    model_df = pd.DataFrame(
        [
            {"game_date": "2025-07-30", "strikeouts": 5},
            {"game_date": "2025-08-02", "strikeouts": 7},
        ]
    )
    train_df = pd.DataFrame([{"game_date": "2025-07-30", "strikeouts": 5}])
    test_df = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob deGrom",
                "strikeouts": 7,
            }
        ]
    )
    train_output = {
        "model": FakeSingleTestPredictModel(),
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
            ]
        ),
        "y_train": pd.Series([5.0, 6.0], dtype="float64"),
        "y_test": pd.Series([7.0], dtype="float64"),
    }
    historical_lines_df = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 6.5,
                "price": -120,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "selection_rule": "latest_pregame_snapshot_per_game_player_book_side",
                "source": "fixture",
                "pulled_at": "2025-08-02T22:50:00Z",
                "snapshot_type": "selected",
                "is_closing_line": True,
                "snapshot_rank": 1,
            }
        ]
    )

    metadata = training_job.build_training_metadata(
        model_df=model_df,
        train_df=train_df,
        test_df=test_df,
        train_output=train_output,
        historical_lines_df=historical_lines_df,
    )

    workflow_backtest = metadata["evaluation_metrics"]["workflow_backtest"]
    assert workflow_backtest["available"] is True
    assert workflow_backtest["overall"][0]["picks"] == 1
    assert workflow_backtest["by_book"][0]["book"] == "DraftKings"


def test_train_pitcher_k_model_filters_to_starter_like_appearances(monkeypatch):
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2025-07-30",
                "game_pk": 1,
                "pitcher": 111,
                "player_name": "Starter One",
                "strikeouts": 6,
                "pitches": 91,
                "batters_faced": 25,
                "pitches_last3": 92.0,
                "pitches_last10": 94.0,
                "whiff_per_pitch_last3": 0.14,
                "avg_velo_last3": 97.1,
                "avg_spin_last3": 2450.0,
                "k_per_pitch_last10": 0.08,
                "k_rate_last10": 0.28,
                "opp_strikeouts_per_game_last10": 9.0,
                "opp_k_rate_last10": 0.25,
            },
            {
                "game_date": "2025-08-02",
                "game_pk": 2,
                "pitcher": 111,
                "player_name": "Starter One",
                "strikeouts": 1,
                "pitches": 19,
                "batters_faced": 5,
                "pitches_last3": 19.0,
                "pitches_last10": 19.0,
                "whiff_per_pitch_last3": 0.08,
                "avg_velo_last3": 96.0,
                "avg_spin_last3": 2430.0,
                "k_per_pitch_last10": 0.04,
                "k_rate_last10": 0.20,
                "opp_strikeouts_per_game_last10": 8.4,
                "opp_k_rate_last10": 0.23,
            },
        ]
    )
    historical_lines_df = empty_historical_lines_df()

    captured = {}

    def fake_train_model(train_df, test_df):
        captured["train_df"] = train_df.copy()
        captured["test_df"] = test_df.copy()
        return {
            "model": FakePredictModel(),
            "dtrain": "train",
            "dtest": "test",
            "X_train": pd.DataFrame([{"pitches_last3": 92.0, "strikeouts_stddev_last10": 1.1}]),
            "X_test": pd.DataFrame([{"pitches_last3": 94.0, "strikeouts_stddev_last10": 1.2}]),
            "y_train": pd.Series([6.0], dtype="float64"),
            "y_test": pd.Series([7.0], dtype="float64"),
        }

    def fake_time_split(model_df):
        captured["model_df"] = model_df.copy()
        return model_df.iloc[:1].copy(), model_df.iloc[:1].copy()

    monkeypatch.setattr(training_job, "time_split", fake_time_split)
    monkeypatch.setattr(training_job, "train_model", fake_train_model)
    monkeypatch.setattr(training_job, "build_training_metadata", lambda **kwargs: {"ok": True})

    _, model_df, metadata = training_job.train_pitcher_k_model(
        pitcher_games,
        historical_lines_df=historical_lines_df,
    )

    assert metadata == {"ok": True}
    assert len(model_df) == 1
    assert model_df["game_pk"].tolist() == [1]
    assert captured["model_df"]["game_pk"].tolist() == [1]
