# MLB/src/jobs/build_training_artifacts.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from pitcher_k.config import RAW_STATCAST_START, RAW_STATCAST_END
from pitcher_k.data_loader import load_statcast_data
from pitcher_k.preprocessing import add_outcome_flags
from pitcher_k.feature_engineering import (
    build_pitcher_game_table,
    add_pitcher_team_info,
    add_opponent_k_features,
    add_rolling_pitcher_features,
    add_rate_features,
)
from pitcher_k.feature_model import build_model_df
from pitcher_k.train import time_split, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

PITCHER_GAMES_PATH = ARTIFACTS_DIR / "pitcher_games.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.ubj"
MODEL_DF_PATH = ARTIFACTS_DIR / "model_df.csv"


def ensure_artifact_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_historical_pitcher_games() -> pd.DataFrame:
    sc = load_statcast_data(RAW_STATCAST_START, RAW_STATCAST_END)
    sc = add_outcome_flags(sc)

    pitcher_games = build_pitcher_game_table(sc)
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    pitcher_games = add_opponent_k_features(pitcher_games, sc)
    pitcher_games = add_rolling_pitcher_features(pitcher_games)
    pitcher_games = add_rate_features(pitcher_games)

    return pitcher_games


def train_pitcher_k_model(pitcher_games: pd.DataFrame):
    model_df = build_model_df(pitcher_games)
    train_df, test_df = time_split(model_df)
    train_output = train_model(train_df, test_df)
    return train_output["model"], model_df


def save_pitcher_games_artifact(pitcher_games: pd.DataFrame) -> Path:
    pitcher_games.to_csv(PITCHER_GAMES_PATH, index=False)
    return PITCHER_GAMES_PATH


def save_model_df_artifact(model_df: pd.DataFrame) -> Path:
    model_df.to_csv(MODEL_DF_PATH, index=False)
    return MODEL_DF_PATH


def save_model_artifact(model) -> Path:
    model.save_model(str(MODEL_PATH))
    return MODEL_PATH


def build_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    ensure_artifact_dir()

    pitcher_games = build_historical_pitcher_games()
    model, model_df = train_pitcher_k_model(pitcher_games)

    pitcher_games_path = save_pitcher_games_artifact(pitcher_games)
    model_path = save_model_artifact(model)
    save_model_df_artifact(model_df)

    return pitcher_games, model_df, pitcher_games_path, model_path


if __name__ == "__main__":
    pitcher_games, model_df, pitcher_games_path, model_path = build_training_artifacts()

    print("Saved training artifacts:")
    print(f"- pitcher_games: {pitcher_games_path}")
    print(f"- model: {model_path}")
    print(f"- pitcher_games shape: {pitcher_games.shape}")
    print(f"- model_df shape: {model_df.shape}")