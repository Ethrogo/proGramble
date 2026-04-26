# MLB/src/jobs/run_daily_card.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import xgboost as xgb

from starters.today_starters import get_today_starters_df, save_today_starters_csv
from pitcher_k.feature_engineering import build_team_context
from pitcher_k.feature_tomorrow import build_tomorrow_features
from pitcher_k.predict import predict_on_dataframe
from pitcher_k.config import PITCHER_K_PROP_MARKET

from odds.run_edges import run_edge_pipeline
from odds.create_picks import build_daily_picks, filter_postable_picks
from common.contracts import (
    validate_starters_contract,
    validate_pitcher_games_contract,
    validate_joined_odds_contract,
    validate_final_picks_contract,
    assert_non_empty,
    require_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
LATEST_ARTIFACTS_DIR = ARTIFACTS_DIR / "latest"
PREVIOUS_ARTIFACTS_DIR = ARTIFACTS_DIR / "previous"

MODEL_PATH = LATEST_ARTIFACTS_DIR / "model.ubj"
PITCHER_GAMES_PATH = LATEST_ARTIFACTS_DIR / "pitcher_games.csv"
METADATA_FILENAME = "metadata.json"

OUTPUT_DIR = DATA_DIR / "outputs"

PROJECTIONS_DIR = OUTPUT_DIR / "projections"
EDGES_DIR = OUTPUT_DIR / "edges"
PICKS_DIR = OUTPUT_DIR / "picks"

BuildPicksFn = Callable[[pd.DataFrame], pd.DataFrame]
FilterPostablePicksFn = Callable[[pd.DataFrame], pd.DataFrame]


def ensure_output_dirs() -> None:
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_artifact_path(filename: str) -> Path:
    candidate_paths = [
        LATEST_ARTIFACTS_DIR / filename,
        PREVIOUS_ARTIFACTS_DIR / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing {filename} artifact in both latest/ and previous/."
    )


def load_pitcher_games_artifact() -> pd.DataFrame:
    path = resolve_artifact_path("pitcher_games.csv")
    pitcher_games = pd.read_csv(path)
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
    validate_pitcher_games_contract(pitcher_games)
    print(f"Loaded pitcher_games artifact from: {path}")
    return pitcher_games


def load_model_artifact():
    path = resolve_artifact_path("model.ubj")
    model = xgb.Booster()
    model.load_model(str(path))
    print(f"Loaded model artifact from: {path}")
    return model


def load_model_metadata() -> dict:
    model_path = resolve_artifact_path("model.ubj")
    metadata_path = model_path.with_name(METADATA_FILENAME)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata artifact paired with model: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Loaded model metadata from: {metadata_path}")
    print("Model metadata:")
    print(json.dumps(metadata, indent=2))
    return metadata


def build_today_predictions(starters_df: pd.DataFrame, pitcher_games: pd.DataFrame, model):
    validate_starters_contract(starters_df)
    validate_pitcher_games_contract(pitcher_games)

    as_of_date = starters_df["game_date"].min()
    team_context = build_team_context(pitcher_games, as_of_date=as_of_date)

    today_features = build_tomorrow_features(
        slate_df=starters_df,
        pitcher_games=pitcher_games,
        team_context=team_context,
    )

    if today_features.empty:
        return today_features

    today_preds = predict_on_dataframe(model, today_features)
    assert_non_empty(today_preds, "today_preds")
    require_columns(
        today_preds,
        ["player_name", "predicted_strikeouts"],
        "today_preds",
    )
    return today_preds


def save_outputs(
    starters_df: pd.DataFrame,
    today_preds: pd.DataFrame,
    joined_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> None:
    save_today_starters_csv(starters_df)

    today_preds.to_csv(PROJECTIONS_DIR / "today_projections.csv", index=False)
    joined_df.to_csv(EDGES_DIR / "today_joined_edges.csv", index=False)
    picks_df.to_csv(PICKS_DIR / "today_all_picks.csv", index=False)
    post_df.to_csv(PICKS_DIR / "today_postable_picks.csv", index=False)


def run_daily_card(
    *,
    market: str = PITCHER_K_PROP_MARKET,
    build_picks_fn: BuildPicksFn | None = None,
    filter_postable_picks_fn: FilterPostablePicksFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()

    if build_picks_fn is None:
        build_picks_fn = build_daily_picks

    if filter_postable_picks_fn is None:
        def filter_postable_picks_fn(picks_df: pd.DataFrame) -> pd.DataFrame:
            return filter_postable_picks(
                picks_df,
                max_official=3,
                max_leans=1,
            )

    starters_df = get_today_starters_df()
    validate_starters_contract(starters_df)
    pitcher_games = load_pitcher_games_artifact()
    model = load_model_artifact()
    load_model_metadata()

    today_preds = build_today_predictions(
        starters_df=starters_df,
        pitcher_games=pitcher_games,
        model=model,
    )

    if today_preds.empty:
        raise ValueError("No today predictions were generated.")

    joined_df, _ = run_edge_pipeline(today_preds, market)
    validate_joined_odds_contract(joined_df)

    picks_df = build_picks_fn(joined_df)
    validate_final_picks_contract(picks_df)

    post_df = filter_postable_picks_fn(picks_df)
    validate_final_picks_contract(post_df)

    save_outputs(
        starters_df=starters_df,
        today_preds=today_preds,
        joined_df=joined_df,
        picks_df=picks_df,
        post_df=post_df,
    )

    return starters_df, today_preds, picks_df, post_df


if __name__ == "__main__":
    _, _, picks_df, post_df = run_daily_card()

    print("\nTop postable picks:")
    if post_df.empty:
        print("No postable picks found.")
    else:
        print(
            post_df[
                [
                    "player_name",
                    "book",
                    "pick_side",
                    "line",
                    "price",
                    "predicted_strikeouts",
                    "edge",
                    "pick_type",
                ]
            ].to_string(index=False)
        )
