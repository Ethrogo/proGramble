# MLB/src/jobs/run_daily_card.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from starters.today_starters import get_today_starters_df, save_today_starters_csv
from pitcher_k.feature_engineering import build_team_context
from pitcher_k.feature_tomorrow import build_tomorrow_features
from pitcher_k.predict import predict_on_dataframe

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

OUTPUT_DIR = DATA_DIR / "outputs"

PROJECTIONS_DIR = OUTPUT_DIR / "projections"
EDGES_DIR = OUTPUT_DIR / "edges"
PICKS_DIR = OUTPUT_DIR / "picks"


def ensure_output_dirs() -> None:
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_DIR.mkdir(parents=True, exist_ok=True)


def load_pitcher_games_artifact() -> pd.DataFrame:
    candidate_paths = [
        LATEST_ARTIFACTS_DIR / "pitcher_games.csv",
        PREVIOUS_ARTIFACTS_DIR / "pitcher_games.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            pitcher_games = pd.read_csv(path)
            pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
            validate_pitcher_games_contract(pitcher_games)
            print(f"Loaded pitcher_games artifact from: {path}")
            return pitcher_games

    raise FileNotFoundError(
        "Missing pitcher_games artifact in both latest/ and previous/."
    )


def load_model_artifact():
    candidate_paths = [
        LATEST_ARTIFACTS_DIR / "model.ubj",
        PREVIOUS_ARTIFACTS_DIR / "model.ubj",
    ]

    for path in candidate_paths:
        if path.exists():
            model = xgb.Booster()
            model.load_model(str(path))
            print(f"Loaded model artifact from: {path}")
            return model

    raise FileNotFoundError(
        "Missing model artifact in both latest/ and previous/."
    )


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


def run_daily_card() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()

    starters_df = get_today_starters_df()
    validate_starters_contract(starters_df)
    pitcher_games = load_pitcher_games_artifact()
    model = load_model_artifact()

    today_preds = build_today_predictions(
        starters_df=starters_df,
        pitcher_games=pitcher_games,
        model=model,
    )

    if today_preds.empty:
        raise ValueError("No today predictions were generated.")

    joined_df, _ = run_edge_pipeline(today_preds)
    validate_joined_odds_contract(joined_df)

    picks_df = build_daily_picks(joined_df)
    validate_final_picks_contract(picks_df)

    post_df = filter_postable_picks(picks_df, max_official=3, max_leans=1)
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