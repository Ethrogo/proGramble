from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from common.contracts import validate_pitcher_games_contract
from common.workflows import ModelingWorkflowSpec, PropFieldSpec, ProjectionOddsJoinKeys, WorkflowArtifactSpec
from odds.policy import DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY, PostablePickLimits
from pitcher_bb.config import PITCHER_BB_PROP_MARKET
from pitcher_bb.feature_tomorrow import build_tomorrow_features
from pitcher_bb.predict import predict_on_dataframe


def load_pitcher_history_artifact(path: Path) -> pd.DataFrame:
    pitcher_games = pd.read_csv(path)
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
    validate_pitcher_games_contract(pitcher_games)
    return pitcher_games


def load_xgboost_model_artifact(path: Path):
    model = xgb.Booster()
    model.load_model(str(path))
    return model


def build_mlb_pitcher_walk_features(
    starters_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
) -> pd.DataFrame:
    as_of_date = pd.to_datetime(starters_df["game_date"]).min()
    team_context = build_team_context(pitcher_games, as_of_date=as_of_date)
    return build_tomorrow_features(
        slate_df=starters_df,
        pitcher_games=pitcher_games,
        team_context=team_context,
    )


def build_team_context(pitcher_games: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    df = pitcher_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    as_of_date = pd.to_datetime(as_of_date)
    df = df[df["game_date"] < as_of_date].copy()
    df = df.sort_values(["opponent_team", "game_date", "game_pk"])
    team_context = (
        df.groupby("opponent_team", as_index=False)
        .tail(1)[["opponent_team", "opp_walks_per_game_last10", "opp_bb_rate_last10"]]
        .drop_duplicates(subset=["opponent_team"])
        .reset_index(drop=True)
    )
    return team_context


def apply_pitcher_bb_metadata_uncertainty(
    today_preds: pd.DataFrame,
    metadata: dict | None,
) -> pd.DataFrame:
    if today_preds.empty:
        return today_preds

    interval_config = (metadata or {}).get("uncertainty_model", {})
    multiplier = float(interval_config.get("interval_multiplier", 1.0))
    adjusted = today_preds.copy()
    adjusted["raw_std_dev"] = pd.to_numeric(adjusted["std_dev"], errors="coerce").fillna(0.0).clip(lower=0.0)
    adjusted["std_dev"] = adjusted["raw_std_dev"] * multiplier
    adjusted["lower_bound"] = (adjusted["predicted_walks"] - adjusted["std_dev"]).clip(lower=0.0)
    adjusted["upper_bound"] = adjusted["predicted_walks"] + adjusted["std_dev"]
    if interval_config:
        adjusted["interval_coverage_target"] = float(interval_config.get("nominal_coverage", 0.8))
        adjusted["interval_multiplier"] = multiplier
    return adjusted


MLB_PITCHER_WALK_WORKFLOW = ModelingWorkflowSpec(
    prop_type="pitcher_bb",
    sport="MLB",
    participant_key="player_name",
    market_key=PITCHER_BB_PROP_MARKET,
    prop_fields=PropFieldSpec(
        prediction="predicted_walks",
        actual="actual_walks",
    ),
    artifacts=WorkflowArtifactSpec(
        history_filename="pitcher_games.csv",
        history_loader=load_pitcher_history_artifact,
        model_filename="model.ubj",
        model_loader=load_xgboost_model_artifact,
        artifact_subdir="pitcher_bb",
    ),
    feature_builder=build_mlb_pitcher_walk_features,
    predictor=predict_on_dataframe,
    projection_odds_join_keys=ProjectionOddsJoinKeys(
        projection="player_name_norm",
        odds="player_name_norm",
    ),
    pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    prediction_columns=(
        "player_name",
        "predicted_walks",
        "lower_bound",
        "upper_bound",
        "std_dev",
    ),
    prediction_metadata_adjuster=apply_pitcher_bb_metadata_uncertainty,
    postable_limits=PostablePickLimits(
        max_official=3,
        max_leans=1,
    ),
)
