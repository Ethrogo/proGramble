from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from common.contracts import validate_pitcher_games_contract
from common.workflows import (
    ModelingWorkflowSpec,
    ProjectionOddsJoinKeys,
    WorkflowArtifactSpec,
)
from odds.policy import DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY, PostablePickLimits
from pitcher_k.config import PITCHER_K_PROP_MARKET
from pitcher_k.feature_engineering import build_team_context
from pitcher_k.feature_tomorrow import build_tomorrow_features
from pitcher_k.predict import predict_on_dataframe


def load_pitcher_history_artifact(path: Path) -> pd.DataFrame:
    pitcher_games = pd.read_csv(path)
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
    validate_pitcher_games_contract(pitcher_games)
    return pitcher_games


def load_xgboost_model_artifact(path: Path):
    model = xgb.Booster()
    model.load_model(str(path))
    return model


def build_mlb_pitcher_strikeout_features(
    starters_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
) -> pd.DataFrame:
    as_of_date = starters_df["game_date"].min()
    team_context = build_team_context(pitcher_games, as_of_date=as_of_date)
    return build_tomorrow_features(
        slate_df=starters_df,
        pitcher_games=pitcher_games,
        team_context=team_context,
    )


MLB_PITCHER_STRIKEOUT_WORKFLOW = ModelingWorkflowSpec(
    sport="MLB",
    participant_key="player_name",
    market_key=PITCHER_K_PROP_MARKET,
    artifacts=WorkflowArtifactSpec(
        history_filename="pitcher_games.csv",
        history_loader=load_pitcher_history_artifact,
        model_filename="model.ubj",
        model_loader=load_xgboost_model_artifact,
    ),
    feature_builder=build_mlb_pitcher_strikeout_features,
    predictor=predict_on_dataframe,
    projection_odds_join_keys=ProjectionOddsJoinKeys(
        projection="player_name_norm",
        odds="player_name_norm",
    ),
    pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    prediction_columns=(
        "player_name",
        "predicted_strikeouts",
        "lower_bound",
        "upper_bound",
        "std_dev",
    ),
    postable_limits=PostablePickLimits(
        max_official=3,
        max_leans=1,
    ),
)
