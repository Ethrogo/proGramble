from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from odds.policy import (
    DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    PickRankingPolicy,
    PostablePickLimits,
)
from pitcher_k.config import PITCHER_K_PROP_MARKET
from pitcher_k.feature_engineering import build_team_context
from pitcher_k.feature_tomorrow import build_tomorrow_features
from pitcher_k.predict import predict_on_dataframe

FeatureBuilder = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
Predictor = Callable[[object, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ProjectionOddsJoinKeys:
    projection: str
    odds: str


@dataclass(frozen=True)
class ModelingWorkflowSpec:
    sport: str
    participant_key: str
    market_key: str
    feature_builder: FeatureBuilder
    predictor: Predictor
    projection_odds_join_keys: ProjectionOddsJoinKeys
    pick_ranking_policy: PickRankingPolicy
    postable_limits: PostablePickLimits = field(default_factory=PostablePickLimits)

    def resolved_postable_limits(self) -> PostablePickLimits:
        return self.postable_limits


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
    feature_builder=build_mlb_pitcher_strikeout_features,
    predictor=predict_on_dataframe,
    projection_odds_join_keys=ProjectionOddsJoinKeys(
        projection="player_name_norm",
        odds="player_name_norm",
    ),
    pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    postable_limits=PostablePickLimits(
        max_official=3,
        max_leans=1,
    ),
)
