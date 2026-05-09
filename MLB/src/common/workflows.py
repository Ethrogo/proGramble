from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from common.identity import MarketIdentitySpec, ParticipantIdentitySpec
from odds.policy import PickRankingPolicy, PostablePickLimits

FeatureBuilder = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
Predictor = Callable[[object, pd.DataFrame], pd.DataFrame]
ArtifactDataLoader = Callable[[Path], pd.DataFrame]
ArtifactModelLoader = Callable[[Path], object]
PredictionMetadataAdjuster = Callable[[pd.DataFrame, dict | None], pd.DataFrame]


@dataclass(frozen=True)
class ProjectionOddsJoinKeys:
    projection: str
    odds: str


@dataclass(frozen=True)
class PropFieldSpec:
    prediction: str
    actual: str
    stddev: str = "std_dev"
    lower_bound: str = "lower_bound"
    upper_bound: str = "upper_bound"
    shared_prediction: str = "predicted_value"
    shared_actual: str = "actual_value"


@dataclass(frozen=True)
class WorkflowArtifactSpec:
    history_filename: str
    history_loader: ArtifactDataLoader
    model_filename: str
    model_loader: ArtifactModelLoader
    metadata_filename: str = "metadata.json"
    artifact_subdir: str | None = None


@dataclass(frozen=True)
class ModelingWorkflowSpec:
    prop_type: str
    sport: str
    participant_key: str
    market_key: str
    artifacts: WorkflowArtifactSpec
    feature_builder: FeatureBuilder
    predictor: Predictor
    projection_odds_join_keys: ProjectionOddsJoinKeys
    pick_ranking_policy: PickRankingPolicy
    prediction_columns: tuple[str, ...]
    prop_fields: PropFieldSpec = field(
        default_factory=lambda: PropFieldSpec(
            prediction="predicted_strikeouts",
            actual="actual_strikeouts",
        )
    )
    participant_identity: ParticipantIdentitySpec = field(default_factory=ParticipantIdentitySpec)
    market_identity: MarketIdentitySpec = field(default_factory=MarketIdentitySpec)
    prediction_metadata_adjuster: PredictionMetadataAdjuster | None = None
    postable_limits: PostablePickLimits = field(default_factory=PostablePickLimits)
    enabled_in_default_daily_card: bool = True

    def resolved_postable_limits(self) -> PostablePickLimits:
        return self.postable_limits
