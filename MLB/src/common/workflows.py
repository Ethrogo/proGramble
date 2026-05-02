from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

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
class WorkflowArtifactSpec:
    history_filename: str
    history_loader: ArtifactDataLoader
    model_filename: str
    model_loader: ArtifactModelLoader
    metadata_filename: str = "metadata.json"


@dataclass(frozen=True)
class ModelingWorkflowSpec:
    sport: str
    participant_key: str
    market_key: str
    artifacts: WorkflowArtifactSpec
    feature_builder: FeatureBuilder
    predictor: Predictor
    projection_odds_join_keys: ProjectionOddsJoinKeys
    pick_ranking_policy: PickRankingPolicy
    prediction_columns: tuple[str, ...]
    prediction_metadata_adjuster: PredictionMetadataAdjuster | None = None
    postable_limits: PostablePickLimits = field(default_factory=PostablePickLimits)

    def resolved_postable_limits(self) -> PostablePickLimits:
        return self.postable_limits
