from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MarketRankingRule:
    side: str
    sort_by: tuple[str, ...]
    ascending: tuple[bool, ...]


@dataclass(frozen=True)
class PostablePickLimits:
    max_official: int = 4
    max_leans: int = 2


@dataclass(frozen=True)
class PickRankingPolicy:
    version: str
    official_edge_threshold: float
    lean_edge_threshold: float
    confidence_tier_thresholds: tuple[tuple[str, float], ...]
    confidence_default_tier: str
    market_ranking_rules: tuple[MarketRankingRule, ...]
    edge_tie_preference: str
    pick_type_order: tuple[str, ...]
    risk_tier_thresholds: tuple[tuple[str, float], ...] = (
        ("low", 0.58),
        ("medium", 0.50),
    )
    risk_default_tier: str = "high"
    market_selection_metric_column: str | None = None
    side_choice_metric_column: str = "edge"
    pick_type_metric_column: str = "edge"
    confidence_metric_column: str = "value_score"
    ranking_metric_column: str = "adjusted_value_score"
    postable_limits: PostablePickLimits = field(default_factory=PostablePickLimits)

    def classify_pick_type(self, edge: float) -> str:
        abs_edge = abs(edge)

        if abs_edge >= self.official_edge_threshold:
            return "official"
        if abs_edge >= self.lean_edge_threshold:
            return "lean"
        return "pass"

    def classify_confidence_tier(self, value_score: float) -> str:
        for tier_name, threshold in self.confidence_tier_thresholds:
            if value_score >= threshold:
                return tier_name
        return self.confidence_default_tier

    def classify_risk_tier(self, implied_probability: float) -> str:
        for tier_name, threshold in self.risk_tier_thresholds:
            if implied_probability >= threshold:
                return tier_name
        return self.risk_default_tier

    def select_best_market(self, player_df: pd.DataFrame, side: str) -> pd.Series:
        rule = self._market_rule_for_side(side)
        side_df = player_df[player_df["side_norm"] == rule.side].copy()
        if side_df.empty:
            return pd.Series(dtype="object")

        sort_by = list(rule.sort_by)
        ascending = list(rule.ascending)
        if self.market_selection_metric_column and self.market_selection_metric_column in side_df.columns:
            sort_by = [self.market_selection_metric_column, *sort_by]
            ascending = [False, *ascending]

        side_df = side_df.sort_values(by=sort_by, ascending=ascending)
        return side_df.iloc[0]

    def choose_pick_side(
        self,
        *,
        best_over: pd.Series,
        best_under: pd.Series,
        predicted: float,
    ) -> pd.Series:
        over_edge = None
        under_edge = None

        if not best_over.empty:
            over_edge = predicted - float(best_over["line"])

        if not best_under.empty:
            under_edge = float(best_under["line"]) - predicted

        over_metric = self._resolve_side_choice_metric(best_over, fallback=over_edge)
        under_metric = self._resolve_side_choice_metric(best_under, fallback=under_edge)

        if over_metric is None and under_metric is None:
            return pd.Series(dtype="object")

        if over_metric is None:
            return self._finalize_choice(best_under, edge=under_edge, pick_side="under")

        if under_metric is None:
            return self._finalize_choice(best_over, edge=over_edge, pick_side="over")

        if over_metric > under_metric:
            return self._finalize_choice(best_over, edge=over_edge, pick_side="over")

        if under_metric > over_metric:
            return self._finalize_choice(best_under, edge=under_edge, pick_side="under")

        if self.edge_tie_preference == "under":
            return self._finalize_choice(best_under, edge=under_edge, pick_side="under")

        return self._finalize_choice(best_over, edge=over_edge, pick_side="over")

    def sort_picks(self, picks_df: pd.DataFrame) -> pd.DataFrame:
        order_lookup = {
            pick_type: rank
            for rank, pick_type in enumerate(self.pick_type_order)
        }
        picks = picks_df.copy()
        picks["pick_type_order"] = picks["pick_type"].map(order_lookup).fillna(99)
        ranking_column = self.ranking_metric_column
        if ranking_column not in picks.columns:
            ranking_column = "adjusted_value_score" if "adjusted_value_score" in picks.columns else "value_score"
        return (
            picks.sort_values(
                by=["pick_type_order", ranking_column],
                ascending=[True, False],
            )
            .drop(columns=["pick_type_order"])
            .reset_index(drop=True)
        )

    def resolved_postable_limits(
        self,
        *,
        max_official: int | None = None,
        max_leans: int | None = None,
    ) -> PostablePickLimits:
        return PostablePickLimits(
            max_official=(
                self.postable_limits.max_official
                if max_official is None
                else max_official
            ),
            max_leans=(
                self.postable_limits.max_leans
                if max_leans is None
                else max_leans
            ),
        )

    def _market_rule_for_side(self, side: str) -> MarketRankingRule:
        for rule in self.market_ranking_rules:
            if rule.side == side:
                return rule
        raise ValueError(f"No market ranking rule configured for side '{side}'.")

    def _resolve_side_choice_metric(
        self,
        row: pd.Series,
        *,
        fallback: float | None,
    ) -> float | None:
        if row.empty:
            return fallback
        if self.side_choice_metric_column in row and pd.notna(row[self.side_choice_metric_column]):
            return float(row[self.side_choice_metric_column])
        return fallback

    @staticmethod
    def _finalize_choice(row: pd.Series, *, edge: float, pick_side: str) -> pd.Series:
        chosen = row.copy()
        chosen["edge"] = edge
        chosen["pick_side"] = pick_side
        return chosen


DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY = PickRankingPolicy(
    version="mlb_pitcher_props_policy_v1",
    official_edge_threshold=0.75,
    lean_edge_threshold=0.40,
    confidence_tier_thresholds=(
        ("high", 0.50),
        ("medium", 0.30),
        ("low", 0.15),
    ),
    confidence_default_tier="thin",
    risk_tier_thresholds=(
        ("low", 0.58),
        ("medium", 0.50),
    ),
    risk_default_tier="high",
    market_ranking_rules=(
        MarketRankingRule(
            side="over",
            sort_by=("line", "price_sort_key"),
            ascending=(True, False),
        ),
        MarketRankingRule(
            side="under",
            sort_by=("line", "price_sort_key"),
            ascending=(False, False),
        ),
    ),
    edge_tie_preference="over",
    pick_type_order=("official", "lean", "pass"),
    postable_limits=PostablePickLimits(
        max_official=4,
        max_leans=2,
    ),
)

DEFAULT_MLB_PITCHER_WALKS_POLICY = PickRankingPolicy(
    version="mlb_pitcher_walks_ev_policy_v1",
    official_edge_threshold=0.08,
    lean_edge_threshold=0.02,
    confidence_tier_thresholds=(
        ("high", 0.08),
        ("medium", 0.04),
        ("low", 0.02),
    ),
    confidence_default_tier="thin",
    risk_tier_thresholds=(
        ("low", 0.58),
        ("medium", 0.50),
    ),
    risk_default_tier="high",
    market_ranking_rules=(
        MarketRankingRule(
            side="over",
            sort_by=("line", "price_sort_key"),
            ascending=(True, False),
        ),
        MarketRankingRule(
            side="under",
            sort_by=("line", "price_sort_key"),
            ascending=(False, False),
        ),
    ),
    edge_tie_preference="over",
    pick_type_order=("official", "lean", "pass"),
    market_selection_metric_column="selection_expected_return",
    side_choice_metric_column="selection_expected_return",
    pick_type_metric_column="expected_return",
    confidence_metric_column="expected_return",
    ranking_metric_column="adjusted_expected_return",
    postable_limits=PostablePickLimits(
        max_official=3,
        max_leans=1,
    ),
)
