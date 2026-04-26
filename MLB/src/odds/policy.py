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
    official_edge_threshold: float
    lean_edge_threshold: float
    confidence_tier_thresholds: tuple[tuple[str, float], ...]
    confidence_default_tier: str
    market_ranking_rules: tuple[MarketRankingRule, ...]
    edge_tie_preference: str
    pick_type_order: tuple[str, ...]
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

    def select_best_market(self, player_df: pd.DataFrame, side: str) -> pd.Series:
        rule = self._market_rule_for_side(side)
        side_df = player_df[player_df["side_norm"] == rule.side].copy()
        if side_df.empty:
            return pd.Series(dtype="object")

        side_df = side_df.sort_values(
            by=list(rule.sort_by),
            ascending=list(rule.ascending),
        )
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

        if over_edge is None and under_edge is None:
            return pd.Series(dtype="object")

        if over_edge is None:
            return self._finalize_choice(best_under, edge=under_edge, pick_side="under")

        if under_edge is None:
            return self._finalize_choice(best_over, edge=over_edge, pick_side="over")

        if over_edge > under_edge:
            return self._finalize_choice(best_over, edge=over_edge, pick_side="over")

        if under_edge > over_edge:
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
        return (
            picks.sort_values(
                by=["pick_type_order", "value_score"],
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

    @staticmethod
    def _finalize_choice(row: pd.Series, *, edge: float, pick_side: str) -> pd.Series:
        chosen = row.copy()
        chosen["edge"] = edge
        chosen["pick_side"] = pick_side
        return chosen


DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY = PickRankingPolicy(
    official_edge_threshold=0.75,
    lean_edge_threshold=0.40,
    confidence_tier_thresholds=(
        ("high", 0.50),
        ("medium", 0.30),
        ("low", 0.15),
    ),
    confidence_default_tier="thin",
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
