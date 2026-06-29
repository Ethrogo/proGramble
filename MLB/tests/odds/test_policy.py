import pandas as pd

from odds.policy import (
    DEFAULT_MLB_PITCHER_WALKS_POLICY,
    DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    MarketRankingRule,
    PickRankingPolicy,
    PostablePickLimits,
)


def test_default_policy_preserves_thresholds_and_caps():
    policy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY

    assert policy.version == "mlb_pitcher_props_policy_v1"
    assert policy.classify_pick_type(0.75) == "official"
    assert policy.classify_pick_type(0.40) == "lean"
    assert policy.classify_pick_type(0.39) == "pass"
    assert policy.classify_pick_type(-0.75) == "pass"
    assert policy.classify_confidence_tier(0.50) == "high"
    assert policy.classify_confidence_tier(0.30) == "medium"
    assert policy.classify_confidence_tier(0.15) == "low"
    assert policy.classify_confidence_tier(0.14) == "thin"

    limits = policy.resolved_postable_limits()
    assert limits.max_official == 4
    assert limits.max_leans == 2


def test_default_policy_preserves_market_tie_breakers():
    policy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY
    player_df = pd.DataFrame(
        [
            {"side_norm": "over", "line": 5.5, "price_sort_key": -120},
            {"side_norm": "over", "line": 5.5, "price_sort_key": 100},
            {"side_norm": "over", "line": 6.5, "price_sort_key": 150},
            {"side_norm": "under", "line": 5.5, "price_sort_key": -110},
            {"side_norm": "under", "line": 6.5, "price_sort_key": -125},
            {"side_norm": "under", "line": 6.5, "price_sort_key": 105},
        ]
    )

    best_over = policy.select_best_market(player_df, "over")
    best_under = policy.select_best_market(player_df, "under")

    assert best_over["line"] == 5.5
    assert best_over["price_sort_key"] == 100
    assert best_under["line"] == 6.5
    assert best_under["price_sort_key"] == 105


def test_policy_can_change_edge_tie_break_preference_and_postable_caps():
    policy = PickRankingPolicy(
        version="test_policy_v1",
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
        edge_tie_preference="under",
        pick_type_order=("official", "lean", "pass"),
        postable_limits=PostablePickLimits(max_official=2, max_leans=1),
    )
    best_over = pd.Series({"line": 5.5, "bookmaker": "DraftKings"})
    best_under = pd.Series({"line": 6.5, "bookmaker": "FanDuel"})

    chosen = policy.choose_pick_side(
        best_over=best_over,
        best_under=best_under,
        predicted=6.0,
    )
    limits = policy.resolved_postable_limits()

    assert chosen["pick_side"] == "under"
    assert chosen["edge"] == 0.5
    assert limits.max_official == 2
    assert limits.max_leans == 1


def test_choose_pick_side_returns_empty_when_neither_side_has_positive_edge():
    policy = DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY
    best_over = pd.Series({"line": 6.5, "bookmaker": "DraftKings"})
    best_under = pd.Series({"line": 4.5, "bookmaker": "FanDuel"})

    chosen = policy.choose_pick_side(
        best_over=best_over,
        best_under=best_under,
        predicted=5.0,
    )

    assert chosen.empty


def test_default_pitcher_walks_policy_uses_expected_return_metrics():
    policy = DEFAULT_MLB_PITCHER_WALKS_POLICY

    assert policy.version == "mlb_pitcher_walks_ev_policy_v1"
    assert policy.market_selection_metric_column == "selection_expected_return"
    assert policy.side_choice_metric_column == "selection_expected_return"
    assert policy.pick_type_metric_column == "expected_return"
    assert policy.confidence_metric_column == "expected_return"
    assert policy.ranking_metric_column == "adjusted_expected_return"
    assert policy.classify_pick_type(0.08) == "official"
    assert policy.classify_pick_type(0.02) == "lean"
    assert policy.classify_pick_type(0.01) == "pass"
    assert policy.classify_pick_type(-0.08) == "pass"
