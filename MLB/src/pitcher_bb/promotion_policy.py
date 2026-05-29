from __future__ import annotations

from pitcher_k import promotion_policy as pitcher_k_promotion_policy


REVIEW_STATUS_PROMOTE = pitcher_k_promotion_policy.REVIEW_STATUS_PROMOTE
REVIEW_STATUS_HOLD = pitcher_k_promotion_policy.REVIEW_STATUS_HOLD
REVIEW_STATUS_INSUFFICIENT_EVIDENCE = (
    pitcher_k_promotion_policy.REVIEW_STATUS_INSUFFICIENT_EVIDENCE
)
REVIEW_STATUS_ROLLBACK_CANDIDATE = (
    pitcher_k_promotion_policy.REVIEW_STATUS_ROLLBACK_CANDIDATE
)

CHAMPION_STATUS_STABLE = pitcher_k_promotion_policy.CHAMPION_STATUS_STABLE
CHAMPION_STATUS_PROVISIONAL = pitcher_k_promotion_policy.CHAMPION_STATUS_PROVISIONAL
CHAMPION_STATUS_RECENTLY_PROMOTED = (
    pitcher_k_promotion_policy.CHAMPION_STATUS_RECENTLY_PROMOTED
)

ShadowPromotionPolicy = pitcher_k_promotion_policy.ShadowPromotionPolicy

DEFAULT_SHADOW_PROMOTION_POLICY = ShadowPromotionPolicy(
    version="pitcher_bb_shadow_promotion_policy_v1",
)


def build_shadow_promotion_review(
    summary: dict,
    *,
    champion_name: str,
    challenger_name: str,
    policy: ShadowPromotionPolicy = DEFAULT_SHADOW_PROMOTION_POLICY,
    review_context: dict | None = None,
) -> dict[str, object]:
    return pitcher_k_promotion_policy.build_shadow_promotion_review(
        summary,
        champion_name=champion_name,
        challenger_name=challenger_name,
        policy=policy,
        review_context=review_context,
    )
