from __future__ import annotations

from dataclasses import asdict, dataclass


REVIEW_STATUS_PROMOTE = "promote"
REVIEW_STATUS_HOLD = "hold"
REVIEW_STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REVIEW_STATUS_ROLLBACK_CANDIDATE = "rollback_candidate"

CHAMPION_STATUS_STABLE = "stable_champion"
CHAMPION_STATUS_PROVISIONAL = "provisional_champion"
CHAMPION_STATUS_RECENTLY_PROMOTED = "recently_promoted"


@dataclass(frozen=True)
class ShadowPromotionPolicy:
    version: str = "pitcher_k_shadow_promotion_policy_v1"
    minimum_overlapping_days: int = 21
    minimum_overlapping_pick_count: int = 30
    minimum_disagreement_rows: int = 10
    minimum_metric_advantages_for_promotion: int = 2
    minimum_regression_advantages_for_promotion: int = 1
    minimum_workflow_advantages_for_promotion: int = 1
    mae_tie_margin: float = 0.02
    rmse_tie_margin: float = 0.03
    calibration_gap_tie_margin: float = 0.01
    roi_per_pick_tie_margin: float = 0.02
    profit_units_tie_margin: float = 1.0
    disagreement_roi_tie_margin: float = 0.03
    pick_count_tie_margin: int = 5
    manual_approval_required: bool = True


DEFAULT_SHADOW_PROMOTION_POLICY = ShadowPromotionPolicy()


def _compare_metric(
    champion_value: float | int | None,
    challenger_value: float | int | None,
    *,
    lower_is_better: bool,
    tie_margin: float | int,
) -> dict[str, object]:
    if champion_value is None or challenger_value is None:
        return {
            "champion": champion_value,
            "challenger": challenger_value,
            "winner": "unknown",
            "delta_challenger_minus_champion": None,
            "material_difference": False,
        }

    champion_numeric = float(champion_value)
    challenger_numeric = float(challenger_value)
    delta = challenger_numeric - champion_numeric

    if abs(delta) <= float(tie_margin):
        winner = "tie"
        material_difference = False
    else:
        material_difference = True
        if lower_is_better:
            winner = "challenger" if challenger_numeric < champion_numeric else "champion"
        else:
            winner = "challenger" if challenger_numeric > champion_numeric else "champion"

    return {
        "champion": champion_numeric,
        "challenger": challenger_numeric,
        "winner": winner,
        "delta_challenger_minus_champion": delta,
        "material_difference": material_difference,
    }


def _disagreement_slice(summary: dict) -> dict[str, object] | None:
    for record in summary.get("agreement_slices", []):
        if str(record.get("comparison_slice")) == "disagreement":
            return record
    return None


def _default_review_context(
    *,
    champion_name: str,
    challenger_name: str,
) -> dict[str, object]:
    return {
        "champion_name": champion_name,
        "challenger_name": challenger_name,
        "champion_status": CHAMPION_STATUS_STABLE,
        "manual_approval_required": True,
    }


def build_shadow_promotion_review(
    summary: dict,
    *,
    champion_name: str,
    challenger_name: str,
    policy: ShadowPromotionPolicy = DEFAULT_SHADOW_PROMOTION_POLICY,
    review_context: dict | None = None,
) -> dict[str, object]:
    review_context_payload = _default_review_context(
        champion_name=champion_name,
        challenger_name=challenger_name,
    )
    if review_context:
        review_context_payload.update(review_context)
    review_context_payload["manual_approval_required"] = bool(policy.manual_approval_required)

    models = summary.get("models", {})
    champion = models.get(champion_name, {})
    challenger = models.get(challenger_name, {})
    overlap_sample = summary.get("overlap_sample", {})
    disagreement = _disagreement_slice(summary) or {}

    evidence = {
        "overlapping_days": int(overlap_sample.get("unique_dates", 0) or 0),
        "minimum_overlapping_days": int(policy.minimum_overlapping_days),
        "champion_pick_count": int(champion.get("pick_count", 0) or 0),
        "challenger_pick_count": int(challenger.get("pick_count", 0) or 0),
        "minimum_overlapping_pick_count": int(policy.minimum_overlapping_pick_count),
        "disagreement_rows": int(disagreement.get("rows", 0) or 0),
        "minimum_disagreement_rows": int(policy.minimum_disagreement_rows),
    }

    metric_comparison = {
        "mae": _compare_metric(
            champion.get("mae"),
            challenger.get("mae"),
            lower_is_better=True,
            tie_margin=policy.mae_tie_margin,
        ),
        "rmse": _compare_metric(
            champion.get("rmse"),
            challenger.get("rmse"),
            lower_is_better=True,
            tie_margin=policy.rmse_tie_margin,
        ),
        "calibration_gap": _compare_metric(
            champion.get("calibration_gap"),
            challenger.get("calibration_gap"),
            lower_is_better=True,
            tie_margin=policy.calibration_gap_tie_margin,
        ),
        "roi_per_pick": _compare_metric(
            champion.get("roi_per_pick"),
            challenger.get("roi_per_pick"),
            lower_is_better=False,
            tie_margin=policy.roi_per_pick_tie_margin,
        ),
        "profit_units": _compare_metric(
            champion.get("profit_units"),
            challenger.get("profit_units"),
            lower_is_better=False,
            tie_margin=policy.profit_units_tie_margin,
        ),
        "pick_count": _compare_metric(
            champion.get("pick_count"),
            challenger.get("pick_count"),
            lower_is_better=False,
            tie_margin=policy.pick_count_tie_margin,
        ),
        "disagreement_roi_per_pick": _compare_metric(
            disagreement.get(f"{champion_name}_roi_per_pick"),
            disagreement.get(f"{challenger_name}_roi_per_pick"),
            lower_is_better=False,
            tie_margin=policy.disagreement_roi_tie_margin,
        ),
    }

    regression_metric_names = ["mae", "rmse", "calibration_gap"]
    workflow_metric_names = ["roi_per_pick", "profit_units", "disagreement_roi_per_pick", "pick_count"]
    challenger_regression_advantages = sum(
        1 for name in regression_metric_names if metric_comparison[name]["winner"] == "challenger"
    )
    challenger_workflow_advantages = sum(
        1 for name in workflow_metric_names if metric_comparison[name]["winner"] == "challenger"
    )
    champion_workflow_advantages = sum(
        1 for name in workflow_metric_names if metric_comparison[name]["winner"] == "champion"
    )
    challenger_total_advantages = sum(
        1 for record in metric_comparison.values() if record["winner"] == "challenger"
    )
    champion_total_advantages = sum(
        1 for record in metric_comparison.values() if record["winner"] == "champion"
    )
    ties = sum(1 for record in metric_comparison.values() if record["winner"] == "tie")

    rationale: list[str] = []
    conflicting_signals: list[str] = []
    evidence_failures: list[str] = []

    if evidence["overlapping_days"] < policy.minimum_overlapping_days:
        evidence_failures.append(
            f"Needs at least {policy.minimum_overlapping_days} overlapping forward days; only has {evidence['overlapping_days']}."
        )
    if min(evidence["champion_pick_count"], evidence["challenger_pick_count"]) < policy.minimum_overlapping_pick_count:
        evidence_failures.append(
            "Needs at least "
            f"{policy.minimum_overlapping_pick_count} overlapping workflow picks per model; has "
            f"{evidence['champion_pick_count']} for champion and {evidence['challenger_pick_count']} for challenger."
        )
    if evidence["disagreement_rows"] < policy.minimum_disagreement_rows:
        evidence_failures.append(
            f"Needs at least {policy.minimum_disagreement_rows} disagreement-slice rows; only has {evidence['disagreement_rows']}."
        )

    if metric_comparison["mae"]["winner"] == "challenger" and metric_comparison["roi_per_pick"]["winner"] == "champion":
        conflicting_signals.append("Challenger improved point accuracy, but champion still leads on ROI per pick.")
    if metric_comparison["calibration_gap"]["winner"] == "challenger" and metric_comparison["pick_count"]["winner"] == "champion":
        conflicting_signals.append("Challenger improved calibration, but champion still captures meaningfully more picks.")
    if metric_comparison["roi_per_pick"]["winner"] == "challenger" and evidence["challenger_pick_count"] < policy.minimum_overlapping_pick_count:
        conflicting_signals.append("Challenger ROI improvement came on too few picks to justify promotion.")
    if metric_comparison["profit_units"]["winner"] == "champion" and metric_comparison["mae"]["winner"] == "challenger":
        conflicting_signals.append("Challenger leads on MAE, but champion still leads on realized profit units.")

    if evidence_failures:
        rationale.extend(evidence_failures)
        decision = REVIEW_STATUS_INSUFFICIENT_EVIDENCE
        recommended_action = REVIEW_STATUS_HOLD
    else:
        challenger_promotion_ready = (
            challenger_total_advantages >= policy.minimum_metric_advantages_for_promotion
            and challenger_regression_advantages >= policy.minimum_regression_advantages_for_promotion
            and challenger_workflow_advantages >= policy.minimum_workflow_advantages_for_promotion
            and champion_workflow_advantages == 0
        )
        champion_status = str(review_context_payload.get("champion_status", CHAMPION_STATUS_STABLE))
        champion_is_provisional = champion_status in {
            CHAMPION_STATUS_PROVISIONAL,
            CHAMPION_STATUS_RECENTLY_PROMOTED,
        }
        rollback_ready = champion_is_provisional and challenger_promotion_ready

        if rollback_ready:
            decision = REVIEW_STATUS_ROLLBACK_CANDIDATE
            recommended_action = REVIEW_STATUS_ROLLBACK_CANDIDATE
            rationale.append(
                "Current champion is marked as provisional/recently promoted and the challenger has a clear multi-metric advantage."
            )
        elif challenger_promotion_ready and not conflicting_signals:
            decision = REVIEW_STATUS_PROMOTE
            recommended_action = REVIEW_STATUS_PROMOTE
            rationale.append(
                "Challenger leads on both regression and workflow metrics without a material workflow regression."
            )
        else:
            decision = REVIEW_STATUS_HOLD
            recommended_action = REVIEW_STATUS_HOLD
            if conflicting_signals:
                rationale.append(
                    "Signals are mixed across accuracy, calibration, and workflow outcomes, so the conservative action is to hold."
                )
            else:
                rationale.append(
                    "Challenger does not clear the minimum multi-metric advantage threshold required for promotion."
                )

    if policy.manual_approval_required:
        rationale.append("Manual approval is always required before any production promotion.")

    return {
        "policy_version": policy.version,
        "manual_approval_required": bool(policy.manual_approval_required),
        "review_status": decision,
        "recommended_action": recommended_action,
        "eligible_for_review": not evidence_failures,
        "review_context": review_context_payload,
        "policy_thresholds": asdict(policy),
        "evidence": evidence,
        "metric_comparison": metric_comparison,
        "advantage_counts": {
            "challenger_total_advantages": int(challenger_total_advantages),
            "champion_total_advantages": int(champion_total_advantages),
            "ties": int(ties),
            "challenger_regression_advantages": int(challenger_regression_advantages),
            "challenger_workflow_advantages": int(challenger_workflow_advantages),
            "champion_workflow_advantages": int(champion_workflow_advantages),
        },
        "conflicting_signals": conflicting_signals,
        "rationale": rationale,
    }
