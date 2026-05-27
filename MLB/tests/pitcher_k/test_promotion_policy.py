from pitcher_k.promotion_policy import (
    CHAMPION_STATUS_RECENTLY_PROMOTED,
    REVIEW_STATUS_HOLD,
    REVIEW_STATUS_INSUFFICIENT_EVIDENCE,
    REVIEW_STATUS_PROMOTE,
    REVIEW_STATUS_ROLLBACK_CANDIDATE,
    build_shadow_promotion_review,
)


def _base_summary() -> dict:
    return {
        "models": {
            "xgboost_champion": {
                "mae": 1.74,
                "rmse": 2.20,
                "calibration_gap": 0.015,
                "pick_count": 50,
                "roi_per_pick": 0.01,
                "profit_units": 0.5,
            },
            "ridge_challenger": {
                "mae": 1.70,
                "rmse": 2.16,
                "calibration_gap": 0.005,
                "pick_count": 52,
                "roi_per_pick": 0.05,
                "profit_units": 3.0,
            },
        },
        "overlap_sample": {
            "unique_dates": 30,
        },
        "agreement_slices": [
            {
                "comparison_slice": "disagreement",
                "rows": 16,
                "xgboost_champion_roi_per_pick": -0.02,
                "ridge_challenger_roi_per_pick": 0.07,
            }
        ],
    }


def test_build_shadow_promotion_review_marks_insufficient_evidence():
    summary = _base_summary()
    summary["overlap_sample"]["unique_dates"] = 8
    summary["models"]["xgboost_champion"]["pick_count"] = 12
    summary["models"]["ridge_challenger"]["pick_count"] = 11
    summary["agreement_slices"][0]["rows"] = 4

    review = build_shadow_promotion_review(
        summary,
        champion_name="xgboost_champion",
        challenger_name="ridge_challenger",
    )

    assert review["review_status"] == REVIEW_STATUS_INSUFFICIENT_EVIDENCE
    assert review["recommended_action"] == REVIEW_STATUS_HOLD
    assert review["manual_approval_required"] is True
    assert review["eligible_for_review"] is False


def test_build_shadow_promotion_review_marks_promote_when_challenger_clearly_wins():
    review = build_shadow_promotion_review(
        _base_summary(),
        champion_name="xgboost_champion",
        challenger_name="ridge_challenger",
    )

    assert review["review_status"] == REVIEW_STATUS_PROMOTE
    assert review["recommended_action"] == REVIEW_STATUS_PROMOTE
    assert review["eligible_for_review"] is True
    assert review["advantage_counts"]["challenger_regression_advantages"] >= 1
    assert review["advantage_counts"]["challenger_workflow_advantages"] >= 1


def test_build_shadow_promotion_review_marks_hold_when_signals_conflict():
    summary = _base_summary()
    summary["models"]["ridge_challenger"]["roi_per_pick"] = -0.03
    summary["models"]["ridge_challenger"]["profit_units"] = -2.0
    summary["agreement_slices"][0]["ridge_challenger_roi_per_pick"] = -0.04

    review = build_shadow_promotion_review(
        summary,
        champion_name="xgboost_champion",
        challenger_name="ridge_challenger",
    )

    assert review["review_status"] == REVIEW_STATUS_HOLD
    assert review["recommended_action"] == REVIEW_STATUS_HOLD
    assert review["conflicting_signals"]


def test_build_shadow_promotion_review_marks_rollback_candidate_for_provisional_champion():
    review = build_shadow_promotion_review(
        _base_summary(),
        champion_name="xgboost_champion",
        challenger_name="ridge_challenger",
        review_context={
            "champion_status": CHAMPION_STATUS_RECENTLY_PROMOTED,
        },
    )

    assert review["review_status"] == REVIEW_STATUS_ROLLBACK_CANDIDATE
    assert review["recommended_action"] == REVIEW_STATUS_ROLLBACK_CANDIDATE
