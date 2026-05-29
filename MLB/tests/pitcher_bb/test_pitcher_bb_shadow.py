import pandas as pd
import pytest

from common.identity import MARKET_OFFER_KEY_COLUMN, MARKET_SELECTION_KEY_COLUMN
from pitcher_bb import shadow


def _base_shadow_row(**overrides) -> dict:
    row = {
        "shadow_row_key": "",
        "candidate_key": "",
        "game_date": "2026-05-10",
        "player_name": "Zac Gallen",
        "participant_join_key": "mlbam_player:321",
        "participant_id": "mlbam_player:321",
        "participant_source_id": "321",
        "participant_source_id_type": "mlbam_player",
        "participant_name_norm": "zac gallen",
        "sport": "MLB",
        "market_key": "pitcher_walks",
        "market_family": "player_prop",
        "prop_type": "pitcher_bb",
        "team": "ARI",
        "opponent": "LAD",
        "book": "DraftKings",
        "bookmaker_key": "draftkings",
        "event_id": "evt_1",
        "side": "under",
        "side_norm": "under",
        "line": 2.5,
        "price": -110,
        MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|under|2.5",
        MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|under|2.5|draftkings",
        "model_name": shadow.CHAMPION_MODEL_NAME,
        "model_role": shadow.CHAMPION_MODEL_ROLE,
        "model_version": "champion_v1",
        "policy_version": "policy_v1",
        "predicted_value": 2.0,
        "predicted_walks": 2.0,
        "lower_bound": 1.4,
        "upper_bound": 2.6,
        "std_dev": 0.6,
        "edge": 0.5,
        "would_pick": True,
        "pick_rank": 1,
        "pick_type": "official",
        "confidence_tier": "high",
        "actual_value": "1",
        "result": "W",
        "profit_units": 0.9090909091,
        "graded_at": "2026-05-11T10:00:00-04:00",
        "tracking_regime": "current_workflow",
    }
    row.update(overrides)
    if not row["candidate_key"]:
        row["candidate_key"] = f"{row['game_date']}|{row[MARKET_OFFER_KEY_COLUMN]}"
    if not row["shadow_row_key"]:
        row["shadow_row_key"] = f"{row['game_date']}|{row['model_name']}|{row['candidate_key']}"
    return row


def test_build_shadow_candidate_rows_marks_walk_pick_and_rank():
    joined_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "player_name": "Zac Gallen",
                "participant_join_key": "mlbam_player:321",
                "participant_id": "mlbam_player:321",
                "participant_source_id": "321",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "zac gallen",
                "sport": "MLB",
                "market_key": "pitcher_walks",
                "market_family": "player_prop",
                "team": "ARI",
                "opponent": "LAD",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_1",
                "side": "under",
                "line": 2.5,
                "price": -110,
                MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|under|2.5",
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|under|2.5|draftkings",
                "predicted_value": 2.0,
                "predicted_walks": 2.0,
                "lower_bound": 1.4,
                "upper_bound": 2.6,
                "std_dev": 0.6,
            },
            {
                "game_date": "2026-05-10",
                "player_name": "Zac Gallen",
                "participant_join_key": "mlbam_player:321",
                "participant_id": "mlbam_player:321",
                "participant_source_id": "321",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "zac gallen",
                "sport": "MLB",
                "market_key": "pitcher_walks",
                "market_family": "player_prop",
                "team": "ARI",
                "opponent": "LAD",
                "bookmaker": "BetMGM",
                "bookmaker_key": "betmgm",
                "event_id": "evt_1",
                "side": "over",
                "line": 1.5,
                "price": 100,
                MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|over|1.5",
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|over|1.5|betmgm",
                "predicted_value": 2.0,
                "predicted_walks": 2.0,
                "lower_bound": 1.4,
                "upper_bound": 2.6,
                "std_dev": 0.6,
            },
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "player_name": "Zac Gallen",
                "participant_join_key": "mlbam_player:321",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "pick_side": "under",
                "line": 2.5,
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_walks|mlbam_player:321|under|2.5|draftkings",
                "pick_type": "official",
                "confidence_tier": "high",
            }
        ]
    )

    rows = shadow.build_shadow_candidate_rows(
        joined_df,
        picks_df,
        model_name=shadow.CHAMPION_MODEL_NAME,
        model_role=shadow.CHAMPION_MODEL_ROLE,
        model_version="champion_v1",
        policy_version="policy_v1",
        tracking_regime="current_workflow",
        game_date="2026-05-10",
    )

    assert len(rows) == 2
    selected = rows[rows["would_pick"]].reset_index(drop=True)
    skipped = rows[~rows["would_pick"]].reset_index(drop=True)
    assert len(selected) == 1
    assert selected.loc[0, "pick_rank"] == pytest.approx(1)
    assert selected.loc[0, "predicted_walks"] == pytest.approx(2.0)
    assert skipped.loc[0, "pick_type"] == ""


def test_build_shadow_comparison_report_uses_walk_labels_and_policy():
    rows = pd.DataFrame(
        [
            _base_shadow_row(
                candidate_key="2026-05-10|offer_agree",
                shadow_row_key="2026-05-10|xgboost_champion|2026-05-10|offer_agree",
            ),
            _base_shadow_row(
                candidate_key="2026-05-10|offer_agree",
                shadow_row_key="2026-05-10|ridge_challenger|2026-05-10|offer_agree",
                model_name=shadow.CHALLENGER_MODEL_NAME,
                model_role=shadow.CHALLENGER_MODEL_ROLE,
                model_version="ridge_v1",
                predicted_value=1.9,
                predicted_walks=1.9,
                lower_bound=1.2,
                upper_bound=2.6,
            ),
        ]
    )

    report = shadow.build_shadow_comparison_report(rows)

    assert report["available"] is True
    assert not report["overlap_df"].empty
    assert "predicted_walks" in report["overlap_df"].columns
    assert report["summary"]["analysis_type"] == "pitcher_bb_shadow_comparison"
    assert (
        report["summary"]["promotion_review"]["policy_version"]
        == "pitcher_bb_shadow_promotion_policy_v1"
    )
