import pandas as pd
import pytest

from common.identity import MARKET_OFFER_KEY_COLUMN, MARKET_SELECTION_KEY_COLUMN
from pitcher_k import shadow


def _base_shadow_row(**overrides) -> dict:
    row = {
        "shadow_row_key": "",
        "candidate_key": "",
        "game_date": "2026-05-10",
        "player_name": "Jacob deGrom",
        "participant_join_key": "mlbam_player:123",
        "participant_id": "mlbam_player:123",
        "participant_source_id": "123",
        "participant_source_id_type": "mlbam_player",
        "participant_name_norm": "jacob degrom",
        "sport": "MLB",
        "market_key": "pitcher_strikeouts",
        "market_family": "player_prop",
        "prop_type": "pitcher_k",
        "team": "TEX",
        "opponent": "SEA",
        "book": "DraftKings",
        "bookmaker_key": "draftkings",
        "event_id": "evt_1",
        "side": "over",
        "side_norm": "over",
        "line": 5.5,
        "price": -120,
        MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5",
        MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5|draftkings",
        "model_name": shadow.CHAMPION_MODEL_NAME,
        "model_role": shadow.CHAMPION_MODEL_ROLE,
        "model_version": "champion_v1",
        "policy_version": "policy_v1",
        "predicted_value": 6.8,
        "predicted_strikeouts": 6.8,
        "lower_bound": 5.7,
        "upper_bound": 7.9,
        "std_dev": 1.1,
        "edge": 1.3,
        "would_pick": True,
        "pick_rank": 1,
        "pick_type": "official",
        "confidence_tier": "high",
        "actual_value": "7",
        "result": "W",
        "profit_units": 0.8333333333,
        "graded_at": "2026-05-11T10:00:00-04:00",
        "tracking_regime": "current_workflow",
    }
    row.update(overrides)
    if not row["candidate_key"]:
        row["candidate_key"] = f"{row['game_date']}|{row[MARKET_OFFER_KEY_COLUMN]}"
    if not row["shadow_row_key"]:
        row["shadow_row_key"] = f"{row['game_date']}|{row['model_name']}|{row['candidate_key']}"
    return row


def test_build_shadow_candidate_rows_marks_actionable_pick_and_rank():
    joined_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "player_name": "Jacob deGrom",
                "participant_join_key": "mlbam_player:123",
                "participant_id": "mlbam_player:123",
                "participant_source_id": "123",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "jacob degrom",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "team": "TEX",
                "opponent": "SEA",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_1",
                "side": "over",
                "line": 5.5,
                "price": -120,
                MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5",
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5|draftkings",
                "predicted_value": 6.8,
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.7,
                "upper_bound": 7.9,
                "std_dev": 1.1,
            },
            {
                "game_date": "2026-05-10",
                "player_name": "Jacob deGrom",
                "participant_join_key": "mlbam_player:123",
                "participant_id": "mlbam_player:123",
                "participant_source_id": "123",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "jacob degrom",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "team": "TEX",
                "opponent": "SEA",
                "bookmaker": "BetMGM",
                "bookmaker_key": "betmgm",
                "event_id": "evt_1",
                "side": "under",
                "line": 6.5,
                "price": -110,
                MARKET_SELECTION_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|under|6.5",
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|under|6.5|betmgm",
                "predicted_value": 6.8,
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.7,
                "upper_bound": 7.9,
                "std_dev": 1.1,
            },
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "player_name": "Jacob deGrom",
                "participant_join_key": "mlbam_player:123",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "pick_side": "over",
                "line": 5.5,
                MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5|draftkings",
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
    assert selected.loc[0, "pick_type"] == "official"
    assert selected.loc[0, "book"] == "DraftKings"
    assert skipped.loc[0, "pick_type"] == ""
    assert skipped.loc[0, "actual_value"] == ""


def test_build_shadow_comparison_report_summarizes_overlap_and_disagreement():
    rows = pd.DataFrame(
        [
            _base_shadow_row(
                candidate_key="2026-05-10|offer_agree",
                shadow_row_key="2026-05-10|xgboost_champion|2026-05-10|offer_agree",
                model_name=shadow.CHAMPION_MODEL_NAME,
                model_role=shadow.CHAMPION_MODEL_ROLE,
                player_name="Jacob deGrom",
                participant_join_key="mlbam_player:123",
                line=5.5,
                side="over",
                side_norm="over",
                book="DraftKings",
                bookmaker_key="draftkings",
                edge=1.3,
                actual_value="7",
                result="W",
                profit_units=0.83,
                would_pick=True,
            ),
            _base_shadow_row(
                candidate_key="2026-05-10|offer_agree",
                shadow_row_key="2026-05-10|ridge_challenger|2026-05-10|offer_agree",
                model_name=shadow.CHALLENGER_MODEL_NAME,
                model_role=shadow.CHALLENGER_MODEL_ROLE,
                model_version="ridge_v1",
                player_name="Jacob deGrom",
                participant_join_key="mlbam_player:123",
                predicted_value=6.4,
                predicted_strikeouts=6.4,
                lower_bound=5.2,
                upper_bound=7.6,
                std_dev=1.2,
                edge=0.9,
                actual_value="7",
                result="W",
                profit_units=0.83,
                would_pick=True,
            ),
            _base_shadow_row(
                candidate_key="2026-05-11|offer_over",
                shadow_row_key="2026-05-11|xgboost_champion|2026-05-11|offer_over",
                model_name=shadow.CHAMPION_MODEL_NAME,
                model_role=shadow.CHAMPION_MODEL_ROLE,
                player_name="Max Fried",
                participant_join_key="mlbam_player:456",
                line=5.5,
                side="over",
                side_norm="over",
                book="FanDuel",
                bookmaker_key="fanduel",
                actual_value="3",
                result="L",
                profit_units=-1.0,
                would_pick=True,
            ),
            _base_shadow_row(
                candidate_key="2026-05-11|offer_over",
                shadow_row_key="2026-05-11|ridge_challenger|2026-05-11|offer_over",
                model_name=shadow.CHALLENGER_MODEL_NAME,
                model_role=shadow.CHALLENGER_MODEL_ROLE,
                model_version="ridge_v1",
                player_name="Max Fried",
                participant_join_key="mlbam_player:456",
                line=5.5,
                side="over",
                side_norm="over",
                book="FanDuel",
                bookmaker_key="fanduel",
                predicted_value=4.2,
                predicted_strikeouts=4.2,
                lower_bound=3.3,
                upper_bound=5.1,
                std_dev=0.9,
                edge=-1.3,
                actual_value="3",
                result="",
                profit_units="",
                would_pick=False,
                pick_rank="",
                pick_type="",
                confidence_tier="",
            ),
            _base_shadow_row(
                candidate_key="2026-05-11|offer_under",
                shadow_row_key="2026-05-11|xgboost_champion|2026-05-11|offer_under",
                model_name=shadow.CHAMPION_MODEL_NAME,
                model_role=shadow.CHAMPION_MODEL_ROLE,
                player_name="Max Fried",
                participant_join_key="mlbam_player:456",
                line=5.5,
                side="under",
                side_norm="under",
                book="FanDuel",
                bookmaker_key="fanduel",
                actual_value="3",
                result="",
                profit_units="",
                would_pick=False,
                pick_rank="",
                pick_type="",
                confidence_tier="",
            ),
            _base_shadow_row(
                candidate_key="2026-05-11|offer_under",
                shadow_row_key="2026-05-11|ridge_challenger|2026-05-11|offer_under",
                model_name=shadow.CHALLENGER_MODEL_NAME,
                model_role=shadow.CHALLENGER_MODEL_ROLE,
                model_version="ridge_v1",
                player_name="Max Fried",
                participant_join_key="mlbam_player:456",
                line=5.5,
                side="under",
                side_norm="under",
                book="FanDuel",
                bookmaker_key="fanduel",
                predicted_value=4.2,
                predicted_strikeouts=4.2,
                lower_bound=3.3,
                upper_bound=5.1,
                std_dev=0.9,
                edge=1.3,
                actual_value="3",
                result="W",
                profit_units=0.91,
                would_pick=True,
                pick_rank=1,
                pick_type="official",
                confidence_tier="high",
            ),
        ]
    )

    report = shadow.build_shadow_comparison_report(rows)

    assert report["available"] is True
    assert not report["overlap_df"].empty
    assert report["summary"]["models"][shadow.CHAMPION_MODEL_NAME]["pick_count"] == 2
    assert report["summary"]["models"][shadow.CHALLENGER_MODEL_NAME]["pick_count"] == 2
    slices = {row["comparison_slice"]: row for row in report["summary"]["agreement_slices"]}
    assert slices["agreement"]["rows"] == 1
    assert slices["disagreement"]["rows"] == 1
    assert report["summary"]["rolling_regression_windows"]
    assert report["summary"]["rolling_workflow_windows"]
