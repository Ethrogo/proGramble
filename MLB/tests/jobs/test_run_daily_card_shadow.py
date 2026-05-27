import json
from dataclasses import replace

import pandas as pd

from common.identity import MARKET_OFFER_KEY_COLUMN, MARKET_SELECTION_KEY_COLUMN
from jobs import run_daily_card as daily_card
from pitcher_k.workflow import MLB_PITCHER_STRIKEOUT_WORKFLOW


def test_persist_pitcher_k_shadow_predictions_writes_champion_and_challenger_rows(monkeypatch):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "game_pk": 1,
                "pitcher": 123,
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "home_team": "TEX",
                "away_team": "SEA",
                "is_home": 1,
                "p_throws": "R",
            }
        ]
    )
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2026-05-09",
                "game_pk": 99,
                "pitcher": 123,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "strikeouts": 7,
                "pitches_last10": 95,
                "whiff_per_pitch_last3": 0.14,
                "pitches_trend_last3_vs_last10": 2.0,
                "avg_velo_last3": 97.5,
                "avg_spin_last3": 2480.0,
                "k_rate_last10": 0.31,
                "pitches_per_batter_last10": 3.9,
                "opp_strikeouts_per_game_last10": 9.1,
                "strikeouts_stddev_last10": 1.3,
            }
        ]
    )
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
            }
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "player_name": "Jacob deGrom",
                "participant_join_key": "mlbam_player:123",
                "participant_id": "mlbam_player:123",
                "participant_source_id": "123",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "jacob degrom",
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

    monkeypatch.setattr(
        daily_card.pitcher_k_shadow,
        "build_ridge_shadow_predictions",
        lambda today_features, pitcher_games: (
            pd.DataFrame(
                [
                    {
                        "player_name": "Jacob deGrom",
                        "pitcher": 123,
                        "participant_join_key": "mlbam_player:123",
                        "participant_id": "mlbam_player:123",
                        "participant_source_id": "123",
                        "participant_source_id_type": "mlbam_player",
                        "participant_name_norm": "jacob degrom",
                        "predicted_value": 6.2,
                        "predicted_strikeouts": 6.2,
                        "lower_bound": 5.0,
                        "upper_bound": 7.4,
                        "std_dev": 1.2,
                    }
                ]
            ),
            {
                "model_name": "ridge_challenger",
                "model_role": "challenger",
                "model_version": "ridge_shadow_v1",
            },
        ),
    )

    def fake_build_picks_fn(df: pd.DataFrame) -> pd.DataFrame:
        predicted_value = float(df["predicted_value"].iloc[0])
        pick_side = "over" if predicted_value >= 6.0 else "under"
        return pd.DataFrame(
            [
                {
                    "game_date": "2026-05-10",
                    "player_name": "Jacob deGrom",
                    "participant_join_key": "mlbam_player:123",
                    "participant_id": "mlbam_player:123",
                    "participant_source_id": "123",
                    "participant_source_id_type": "mlbam_player",
                    "participant_name_norm": "jacob degrom",
                    "book": "DraftKings",
                    "bookmaker_key": "draftkings",
                    "pick_side": pick_side,
                    "line": 5.5,
                    MARKET_OFFER_KEY_COLUMN: "MLB|pitcher_strikeouts|mlbam_player:123|over|5.5|draftkings",
                    "pick_type": "official",
                    "confidence_tier": "high",
                }
            ]
        )

    workflow = replace(
        MLB_PITCHER_STRIKEOUT_WORKFLOW,
        feature_builder=lambda starters_df, pitcher_games: pd.DataFrame(
            [
                {
                    "player_name": "Jacob deGrom",
                    "pitcher": 123,
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
                    "predicted_value": 6.8,
                    "predicted_strikeouts": 6.8,
                    "lower_bound": 5.7,
                    "upper_bound": 7.9,
                    "std_dev": 1.1,
                }
            ]
        ),
    )

    output_path = daily_card.persist_pitcher_k_shadow_predictions(
        starters_df=starters_df,
        pitcher_games=pitcher_games,
        joined_df=joined_df,
        picks_df=picks_df,
        workflow=workflow,
        metadata={"artifact_version": 2, "training_window": {"train_split_date": "2025-08-01"}},
        build_picks_fn=fake_build_picks_fn,
    )

    saved = pd.read_csv(output_path, keep_default_na=False)
    assert output_path.exists()
    assert set(saved["model_name"]) == {"xgboost_champion", "ridge_challenger"}
    assert len(saved) == 2
    assert saved["would_pick"].astype(str).str.lower().eq("true").all()


def test_grade_pitcher_k_shadow_predictions_and_persist_report(monkeypatch):
    shadow_rows = pd.DataFrame(
        [
            {
                "shadow_row_key": "2026-05-10|xgboost_champion|row_a",
                "candidate_key": "row_a",
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
                MARKET_SELECTION_KEY_COLUMN: "sel_a",
                MARKET_OFFER_KEY_COLUMN: "offer_a",
                "model_name": "xgboost_champion",
                "model_role": "champion",
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
                "actual_value": "",
                "result": "",
                "profit_units": "",
                "graded_at": "",
                "tracking_regime": "current_workflow",
            },
            {
                "shadow_row_key": "2026-05-10|ridge_challenger|row_a",
                "candidate_key": "row_a",
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
                MARKET_SELECTION_KEY_COLUMN: "sel_a",
                MARKET_OFFER_KEY_COLUMN: "offer_a",
                "model_name": "ridge_challenger",
                "model_role": "challenger",
                "model_version": "ridge_v1",
                "policy_version": "policy_v1",
                "predicted_value": 6.2,
                "predicted_strikeouts": 6.2,
                "lower_bound": 5.0,
                "upper_bound": 7.4,
                "std_dev": 1.2,
                "edge": 0.7,
                "would_pick": True,
                "pick_rank": 1,
                "pick_type": "official",
                "confidence_tier": "high",
                "actual_value": "",
                "result": "",
                "profit_units": "",
                "graded_at": "",
                "tracking_regime": "current_workflow",
            },
            ]
        )
    daily_card.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    shadow_rows.to_csv(daily_card.PITCHER_K_SHADOW_TRACKING_PATH, index=False)

    monkeypatch.setattr(
        daily_card,
        "load_pitcher_results_from_statcast",
        lambda game_date: pd.DataFrame(
            [
                {
                    "game_date": game_date,
                    "pitcher": 123,
                    "player_name": "Jacob deGrom",
                    "strikeouts": 7,
                    "walks": 1,
                }
            ]
        ),
    )

    grade_result = daily_card.grade_pitcher_k_shadow_predictions_from_statcast(game_date="2026-05-10")
    report = daily_card.persist_pitcher_k_shadow_comparison_report()

    loaded = pd.read_csv(daily_card.PITCHER_K_SHADOW_TRACKING_PATH, keep_default_na=False)
    summary = json.loads(daily_card.PITCHER_K_SHADOW_SUMMARY_PATH.read_text(encoding="utf-8"))

    assert grade_result["updated_rows"] == 2
    assert loaded["actual_value"].astype(str).tolist() == ["7", "7"]
    assert loaded["result"].tolist() == ["W", "W"]
    assert report["summary"]["available"] is True
    assert summary["available"] is True
    assert summary["promotion_review"]["review_status"] == "insufficient_evidence"
    assert summary["promotion_review"]["recommended_action"] == "hold"
    assert summary["model_registry"]["champion"]["model_name"] == "xgboost_champion"
    assert summary["model_registry"]["challenger"]["model_name"] == "ridge_challenger"
    assert daily_card.PITCHER_K_SHADOW_OVERLAP_PATH.exists()
    assert daily_card.PITCHER_K_SHADOW_REGRESSION_PLOT_PATH.exists()
    assert daily_card.PITCHER_K_SHADOW_WORKFLOW_PLOT_PATH.exists()
