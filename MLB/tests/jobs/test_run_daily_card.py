import json

import pandas as pd
import pytest
import requests

from jobs import run_daily_card as daily_card
from common.workflows import ModelingWorkflowSpec, PropFieldSpec, ProjectionOddsJoinKeys, WorkflowArtifactSpec
from odds.policy import (
    DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    PostablePickLimits,
)
from pitcher_bb.config import PITCHER_BB_PROP_MARKET
from pitcher_bb.workflow import MLB_PITCHER_WALK_WORKFLOW
from pitcher_k.config import PITCHER_K_PROP_MARKET


def test_tracking_artifact_paths_are_isolated_from_committed_repo_files(tmp_path):
    committed_tracking_dir = daily_card.PROJECT_ROOT / "data" / "tracking"

    assert daily_card.TRACKING_DIR != committed_tracking_dir
    assert daily_card.OFFICIAL_PICKS_HISTORY_PATH.parent == daily_card.TRACKING_DIR
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_HISTORY_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_GRADES_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_BOOK_SUMMARY_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_SKIPPED_PATH.parents
    assert committed_tracking_dir not in daily_card.OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.parents


def test_run_daily_card_writes_outputs_with_mocked_dependencies(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )

    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
            }
        ]
    )

    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    picks_df = pd.DataFrame(
    [
        {
            "player_name": "Jacob deGrom",
            "prop_type": "pitcher_k",
            "team": "TEX",
            "opponent": "SEA",
            "predicted_strikeouts": 6.8,
            "book": "DraftKings",
            "pick_side": "over",
            "line": 5.5,
            "price": -120,
            "edge": 1.3,
            "implied_probability": 120 / 220,
            "value_score": 1.3 * (1 - (120 / 220)),
            "adjusted_value_score": 1.3 * (1 - (120 / 220)),
            "archetype_risk_score": 0.0,
            "confidence_tier": "medium",
            "pick_type": "official",
        }
    ])

    post_df = picks_df.copy()

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(
        daily_card,
        "load_model_metadata",
        lambda workflow=None: {"target": "strikeouts", "features": ["pitches_last3"]},
    )
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )

    def fake_run_edge_pipeline(preds, market, **kwargs):
        assert market == PITCHER_K_PROP_MARKET
        return joined_df, joined_df, {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1}

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)
    monkeypatch.setattr(
        daily_card,
        "build_daily_picks",
        lambda joined, policy, prediction_column="predicted_value": picks_df,
    )
    monkeypatch.setattr(
        daily_card,
        "filter_postable_picks",
        lambda picks, max_official=3, max_leans=1, policy=None: post_df,
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_GRADES_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_report.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_BOOK_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_by_book.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary_all_time.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary_current_regime.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_SKIPPED_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_skipped.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH",
        tmp_path / "data" / "tracking" / "official_picks_concentration_audit.json",
    )

    saved_starters = {}

    def fake_save_today_starters_csv(df, output_dir=None, filename=None):
        out_dir = output_dir or (tmp_path / "data" / "inputs" / "starters")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (filename or "today_starters.csv")
        df.to_csv(out_path, index=False)
        saved_starters["path"] = out_path
        return out_path

    monkeypatch.setattr(daily_card, "save_today_starters_csv", fake_save_today_starters_csv)

    result_starters, result_preds, result_picks, result_post = daily_card.run_daily_card()

    assert not result_starters.empty
    assert not result_preds.empty
    assert not result_picks.empty
    assert not result_post.empty

    assert saved_starters["path"].exists()
    assert (daily_card.PROJECTIONS_DIR / "today_projections.csv").exists()
    assert (daily_card.EDGES_DIR / "today_joined_edges.csv").exists()
    assert (daily_card.PICKS_DIR / "today_all_picks.csv").exists()
    assert (daily_card.PICKS_DIR / "today_postable_picks.csv").exists()
    assert daily_card.OFFICIAL_PICKS_HISTORY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_GRADES_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_BOOK_SUMMARY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_SKIPPED_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.exists()

    loaded_post = pd.read_csv(daily_card.PICKS_DIR / "today_postable_picks.csv")
    loaded_history = pd.read_csv(daily_card.OFFICIAL_PICKS_HISTORY_PATH, keep_default_na=False)
    loaded_grades = pd.read_csv(daily_card.OFFICIAL_PICKS_GRADES_PATH)
    loaded_book_summary = pd.read_csv(daily_card.OFFICIAL_PICKS_BOOK_SUMMARY_PATH)
    loaded_overall = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    loaded_all_time_summary = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    loaded_current_regime_summary = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    assert len(loaded_post) == 1
    assert loaded_post.loc[0, "player_name"] == "Jacob deGrom"
    assert loaded_post.loc[0, "prop_type"] == "pitcher_k"
    assert loaded_post.loc[0, "pick_type"] == "official"
    assert "value_score" in loaded_post.columns
    assert "adjusted_value_score" in loaded_post.columns
    assert "archetype_risk_score" in loaded_post.columns
    assert len(loaded_history) == 1
    assert loaded_history.loc[0, "game_date"] == "2026-04-19"
    assert str(loaded_history.loc[0, "odds"]) == "-120"
    assert loaded_history.loc[0, "pick_key"] == "2026-04-19|jacob degrom"
    assert loaded_grades.empty
    assert loaded_book_summary.empty
    assert loaded_overall["current_regime_rule"] == {
        "type": "start_date",
        "start_date": daily_card.CURRENT_REGIME_START_DATE,
    }
    assert loaded_overall["summary_views"]["all_time"]["picks"] == 0
    assert loaded_overall["summary_views"]["all_time"]["skipped_rows"] == 1
    assert loaded_overall["summary_views"]["current_regime"]["picks"] == 0
    assert loaded_overall["published_view_artifacts"] == {
        "all_time": "official_picks_profit_summary_all_time.json",
        "current_regime": "official_picks_profit_summary_current_regime.json",
    }
    assert loaded_all_time_summary["summary_scope"] == "all_time"
    assert loaded_all_time_summary["summary_metrics"]["skipped_rows"] == 1
    assert loaded_current_regime_summary["summary_scope"] == "current_regime"
    assert loaded_current_regime_summary["summary_metrics"]["picks"] == 0

    loaded_audit = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH.read_text(encoding="utf-8")
    )
    assert loaded_audit["artifact_type"] == "official_picks_concentration_audit"
    assert loaded_audit["scopes"]["all_time"]["summary"]["official_picks"] == 1
    assert loaded_audit["scopes"]["all_time"]["questions"]["largest_share_of_official_picks"][0]["player_name"] == "Jacob deGrom"


def test_run_daily_card_allows_explicit_market_and_workflow_behavior(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )

    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
            }
        ]
    )

    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )

    picks_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "implied_probability": 120 / 220,
                "value_score": 1.3 * (1 - (120 / 220)),
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )
    post_df = picks_df.copy()

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")

    custom_market = "custom_market"
    calls = {}

    def fake_run_edge_pipeline(preds, market, **kwargs):
        calls["market"] = market
        calls["join_kwargs"] = kwargs
        return joined_df, joined_df, {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1}

    def fake_build_picks(joined):
        calls["build_picks_joined"] = joined.copy()
        return picks_df

    def fake_filter_postable(picks):
        calls["filter_postable_picks"] = picks.copy()
        return post_df

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)

    daily_card.run_daily_card(
        market=custom_market,
        build_picks_fn=fake_build_picks,
        filter_postable_picks_fn=fake_filter_postable,
    )

    assert calls["market"] == custom_market
    assert calls["join_kwargs"] == {
        "participant_key": "player_name",
        "prediction_column": "predicted_value",
        "projection_join_key": "player_name_norm",
        "odds_join_key": "player_name_norm",
        "sport": "MLB",
    }
    pd.testing.assert_frame_equal(
        calls["build_picks_joined"],
        joined_df.assign(
            predicted_value=6.8,
            prop_type="pitcher_k",
            record_source="run_daily_card",
            model_version="pitcher_k_artifact_v1",
            policy_version="mlb_pitcher_props_policy_v1",
            tracking_regime="legacy_workflow",
        ),
    )
    pd.testing.assert_frame_equal(
        calls["filter_postable_picks"],
        picks_df.assign(
            prop_type="pitcher_k",
            record_source="run_daily_card",
            model_version="pitcher_k_artifact_v1",
            policy_version="mlb_pitcher_props_policy_v1",
            tracking_regime="legacy_workflow",
        ),
    )


def test_run_daily_card_raises_when_today_predictions_are_empty(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(
        daily_card,
        "load_model_metadata",
        lambda workflow=None: {"target": "strikeouts", "features": ["pitches_last3"]},
    )
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: pd.DataFrame(),
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )

    with pytest.raises(ValueError, match="No today predictions were generated."):
        daily_card.run_daily_card()

def test_run_daily_card_raises_when_pitcher_games_artifact_is_missing(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)

    def raise_missing_pitcher_games(workflow):
        raise FileNotFoundError("Missing pitcher_games artifact: fake/path/pitcher_games.csv")

    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", raise_missing_pitcher_games)

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )

    with pytest.raises(FileNotFoundError, match="Missing pitcher_games artifact"):
        daily_card.run_daily_card()


def test_run_daily_card_uses_workflow_spec_for_market_policy_and_limits(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
                "std_dev": 1.0,
            }
        ]
    )
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "pick_type": "official",
                "implied_probability": 120 / 220,
                "value_score": 1.3 * (1 - (120 / 220)),
                "confidence_tier": "medium",
            }
        ]
    )
    post_df = picks_df.copy()
    calls: dict[str, object] = {}

    workflow = ModelingWorkflowSpec(
        prop_type="custom_prop",
        sport="MLB",
        participant_key="player_name",
        market_key="workflow_market",
        artifacts=WorkflowArtifactSpec(
            history_filename="history.csv",
            history_loader=lambda path: pd.DataFrame(),
            model_filename="model.bin",
            model_loader=lambda path: "unused-model",
        ),
        feature_builder=lambda starters, history: starters,
        predictor=lambda model, features: features,
        projection_odds_join_keys=ProjectionOddsJoinKeys(
            projection="participant_join_key",
            odds="participant_join_key",
        ),
        pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
        prediction_columns=(
            "player_name",
            "predicted_strikeouts",
            "lower_bound",
            "upper_bound",
            "std_dev",
        ),
        postable_limits=PostablePickLimits(max_official=1, max_leans=0),
    )

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")

    def fake_run_edge_pipeline(preds, market, **kwargs):
        calls["market"] = market
        calls["join_kwargs"] = kwargs
        return joined_df, joined_df, {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1}

    def fake_build_daily_picks(df, policy, prediction_column="predicted_value"):
        calls["build_policy"] = policy
        calls["prediction_column"] = prediction_column
        return picks_df

    def fake_filter_postable_picks(df, max_official, max_leans, policy):
        calls["filter_limits"] = (max_official, max_leans)
        calls["filter_policy"] = policy
        return post_df

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)
    monkeypatch.setattr(daily_card, "build_daily_picks", fake_build_daily_picks)
    monkeypatch.setattr(daily_card, "filter_postable_picks", fake_filter_postable_picks)

    daily_card.run_daily_card(workflow=workflow)

    assert calls["market"] == "workflow_market"
    assert calls["join_kwargs"] == {
        "participant_key": "player_name",
        "prediction_column": "predicted_value",
        "projection_join_key": "participant_join_key",
        "odds_join_key": "participant_join_key",
        "sport": "MLB",
    }
    assert calls["build_policy"] is workflow.pick_ranking_policy
    assert calls["prediction_column"] == "predicted_value"
    assert calls["filter_policy"] is workflow.pick_ranking_policy
    assert calls["filter_limits"] == (1, 0)


def test_run_daily_card_default_workflow_joins_live_odds_on_normalized_name(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )

    today_preds = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
                "std_dev": 1.0,
            }
        ]
    )

    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "prop_type": "pitcher_k",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "pick_type": "official",
                "implied_probability": 0.545,
                "value_score": 0.5915,
                "confidence_tier": "medium",
            }
        ]
    )

    calls = {}

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(
        daily_card,
        "load_model_metadata",
        lambda workflow=None: {"target": "strikeouts", "features": ["pitches_last3"]},
    )
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )

    def fake_run_edge_pipeline(preds, market, **kwargs):
        calls["joined_preds"] = preds.copy()
        calls["join_kwargs"] = kwargs
        return joined_df, joined_df, {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1}

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)
    monkeypatch.setattr(
        daily_card,
        "build_daily_picks",
        lambda joined, policy, prediction_column="predicted_value": picks_df,
    )
    monkeypatch.setattr(
        daily_card,
        "filter_postable_picks",
        lambda picks, max_official=3, max_leans=1, policy=None: picks,
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")

    daily_card.run_daily_card()

    assert calls["join_kwargs"] == {
        "participant_key": "player_name",
        "prediction_column": "predicted_value",
        "projection_join_key": "player_name_norm",
        "odds_join_key": "player_name_norm",
        "sport": "MLB",
    }
    assert calls["joined_preds"].loc[0, "pitcher"] == 1
    assert calls["joined_preds"].loc[0, "player_name_norm"] == "jacob degrom"
    assert calls["joined_preds"].loc[0, "prop_type"] == "pitcher_k"


def test_build_today_predictions_for_pitcher_walk_workflow_adds_shared_prediction_and_market_identity():
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 5,
                "player_name": "Tarik Skubal",
                "team": "DET",
                "opponent": "CLE",
                "home_team": "DET",
                "away_team": "CLE",
                "is_home": 1,
                "p_throws": "L",
            }
        ]
    )
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 5,
                "player_name": "Tarik Skubal",
                "pitching_team": "DET",
                "opponent_team": "CLE",
            }
        ]
    )

    workflow = ModelingWorkflowSpec(
        prop_type="pitcher_bb",
        sport="MLB",
        participant_key="player_name",
        market_key=PITCHER_BB_PROP_MARKET,
        artifacts=WorkflowArtifactSpec(
            history_filename="bb_history.csv",
            history_loader=lambda path: pd.DataFrame(),
            model_filename="bb_model.ubj",
            model_loader=lambda path: "unused",
        ),
        feature_builder=lambda starters, history: starters.assign(walks_last10=2.0),
        predictor=lambda model, features: features.assign(
            predicted_walks=2.2,
            lower_bound=1.4,
            upper_bound=3.0,
            std_dev=0.8,
        ),
        projection_odds_join_keys=ProjectionOddsJoinKeys(
            projection="player_name_norm",
            odds="player_name_norm",
        ),
        pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
        prediction_columns=("player_name", "predicted_walks", "lower_bound", "upper_bound", "std_dev"),
        prop_fields=PropFieldSpec(
            prediction="predicted_walks",
            actual="actual_walks",
        ),
    )

    today_preds = daily_card.build_today_predictions_for_workflow(
        starters_df=starters_df,
        pitcher_games=pitcher_games,
        model="fake_model",
        workflow=workflow,
    )

    assert today_preds.loc[0, "predicted_walks"] == pytest.approx(2.2)
    assert today_preds.loc[0, "predicted_value"] == pytest.approx(2.2)
    assert today_preds.loc[0, "market_key"] == PITCHER_BB_PROP_MARKET
    assert today_preds.loc[0, "participant_name_norm"] == "tarik skubal"


def test_resolve_daily_card_workflows_includes_pitcher_walks_only_when_ready(monkeypatch):
    monkeypatch.setattr(
        daily_card,
        "DEFAULT_DAILY_CARD_WORKFLOWS",
        [daily_card.MLB_PITCHER_STRIKEOUT_WORKFLOW, daily_card.MLB_PITCHER_WALK_WORKFLOW],
    )

    monkeypatch.setattr(
        daily_card,
        "workflow_is_ready",
        lambda workflow: workflow is daily_card.MLB_PITCHER_STRIKEOUT_WORKFLOW,
    )
    workflows_without_bb = daily_card.resolve_daily_card_workflows()
    assert workflows_without_bb == [daily_card.MLB_PITCHER_STRIKEOUT_WORKFLOW]

    monkeypatch.setattr(daily_card, "workflow_is_ready", lambda workflow: True)
    workflows_with_bb = daily_card.resolve_daily_card_workflows()
    assert workflows_with_bb == [
        daily_card.MLB_PITCHER_STRIKEOUT_WORKFLOW,
        daily_card.MLB_PITCHER_WALK_WORKFLOW,
    ]


def test_build_no_edges_message_distinguishes_empty_odds_from_join_miss():
    no_odds_message = daily_card._build_no_edges_message(
        workflow=MLB_PITCHER_WALK_WORKFLOW,
        diagnostics={
            "fetch_scope": "all_region_books",
            "raw_event_count": 0,
            "normalized_odds_rows": 0,
            "joined_rows": 0,
            "initial_fetch": {"normalized_odds_rows": 0},
        },
    )
    no_join_message = daily_card._build_no_edges_message(
        workflow=MLB_PITCHER_WALK_WORKFLOW,
        diagnostics={
            "fetch_scope": "configured_books",
            "raw_event_count": 7,
            "normalized_odds_rows": 24,
            "joined_rows": 0,
        },
    )

    assert "No live odds rows were returned" in no_odds_message
    assert "none matched today's projections" in no_join_message


def test_run_daily_card_combines_ready_workflow_outputs(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")
    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_GRADES_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_report.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_BOOK_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_by_book.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_SKIPPED_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_skipped.csv",
    )

    workflows = [
        ModelingWorkflowSpec(
            prop_type="pitcher_k",
            sport="MLB",
            participant_key="player_name",
            market_key="pitcher_strikeouts",
            artifacts=WorkflowArtifactSpec(
                history_filename="k_history.csv",
                history_loader=lambda path: pd.DataFrame(),
                model_filename="k_model.ubj",
                model_loader=lambda path: "unused",
            ),
            feature_builder=lambda starters, history: starters,
            predictor=lambda model, features: features,
            projection_odds_join_keys=ProjectionOddsJoinKeys(
                projection="player_name_norm",
                odds="player_name_norm",
            ),
            pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
            prediction_columns=("player_name", "predicted_strikeouts", "lower_bound", "upper_bound", "std_dev"),
        ),
        ModelingWorkflowSpec(
            prop_type="pitcher_bb",
            sport="MLB",
            participant_key="player_name",
            market_key=PITCHER_BB_PROP_MARKET,
            artifacts=WorkflowArtifactSpec(
                history_filename="bb_history.csv",
                history_loader=lambda path: pd.DataFrame(),
                model_filename="bb_model.ubj",
                model_loader=lambda path: "unused",
            ),
            feature_builder=lambda starters, history: starters,
            predictor=lambda model, features: features,
            projection_odds_join_keys=ProjectionOddsJoinKeys(
                projection="player_name_norm",
                odds="player_name_norm",
            ),
            pick_ranking_policy=DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
            prediction_columns=("player_name", "predicted_walks", "lower_bound", "upper_bound", "std_dev"),
            prop_fields=PropFieldSpec(
                prediction="predicted_walks",
                actual="actual_walks",
            ),
        ),
    ]

    def fake_run_workflow_daily_card(*, starters_df, workflow, market, build_picks_fn, filter_postable_picks_fn):
        if workflow.prop_type == "pitcher_k":
            return (
                pd.DataFrame([{"player_name": "Jacob deGrom", "predicted_strikeouts": 6.8, "prop_type": "pitcher_k"}]),
                pd.DataFrame([{"player_name_proj": "Jacob deGrom", "predicted_strikeouts": 6.8, "bookmaker": "DraftKings", "side": "Over", "line": 5.5, "price": -120, "prop_type": "pitcher_k"}]),
                pd.DataFrame([{"player_name": "Jacob deGrom", "prop_type": "pitcher_k", "book": "DraftKings", "pick_side": "over", "line": 5.5, "price": -120, "edge": 1.3, "pick_type": "official", "implied_probability": 0.545, "value_score": 0.59, "confidence_tier": "medium"}]),
                pd.DataFrame([{"player_name": "Jacob deGrom", "prop_type": "pitcher_k", "book": "DraftKings", "pick_side": "over", "line": 5.5, "price": -120, "edge": 1.3, "pick_type": "official", "implied_probability": 0.545, "value_score": 0.59, "confidence_tier": "medium"}]),
                "success",
                None,
            )
        return (
            pd.DataFrame([{"player_name": "Tarik Skubal", "predicted_walks": 2.2, "predicted_value": 2.2, "prop_type": "pitcher_bb"}]),
            pd.DataFrame([{"player_name_proj": "Tarik Skubal", "predicted_walks": 2.2, "predicted_value": 2.2, "bookmaker": "FanDuel", "side": "Under", "line": 2.5, "price": -105, "prop_type": "pitcher_bb"}]),
            pd.DataFrame([{"player_name": "Tarik Skubal", "prop_type": "pitcher_bb", "book": "FanDuel", "pick_side": "under", "line": 2.5, "price": -105, "edge": 0.3, "pick_type": "lean", "implied_probability": 0.512, "value_score": 0.146, "confidence_tier": "low"}]),
            pd.DataFrame([{"player_name": "Tarik Skubal", "prop_type": "pitcher_bb", "book": "FanDuel", "pick_side": "under", "line": 2.5, "price": -105, "edge": 0.3, "pick_type": "lean", "implied_probability": 0.512, "value_score": 0.146, "confidence_tier": "low"}]),
            "success",
            None,
        )

    monkeypatch.setattr(daily_card, "run_workflow_daily_card", fake_run_workflow_daily_card)

    _, result_preds, result_picks, result_post = daily_card.run_daily_card(workflows=workflows)

    assert set(result_preds["prop_type"]) == {"pitcher_k", "pitcher_bb"}
    assert set(result_picks["prop_type"]) == {"pitcher_k", "pitcher_bb"}
    assert set(result_post["prop_type"]) == {"pitcher_k", "pitcher_bb"}
    assert list(result_post["player_name"]) == ["Jacob deGrom", "Tarik Skubal"]
    assert result_preds.loc[result_preds["prop_type"] == "pitcher_bb", "predicted_walks"].iloc[0] == pytest.approx(2.2)


def test_run_workflow_daily_card_uses_pitcher_walk_market_and_validates_joined_odds(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 5,
                "player_name": "Tarik Skubal",
                "team": "DET",
                "opponent": "CLE",
                "home_team": "DET",
                "away_team": "CLE",
                "is_home": 1,
                "p_throws": "L",
            }
        ]
    )
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 5,
                "player_name": "Tarik Skubal",
                "pitching_team": "DET",
                "opponent_team": "CLE",
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Tarik Skubal",
                "player_name_norm": "tarik skubal",
                "team": "DET",
                "opponent": "CLE",
                "predicted_walks": 2.2,
                "predicted_value": 2.2,
                "lower_bound": 1.4,
                "upper_bound": 3.0,
                "std_dev": 0.8,
                "market_key": PITCHER_BB_PROP_MARKET,
            }
        ]
    )
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Tarik Skubal",
                "player_name_norm": "tarik skubal",
                "predicted_walks": 2.2,
                "predicted_value": 2.2,
                "bookmaker": "FanDuel",
                "side": "Under",
                "line": 2.5,
                "price": -105,
                "market_key": PITCHER_BB_PROP_MARKET,
            }
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "Tarik Skubal",
                "book": "FanDuel",
                "pick_side": "under",
                "line": 2.5,
                "price": -105,
                "edge": 0.3,
                "pick_type": "lean",
                "implied_probability": 0.512,
                "value_score": 0.146,
                "confidence_tier": "low",
            }
        ]
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "walks"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )

    def fake_run_edge_pipeline(preds, market, **kwargs):
        calls["market"] = market
        calls["join_kwargs"] = kwargs
        calls["preds"] = preds.copy()
        return joined_df, joined_df, {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1}

    def fake_build_daily_picks(df, policy, prediction_column="predicted_value"):
        calls["prediction_column"] = prediction_column
        calls["validated_joined"] = df.copy()
        return picks_df

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)
    monkeypatch.setattr(daily_card, "build_daily_picks", fake_build_daily_picks)
    monkeypatch.setattr(
        daily_card,
        "filter_postable_picks",
        lambda df, max_official=3, max_leans=1, policy=None: df,
    )

    _, result_joined, result_picks, result_post, result_status, result_message = daily_card.run_workflow_daily_card(
        starters_df=starters_df,
        workflow=MLB_PITCHER_WALK_WORKFLOW,
    )

    assert calls["market"] == PITCHER_BB_PROP_MARKET
    assert calls["join_kwargs"] == {
        "participant_key": "player_name",
        "prediction_column": "predicted_value",
        "projection_join_key": "player_name_norm",
        "odds_join_key": "player_name_norm",
        "sport": "MLB",
    }
    assert calls["preds"].loc[0, "predicted_walks"] == pytest.approx(2.2)
    assert calls["preds"].loc[0, "predicted_value"] == pytest.approx(2.2)
    assert calls["prediction_column"] == "predicted_value"
    assert result_joined.loc[0, "prop_type"] == "pitcher_bb"
    assert result_picks.loc[0, "prop_type"] == "pitcher_bb"
    assert result_post.loc[0, "prop_type"] == "pitcher_bb"
    assert result_status == "success"
    assert result_message is None


def test_run_workflow_daily_card_allows_empty_postable_picks(monkeypatch):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "pitcher": 1,
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
            }
        ]
    )
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Jacob deGrom",
                "player_name_norm": "jacob degrom",
                "predicted_value": 6.8,
                "predicted_strikeouts": 6.8,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 5.5,
                "price": -120,
            }
        ]
    )
    picks_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "pick_type": "official",
                "implied_probability": 0.545,
                "value_score": 0.5915,
                "confidence_tier": "medium",
            }
        ]
    )

    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(
        daily_card,
        "run_edge_pipeline",
        lambda preds, market, **kwargs: (
            joined_df,
            joined_df,
            {"raw_event_count": 1, "normalized_odds_rows": 1, "joined_rows": 1},
        ),
    )
    monkeypatch.setattr(
        daily_card,
        "build_daily_picks",
        lambda joined, policy, prediction_column="predicted_value": picks_df,
    )
    monkeypatch.setattr(
        daily_card,
        "filter_postable_picks",
        lambda picks, max_official=3, max_leans=1, policy=None: daily_card.empty_final_picks_df(),
    )

    _, _, result_picks, result_post, result_status, result_message = daily_card.run_workflow_daily_card(
        starters_df=starters_df,
        workflow=daily_card.MLB_PITCHER_STRIKEOUT_WORKFLOW,
    )

    assert len(result_picks) == 1
    assert result_post.empty
    assert set(daily_card.empty_final_picks_df().columns).issubset(result_post.columns)
    assert result_status == "success"
    assert result_message is None


def test_run_daily_card_preserves_other_workflow_picks_when_one_workflow_has_no_postable_rows(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
    strikeout_picks = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "prop_type": "pitcher_k",
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "pick_type": "official",
                "implied_probability": 0.545,
                "value_score": 0.5915,
                "confidence_tier": "medium",
                "record_source": "run_daily_card",
                "model_version": "pitcher_k_artifact_v1",
                "policy_version": "mlb_pitcher_props_policy_v1",
                "tracking_regime": "legacy_workflow",
            }
        ]
    )
    walk_picks = daily_card.empty_final_picks_df().assign(
        prop_type=pd.Series(dtype="object"),
        record_source=pd.Series(dtype="object"),
        model_version=pd.Series(dtype="object"),
        policy_version=pd.Series(dtype="object"),
        tracking_regime=pd.Series(dtype="object"),
    )
    calls = {"count": 0}

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")
    monkeypatch.setattr(daily_card, "grade_official_picks_from_statcast", lambda: None)
    monkeypatch.setattr(daily_card, "persist_official_picks_profit_reports", lambda *args, **kwargs: {})

    def fake_run_workflow_daily_card(*, starters_df, workflow, market, build_picks_fn, filter_postable_picks_fn):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                pd.DataFrame([{"player_name": "Jacob deGrom", "prop_type": "pitcher_k"}]),
                daily_card.empty_joined_odds_df(),
                strikeout_picks,
                strikeout_picks.copy(),
                "success",
                None,
            )
        return (
            pd.DataFrame([{"player_name": "Jacob deGrom", "prop_type": "pitcher_bb"}]),
            daily_card.empty_joined_odds_df(),
            walk_picks.copy(),
            walk_picks.copy(),
            "success",
            None,
        )

    monkeypatch.setattr(daily_card, "run_workflow_daily_card", fake_run_workflow_daily_card)

    _, _, result_picks, result_post = daily_card.run_daily_card()

    assert list(result_picks["prop_type"]) == ["pitcher_k"]
    assert list(result_post["prop_type"]) == ["pitcher_k"]
    saved_post = pd.read_csv(tmp_path / "data" / "outputs" / "picks" / "today_postable_picks.csv")
    assert list(saved_post["prop_type"]) == ["pitcher_k"]


def test_persist_official_picks_history_is_idempotent_and_preserves_manual_results(tmp_path, monkeypatch):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
    post_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )

    tracking_dir = tmp_path / "data" / "tracking"
    history_path = tracking_dir / "official_picks_history.csv"

    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    seed_history = pd.DataFrame(
        [
            {
                "pick_key": "2026-04-19|jacob degrom",
                "game_date": "2026-04-19",
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "book": "DraftKings",
                "odds": "-120",
                "price": -120,
                "pick_side": "over",
                "line": 5.5,
                "predicted_strikeouts": 6.6,
                "edge": 1.1,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "W",
                "actual_strikeouts": "7",
                "record_source": "manual_seed",
            }
        ]
    )
    seed_history.to_csv(history_path, index=False)

    daily_card.persist_official_picks_history(starters_df, post_df)
    daily_card.persist_official_picks_history(starters_df, post_df)

    loaded_history = pd.read_csv(history_path, keep_default_na=False)

    assert len(loaded_history) == 1
    assert loaded_history.loc[0, "result"] == "W"
    assert str(loaded_history.loc[0, "actual_strikeouts"]) == "7"
    assert loaded_history.loc[0, "predicted_strikeouts"] == pytest.approx(6.8)
    assert loaded_history.loc[0, "edge"] == pytest.approx(1.3)
    assert loaded_history.loc[0, "record_source"] == "manual_seed"
    assert loaded_history.loc[0, "model_version"] == daily_card.LEGACY_MANUAL_MODEL_VERSION
    assert loaded_history.loc[0, "policy_version"] == daily_card.LEGACY_MANUAL_POLICY_VERSION
    assert loaded_history.loc[0, "tracking_regime"] == daily_card.TRACKING_REGIME_MANUAL_BACKFILL


def test_build_official_picks_history_rows_handles_existing_game_date_in_post_df():
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
    post_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )

    history_rows = daily_card.build_official_picks_history_rows(starters_df, post_df)

    assert len(history_rows) == 1
    assert history_rows.loc[0, "game_date"] == "2026-04-19"


def test_build_official_picks_history_rows_preserves_explicit_provenance_fields():
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-10",
                "pitcher": 1,
                "player_name": "Tarik Skubal",
                "team": "DET",
                "opponent": "CLE",
            }
        ]
    )
    post_df = pd.DataFrame(
        [
            {
                "player_name": "Tarik Skubal",
                "team": "DET",
                "opponent": "CLE",
                "predicted_value": 2.2,
                "book": "FanDuel",
                "pick_side": "under",
                "line": 2.5,
                "price": -105,
                "edge": 0.3,
                "confidence_tier": "low",
                "pick_type": "official",
                "record_source": "run_daily_card",
                "model_version": "pitcher_bb_model_v1",
                "policy_version": "mlb_pitcher_props_policy_v1",
                "tracking_regime": "current_workflow",
            }
        ]
    )

    history_rows = daily_card.build_official_picks_history_rows(starters_df, post_df)

    assert history_rows.loc[0, "record_source"] == "run_daily_card"
    assert history_rows.loc[0, "model_version"] == "pitcher_bb_model_v1"
    assert history_rows.loc[0, "policy_version"] == "mlb_pitcher_props_policy_v1"
    assert history_rows.loc[0, "tracking_regime"] == "current_workflow"
    assert history_rows.loc[0, "pick_key"] == "2026-05-10|tarik skubal"


def test_build_official_picks_history_rows_falls_back_to_unique_starter_date_when_merge_keeps_only_suffixed_dates():
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
    post_df = pd.DataFrame(
        [
            {
                "game_date_x": "2026-04-19",
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "book": "DraftKings",
                "pick_side": "over",
                "line": 5.5,
                "price": -120,
                "edge": 1.3,
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )

    history_rows = daily_card.build_official_picks_history_rows(starters_df, post_df)

    assert len(history_rows) == 1
    assert history_rows.loc[0, "game_date"] == "2026-04-19"
    assert history_rows.loc[0, "pick_key"] == "2026-04-19|jacob degrom"


def test_build_official_picks_history_rows_uses_canonical_offer_identity_when_available():
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
    post_df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "participant_join_key": "mlbam_player:1",
                "participant_id": "mlbam_player:1",
                "participant_source_id": "1",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "jacob degrom",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "team": "TEX",
                "opponent": "SEA",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_1",
                "pick_side": "over",
                "line": 5.5,
                "market_selection_key": "MLB|pitcher_strikeouts|mlbam_player:1|over|5.5",
                "market_offer_key": "MLB|pitcher_strikeouts|mlbam_player:1|over|5.5|draftkings",
                "price": -120,
                "predicted_strikeouts": 6.8,
                "edge": 1.3,
                "confidence_tier": "medium",
                "pick_type": "official",
            }
        ]
    )

    history_rows = daily_card.build_official_picks_history_rows(starters_df, post_df)

    assert len(history_rows) == 1
    assert history_rows.loc[0, "participant_join_key"] == "mlbam_player:1"
    assert history_rows.loc[0, "market_key"] == "pitcher_strikeouts"
    assert history_rows.loc[0, "pick_key"] == (
        "2026-04-19|MLB|pitcher_strikeouts|mlbam_player:1|over|5.5|draftkings"
    )


def test_run_daily_card_handles_live_odds_http_error_gracefully(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
                "std_dev": 1.0,
            }
        ]
    )

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(
        daily_card,
        "run_edge_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.HTTPError("401 Client Error")),
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_GRADES_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_report.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_BOOK_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_by_book.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_SKIPPED_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_skipped.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "RUN_STATUS_PATH",
        tmp_path / "data" / "outputs" / "run_daily_card_status.json",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")

    _, result_preds, result_picks, result_post = daily_card.run_daily_card()

    assert not result_preds.empty
    assert result_picks.empty
    assert result_post.empty

    loaded_edges = pd.read_csv(daily_card.EDGES_DIR / "today_joined_edges.csv")
    loaded_picks = pd.read_csv(daily_card.PICKS_DIR / "today_all_picks.csv")
    loaded_post = pd.read_csv(daily_card.PICKS_DIR / "today_postable_picks.csv")
    status_payload = daily_card.json.loads(daily_card.RUN_STATUS_PATH.read_text(encoding="utf-8"))

    assert loaded_edges.empty
    assert loaded_picks.empty
    assert loaded_post.empty
    assert status_payload["status"] == "degraded"
    assert "Live odds fetch failed" in status_payload["message"]


def test_run_daily_card_handles_empty_joined_odds_gracefully(monkeypatch, tmp_path):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
                "std_dev": 1.0,
            }
        ]
    )

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(
        daily_card,
        "run_edge_pipeline",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            {"raw_event_count": 0, "normalized_odds_rows": 0, "joined_rows": 0},
        ),
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tmp_path / "data" / "tracking")
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_HISTORY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_history.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_GRADES_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_report.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_BOOK_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_by_book.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_summary.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_SKIPPED_PATH",
        tmp_path / "data" / "tracking" / "official_picks_profit_skipped.csv",
    )
    monkeypatch.setattr(
        daily_card,
        "RUN_STATUS_PATH",
        tmp_path / "data" / "outputs" / "run_daily_card_status.json",
    )
    monkeypatch.setattr(daily_card, "save_today_starters_csv", lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv")

    _, result_preds, result_picks, result_post = daily_card.run_daily_card()

    assert not result_preds.empty
    assert result_picks.empty
    assert result_post.empty

    loaded_edges = pd.read_csv(daily_card.EDGES_DIR / "today_joined_edges.csv")
    loaded_picks = pd.read_csv(daily_card.PICKS_DIR / "today_all_picks.csv")
    loaded_post = pd.read_csv(daily_card.PICKS_DIR / "today_postable_picks.csv")
    status_payload = daily_card.json.loads(daily_card.RUN_STATUS_PATH.read_text(encoding="utf-8"))

    assert loaded_edges.empty
    assert loaded_picks.empty
    assert loaded_post.empty
    assert status_payload["status"] == "degraded"
    assert "No live odds rows were returned" in status_payload["message"]


def test_run_daily_card_degraded_run_does_not_overwrite_existing_tracking_summaries(
    monkeypatch,
    tmp_path,
):
    starters_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 1,
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
                "game_date": "2026-04-18",
                "game_pk": 111111,
                "pitcher": 1,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "predicted_strikeouts": 6.8,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
                "std_dev": 1.0,
            }
        ]
    )

    tracking_dir = tmp_path / "data" / "tracking"
    outputs_dir = tmp_path / "data" / "outputs"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    all_time_summary_path = tracking_dir / "official_picks_profit_summary_all_time.json"
    current_regime_summary_path = tracking_dir / "official_picks_profit_summary_current_regime.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"
    history_path = tracking_dir / "official_picks_history.csv"

    tracking_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"pick_key": "seed", "player_name": "Pitcher A"}]).to_csv(grades_path, index=False)
    pd.DataFrame([{"book": "DraftKings", "picks": 1}]).to_csv(by_book_path, index=False)
    pd.DataFrame([{"pick_key": "pending-seed", "player_name": "Pitcher B"}]).to_csv(skipped_path, index=False)
    summary_path.write_text(json.dumps({"picks": 1, "skipped_rows": 1}, indent=2), encoding="utf-8")

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "load_workflow_history_artifact", lambda workflow: pitcher_games)
    monkeypatch.setattr(daily_card, "load_workflow_model_artifact", lambda workflow: "fake_model")
    monkeypatch.setattr(daily_card, "load_model_metadata", lambda workflow=None: {"target": "strikeouts"})
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions_for_workflow",
        lambda *, starters_df, pitcher_games, model, workflow: today_preds,
    )
    monkeypatch.setattr(
        daily_card,
        "run_edge_pipeline",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            {"raw_event_count": 0, "normalized_odds_rows": 0, "joined_rows": 0},
        ),
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", outputs_dir)
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", outputs_dir / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", outputs_dir / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", outputs_dir / "picks")
    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH", all_time_summary_path)
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH",
        current_regime_summary_path,
    )
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)
    monkeypatch.setattr(daily_card, "RUN_STATUS_PATH", outputs_dir / "run_daily_card_status.json")
    monkeypatch.setattr(
        daily_card,
        "save_today_starters_csv",
        lambda df, output_dir=None, filename=None: tmp_path / "today_starters.csv",
    )

    with pytest.raises(ValueError, match="Refusing to overwrite non-empty tracking summaries"):
        daily_card.run_daily_card()

    assert len(pd.read_csv(grades_path)) == 1
    assert len(pd.read_csv(by_book_path)) == 1
    assert len(pd.read_csv(skipped_path)) == 1
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["picks"] == 1
    assert not all_time_summary_path.exists()
    assert not current_regime_summary_path.exists()


def test_load_model_metadata_reads_matching_file_from_selected_artifact_dir(tmp_path, monkeypatch, capsys):
    latest_dir = tmp_path / "artifacts" / "latest"
    previous_dir = tmp_path / "artifacts" / "previous"
    latest_dir.mkdir(parents=True, exist_ok=True)
    previous_dir.mkdir(parents=True, exist_ok=True)

    latest_model = latest_dir / "model.ubj"
    latest_model.write_text("placeholder", encoding="utf-8")
    (latest_dir / "metadata.json").write_text(
        '{"target": "strikeouts", "features": ["pitches_last3"], "evaluation_metrics": {"mae": 0.9}}',
        encoding="utf-8",
    )
    (previous_dir / "model.ubj").write_text("older-placeholder", encoding="utf-8")
    (previous_dir / "metadata.json").write_text(
        '{"target": "old_target"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(daily_card, "LATEST_ARTIFACTS_DIR", latest_dir)
    monkeypatch.setattr(daily_card, "PREVIOUS_ARTIFACTS_DIR", previous_dir)

    metadata = daily_card.load_model_metadata()
    captured = capsys.readouterr()

    assert metadata["target"] == "strikeouts"
    assert metadata["evaluation_metrics"]["mae"] == 0.9
    assert str(latest_dir / "metadata.json") in captured.out
    assert '"target": "strikeouts"' in captured.out


def test_apply_metadata_uncertainty_uses_saved_interval_calibration():
    today_preds = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "predicted_strikeouts": 6.8,
                "std_dev": 1.0,
                "lower_bound": 5.8,
                "upper_bound": 7.8,
            }
        ]
    )
    metadata = {
        "uncertainty_model": {
            "interval_multiplier": 1.4,
            "nominal_coverage": 0.8,
        }
    }

    adjusted = daily_card.apply_metadata_uncertainty(today_preds, metadata)

    assert adjusted.loc[0, "raw_std_dev"] == pytest.approx(1.0)
    assert adjusted.loc[0, "std_dev"] == pytest.approx(1.4)
    assert adjusted.loc[0, "lower_bound"] == pytest.approx(5.4)
    assert adjusted.loc[0, "upper_bound"] == pytest.approx(8.2)


def test_build_official_picks_profit_report_grades_units_and_aggregates_by_book():
    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-01|pitcher a",
                "game_date": "2026-05-01",
                "player_name": "Pitcher A",
                "team": "AAA",
                "opponent": "BBB",
                "book": "DraftKings",
                "odds": "-110",
                "price": -110,
                "pick_side": "over",
                "line": 5.5,
                "predicted_strikeouts": 6.4,
                "edge": 0.9,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "W",
                "actual_strikeouts": "7",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-02|pitcher b",
                "game_date": "2026-05-02",
                "player_name": "Pitcher B",
                "team": "CCC",
                "opponent": "DDD",
                "book": "FanDuel",
                "odds": "",
                "price": 120,
                "pick_side": "under",
                "line": 6.5,
                "predicted_strikeouts": 5.5,
                "edge": 1.0,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "L",
                "actual_strikeouts": "8",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-03|pitcher c",
                "game_date": "2026-05-03",
                "player_name": "Pitcher C",
                "team": "EEE",
                "opponent": "FFF",
                "book": "BetMGM",
                "odds": "",
                "price": "",
                "pick_side": "over",
                "line": 4.5,
                "predicted_strikeouts": 5.1,
                "edge": 0.6,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "Push",
                "actual_strikeouts": "4.5",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-04|pitcher d",
                "game_date": "2026-05-04",
                "player_name": "Pitcher D",
                "team": "GGG",
                "opponent": "HHH",
                "book": "Caesars",
                "odds": "",
                "price": "",
                "pick_side": "over",
                "line": 4.5,
                "predicted_strikeouts": 5.1,
                "edge": 0.6,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "W",
                "actual_strikeouts": "6",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-05|lean row",
                "game_date": "2026-05-05",
                "player_name": "Lean Row",
                "team": "III",
                "opponent": "JJJ",
                "book": "DraftKings",
                "odds": "+100",
                "price": 100,
                "pick_side": "over",
                "line": 3.5,
                "predicted_strikeouts": 4.0,
                "edge": 0.5,
                "confidence_tier": "medium",
                "pick_type": "lean",
                "result": "W",
                "actual_strikeouts": "5",
                "record_source": "run_daily_card",
            },
        ]
    )

    report = daily_card.build_official_picks_profit_report(history_df)

    graded_df = report["graded_df"]
    summary_by_book_df = report["summary_by_book_df"]
    overall_summary = report["overall_summary"]
    skipped_df = report["skipped_df"]

    assert len(graded_df) == 3
    pitcher_a = graded_df.loc[graded_df["player_name"] == "Pitcher A"].iloc[0]
    pitcher_b = graded_df.loc[graded_df["player_name"] == "Pitcher B"].iloc[0]
    pitcher_c = graded_df.loc[graded_df["player_name"] == "Pitcher C"].iloc[0]
    assert pitcher_a["units_result"] == pytest.approx(100 / 110)
    assert pitcher_a["units_risked"] == pytest.approx(1.0)
    assert pitcher_b["units_result"] == pytest.approx(-1.0)
    assert pitcher_b["units_risked"] == pytest.approx(1.0)
    assert pitcher_c["units_result"] == pytest.approx(0.0)
    assert pitcher_c["units_risked"] == pytest.approx(0.0)

    by_scope_and_book = {
        (row["summary_scope"], row["book"]): row
        for row in summary_by_book_df.to_dict(orient="records")
    }
    assert by_scope_and_book[("all_time", "DraftKings")]["units_profit"] == pytest.approx(100 / 110)
    assert by_scope_and_book[("all_time", "FanDuel")]["units_profit"] == pytest.approx(-1.0)
    assert by_scope_and_book[("all_time", "BetMGM")]["units_profit"] == pytest.approx(0.0)
    assert by_scope_and_book[("all_time", "DraftKings")]["roi"] == pytest.approx(100 / 110)
    assert by_scope_and_book[("all_time", "FanDuel")]["roi"] == pytest.approx(-1.0)
    assert pd.isna(by_scope_and_book[("all_time", "BetMGM")]["roi"])

    expected_profit = (100 / 110) - 1.0
    assert overall_summary["current_regime_rule"] == {
        "type": "start_date",
        "start_date": daily_card.CURRENT_REGIME_START_DATE,
    }
    assert overall_summary["summary_views"]["all_time"]["picks"] == 3
    assert overall_summary["summary_views"]["all_time"]["wins"] == 1
    assert overall_summary["summary_views"]["all_time"]["losses"] == 1
    assert overall_summary["summary_views"]["all_time"]["pushes"] == 1
    assert overall_summary["summary_views"]["all_time"]["decisions"] == 2
    assert overall_summary["summary_views"]["all_time"]["units_risked"] == pytest.approx(2.0)
    assert overall_summary["summary_views"]["all_time"]["units_profit"] == pytest.approx(expected_profit)
    assert overall_summary["summary_views"]["all_time"]["roi"] == pytest.approx(expected_profit / 2.0)
    assert overall_summary["summary_views"]["all_time"]["skipped_rows"] == 1
    assert overall_summary["summary_views"]["current_regime"]["picks"] == 0
    assert overall_summary["segmented_views"]["record_source"]["run_daily_card"]["picks"] == 3
    assert overall_summary["segmented_views"]["model_version"][daily_card.LEGACY_WORKFLOW_MODEL_VERSION]["picks"] == 3
    assert overall_summary["segmented_views"]["policy_version"][daily_card.LEGACY_WORKFLOW_POLICY_VERSION]["picks"] == 3
    assert overall_summary["segmented_views"]["tracking_regime"][daily_card.TRACKING_REGIME_LEGACY_WORKFLOW]["picks"] == 3
    assert list(skipped_df["player_name"]) == ["Pitcher D"]

    segmented_rows = summary_by_book_df.loc[
        summary_by_book_df["segment_type"] == "record_source"
    ]
    assert set(segmented_rows["segment_value"]) == {"run_daily_card"}


def test_persist_official_picks_profit_reports_writes_tracking_artifacts(tmp_path, monkeypatch):
    tracking_dir = tmp_path / "data" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    history_path = tracking_dir / "official_picks_history.csv"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    all_time_summary_path = tracking_dir / "official_picks_profit_summary_all_time.json"
    current_regime_summary_path = tracking_dir / "official_picks_profit_summary_current_regime.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"
    audit_path = tracking_dir / "official_picks_concentration_audit.json"

    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-01|pitcher a",
                "game_date": "2026-05-01",
                "player_name": "Pitcher A",
                "team": "AAA",
                "opponent": "BBB",
                "book": "DraftKings",
                "odds": "-110",
                "price": -110,
                "pick_side": "over",
                "line": 5.5,
                "predicted_strikeouts": 6.4,
                "edge": 0.9,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "W",
                "actual_strikeouts": "7",
                "record_source": "run_daily_card",
            }
        ]
    )
    history_df.to_csv(history_path, index=False)

    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH",
        tracking_dir / "official_picks_profit_summary_all_time.json",
    )
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH",
        tracking_dir / "official_picks_profit_summary_current_regime.json",
    )
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_CONCENTRATION_AUDIT_PATH", audit_path)

    daily_card.persist_official_picks_profit_reports()

    assert grades_path.exists()
    assert by_book_path.exists()
    assert summary_path.exists()
    assert daily_card.OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.exists()
    assert daily_card.OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.exists()
    assert skipped_path.exists()
    assert audit_path.exists()

    grades_df = pd.read_csv(grades_path)
    by_book_df = pd.read_csv(by_book_path)
    summary_payload = daily_card.json.loads(summary_path.read_text(encoding="utf-8"))
    all_time_summary_payload = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    current_regime_summary_payload = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    skipped_df = pd.read_csv(skipped_path)
    audit_payload = daily_card.json.loads(audit_path.read_text(encoding="utf-8"))

    assert grades_df.loc[0, "units_result"] == pytest.approx(100 / 110)
    assert by_book_df.loc[0, "summary_scope"] == "all_time"
    assert by_book_df.loc[0, "segment_type"] == "summary_scope"
    assert by_book_df.loc[0, "segment_value"] == "all_time"
    assert by_book_df.loc[0, "book"] == "DraftKings"
    assert by_book_df.loc[0, "units_profit"] == pytest.approx(100 / 110)
    assert summary_payload["summary_views"]["all_time"]["picks"] == 1
    assert summary_payload["summary_views"]["all_time"]["units_profit"] == pytest.approx(100 / 110)
    assert summary_payload["summary_views"]["current_regime"]["picks"] == 0
    assert all_time_summary_payload["summary_scope"] == "all_time"
    assert all_time_summary_payload["summary_metrics"]["picks"] == 1
    assert all_time_summary_payload["summary_metrics"]["units_profit"] == pytest.approx(100 / 110)
    assert current_regime_summary_payload["summary_scope"] == "current_regime"
    assert current_regime_summary_payload["summary_metrics"]["picks"] == 0
    assert summary_payload["segmented_views"]["record_source"]["run_daily_card"]["picks"] == 1
    assert audit_payload["scopes"]["all_time"]["summary"]["official_picks"] == 1
    assert audit_payload["provenance_groupings"]["record_source"][0]["record_source"] == "run_daily_card"
    assert skipped_df.empty


def test_build_official_picks_profit_report_includes_current_regime_view():
    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-06|pitcher a",
                "game_date": "2026-05-06",
                "player_name": "Pitcher A",
                "book": "DraftKings",
                "odds": "-110",
                "price": -110,
                "pick_side": "over",
                "line": 5.5,
                "predicted_strikeouts": 6.4,
                "edge": 0.9,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "W",
                "actual_strikeouts": "7",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": f"{daily_card.CURRENT_REGIME_START_DATE}|pitcher b",
                "game_date": daily_card.CURRENT_REGIME_START_DATE,
                "player_name": "Pitcher B",
                "book": "FanDuel",
                "odds": "+100",
                "price": 100,
                "pick_side": "under",
                "line": 4.5,
                "predicted_strikeouts": 3.9,
                "edge": 0.6,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "L",
                "actual_strikeouts": "5",
                "record_source": "run_daily_card",
            },
        ]
    )

    report = daily_card.build_official_picks_profit_report(history_df)

    assert report["overall_summary"]["summary_views"]["all_time"]["picks"] == 2
    assert report["overall_summary"]["summary_views"]["current_regime"]["picks"] == 1
    assert report["overall_summary"]["summary_views"]["current_regime"]["losses"] == 1
    assert report["published_summary_views"]["all_time"]["summary_scope"] == "all_time"
    assert report["published_summary_views"]["current_regime"]["summary_scope"] == "current_regime"
    assert report["published_summary_views"]["current_regime"]["summary_metrics"]["picks"] == 1

    by_scope = report["summary_by_book_df"]["summary_scope"].tolist()
    assert "all_time" in by_scope
    assert "current_regime" in by_scope


def test_build_official_picks_concentration_audit_answers_concentration_questions():
    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-01|zac gallen over",
                "game_date": "2026-05-01",
                "player_name": "Zac Gallen",
                "prop_type": "pitcher_k",
                "book": "DraftKings",
                "odds": "-110",
                "price": -110,
                "pick_side": "over",
                "line": 7.5,
                "edge": 0.9,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "L",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-08|zac gallen over",
                "game_date": "2026-05-08",
                "player_name": "Zac Gallen",
                "prop_type": "pitcher_k",
                "book": "FanDuel",
                "odds": "-105",
                "price": -105,
                "pick_side": "over",
                "line": 7.5,
                "edge": 0.8,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "L",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-09|shohei ohtani under",
                "game_date": "2026-05-09",
                "player_name": "Shohei Ohtani",
                "prop_type": "pitcher_k",
                "book": "BetMGM",
                "odds": "-120",
                "price": -120,
                "pick_side": "under",
                "line": 7.5,
                "edge": 0.7,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "L",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-10|chris sale over",
                "game_date": "2026-05-10",
                "player_name": "Chris Sale",
                "prop_type": "pitcher_k",
                "book": "DraftKings",
                "odds": "+100",
                "price": 100,
                "pick_side": "over",
                "line": 7.5,
                "edge": 0.85,
                "confidence_tier": "medium",
                "pick_type": "official",
                "result": "W",
                "record_source": "run_daily_card",
            },
            {
                "pick_key": "2026-05-11|simeon under walks",
                "game_date": "2026-05-11",
                "player_name": "Simeon Woods Richardson",
                "prop_type": "pitcher_bb",
                "book": "FanDuel",
                "odds": "-110",
                "price": -110,
                "pick_side": "under",
                "line": 2.5,
                "edge": 0.5,
                "confidence_tier": "low",
                "pick_type": "official",
                "result": "W",
                "record_source": "run_daily_card",
            },
        ]
    )

    audit = daily_card.build_official_picks_concentration_audit(history_df)

    assert audit["artifact_type"] == "official_picks_concentration_audit"
    assert audit["current_regime_rule"]["start_date"] == daily_card.CURRENT_REGIME_START_DATE

    all_time_scope = audit["scopes"]["all_time"]
    current_regime_scope = audit["scopes"]["current_regime"]
    assert all_time_scope["summary"]["official_picks"] == 5
    assert current_regime_scope["summary"]["official_picks"] == 4

    top_pick_share = all_time_scope["questions"]["largest_share_of_official_picks"][0]
    assert top_pick_share["player_name"] == "Zac Gallen"
    assert top_pick_share["picks"] == 2

    top_losses = all_time_scope["questions"]["largest_share_of_losses_or_negative_units"]["by_losses"][0]
    assert top_losses["player_name"] == "Zac Gallen"
    assert top_losses["losses"] == 2

    top_negative_units = all_time_scope["questions"]["largest_share_of_losses_or_negative_units"]["by_negative_units"][0]
    assert top_negative_units["player_name"] == "Zac Gallen"

    overselected_archetypes = all_time_scope["questions"]["overselected_archetypes_relative_to_performance"]
    assert overselected_archetypes[0]["archetype"] == "high-K ace"

    failing_combo = all_time_scope["questions"]["repeatedly_failing_combos"][0]
    assert failing_combo["prop_type"] == "pitcher_k"
    assert failing_combo["pick_side"] == "over"
    assert failing_combo["line_bucket"] == "7.5+"
    assert failing_combo["archetype"] == "high-K ace"

    persisted_archetypes = audit["regime_comparison"]["by_archetype"]
    assert persisted_archetypes[0]["archetype"] == "high-K ace"
    assert persisted_archetypes[0]["current_regime_picks"] >= 1


def test_persist_official_picks_profit_reports_refuses_to_overwrite_non_empty_summaries_with_empty_history(
    tmp_path,
    monkeypatch,
):
    tracking_dir = tmp_path / "data" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    history_path = tracking_dir / "official_picks_history.csv"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    all_time_summary_path = tracking_dir / "official_picks_profit_summary_all_time.json"
    current_regime_summary_path = tracking_dir / "official_picks_profit_summary_current_regime.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"

    pd.DataFrame([{"pick_key": "seed", "player_name": "Pitcher A"}]).to_csv(grades_path, index=False)
    pd.DataFrame([{"book": "DraftKings", "picks": 1}]).to_csv(by_book_path, index=False)
    pd.DataFrame([{"pick_key": "pending-seed", "player_name": "Pitcher B"}]).to_csv(skipped_path, index=False)
    summary_path.write_text(json.dumps({"picks": 1, "skipped_rows": 1}, indent=2), encoding="utf-8")
    pd.DataFrame(columns=daily_card.OFFICIAL_PICKS_HISTORY_COLUMNS).to_csv(history_path, index=False)

    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH", all_time_summary_path)
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH",
        current_regime_summary_path,
    )
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)

    with pytest.raises(ValueError, match="Refusing to overwrite non-empty tracking summaries"):
        daily_card.persist_official_picks_profit_reports()

    assert len(pd.read_csv(grades_path)) == 1
    assert len(pd.read_csv(by_book_path)) == 1
    assert len(pd.read_csv(skipped_path)) == 1
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["picks"] == 1
    assert not all_time_summary_path.exists()
    assert not current_regime_summary_path.exists()


def test_persist_official_picks_profit_reports_allows_empty_outputs_in_isolated_temp_paths(
    tmp_path,
    monkeypatch,
):
    tracking_dir = tmp_path / "data" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    history_path = tracking_dir / "official_picks_history.csv"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    all_time_summary_path = tracking_dir / "official_picks_profit_summary_all_time.json"
    current_regime_summary_path = tracking_dir / "official_picks_profit_summary_current_regime.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"

    pd.DataFrame(columns=daily_card.OFFICIAL_PICKS_HISTORY_COLUMNS).to_csv(history_path, index=False)

    monkeypatch.setattr(daily_card, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_ALL_TIME_SUMMARY_PATH", all_time_summary_path)
    monkeypatch.setattr(
        daily_card,
        "OFFICIAL_PICKS_CURRENT_REGIME_SUMMARY_PATH",
        current_regime_summary_path,
    )
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)

    daily_card.persist_official_picks_profit_reports()

    assert grades_path.exists()
    assert by_book_path.exists()
    assert skipped_path.exists()
    assert summary_path.exists()
    assert all_time_summary_path.exists()
    assert current_regime_summary_path.exists()
    assert pd.read_csv(grades_path).empty
    assert pd.read_csv(by_book_path).empty
    assert pd.read_csv(skipped_path).empty
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    all_time_summary_payload = json.loads(all_time_summary_path.read_text(encoding="utf-8"))
    current_regime_summary_payload = json.loads(current_regime_summary_path.read_text(encoding="utf-8"))
    assert summary_payload["summary_views"]["all_time"]["picks"] == 0
    assert all_time_summary_payload["summary_metrics"]["picks"] == 0
    assert current_regime_summary_payload["summary_metrics"]["picks"] == 0


def test_apply_statcast_results_to_official_picks_history_updates_yesterday_pick_results():
    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-06|pitcher-a",
                "game_date": "2026-05-06",
                "player_name": "Pitcher A",
                "participant_join_key": "mlbam_player:101",
                "participant_id": "mlbam_player:101",
                "participant_source_id": "101",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "pitcher a",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "team": "AAA",
                "opponent": "BBB",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_1",
                "odds": "-110",
                "price": -110,
                "pick_side": "over",
                "line": 5.5,
                "market_selection_key": "MLB|pitcher_strikeouts|mlbam_player:101|over|5.5",
                "market_offer_key": "MLB|pitcher_strikeouts|mlbam_player:101|over|5.5|draftkings",
                "predicted_strikeouts": 6.2,
                "edge": 0.7,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "",
                "actual_strikeouts": "",
                "record_source": "run_daily_card",
            }
        ]
    )
    pitcher_results_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-06",
                "pitcher": 101,
                "player_name": "Pitcher A",
                "strikeouts": 6,
            }
        ]
    )

    updated = daily_card.apply_statcast_results_to_official_picks_history(
        history_df,
        pitcher_results_df,
        game_date="2026-05-06",
    )

    assert updated.loc[0, "actual_strikeouts"] == "6"
    assert updated.loc[0, "result"] == "W"


def test_apply_statcast_results_to_official_picks_history_updates_pitcher_walk_results():
    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-06|pitcher-walks-a",
                "game_date": "2026-05-06",
                "player_name": "Pitcher Walks A",
                "participant_join_key": "mlbam_player:303",
                "participant_id": "mlbam_player:303",
                "participant_source_id": "303",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "pitcher walks a",
                "sport": "MLB",
                "market_key": "pitcher_walks",
                "market_family": "player_prop",
                "team": "AAA",
                "opponent": "BBB",
                "book": "FanDuel",
                "bookmaker_key": "fanduel",
                "event_id": "evt_walks",
                "odds": "-105",
                "price": -105,
                "pick_side": "under",
                "line": 2.5,
                "market_selection_key": "MLB|pitcher_walks|mlbam_player:303|under|2.5",
                "market_offer_key": "MLB|pitcher_walks|mlbam_player:303|under|2.5|fanduel",
                "predicted_value": 2.2,
                "edge": 0.3,
                "confidence_tier": "low",
                "pick_type": "official",
                "result": "",
                "actual_value": "",
                "actual_strikeouts": "",
                "record_source": "run_daily_card",
            }
        ]
    )
    pitcher_results_df = pd.DataFrame(
        [
            {
                "game_date": "2026-05-06",
                "pitcher": 303,
                "player_name": "Pitcher Walks A",
                "strikeouts": 7,
                "walks": 1,
            }
        ]
    )

    updated = daily_card.apply_statcast_results_to_official_picks_history(
        history_df,
        pitcher_results_df,
        game_date="2026-05-06",
    )

    assert updated.loc[0, "actual_value"] == "1"
    assert updated.loc[0, "actual_strikeouts"] == ""
    assert updated.loc[0, "result"] == "W"


def test_grade_official_picks_from_statcast_persists_updates_and_profit_reports(tmp_path, monkeypatch):
    tracking_dir = tmp_path / "data" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    history_path = tracking_dir / "official_picks_history.csv"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"

    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-06|pitcher-a",
                "game_date": "2026-05-06",
                "player_name": "Pitcher A",
                "participant_join_key": "mlbam_player:101",
                "participant_id": "mlbam_player:101",
                "participant_source_id": "101",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "pitcher a",
                "sport": "MLB",
                "market_key": "pitcher_strikeouts",
                "market_family": "player_prop",
                "team": "AAA",
                "opponent": "BBB",
                "book": "DraftKings",
                "bookmaker_key": "draftkings",
                "event_id": "evt_1",
                "odds": "-110",
                "price": -110,
                "pick_side": "under",
                "line": 5.5,
                "market_selection_key": "MLB|pitcher_strikeouts|mlbam_player:101|under|5.5",
                "market_offer_key": "MLB|pitcher_strikeouts|mlbam_player:101|under|5.5|draftkings",
                "predicted_strikeouts": 4.7,
                "edge": 0.8,
                "confidence_tier": "high",
                "pick_type": "official",
                "result": "",
                "actual_strikeouts": "",
                "record_source": "run_daily_card",
            }
        ]
    )
    history_df.to_csv(history_path, index=False)

    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)
    monkeypatch.setattr(
        daily_card,
        "load_pitcher_results_from_statcast",
        lambda game_date: pd.DataFrame(
            [
                {
                    "game_date": game_date,
                    "pitcher": 101,
                    "player_name": "Pitcher A",
                    "strikeouts": 7,
                }
            ]
        ),
    )

    result = daily_card.grade_official_picks_from_statcast(game_date="2026-05-06")

    loaded_history = pd.read_csv(history_path, keep_default_na=False)
    grades_df = pd.read_csv(grades_path)
    summary_payload = daily_card.json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["updated_rows"] == 1
    assert str(loaded_history.loc[0, "actual_strikeouts"]) == "7"
    assert loaded_history.loc[0, "result"] == "L"
    assert grades_df.loc[0, "result"] == "L"
    assert summary_payload["summary_views"]["all_time"]["losses"] == 1
    assert summary_payload["summary_views"]["current_regime"]["losses"] == 0


def test_grade_official_picks_from_statcast_persists_pitcher_walk_updates(tmp_path, monkeypatch):
    tracking_dir = tmp_path / "data" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    history_path = tracking_dir / "official_picks_history.csv"
    grades_path = tracking_dir / "official_picks_profit_report.csv"
    by_book_path = tracking_dir / "official_picks_profit_by_book.csv"
    summary_path = tracking_dir / "official_picks_profit_summary.json"
    skipped_path = tracking_dir / "official_picks_profit_skipped.csv"

    history_df = pd.DataFrame(
        [
            {
                "pick_key": "2026-05-06|pitcher-walks-a",
                "game_date": "2026-05-06",
                "player_name": "Pitcher Walks A",
                "participant_join_key": "mlbam_player:303",
                "participant_id": "mlbam_player:303",
                "participant_source_id": "303",
                "participant_source_id_type": "mlbam_player",
                "participant_name_norm": "pitcher walks a",
                "sport": "MLB",
                "market_key": "pitcher_walks",
                "market_family": "player_prop",
                "team": "AAA",
                "opponent": "BBB",
                "book": "FanDuel",
                "bookmaker_key": "fanduel",
                "event_id": "evt_walks",
                "odds": "-105",
                "price": -105,
                "pick_side": "under",
                "line": 2.5,
                "market_selection_key": "MLB|pitcher_walks|mlbam_player:303|under|2.5",
                "market_offer_key": "MLB|pitcher_walks|mlbam_player:303|under|2.5|fanduel",
                "predicted_value": 2.2,
                "edge": 0.3,
                "confidence_tier": "low",
                "pick_type": "official",
                "result": "",
                "actual_value": "",
                "actual_strikeouts": "",
                "record_source": "run_daily_card",
            }
        ]
    )
    history_df.to_csv(history_path, index=False)

    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_HISTORY_PATH", history_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_GRADES_PATH", grades_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_BOOK_SUMMARY_PATH", by_book_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_OVERALL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)
    monkeypatch.setattr(
        daily_card,
        "load_pitcher_results_from_statcast",
        lambda game_date: pd.DataFrame(
            [
                {
                    "game_date": game_date,
                    "pitcher": 303,
                    "player_name": "Pitcher Walks A",
                    "strikeouts": 7,
                    "walks": 1,
                }
            ]
        ),
    )

    result = daily_card.grade_official_picks_from_statcast(game_date="2026-05-06")

    loaded_history = pd.read_csv(history_path, keep_default_na=False)
    grades_df = pd.read_csv(grades_path)
    summary_payload = daily_card.json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["updated_rows"] == 1
    assert str(loaded_history.loc[0, "actual_value"]) == "1"
    assert loaded_history.loc[0, "result"] == "W"
    assert grades_df.loc[0, "result"] == "W"
    assert summary_payload["summary_views"]["all_time"]["wins"] == 1
    assert summary_payload["summary_views"]["current_regime"]["wins"] == 0
