import pandas as pd
import pytest
import requests

from jobs import run_daily_card as daily_card
from common.workflows import ModelingWorkflowSpec, ProjectionOddsJoinKeys, WorkflowArtifactSpec
from odds.policy import (
    DEFAULT_MLB_PITCHER_STRIKEOUT_POLICY,
    PostablePickLimits,
)
from pitcher_k.config import PITCHER_K_PROP_MARKET


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
        return joined_df, joined_df

    monkeypatch.setattr(daily_card, "run_edge_pipeline", fake_run_edge_pipeline)
    monkeypatch.setattr(daily_card, "build_daily_picks", lambda joined, policy: picks_df)
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
    assert daily_card.OFFICIAL_PICKS_SKIPPED_PATH.exists()

    loaded_post = pd.read_csv(daily_card.PICKS_DIR / "today_postable_picks.csv")
    loaded_history = pd.read_csv(daily_card.OFFICIAL_PICKS_HISTORY_PATH, keep_default_na=False)
    loaded_grades = pd.read_csv(daily_card.OFFICIAL_PICKS_GRADES_PATH)
    loaded_book_summary = pd.read_csv(daily_card.OFFICIAL_PICKS_BOOK_SUMMARY_PATH)
    loaded_overall = daily_card.json.loads(
        daily_card.OFFICIAL_PICKS_OVERALL_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    assert len(loaded_post) == 1
    assert loaded_post.loc[0, "player_name"] == "Jacob deGrom"
    assert loaded_post.loc[0, "pick_type"] == "official"
    assert len(loaded_history) == 1
    assert loaded_history.loc[0, "game_date"] == "2026-04-19"
    assert str(loaded_history.loc[0, "odds"]) == "-120"
    assert loaded_history.loc[0, "pick_key"] == "2026-04-19|jacob degrom"
    assert loaded_grades.empty
    assert loaded_book_summary.empty
    assert loaded_overall["picks"] == 0
    assert loaded_overall["skipped_rows"] == 1


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
        return joined_df, joined_df

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
        "projection_join_key": "player_name_norm",
        "odds_join_key": "player_name_norm",
    }
    pd.testing.assert_frame_equal(calls["build_picks_joined"], joined_df)
    pd.testing.assert_frame_equal(calls["filter_postable_picks"], picks_df)


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
            projection="player_name_norm",
            odds="player_name_norm",
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
        return joined_df, joined_df

    def fake_build_daily_picks(df, policy):
        calls["build_policy"] = policy
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
        "projection_join_key": "player_name_norm",
        "odds_join_key": "player_name_norm",
    }
    assert calls["build_policy"] is workflow.pick_ranking_policy
    assert calls["filter_policy"] is workflow.pick_ranking_policy
    assert calls["filter_limits"] == (1, 0)


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
    assert history_rows.loc[0, "pick_key"] == "2026-04-19|jacob degrom"


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

    by_book = {row["book"]: row for row in summary_by_book_df.to_dict(orient="records")}
    assert by_book["DraftKings"]["units_profit"] == pytest.approx(100 / 110)
    assert by_book["FanDuel"]["units_profit"] == pytest.approx(-1.0)
    assert by_book["BetMGM"]["units_profit"] == pytest.approx(0.0)
    assert by_book["DraftKings"]["roi"] == pytest.approx(100 / 110)
    assert by_book["FanDuel"]["roi"] == pytest.approx(-1.0)
    assert pd.isna(by_book["BetMGM"]["roi"])

    expected_profit = (100 / 110) - 1.0
    assert overall_summary["picks"] == 3
    assert overall_summary["wins"] == 1
    assert overall_summary["losses"] == 1
    assert overall_summary["pushes"] == 1
    assert overall_summary["decisions"] == 2
    assert overall_summary["units_risked"] == pytest.approx(2.0)
    assert overall_summary["units_profit"] == pytest.approx(expected_profit)
    assert overall_summary["roi"] == pytest.approx(expected_profit / 2.0)
    assert overall_summary["skipped_rows"] == 1
    assert list(skipped_df["player_name"]) == ["Pitcher D"]


def test_persist_official_picks_profit_reports_writes_tracking_artifacts(tmp_path, monkeypatch):
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
    monkeypatch.setattr(daily_card, "OFFICIAL_PICKS_SKIPPED_PATH", skipped_path)

    daily_card.persist_official_picks_profit_reports()

    assert grades_path.exists()
    assert by_book_path.exists()
    assert summary_path.exists()
    assert skipped_path.exists()

    grades_df = pd.read_csv(grades_path)
    by_book_df = pd.read_csv(by_book_path)
    summary_payload = daily_card.json.loads(summary_path.read_text(encoding="utf-8"))
    skipped_df = pd.read_csv(skipped_path)

    assert grades_df.loc[0, "units_result"] == pytest.approx(100 / 110)
    assert by_book_df.loc[0, "book"] == "DraftKings"
    assert by_book_df.loc[0, "units_profit"] == pytest.approx(100 / 110)
    assert summary_payload["picks"] == 1
    assert summary_payload["units_profit"] == pytest.approx(100 / 110)
    assert skipped_df.empty
