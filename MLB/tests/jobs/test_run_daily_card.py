# MLB/tests/jobs/test_run_daily_card.py

from pathlib import Path

import pandas as pd

from jobs import run_daily_card as daily_card


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
                "pitches": 95,
                "strikeouts": 8,
                "whiffs": 14,
                "avg_velo": 97.2,
                "avg_spin": 2450,
                "batters_faced": 27,
                "home_team": "TEX",
                "away_team": "SEA",
                "p_throws": "R",
                "whiff_per_pitch": 14 / 95,
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "strikeouts_last3": 7.0,
                "strikeouts_last10": 6.8,
                "pitches_last3": 92.0,
                "pitches_last10": 94.0,
                "batters_faced_last3": 25.0,
                "batters_faced_last10": 26.0,
                "whiff_per_pitch_last3": 0.14,
                "whiff_per_pitch_last10": 0.135,
                "avg_velo_last3": 97.0,
                "avg_velo_last10": 96.8,
                "avg_spin_last3": 2440.0,
                "avg_spin_last10": 2435.0,
                "k_per_pitch_last10": 6.8 / 94.0,
                "k_rate_last10": 6.8 / 26.0,
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
                "pick_type": "official",
            }
        ]
    )

    post_df = picks_df.copy()

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "build_historical_pitcher_games", lambda: pitcher_games)
    monkeypatch.setattr(daily_card, "train_pitcher_k_model", lambda pg: ("fake_model", pg))
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions",
        lambda starters_df, pitcher_games, model: today_preds,
    )
    monkeypatch.setattr(daily_card, "run_edge_pipeline", lambda preds: (joined_df, joined_df))
    monkeypatch.setattr(daily_card, "build_daily_picks", lambda joined: picks_df)
    monkeypatch.setattr(
        daily_card,
        "filter_postable_picks",
        lambda picks, max_official=3, max_leans=1: post_df,
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")

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

    loaded_post = pd.read_csv(daily_card.PICKS_DIR / "today_postable_picks.csv")
    assert len(loaded_post) == 1
    assert loaded_post.loc[0, "player_name"] == "Jacob deGrom"
    assert loaded_post.loc[0, "pick_type"] == "official"

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
            }
        ]
    )

    monkeypatch.setattr(daily_card, "get_today_starters_df", lambda: starters_df)
    monkeypatch.setattr(daily_card, "build_historical_pitcher_games", lambda: pitcher_games)
    monkeypatch.setattr(daily_card, "train_pitcher_k_model", lambda pg: ("fake_model", pg))
    monkeypatch.setattr(
        daily_card,
        "build_today_predictions",
        lambda starters_df, pitcher_games, model: pd.DataFrame(),
    )

    monkeypatch.setattr(daily_card, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(daily_card, "OUTPUT_DIR", tmp_path / "data" / "outputs")
    monkeypatch.setattr(daily_card, "PROJECTIONS_DIR", tmp_path / "data" / "outputs" / "projections")
    monkeypatch.setattr(daily_card, "EDGES_DIR", tmp_path / "data" / "outputs" / "edges")
    monkeypatch.setattr(daily_card, "PICKS_DIR", tmp_path / "data" / "outputs" / "picks")

    import pytest

    with pytest.raises(ValueError, match="No today predictions were generated."):
        daily_card.run_daily_card()