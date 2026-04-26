import pandas as pd

from odds.backtest import run_historical_workflow_backtest, run_pick_backtest


def test_run_pick_backtest_summarizes_workflow_by_betting_segments():
    joined_df = pd.DataFrame(
        [
            {
                "player_name_proj": "Official Over",
                "predicted_strikeouts": 7.2,
                "bookmaker": "DraftKings",
                "side": "Over",
                "line": 6.0,
                "price": -110,
                "actual_strikeouts": 8,
            },
            {
                "player_name_proj": "Official Over",
                "predicted_strikeouts": 7.2,
                "bookmaker": "FanDuel",
                "side": "Over",
                "line": 6.5,
                "price": 105,
                "actual_strikeouts": 8,
            },
            {
                "player_name_proj": "Lean Under",
                "predicted_strikeouts": 5.0,
                "bookmaker": "Caesars",
                "side": "Under",
                "line": 5.5,
                "price": -115,
                "actual_strikeouts": 4,
            },
            {
                "player_name_proj": "Pass Case",
                "predicted_strikeouts": 4.7,
                "bookmaker": "BetMGM",
                "side": "Over",
                "line": 4.5,
                "price": -110,
                "actual_strikeouts": 3,
            },
        ]
    )

    backtest = run_pick_backtest(joined_df)

    assert backtest["available"] is True
    assert len(backtest["overall"]) == 1
    assert backtest["overall"][0]["picks"] == 3
    assert backtest["overall"][0]["wins"] == 2
    assert backtest["overall"][0]["losses"] == 1

    by_pick_type = {row["pick_type"]: row for row in backtest["by_pick_type"]}
    assert by_pick_type["official"]["wins"] == 1
    assert by_pick_type["lean"]["wins"] == 1
    assert by_pick_type["pass"]["losses"] == 1

    by_book = {row["book"]: row for row in backtest["by_book"]}
    assert set(by_book) == {"DraftKings", "Caesars", "BetMGM"}

    by_side = {row["pick_side"]: row for row in backtest["by_pick_side"]}
    assert by_side["over"]["picks"] == 2
    assert by_side["under"]["picks"] == 1

    by_band = {row["line_band"]: row for row in backtest["by_line_band"]}
    assert by_band["5.5-6.5"]["picks"] == 1
    assert by_band["4.5-5.5"]["picks"] == 1
    assert by_band["<=4.5"]["picks"] == 1

    graded = backtest["graded_picks"]
    assert set(graded["outcome"]) == {"win", "loss"}


def test_run_historical_workflow_backtest_uses_native_lines_artifact_rows():
    projections = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Official Over",
                "predicted_strikeouts": 7.2,
                "actual_strikeouts": 8,
            },
            {
                "game_date": "2025-08-02",
                "player_name": "Lean Under",
                "predicted_strikeouts": 5.0,
                "actual_strikeouts": 4,
            },
        ]
    )
    historical_lines = pd.DataFrame(
        [
            {
                "game_date": "2025-08-02",
                "player_name": "Official Over",
                "player_name_norm": "official over",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "side": "Over",
                "line": 6.0,
                "price": -110,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "selection_rule": "latest_pregame_snapshot_per_game_player_book_side",
                "source": "fixture",
                "pulled_at": "2025-08-02T22:50:00Z",
                "snapshot_type": "selected",
                "is_closing_line": True,
                "snapshot_rank": 1,
            },
            {
                "game_date": "2025-08-02",
                "player_name": "Lean Under",
                "player_name_norm": "lean under",
                "market_key": "pitcher_strikeouts",
                "bookmaker": "Caesars",
                "bookmaker_key": "caesars",
                "side": "Under",
                "line": 5.5,
                "price": -115,
                "event_id": "evt_1",
                "commence_time": "2025-08-02T23:10:00Z",
                "selection_rule": "latest_pregame_snapshot_per_game_player_book_side",
                "source": "fixture",
                "pulled_at": "2025-08-02T22:55:00Z",
                "snapshot_type": "selected",
                "is_closing_line": True,
                "snapshot_rank": 1,
            },
        ]
    )

    backtest = run_historical_workflow_backtest(projections, historical_lines)

    assert backtest["available"] is True
    assert backtest["overall"][0]["picks"] == 2
    assert backtest["overall"][0]["wins"] == 2
