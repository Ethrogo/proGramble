import pandas as pd

from odds.backtest import run_pick_backtest


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
