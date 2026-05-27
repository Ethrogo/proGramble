import pandas as pd

from pitcher_bb import train as train_mod


def _row(game_date: str, walks: int, pitches_last3: float) -> dict:
    pitches_last10 = pitches_last3 + 2.0
    batters_faced_last3 = max(12.0, pitches_last3 / 4.0)
    batters_faced_last10 = batters_faced_last3 + 1.0
    walks_last10 = max(0.5, walks + 0.2)
    return {
        "game_date": game_date,
        "game_pk": int(game_date.replace("-", "")),
        "pitcher": 101,
        "player_name": "Starter One",
        "walks": walks,
        "pitches_last3": pitches_last3,
        "pitches_last10": pitches_last10,
        "batters_faced_last3": batters_faced_last3,
        "batters_faced_last10": batters_faced_last10,
        "walks_last3": float(walks),
        "walks_last10": walks_last10,
        "avg_velo_last3": 96.1,
        "avg_spin_last3": 2450.0,
        "bb_per_pitch_last10": walks_last10 / pitches_last10,
        "bb_rate_last10": walks_last10 / batters_faced_last10,
        "opp_walks_per_game_last10": 3.8,
        "opp_bb_rate_last10": 0.09,
    }


def test_validation_time_split_uses_latest_dates_for_validation():
    train_df = pd.DataFrame(
        [
            _row("2025-07-27", 1, 88.0),
            _row("2025-07-28", 2, 90.0),
            _row("2025-07-29", 1, 92.0),
            _row("2025-07-30", 3, 94.0),
        ]
    )

    subtrain_df, validation_df = train_mod.validation_time_split(
        train_df,
        validation_fraction=0.25,
    )

    assert subtrain_df["game_date"].max() == "2025-07-29"
    assert validation_df["game_date"].min() == "2025-07-30"
    assert len(subtrain_df) == 3
    assert len(validation_df) == 1


def test_train_model_uses_validation_early_stopping_and_refits(monkeypatch):
    train_df = pd.DataFrame(
        [
            _row("2025-07-27", 1, 88.0),
            _row("2025-07-28", 2, 90.0),
            _row("2025-07-29", 1, 92.0),
            _row("2025-07-30", 3, 94.0),
        ]
    )
    test_df = pd.DataFrame(
        [
            _row("2025-08-02", 2, 95.0),
        ]
    )

    calls: list[dict] = []

    class FakeBooster:
        def __init__(self, *, best_iteration=None, best_score=None):
            self.best_iteration = best_iteration
            self.best_score = best_score

    def fake_xgb_train(
        *,
        params,
        dtrain,
        num_boost_round,
        evals=None,
        early_stopping_rounds=None,
        verbose_eval=False,
        evals_result=None,
    ):
        calls.append(
            {
                "params": params,
                "num_boost_round": num_boost_round,
                "evals_count": len(evals or []),
                "early_stopping_rounds": early_stopping_rounds,
            }
        )
        if early_stopping_rounds is not None:
            if evals_result is not None:
                evals_result.update(
                    {
                        "train": {"mae": [1.1, 1.0, 0.9]},
                        "validation": {"mae": [1.2, 1.0, 0.95]},
                    }
                )
            return FakeBooster(best_iteration=5, best_score=0.95)
        return FakeBooster()

    monkeypatch.setattr(train_mod.xgb, "train", fake_xgb_train)

    result = train_mod.train_model(
        train_df=train_df,
        test_df=test_df,
        validation_fraction=0.25,
    )

    assert len(calls) == 2
    assert calls[0]["num_boost_round"] == 200
    assert calls[0]["early_stopping_rounds"] == 25
    assert calls[0]["evals_count"] == 2
    assert calls[1]["num_boost_round"] == 6
    assert calls[1]["early_stopping_rounds"] is None
    assert result["selected_num_boost_round"] == 6
    assert result["best_validation_mae"] == 0.95
    assert len(result["validation_df"]) == 1
