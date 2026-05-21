import pandas as pd

from pitcher_k import train as train_mod


def _row(game_date: str, strikeouts: int, pitches_last3: float) -> dict:
    return {
        "game_date": game_date,
        "game_pk": int(game_date.replace("-", "")),
        "pitcher": 101,
        "player_name": "Starter One",
        "strikeouts": strikeouts,
        "pitches_last3": pitches_last3,
        "pitches_last10": pitches_last3 + 2.0,
        "whiff_per_pitch_last3": 0.12,
        "avg_velo_last3": 96.1,
        "avg_spin_last3": 2450.0,
        "k_per_pitch_last10": 0.08,
        "k_rate_last10": 0.27,
        "opp_strikeouts_per_game_last10": 8.8,
        "opp_k_rate_last10": 0.24,
    }


def test_validation_time_split_uses_latest_dates_for_validation():
    train_df = pd.DataFrame(
        [
            _row("2025-07-27", 5, 88.0),
            _row("2025-07-28", 6, 90.0),
            _row("2025-07-29", 7, 92.0),
            _row("2025-07-30", 8, 94.0),
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
            _row("2025-07-27", 5, 88.0),
            _row("2025-07-28", 6, 90.0),
            _row("2025-07-29", 7, 92.0),
            _row("2025-07-30", 8, 94.0),
        ]
    )
    test_df = pd.DataFrame(
        [
            _row("2025-08-02", 7, 95.0),
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
                        "train": {"mae": [2.0, 1.9, 1.8]},
                        "validation": {"mae": [2.1, 1.85, 1.8]},
                    }
                )
            return FakeBooster(best_iteration=6, best_score=1.8)
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
    assert calls[1]["num_boost_round"] == 7
    assert calls[1]["early_stopping_rounds"] is None
    assert result["selected_num_boost_round"] == 7
    assert result["best_validation_mae"] == 1.8
    assert len(result["validation_df"]) == 1
