import numpy as np
import pandas as pd
import pytest

from pitcher_k.config import BASE_FEATURES
from pitcher_k.predict import predict_on_dataframe


class FakeModel:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = np.array(predictions, dtype=float)

    def predict(self, dmatrix):
        assert dmatrix.num_row() == len(self._predictions)
        return self._predictions


def test_predict_on_dataframe_adds_uncertainty_fields_from_recent_history():
    df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "strikeouts_stddev_last10": 1.25,
                **{feature: 1.0 for feature in BASE_FEATURES},
            }
        ]
    )

    result = predict_on_dataframe(FakeModel([6.8]), df)

    assert result.loc[0, "predicted_strikeouts"] == pytest.approx(6.8)
    assert result.loc[0, "std_dev"] == pytest.approx(1.25)
    assert result.loc[0, "lower_bound"] == pytest.approx(5.55)
    assert result.loc[0, "upper_bound"] == pytest.approx(8.05)


def test_predict_on_dataframe_falls_back_to_zero_uncertainty_when_history_missing():
    df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                **{feature: 1.0 for feature in BASE_FEATURES},
            }
        ]
    )

    result = predict_on_dataframe(FakeModel([4.2]), df)

    assert result.loc[0, "std_dev"] == pytest.approx(0.0)
    assert result.loc[0, "lower_bound"] == pytest.approx(4.2)
    assert result.loc[0, "upper_bound"] == pytest.approx(4.2)


def test_predict_on_dataframe_applies_interval_multiplier_when_provided():
    df = pd.DataFrame(
        [
            {
                "player_name": "Jacob deGrom",
                "strikeouts_stddev_last10": 1.25,
                **{feature: 1.0 for feature in BASE_FEATURES},
            }
        ]
    )

    result = predict_on_dataframe(
        FakeModel([6.8]),
        df,
        interval_config={"interval_multiplier": 1.6, "nominal_coverage": 0.8},
    )

    assert result.loc[0, "raw_std_dev"] == pytest.approx(1.25)
    assert result.loc[0, "std_dev"] == pytest.approx(2.0)
    assert result.loc[0, "lower_bound"] == pytest.approx(4.8)
    assert result.loc[0, "upper_bound"] == pytest.approx(8.8)
    assert result.loc[0, "interval_coverage_target"] == pytest.approx(0.8)
