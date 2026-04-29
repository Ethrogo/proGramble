import math

import pandas as pd
import pytest

from pitcher_k.evaluate import evaluate_predictions


def test_evaluate_predictions_ignores_input_index_alignment():
    y_true = pd.Series([5.0, 8.0], index=[10, 11], dtype="float64")
    y_pred = pd.Series([5.4, 7.1], index=[0, 1], dtype="float64")

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(0.65)
    assert math.isclose(metrics["rmse"], math.sqrt((0.4 ** 2 + (-0.9) ** 2) / 2))
    assert metrics["median_absolute_error"] == pytest.approx(0.65)
    assert metrics["bias"] == pytest.approx(-0.25)
    assert metrics["abs_error_p75"] == pytest.approx(0.775)
    assert metrics["abs_error_p90"] == pytest.approx(0.85)
    assert metrics["pred_min"] == pytest.approx(5.4)
    assert metrics["pred_max"] == pytest.approx(7.1)
    assert metrics["actual_min"] == pytest.approx(5.0)
    assert metrics["actual_max"] == pytest.approx(8.0)
