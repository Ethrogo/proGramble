# src/pitcher_k/evaluate.py

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Return core regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "mae": mae,
        "pred_min": float(min(y_pred)),
        "pred_max": float(max(y_pred)),
    }


def build_prediction_results(X_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    """
    Combine test features, actual values, and predictions into one dataframe.
    """
    results = X_test.copy()
    results["actual_strikeouts"] = y_test.values
    results["predicted_strikeouts"] = y_pred
    results["error"] = results["predicted_strikeouts"] - results["actual_strikeouts"]
    results["abs_error"] = results["error"].abs()
    return results


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Scatter plot of actual vs predicted strikeouts.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Strikeouts")
    plt.ylabel("Predicted Strikeouts")
    plt.title("Actual vs Predicted Strikeouts")
    plt.show()


def get_feature_importance(model) -> pd.DataFrame:
    """
    Return XGBoost feature importance as a dataframe.
    """
    importance = model.get_score(importance_type="gain")

    imp_df = (
        pd.DataFrame(
            {
                "feature": list(importance.keys()),
                "importance_gain": list(importance.values()),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    return imp_df