import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INTERVAL_COVERAGE = 0.8
DEFAULT_INTERVAL_STDDEV_COLUMN = "strikeouts_stddev_last10"


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Return richer regression metrics for model governance.
    """
    y_true_series = pd.Series(y_true, dtype="float64").reset_index(drop=True)
    y_pred_series = pd.Series(y_pred, dtype="float64").reset_index(drop=True)
    errors = y_pred_series - y_true_series
    abs_errors = errors.abs()

    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "median_absolute_error": float(abs_errors.median()),
        "bias": float(errors.mean()),
        "abs_error_p75": float(abs_errors.quantile(0.75)),
        "abs_error_p90": float(abs_errors.quantile(0.90)),
        "pred_min": float(y_pred_series.min()),
        "pred_max": float(y_pred_series.max()),
        "actual_min": float(y_true_series.min()),
        "actual_max": float(y_true_series.max()),
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


def fit_interval_calibration(
    X_calibration: pd.DataFrame,
    y_true,
    y_pred,
    *,
    stddev_column: str = DEFAULT_INTERVAL_STDDEV_COLUMN,
    nominal_coverage: float = DEFAULT_INTERVAL_COVERAGE,
) -> dict:
    """
    Fit a scalar interval multiplier so `prediction +/- multiplier * recent_stddev`
    has an interpretable empirical coverage target.
    """
    calibration_df = pd.DataFrame(
        {
            "actual": pd.Series(y_true, dtype="float64").reset_index(drop=True),
            "predicted": pd.Series(y_pred, dtype="float64").reset_index(drop=True),
        }
    )
    if stddev_column in X_calibration.columns:
        calibration_df["base_std_dev"] = (
            pd.to_numeric(X_calibration[stddev_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .reset_index(drop=True)
        )
    else:
        calibration_df["base_std_dev"] = 0.0

    calibration_df["abs_error"] = (calibration_df["predicted"] - calibration_df["actual"]).abs()
    usable = calibration_df[calibration_df["base_std_dev"] > 0].copy()

    if usable.empty:
        interval_multiplier = 1.0
        calibration_rows = 0
    else:
        usable["scaled_abs_error"] = usable["abs_error"] / usable["base_std_dev"]
        interval_multiplier = float(
            usable["scaled_abs_error"].quantile(nominal_coverage, interpolation="linear")
        )
        if not math.isfinite(interval_multiplier) or interval_multiplier <= 0:
            interval_multiplier = 1.0
        calibration_rows = int(len(usable))

    return {
        "method": "recent_stddev_scaled_by_empirical_residual_quantile",
        "base_stddev_column": stddev_column,
        "nominal_coverage": float(nominal_coverage),
        "interval_multiplier": float(interval_multiplier),
        "calibration_rows": calibration_rows,
        "documented_interpretation": (
            "lower_bound and upper_bound represent an empirical central prediction "
            f"interval targeting {nominal_coverage:.0%} coverage on held-out data when "
            "the recent strikeout standard deviation signal is present."
        ),
    }


def apply_interval_calibration(
    pred_df: pd.DataFrame,
    interval_config: dict | None = None,
    *,
    stddev_column: str = DEFAULT_INTERVAL_STDDEV_COLUMN,
) -> pd.DataFrame:
    """
    Apply calibrated interval widths to a prediction dataframe.
    """
    interval_config = interval_config or {}
    multiplier = float(interval_config.get("interval_multiplier", 1.0))

    calibrated = pred_df.copy()
    if "std_dev" in calibrated.columns:
        base_std_dev = pd.to_numeric(calibrated["std_dev"], errors="coerce").fillna(0.0)
    elif stddev_column in calibrated.columns:
        base_std_dev = pd.to_numeric(calibrated[stddev_column], errors="coerce").fillna(0.0)
    else:
        base_std_dev = pd.Series(0.0, index=calibrated.index, dtype="float64")

    calibrated["raw_std_dev"] = base_std_dev.clip(lower=0.0)
    calibrated["std_dev"] = calibrated["raw_std_dev"] * multiplier
    calibrated["lower_bound"] = (calibrated["predicted_strikeouts"] - calibrated["std_dev"]).clip(lower=0.0)
    calibrated["upper_bound"] = calibrated["predicted_strikeouts"] + calibrated["std_dev"]

    if interval_config:
        calibrated["interval_coverage_target"] = float(
            interval_config.get("nominal_coverage", DEFAULT_INTERVAL_COVERAGE)
        )
        calibrated["interval_multiplier"] = multiplier

    return calibrated


def summarize_interval_coverage(
    X_eval: pd.DataFrame,
    y_true,
    y_pred,
    interval_config: dict | None = None,
    *,
    stddev_column: str = DEFAULT_INTERVAL_STDDEV_COLUMN,
) -> dict:
    """
    Measure empirical coverage and interval width on evaluation data.
    """
    coverage_df = X_eval.copy().reset_index(drop=True)
    coverage_df["predicted_strikeouts"] = pd.Series(y_pred, dtype="float64").reset_index(drop=True)
    coverage_df["actual_strikeouts"] = pd.Series(y_true, dtype="float64").reset_index(drop=True)
    coverage_df = apply_interval_calibration(
        coverage_df,
        interval_config,
        stddev_column=stddev_column,
    )

    within_bounds = coverage_df["actual_strikeouts"].between(
        coverage_df["lower_bound"],
        coverage_df["upper_bound"],
        inclusive="both",
    )

    return {
        "interval_method": (interval_config or {}).get(
            "method",
            "recent_stddev_without_calibration",
        ),
        "nominal_coverage": float(
            (interval_config or {}).get("nominal_coverage", DEFAULT_INTERVAL_COVERAGE)
        ),
        "empirical_coverage": float(within_bounds.mean()),
        "mean_interval_width": float((coverage_df["upper_bound"] - coverage_df["lower_bound"]).mean()),
        "mean_half_width": float(coverage_df["std_dev"].mean()),
        "miss_rate_below_lower_bound": float(
            (coverage_df["actual_strikeouts"] < coverage_df["lower_bound"]).mean()
        ),
        "miss_rate_above_upper_bound": float(
            (coverage_df["actual_strikeouts"] > coverage_df["upper_bound"]).mean()
        ),
        "rows": int(len(coverage_df)),
    }


def build_error_bucket_summary(
    results_df: pd.DataFrame,
    *,
    bucket_column: str = "predicted_strikeouts",
) -> list[dict]:
    """
    Summarize forecast quality across practical strikeout bands.
    """
    working = results_df.copy()
    bins = [-np.inf, 4.5, 5.5, 6.5, 7.5, np.inf]
    labels = ["<=4.5", "4.5-5.5", "5.5-6.5", "6.5-7.5", "7.5+"]
    working["error_bucket"] = pd.cut(
        working[bucket_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    summary = (
        working.groupby("error_bucket", observed=False)
        .agg(
            rows=(bucket_column, "size"),
            mean_actual=("actual_strikeouts", "mean"),
            mean_prediction=("predicted_strikeouts", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["rows"] > 0]
    summary["error_bucket"] = summary["error_bucket"].astype(str)

    return summary.to_dict(orient="records")


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
