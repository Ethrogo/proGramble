import pandas as pd
import xgboost as xgb

from .config import BASE_FEATURES


def _add_projection_uncertainty(
    pred_df: pd.DataFrame,
    interval_config: dict | None = None,
) -> pd.DataFrame:
    pred_df = pred_df.copy()

    if "walks_stddev_last10" not in pred_df.columns:
        pred_df["std_dev"] = 0.0
    else:
        pred_df["std_dev"] = pred_df["walks_stddev_last10"].fillna(0.0).clip(lower=0.0)

    interval_config = interval_config or {}
    multiplier = float(interval_config.get("interval_multiplier", 1.0))
    pred_df["raw_std_dev"] = pred_df["std_dev"]
    pred_df["std_dev"] = pred_df["raw_std_dev"] * multiplier
    pred_df["lower_bound"] = (pred_df["predicted_walks"] - pred_df["std_dev"]).clip(lower=0.0)
    pred_df["upper_bound"] = pred_df["predicted_walks"] + pred_df["std_dev"]

    if interval_config:
        pred_df["interval_coverage_target"] = float(interval_config.get("nominal_coverage", 0.8))
        pred_df["interval_multiplier"] = multiplier

    return pred_df


def predict_on_dataframe(
    model,
    df: pd.DataFrame,
    features: list[str] = BASE_FEATURES,
    interval_config: dict | None = None,
) -> pd.DataFrame:
    pred_df = df.copy()
    dmatrix = xgb.DMatrix(pred_df[features], feature_names=features)
    pred_df["predicted_walks"] = model.predict(dmatrix)
    return _add_projection_uncertainty(pred_df, interval_config)
