# src/pitcher_k/predict.py

import pandas as pd
import xgboost as xgb

from MLB.src.config import BASE_FEATURES


def predict_on_dataframe(model, df: pd.DataFrame, features: list[str] = BASE_FEATURES) -> pd.DataFrame:
    """
    Generate predictions for any dataframe containing the model features.
    """
    pred_df = df.copy()
    dmatrix = xgb.DMatrix(pred_df[features], feature_names=features)
    pred_df["predicted_strikeouts"] = model.predict(dmatrix)
    return pred_df


def get_latest_pitcher_rows(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent row for each pitcher.
    Useful for next-game style predictions.
    """
    latest_rows = (
        model_df.sort_values("game_date")
        .groupby("pitcher", as_index=False)
        .tail(1)
        .copy()
    )
    return latest_rows


def predict_latest_pitchers(model, model_df: pd.DataFrame, features: list[str] = BASE_FEATURES) -> pd.DataFrame:
    """
    Predict strikeouts for each pitcher's latest available feature row.
    """
    latest_rows = get_latest_pitcher_rows(model_df)
    preds = predict_on_dataframe(model, latest_rows, features)

    cols_to_keep = ["game_date", "pitcher", "strikeouts", "predicted_strikeouts"]
    extra_cols = [col for col in features if col in preds.columns]

    return preds[cols_to_keep + extra_cols].sort_values("predicted_strikeouts", ascending=False)