# src/pitcher_k/train.py

import pandas as pd
import xgboost as xgb

from pitcher_k.config import BASE_FEATURES, TARGET_COL, TRAIN_SPLIT_DATE, XGB_PARAMS


def time_split(model_df: pd.DataFrame, split_date: str = TRAIN_SPLIT_DATE):
    """
    Split model dataframe into train and test by date.
    """
    train_df = model_df[model_df["game_date"] < split_date].copy()
    test_df = model_df[model_df["game_date"] >= split_date].copy()
    return train_df, test_df


def make_xy(df: pd.DataFrame, features: list[str] = BASE_FEATURES, target: str = TARGET_COL):
    """
    Extract X and y from a model dataframe.
    """
    X = df[features].copy()
    y = df[target].copy()
    return X, y


def make_dmats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str] = BASE_FEATURES,
    target: str = TARGET_COL,
):
    """
    Convert train/test dataframes into XGBoost DMatrix objects.
    """
    X_train, y_train = make_xy(train_df, features, target)
    X_test, y_test = make_xy(test_df, features, target)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    return dtrain, dtest, X_train, X_test, y_train, y_test


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str] = BASE_FEATURES,
    target: str = TARGET_COL,
    params: dict = XGB_PARAMS,
    num_boost_round: int = 200,
):
    """
    Train XGBoost model using train/test split.
    """
    dtrain, dtest, X_train, X_test, y_train, y_test = make_dmats(
        train_df=train_df,
        test_df=test_df,
        features=features,
        target=target,
    )

    evals = [(dtrain, "train"), (dtest, "test")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=False,
    )

    return {
        "model": model,
        "dtrain": dtrain,
        "dtest": dtest,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }