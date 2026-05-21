# src/pitcher_k/train.py

import math

import pandas as pd
import xgboost as xgb

from .config import (
    BASE_FEATURES,
    TARGET_COL,
    TRAIN_SPLIT_DATE,
    TRAIN_VALIDATION_FRACTION,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_NUM_BOOST_ROUND,
    XGB_PARAMS,
)


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["game_date", "game_pk", "pitcher"] if col in df.columns]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols, kind="stable").copy()


def time_split(model_df: pd.DataFrame, split_date: str = TRAIN_SPLIT_DATE):
    """
    Split model dataframe into train and test by date.
    """
    sorted_model_df = _sort_chronologically(model_df)
    train_df = sorted_model_df[sorted_model_df["game_date"] < split_date].copy()
    test_df = sorted_model_df[sorted_model_df["game_date"] >= split_date].copy()
    return train_df, test_df


def validation_time_split(
    train_df: pd.DataFrame,
    validation_fraction: float = TRAIN_VALIDATION_FRACTION,
):
    """
    Split a training dataframe into chronological subtrain/validation sets.
    """
    if train_df.empty:
        raise ValueError("Cannot build a validation split from an empty training dataframe.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    sorted_train_df = _sort_chronologically(train_df)
    game_dates = pd.to_datetime(sorted_train_df["game_date"])
    unique_dates = pd.Series(game_dates.dt.normalize().unique()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError(
            "Need at least two unique training dates to create a chronological validation split."
        )

    validation_dates = min(
        max(1, math.ceil(len(unique_dates) * validation_fraction)),
        len(unique_dates) - 1,
    )
    validation_start_date = unique_dates.iloc[-validation_dates]

    subtrain_df = sorted_train_df[game_dates < validation_start_date].copy()
    validation_df = sorted_train_df[game_dates >= validation_start_date].copy()

    if subtrain_df.empty or validation_df.empty:
        raise ValueError(
            "Chronological validation split produced an empty subtrain/validation partition."
        )

    return subtrain_df, validation_df


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
    num_boost_round: int = XGB_NUM_BOOST_ROUND,
    early_stopping_rounds: int | None = XGB_EARLY_STOPPING_ROUNDS,
    validation_fraction: float = TRAIN_VALIDATION_FRACTION,
):
    """
    Train XGBoost model with chronological validation-based early stopping,
    then refit on the full training window using the selected round count.
    """
    subtrain_df, validation_df = validation_time_split(
        train_df,
        validation_fraction=validation_fraction,
    )
    dsubtrain, dvalidation, X_subtrain, X_validation, y_subtrain, y_validation = make_dmats(
        train_df=subtrain_df,
        test_df=validation_df,
        features=features,
        target=target,
    )

    tuning_evals_result: dict[str, dict[str, list[float]]] = {}
    candidate_model = xgb.train(
        params=params,
        dtrain=dsubtrain,
        num_boost_round=num_boost_round,
        evals=[(dsubtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        evals_result=tuning_evals_result,
    )
    best_iteration = getattr(candidate_model, "best_iteration", None)
    selected_num_boost_round = (
        int(best_iteration) + 1 if best_iteration is not None and best_iteration >= 0 else num_boost_round
    )

    dtrain, dtest, X_train, X_test, y_train, y_test = make_dmats(
        train_df=train_df,
        test_df=test_df,
        features=features,
        target=target,
    )
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=selected_num_boost_round,
        verbose_eval=False,
    )

    return {
        "model": model,
        "subtrain_df": subtrain_df,
        "validation_df": validation_df,
        "dsubtrain": dsubtrain,
        "dvalidation": dvalidation,
        "dtrain": dtrain,
        "dtest": dtest,
        "X_subtrain": X_subtrain,
        "X_validation": X_validation,
        "X_train": X_train,
        "X_test": X_test,
        "y_subtrain": y_subtrain,
        "y_validation": y_validation,
        "y_train": y_train,
        "y_test": y_test,
        "candidate_num_boost_round": int(num_boost_round),
        "selected_num_boost_round": int(selected_num_boost_round),
        "early_stopping_rounds": early_stopping_rounds,
        "validation_fraction": float(validation_fraction),
        "best_validation_mae": (
            float(candidate_model.best_score)
            if getattr(candidate_model, "best_score", None) is not None
            else None
        ),
        "tuning_evals_result": tuning_evals_result,
    }
