from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.identity import (
    MARKET_OFFER_KEY_COLUMN,
    MARKET_SELECTION_KEY_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_JOIN_KEY_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    ensure_market_identity,
    ensure_participant_identity,
)
from pitcher_bb import config
from pitcher_bb.feature_model import build_model_df
from pitcher_bb.promotion_policy import (
    CHAMPION_STATUS_STABLE,
    DEFAULT_SHADOW_PROMOTION_POLICY,
    build_shadow_promotion_review,
)
from pitcher_bb.train import validation_time_split
from pitcher_k import shadow as pitcher_k_shadow
from pitcher_k.evaluate import (
    apply_interval_calibration,
    evaluate_predictions,
    fit_interval_calibration,
)
from pitcher_k.feature_engineering import filter_starter_like_appearances


CHAMPION_MODEL_NAME = "xgboost_champion"
CHALLENGER_MODEL_NAME = "ridge_challenger"
CHAMPION_MODEL_ROLE = "champion"
CHALLENGER_MODEL_ROLE = "challenger"
# Walk outcomes are lower-count and noisier than strikeouts, so use a denser
# low-end regularization grid before moving to heavier shrinkage.
RIDGE_ALPHA_CANDIDATES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

SHADOW_TRACKING_COLUMNS = [
    "shadow_row_key",
    "candidate_key",
    "game_date",
    "player_name",
    PARTICIPANT_JOIN_KEY_COLUMN,
    PARTICIPANT_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_COLUMN,
    PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    PARTICIPANT_NAME_NORM_COLUMN,
    "sport",
    "market_key",
    "market_family",
    "prop_type",
    "team",
    "opponent",
    "book",
    "bookmaker_key",
    "event_id",
    "side",
    "side_norm",
    "line",
    "price",
    MARKET_SELECTION_KEY_COLUMN,
    MARKET_OFFER_KEY_COLUMN,
    "model_name",
    "model_role",
    "model_version",
    "policy_version",
    "predicted_value",
    "predicted_walks",
    "lower_bound",
    "upper_bound",
    "std_dev",
    "edge",
    "would_pick",
    "pick_rank",
    "pick_type",
    "confidence_tier",
    "actual_value",
    "result",
    "profit_units",
    "graded_at",
    "tracking_regime",
]


def empty_shadow_tracking_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SHADOW_TRACKING_COLUMNS)


def _normalize_side(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _first_present(df: pd.DataFrame, columns: list[str], default: object = "") -> pd.Series:
    resolved = pd.Series([default] * len(df), index=df.index, dtype="object")
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            mask = series.notna()
        else:
            mask = series.fillna("").astype(str).str.strip().ne("")
        resolved = resolved.where(~mask, series)
    return resolved


def _format_line_key(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value).strip()


def _fallback_candidate_key(
    df: pd.DataFrame,
    *,
    book_col: str,
    side_col: str,
    line_col: str,
) -> pd.Series:
    player_key = _first_present(
        df,
        [PARTICIPANT_JOIN_KEY_COLUMN, PARTICIPANT_NAME_NORM_COLUMN, "player_name", "player_name_proj"],
        "",
    ).fillna("").astype(str).str.strip()
    game_date = pd.to_datetime(_first_present(df, ["game_date"], ""), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    book = _first_present(df, [book_col, "book", "bookmaker"], "").fillna("").astype(str).str.strip().str.lower()
    side = _first_present(df, [side_col, "side", "pick_side"], "").apply(_normalize_side)
    line = _first_present(df, [line_col, "line"], "").apply(_format_line_key)
    return game_date + "|" + player_key + "|" + book + "|" + side + "|" + line


def candidate_key_series(
    df: pd.DataFrame,
    *,
    book_col: str,
    side_col: str,
    line_col: str,
) -> pd.Series:
    game_date = pd.to_datetime(_first_present(df, ["game_date"], ""), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    market_offer_key = _first_present(df, [MARKET_OFFER_KEY_COLUMN], "").fillna("").astype(str).str.strip()
    fallback = _fallback_candidate_key(df, book_col=book_col, side_col=side_col, line_col=line_col)
    base_key = market_offer_key.where(market_offer_key != "", fallback)
    return game_date + "|" + base_key


def shadow_row_key_series(df: pd.DataFrame, *, model_name_col: str = "model_name") -> pd.Series:
    game_date = pd.to_datetime(_first_present(df, ["game_date"], ""), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    model_name = _first_present(df, [model_name_col], "").fillna("").astype(str).str.strip()
    candidate_key = _first_present(df, ["candidate_key"], "").fillna("").astype(str).str.strip()
    return game_date + "|" + model_name + "|" + candidate_key


def _ridge_solver(X_design: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X_design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X_design.T @ X_design + alpha * penalty, X_design.T @ y)


def _design_matrix(
    df: pd.DataFrame,
    features: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    X = df[features].to_numpy(dtype=float)
    X_std = (X - mu) / sigma
    return np.column_stack([np.ones(len(X_std)), X_std])


def fit_ridge_shadow_model(
    pitcher_games: pd.DataFrame,
    *,
    alpha_candidates: tuple[float, ...] = RIDGE_ALPHA_CANDIDATES,
) -> dict[str, object]:
    starter_like = filter_starter_like_appearances(pitcher_games)
    model_df = build_model_df(starter_like)
    if model_df.empty:
        raise ValueError("pitcher_games did not produce any model rows for ridge shadow fitting.")

    features = list(config.BASE_FEATURES)
    subtrain_df, validation_df = validation_time_split(
        model_df,
        validation_fraction=config.TRAIN_VALIDATION_FRACTION,
    )
    if validation_df.empty:
        subtrain_df = model_df.copy()
        validation_df = model_df.copy()

    X_sub = subtrain_df[features].to_numpy(dtype=float)
    y_sub = subtrain_df[config.TARGET_COL].to_numpy(dtype=float)
    y_val = validation_df[config.TARGET_COL].to_numpy(dtype=float)

    mu = X_sub.mean(axis=0)
    sigma = X_sub.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_sub_design = _design_matrix(subtrain_df, features, mu, sigma)
    X_val_design = _design_matrix(validation_df, features, mu, sigma)

    best_alpha = None
    best_metrics = None
    best_validation_pred = None
    for alpha in alpha_candidates:
        coef = _ridge_solver(X_sub_design, y_sub, alpha)
        validation_pred = np.clip(X_val_design @ coef, a_min=0.0, a_max=None)
        metrics = evaluate_predictions(y_val, validation_pred)
        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_alpha = float(alpha)
            best_metrics = metrics
            best_validation_pred = validation_pred

    X_full = model_df[features].to_numpy(dtype=float)
    y_full = model_df[config.TARGET_COL].to_numpy(dtype=float)
    mu_full = X_full.mean(axis=0)
    sigma_full = X_full.std(axis=0)
    sigma_full[sigma_full == 0] = 1.0
    X_full_design = _design_matrix(model_df, features, mu_full, sigma_full)
    coef_full = _ridge_solver(X_full_design, y_full, best_alpha or 0.0)

    interval_config = fit_interval_calibration(
        validation_df,
        y_val,
        best_validation_pred if best_validation_pred is not None else np.clip(X_val_design @ coef_full, a_min=0.0, a_max=None),
    )

    return {
        "features": features,
        "alpha": float(best_alpha or 0.0),
        "coefficients": coef_full,
        "mu": mu_full,
        "sigma": sigma_full,
        "interval_config": interval_config,
        "training_rows": int(len(model_df)),
        "validation_rows": int(len(validation_df)),
        "validation_metrics": best_metrics or {},
    }


def predict_ridge_shadow_model(
    today_features: pd.DataFrame,
    fitted_model: dict[str, object],
) -> pd.DataFrame:
    if today_features.empty:
        return today_features.copy()

    features = list(fitted_model["features"])
    mu = np.asarray(fitted_model["mu"], dtype=float)
    sigma = np.asarray(fitted_model["sigma"], dtype=float)
    coefficients = np.asarray(fitted_model["coefficients"], dtype=float)

    pred_df = today_features.copy()
    X_design = _design_matrix(pred_df, features, mu, sigma)
    pred_df["predicted_walks"] = np.clip(X_design @ coefficients, a_min=0.0, a_max=None)
    pred_df["predicted_value"] = pred_df["predicted_walks"]
    pred_df = apply_interval_calibration(pred_df, fitted_model.get("interval_config"))
    return pred_df


def build_ridge_shadow_predictions(
    today_features: pd.DataFrame,
    pitcher_games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fitted = fit_ridge_shadow_model(pitcher_games)
    pred_df = predict_ridge_shadow_model(today_features, fitted)
    challenger_version = f"{config.TARGET_FORMULATION}_ridge_shadow_alpha_{fitted['alpha']:g}"
    fitted_meta = {
        "model_name": CHALLENGER_MODEL_NAME,
        "model_role": CHALLENGER_MODEL_ROLE,
        "model_version": challenger_version,
        "selected_alpha": float(fitted["alpha"]),
        "interval_config": fitted.get("interval_config", {}),
        "training_rows": int(fitted.get("training_rows", 0)),
        "validation_rows": int(fitted.get("validation_rows", 0)),
        "validation_metrics": fitted.get("validation_metrics", {}),
    }
    return pred_df, fitted_meta


def apply_prediction_frame_to_joined_rows(
    joined_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    prediction_column: str = "predicted_value",
    prop_prediction_column: str = "predicted_walks",
) -> pd.DataFrame:
    if joined_df.empty:
        return joined_df.copy()

    preds = prediction_df.copy()
    preds = ensure_participant_identity(
        preds,
        display_name_col="player_name",
        normalized_name_col=PARTICIPANT_NAME_NORM_COLUMN if PARTICIPANT_NAME_NORM_COLUMN in preds.columns else None,
        source_id_col="pitcher" if "pitcher" in preds.columns else PARTICIPANT_SOURCE_ID_COLUMN,
        source_id_type="mlbam_player",
    )
    preds = ensure_market_identity(
        preds,
        sport="MLB",
        market_key=config.PITCHER_BB_PROP_MARKET,
    )

    mapping_columns = [
        PARTICIPANT_JOIN_KEY_COLUMN,
        prop_prediction_column,
        prediction_column,
        "lower_bound",
        "upper_bound",
        "std_dev",
    ]
    preds = preds[[column for column in mapping_columns if column in preds.columns]].drop_duplicates(
        subset=[PARTICIPANT_JOIN_KEY_COLUMN],
        keep="last",
    )

    working = joined_df.copy()
    working = ensure_participant_identity(
        working,
        display_name_col="player_name" if "player_name" in working.columns else "player_name_proj",
        normalized_name_col=PARTICIPANT_NAME_NORM_COLUMN if PARTICIPANT_NAME_NORM_COLUMN in working.columns else None,
    )
    merged = working.drop(
        columns=[
            column
            for column in [prediction_column, prop_prediction_column, "lower_bound", "upper_bound", "std_dev"]
            if column in working.columns
        ]
    ).merge(
        preds,
        on=PARTICIPANT_JOIN_KEY_COLUMN,
        how="left",
    )
    if prediction_column not in merged.columns and prop_prediction_column in merged.columns:
        merged[prediction_column] = merged[prop_prediction_column]
    return merged


def side_relative_edge(predicted_value: object, line: object, side: object) -> float | None:
    predicted_numeric = pd.to_numeric(pd.Series([predicted_value]), errors="coerce").iloc[0]
    line_numeric = pd.to_numeric(pd.Series([line]), errors="coerce").iloc[0]
    if pd.isna(predicted_numeric) or pd.isna(line_numeric):
        return None
    side_norm = _normalize_side(side)
    if side_norm == "under":
        return float(line_numeric - predicted_numeric)
    return float(predicted_numeric - line_numeric)


def build_shadow_candidate_rows(
    joined_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    *,
    model_name: str,
    model_role: str,
    model_version: str,
    policy_version: str,
    tracking_regime: str,
    game_date: object,
) -> pd.DataFrame:
    if joined_df.empty:
        return empty_shadow_tracking_df()

    working = joined_df.copy()
    working = ensure_participant_identity(
        working,
        display_name_col="player_name" if "player_name" in working.columns else "player_name_proj",
        normalized_name_col=PARTICIPANT_NAME_NORM_COLUMN if PARTICIPANT_NAME_NORM_COLUMN in working.columns else None,
        source_id_col=PARTICIPANT_SOURCE_ID_COLUMN if PARTICIPANT_SOURCE_ID_COLUMN in working.columns else None,
        source_id_type="mlbam_player",
    )
    working = ensure_market_identity(
        working,
        sport="MLB",
        market_key=config.PITCHER_BB_PROP_MARKET,
    )

    working["game_date"] = pd.to_datetime(
        _first_present(working, ["game_date"], game_date),
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    working["player_name"] = _first_present(working, ["player_name", "player_name_proj"], "").fillna("").astype(str)
    working["book"] = _first_present(working, ["book", "bookmaker"], "").fillna("").astype(str)
    working["bookmaker_key"] = _first_present(working, ["bookmaker_key"], "").fillna("").astype(str)
    working["event_id"] = _first_present(working, ["event_id"], "").fillna("").astype(str)
    working["side"] = _first_present(working, ["side"], "").fillna("").astype(str)
    working["side_norm"] = working["side"].apply(_normalize_side)
    working["line"] = pd.to_numeric(_first_present(working, ["line"], np.nan), errors="coerce")
    working["price"] = pd.to_numeric(_first_present(working, ["price"], np.nan), errors="coerce")
    working["team"] = _first_present(working, ["team"], "").fillna("").astype(str)
    working["opponent"] = _first_present(working, ["opponent"], "").fillna("").astype(str)
    working["predicted_walks"] = pd.to_numeric(
        _first_present(working, ["predicted_walks", "predicted_value"], np.nan),
        errors="coerce",
    )
    working["predicted_value"] = pd.to_numeric(
        _first_present(working, ["predicted_value", "predicted_walks"], np.nan),
        errors="coerce",
    )
    working["lower_bound"] = pd.to_numeric(_first_present(working, ["lower_bound"], np.nan), errors="coerce")
    working["upper_bound"] = pd.to_numeric(_first_present(working, ["upper_bound"], np.nan), errors="coerce")
    working["std_dev"] = pd.to_numeric(_first_present(working, ["std_dev"], np.nan), errors="coerce")
    working["candidate_key"] = candidate_key_series(
        working,
        book_col="book",
        side_col="side",
        line_col="line",
    )
    working["edge"] = working.apply(
        lambda row: side_relative_edge(row["predicted_value"], row["line"], row["side"]),
        axis=1,
    )
    working["model_name"] = model_name
    working["model_role"] = model_role
    working["model_version"] = model_version
    working["policy_version"] = policy_version
    working["tracking_regime"] = tracking_regime
    working["prop_type"] = "pitcher_bb"

    pick_lookup = pd.DataFrame(columns=["candidate_key", "would_pick", "pick_rank", "pick_type", "confidence_tier"])
    if not picks_df.empty:
        actionable = picks_df.copy()
        actionable = actionable[actionable["pick_type"].astype(str).ne("pass")].copy()
        if not actionable.empty:
            actionable["game_date"] = pd.to_datetime(
                _first_present(actionable, ["game_date"], game_date),
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")
            actionable["candidate_key"] = candidate_key_series(
                actionable,
                book_col="book",
                side_col="pick_side",
                line_col="line",
            )
            actionable = actionable.reset_index(drop=True)
            actionable["would_pick"] = True
            actionable["pick_rank"] = actionable.index + 1
            pick_lookup = actionable[
                ["candidate_key", "would_pick", "pick_rank", "pick_type", "confidence_tier"]
            ].drop_duplicates(subset=["candidate_key"], keep="first")

    working = working.merge(
        pick_lookup,
        on="candidate_key",
        how="left",
    )
    working["would_pick"] = working["would_pick"].fillna(False).astype(bool)
    working["pick_rank"] = pd.to_numeric(working["pick_rank"], errors="coerce")
    working["pick_type"] = working["pick_type"].fillna("").astype(str)
    working["confidence_tier"] = working["confidence_tier"].fillna("").astype(str)
    working["actual_value"] = ""
    working["result"] = ""
    working["profit_units"] = pd.NA
    working["graded_at"] = ""
    working["shadow_row_key"] = shadow_row_key_series(working)

    for column in SHADOW_TRACKING_COLUMNS:
        if column not in working.columns:
            if column in {"profit_units", "pick_rank", "line", "price", "predicted_value", "predicted_walks", "lower_bound", "upper_bound", "std_dev", "edge"}:
                working[column] = pd.NA
            elif column == "would_pick":
                working[column] = False
            else:
                working[column] = ""

    return working[SHADOW_TRACKING_COLUMNS].copy()


def coerce_shadow_tracking_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_shadow_tracking_df()

    working = df.copy()
    for column in SHADOW_TRACKING_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA if column in {"profit_units", "pick_rank"} else ""

    numeric_columns = [
        "line",
        "price",
        "predicted_value",
        "predicted_walks",
        "lower_bound",
        "upper_bound",
        "std_dev",
        "edge",
        "pick_rank",
        "profit_units",
    ]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if working["would_pick"].dtype != bool:
        working["would_pick"] = working["would_pick"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return working[SHADOW_TRACKING_COLUMNS].copy()


def _to_pitcher_k_shadow_df(shadow_df: pd.DataFrame) -> pd.DataFrame:
    working = coerce_shadow_tracking_df(shadow_df).rename(
        columns={"predicted_walks": "predicted_strikeouts"}
    )
    working["market_key"] = "pitcher_strikeouts"
    working["prop_type"] = "pitcher_k"
    translated = working.reindex(columns=pitcher_k_shadow.SHADOW_TRACKING_COLUMNS, fill_value=pd.NA)
    return pitcher_k_shadow.coerce_shadow_tracking_df(translated)


def _from_pitcher_k_shadow_df(shadow_df: pd.DataFrame) -> pd.DataFrame:
    if shadow_df.empty:
        return empty_shadow_tracking_df()
    working = shadow_df.copy().rename(columns={"predicted_strikeouts": "predicted_walks"})
    working["market_key"] = config.PITCHER_BB_PROP_MARKET
    working["prop_type"] = "pitcher_bb"
    translated = working.reindex(columns=SHADOW_TRACKING_COLUMNS, fill_value=pd.NA)
    return coerce_shadow_tracking_df(translated)


def build_shadow_overlap_df(
    shadow_df: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = (CHAMPION_MODEL_NAME, CHALLENGER_MODEL_NAME),
) -> pd.DataFrame:
    overlap_df = pitcher_k_shadow.build_shadow_overlap_df(
        _to_pitcher_k_shadow_df(shadow_df),
        required_models=required_models,
    )
    return _from_pitcher_k_shadow_df(overlap_df)


def _patched_summary(summary: dict[str, object], *, review_context: dict | None) -> dict[str, object]:
    patched = dict(summary)
    patched["analysis_type"] = "pitcher_bb_shadow_comparison"
    promotion_review = build_shadow_promotion_review(
        patched,
        champion_name=CHAMPION_MODEL_NAME,
        challenger_name=CHALLENGER_MODEL_NAME,
        policy=DEFAULT_SHADOW_PROMOTION_POLICY,
        review_context=review_context
        or {
            "champion_name": CHAMPION_MODEL_NAME,
            "challenger_name": CHALLENGER_MODEL_NAME,
            "champion_status": CHAMPION_STATUS_STABLE,
        },
    )
    patched["promotion_review"] = promotion_review
    model_registry = dict(patched.get("model_registry", {}))
    if "champion" in model_registry:
        champion_registry = dict(model_registry["champion"])
        champion_registry["review_status"] = (
            "under_review" if promotion_review["eligible_for_review"] else "insufficient_evidence"
        )
        model_registry["champion"] = champion_registry
    if "challenger" in model_registry:
        challenger_registry = dict(model_registry["challenger"])
        challenger_registry["review_status"] = promotion_review["review_status"]
        model_registry["challenger"] = challenger_registry
    patched["model_registry"] = model_registry
    return patched


def build_shadow_comparison_report(
    shadow_df: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = (CHAMPION_MODEL_NAME, CHALLENGER_MODEL_NAME),
    review_context: dict | None = None,
) -> dict[str, object]:
    base_report = pitcher_k_shadow.build_shadow_comparison_report(
        _to_pitcher_k_shadow_df(shadow_df),
        required_models=required_models,
        review_context=review_context,
    )
    overlap_df = _from_pitcher_k_shadow_df(base_report["overlap_df"])
    return {
        "available": bool(base_report.get("available")),
        "reason": base_report.get("reason"),
        "overlap_df": overlap_df,
        "daily_regression_df": base_report["daily_regression_df"],
        "daily_workflow_df": base_report["daily_workflow_df"],
        "summary": _patched_summary(base_report["summary"], review_context=review_context),
    }


def write_shadow_regression_plot(
    daily_regression_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    if daily_regression_df.empty:
        return False

    plot_df = daily_regression_df.copy()
    plot_df["game_date"] = pd.to_datetime(plot_df["game_date"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for model_name, group in plot_df.groupby("model_name", dropna=False):
        group = group.sort_values("game_date").copy()
        group["rolling_mae"] = group["mae"].rolling(window=7, min_periods=1).mean()
        group["rolling_rmse"] = group["rmse"].rolling(window=7, min_periods=1).mean()
        axes[0].plot(group["game_date"], group["rolling_mae"], marker="o", label=str(model_name))
        axes[1].plot(group["game_date"], group["rolling_rmse"], marker="o", label=str(model_name))

    axes[0].set_title("Pitcher BB Shadow Regression Performance (7-day rolling)")
    axes[0].set_ylabel("Rolling MAE")
    axes[1].set_ylabel("Rolling RMSE")
    axes[1].set_xlabel("Game date")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def write_shadow_workflow_plot(
    daily_workflow_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    if daily_workflow_df.empty:
        return False

    plot_df = daily_workflow_df.copy()
    plot_df["game_date"] = pd.to_datetime(plot_df["game_date"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for model_name, group in plot_df.groupby("model_name", dropna=False):
        group = group.sort_values("game_date").copy()
        group["cumulative_units"] = group["profit_units"].cumsum()
        group["rolling_roi"] = group["roi_per_pick"].rolling(window=7, min_periods=1).mean()
        axes[0].plot(group["game_date"], group["cumulative_units"], marker="o", label=str(model_name))
        axes[1].plot(group["game_date"], group["rolling_roi"], marker="o", label=str(model_name))

    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Pitcher BB Shadow Workflow ROI (7-day rolling)")
    axes[0].set_ylabel("Cumulative profit units")
    axes[1].set_ylabel("Rolling ROI per pick")
    axes[1].set_xlabel("Game date")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True
