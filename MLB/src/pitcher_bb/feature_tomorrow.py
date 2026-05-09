from __future__ import annotations

import numpy as np
import pandas as pd

from common.contracts import require_columns, validate_pitcher_games_contract, validate_starters_contract
from pitcher_k.feature_engineering import filter_starter_like_appearances, normalize_player_name, _safe_div

from .config import BASE_FEATURES


def build_tomorrow_features(
    slate_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    team_context: pd.DataFrame | None = None,
    min_career_starts: int = 5,
) -> pd.DataFrame:
    slate_df = slate_df.copy()
    pitcher_games = pitcher_games.copy()

    validate_starters_contract(slate_df)
    validate_pitcher_games_contract(pitcher_games)
    require_columns(
        pitcher_games,
        [
            "walks",
            "pitches",
            "batters_faced",
            "avg_velo",
            "avg_spin",
        ],
        "pitcher_games",
    )
    if team_context is not None:
        require_columns(
            team_context,
            ["opponent_team", "opp_walks_per_game_last10", "opp_bb_rate_last10"],
            "team_context",
        )

    slate_df["game_date"] = pd.to_datetime(slate_df["game_date"])
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
    pitcher_games = filter_starter_like_appearances(pitcher_games)

    slate_df["player_name_norm"] = slate_df["player_name"].apply(normalize_player_name)
    pitcher_games["player_name_norm"] = pitcher_games["player_name"].apply(normalize_player_name)
    pitcher_games = pitcher_games.sort_values(["player_name_norm", "game_date", "game_pk"])

    feature_rows = []
    skipped_pitchers = 0

    for _, row in slate_df.iterrows():
        game_date = row["game_date"]
        pitcher_id = row.get("pitcher", np.nan)
        player_name_norm = row["player_name_norm"]

        hist = pd.DataFrame()
        if pd.notna(pitcher_id) and str(pitcher_id).strip() != "":
            hist = pitcher_games[
                (pitcher_games["pitcher"] == pitcher_id)
                & (pitcher_games["game_date"] < game_date)
            ].copy()

        if hist.empty:
            hist = pitcher_games[
                (pitcher_games["player_name_norm"] == player_name_norm)
                & (pitcher_games["game_date"] < game_date)
            ].copy()

        hist = hist.sort_values(["game_date", "game_pk"])
        if len(hist) < min_career_starts:
            skipped_pitchers += 1
            continue

        last3 = hist.tail(3)
        last10 = hist.tail(10)
        feature_row = row.to_dict()

        feature_row["pitches_last3"] = last3["pitches"].mean()
        feature_row["pitches_last10"] = last10["pitches"].mean()
        feature_row["batters_faced_last3"] = last3["batters_faced"].mean()
        feature_row["batters_faced_last10"] = last10["batters_faced"].mean()
        feature_row["walks_last3"] = last3["walks"].mean()
        feature_row["walks_last10"] = last10["walks"].mean()
        feature_row["avg_velo_last3"] = last3["avg_velo"].mean()
        feature_row["avg_spin_last3"] = last3["avg_spin"].mean()
        feature_row["bb_per_pitch_last10"] = _safe_div(last10["walks"].sum(), last10["pitches"].sum())
        feature_row["bb_rate_last10"] = _safe_div(last10["walks"].sum(), last10["batters_faced"].sum())
        feature_row["walks_stddev_last10"] = last10["walks"].std(ddof=0)
        feature_row["walks_p25_last10"] = last10["walks"].quantile(0.25)
        feature_row["walks_p75_last10"] = last10["walks"].quantile(0.75)
        feature_row["is_home"] = int(row["is_home"])
        feature_row["opp_walks_per_game_last10"] = np.nan
        feature_row["opp_bb_rate_last10"] = np.nan
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    if features_df.empty:
        features_df.attrs["skipped_pitchers"] = skipped_pitchers
        return features_df

    if team_context is not None:
        tc = team_context.copy().rename(columns={"opponent_team": "opponent"})
        features_df = features_df.drop(
            columns=["opp_walks_per_game_last10", "opp_bb_rate_last10"],
            errors="ignore",
        ).merge(
            tc[["opponent", "opp_walks_per_game_last10", "opp_bb_rate_last10"]],
            on="opponent",
            how="left",
        )

    base_cols = [
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "is_home",
        "p_throws",
    ]
    feature_cols = [c for c in features_df.columns if c not in base_cols]
    features_df = features_df[base_cols + feature_cols]
    require_columns(
        features_df,
        [
            "game_date",
            "game_pk",
            "pitcher",
            "player_name",
            "team",
            "opponent",
            "is_home",
        ] + BASE_FEATURES,
        "tomorrow_features_df",
    )
    features_df.attrs["skipped_pitchers"] = skipped_pitchers
    return features_df
