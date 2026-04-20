import numpy as np
import pandas as pd
from .feature_engineering import build_pitcher_game_table, add_pitcher_team_info, normalize_player_name, _safe_div
from .feature_model import get_feature_columns
def build_tomorrow_features(
    slate_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    team_context: pd.DataFrame | None = None,
    min_career_starts: int = 5,
) -> pd.DataFrame:
    slate_df = slate_df.copy()
    pitcher_games = pitcher_games.copy()

    slate_df["game_date"] = pd.to_datetime(slate_df["game_date"])
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])

    slate_df["player_name_norm"] = slate_df["player_name"].apply(normalize_player_name)
    pitcher_games["player_name_norm"] = pitcher_games["player_name"].apply(normalize_player_name)

    pitcher_games = pitcher_games.sort_values(["player_name_norm", "game_date", "game_pk"])

    feature_rows = []

    for _, row in slate_df.iterrows():
        game_date = row["game_date"]
        pitcher_id = row.get("pitcher", np.nan)
        player_name_norm = row["player_name_norm"]

        hist = pd.DataFrame()

        # only use ID if it actually matches something historical
        if pd.notna(pitcher_id) and str(pitcher_id).strip() != "":
            hist = pitcher_games[
            (pitcher_games["pitcher"] == pitcher_id)
            & (pitcher_games["game_date"] < game_date)
            ].copy()

        # fallback to normalized name if ID match is empty
        if hist.empty:
            hist = pitcher_games[
            (pitcher_games["player_name_norm"] == player_name_norm)
            & (pitcher_games["game_date"] < game_date)].copy()

        hist = hist.sort_values(["game_date", "game_pk"])

        if len(hist) < min_career_starts:
            continue

        last3 = hist.tail(3)
        last5 = hist.tail(5)
        last10 = hist.tail(10)

        feature_row = row.to_dict()

        feature_row["career_starts"] = len(hist)
        feature_row["starts_last3_available"] = len(last3)
        feature_row["starts_last5_available"] = len(last5)
        feature_row["starts_last10_available"] = len(last10)

        feature_row["strikeouts_per_game"] = hist["strikeouts"].mean()
        feature_row["pitches_per_game"] = hist["pitches"].mean()
        feature_row["batters_faced_per_game"] = hist["batters_faced"].mean()
        feature_row["whiffs_per_game"] = hist["whiffs"].mean()

        feature_row["k_per_pitch"] = _safe_div(hist["strikeouts"].sum(), hist["pitches"].sum())
        feature_row["k_rate"] = _safe_div(hist["strikeouts"].sum(), hist["batters_faced"].sum())
        feature_row["whiff_per_pitch_season"] = _safe_div(hist["whiffs"].sum(), hist["pitches"].sum())

        feature_row["avg_velo_season"] = hist["avg_velo"].mean()
        feature_row["avg_spin_season"] = hist["avg_spin"].mean()

        feature_row["strikeouts_last3"] = last3["strikeouts"].mean()
        feature_row["pitches_last3"] = last3["pitches"].mean()
        feature_row["batters_faced_last3"] = last3["batters_faced"].mean()
        feature_row["whiffs_last3"] = last3["whiffs"].mean()

        feature_row["k_per_pitch_last3"] = _safe_div(last3["strikeouts"].sum(), last3["pitches"].sum())
        feature_row["k_rate_last3"] = _safe_div(last3["strikeouts"].sum(), last3["batters_faced"].sum())
        feature_row["whiff_per_pitch_last3"] = _safe_div(last3["whiffs"].sum(), last3["pitches"].sum())

        feature_row["avg_velo_last3"] = last3["avg_velo"].mean()
        feature_row["avg_spin_last3"] = last3["avg_spin"].mean()

        feature_row["strikeouts_last5"] = last5["strikeouts"].mean()
        feature_row["pitches_last5"] = last5["pitches"].mean()
        feature_row["batters_faced_last5"] = last5["batters_faced"].mean()
        feature_row["whiffs_last5"] = last5["whiffs"].mean()

        feature_row["k_per_pitch_last5"] = _safe_div(last5["strikeouts"].sum(), last5["pitches"].sum())
        feature_row["k_rate_last5"] = _safe_div(last5["strikeouts"].sum(), last5["batters_faced"].sum())
        feature_row["whiff_per_pitch_last5"] = _safe_div(last5["whiffs"].sum(), last5["pitches"].sum())

        feature_row["avg_velo_last5"] = last5["avg_velo"].mean()
        feature_row["avg_spin_last5"] = last5["avg_spin"].mean()

        feature_row["strikeouts_last10"] = last10["strikeouts"].mean()
        feature_row["pitches_last10"] = last10["pitches"].mean()
        feature_row["batters_faced_last10"] = last10["batters_faced"].mean()
        feature_row["whiffs_last10"] = last10["whiffs"].mean()

        feature_row["k_per_pitch_last10"] = _safe_div(last10["strikeouts"].sum(), last10["pitches"].sum())
        feature_row["k_rate_last10"] = _safe_div(last10["strikeouts"].sum(), last10["batters_faced"].sum())
        feature_row["whiff_per_pitch_last10"] = _safe_div(last10["whiffs"].sum(), last10["pitches"].sum())

        feature_row["avg_velo_last10"] = last10["avg_velo"].mean()
        feature_row["avg_spin_last10"] = last10["avg_spin"].mean()

        feature_row["k_trend_last3_vs_last10"] = (
            feature_row["strikeouts_last3"] - feature_row["strikeouts_last10"]
        )
        feature_row["velo_trend_last3_vs_last10"] = (
            feature_row["avg_velo_last3"] - feature_row["avg_velo_last10"]
        )
        feature_row["whiff_trend_last3_vs_last10"] = (
            feature_row["whiff_per_pitch_last3"] - feature_row["whiff_per_pitch_last10"]
        )

        last_game_date = hist["game_date"].max()
        feature_row["days_rest"] = (
            (game_date - last_game_date).days if pd.notna(last_game_date) else np.nan
        )

        feature_row["is_home"] = int(row["is_home"])

        # default opponent context as missing
        feature_row["opp_strikeouts_per_game_last10"] = np.nan
        feature_row["opp_k_rate_last10"] = np.nan

        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)

    if features_df.empty:
        return features_df

    if team_context is not None:
        tc = team_context.copy()
        tc = tc.rename(columns={"opponent_team": "opponent"})
        features_df = features_df.drop(
            columns=["opp_strikeouts_per_game_last10", "opp_k_rate_last10"],
            errors="ignore",
        ).merge(
            tc[["opponent", "opp_strikeouts_per_game_last10", "opp_k_rate_last10"]],
            on="opponent",
            how="left",
        )

    base_cols = [
        "game_date", "game_pk", "pitcher", "player_name",
        "team", "opponent", "home_team", "away_team",
        "is_home", "p_throws"
    ]
    feature_cols = [c for c in features_df.columns if c not in base_cols]
    features_df = features_df[base_cols + feature_cols]

    return features_df
