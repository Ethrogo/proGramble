# src/pitcher_k/features.py

import numpy as np
import pandas as pd


def build_pitcher_game_table(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Statcast data into pitcher-game level stats.
    """
    pitcher_games = (
        sc.groupby(["game_date", "game_pk", "pitcher", "player_name"])
        .agg(
            pitches=("pitch_type", "count"),
            strikeouts=("is_k", "sum"),
            whiffs=("is_whiff", "sum"),
            avg_velo=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            batters_faced=("batter", "nunique"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            p_throws=("p_throws", "first"),
        )
        .reset_index()
    )

    pitcher_games["whiff_per_pitch"] = pitcher_games["whiffs"] / pitcher_games["pitches"]

    return pitcher_games


def build_pitcher_team_lookup(sc: pd.DataFrame) -> pd.DataFrame:
    temp = sc.reset_index(drop=True).copy()

    temp["batting_team"] = np.where(
        temp["inning_topbot"] == "Top",
        temp["away_team"],
        temp["home_team"],
    )

    temp["pitching_team"] = np.where(
        temp["batting_team"] == temp["home_team"],
        temp["away_team"],
        temp["home_team"],
    )

    team_lookup = (
        temp.groupby(["game_date", "game_pk", "pitcher"], as_index=False)
        .agg(
            pitcher_team=("pitching_team", "first"),
            opponent_team=("batting_team", "first"),
            p_throws=("p_throws", "first"),
        )
    )

    return team_lookup


def add_pitcher_team_info(pitcher_games: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    """
    Merge inferred pitcher team/opponent team info onto pitcher-game table.
    """
    team_lookup = build_pitcher_team_lookup(sc)

    pitcher_games = pitcher_games.merge(
        team_lookup,
        on=["game_date", "game_pk", "pitcher"],
        how="left",
    )

    return pitcher_games


def build_team_offense_k_table(sc: pd.DataFrame) -> pd.DataFrame:
    temp = sc.reset_index(drop=True).copy()

    temp["batting_team"] = np.where(
        temp["inning_topbot"] == "Top",
        temp["away_team"],
        temp["home_team"],
    )

    team_offense = (
        temp.groupby(["game_date", "game_pk", "batting_team"], as_index=False)
        .agg(
            team_batter_strikeouts=("is_k", "sum"),
            batters_faced=("batter", "nunique"),
        )
    )

    team_offense["team_k_rate"] = (
        team_offense["team_batter_strikeouts"] / team_offense["batters_faced"]
    )

    team_offense = team_offense.sort_values(["batting_team", "game_date"]).copy()

    team_offense["opp_strikeouts_per_game_last10"] = (
        team_offense.groupby("batting_team")["team_batter_strikeouts"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

    team_offense["opp_k_rate_last10"] = (
        team_offense.groupby("batting_team")["team_k_rate"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

    return team_offense


def add_opponent_k_features(pitcher_games: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent offensive strikeout tendency features to pitcher-game table.
    """
    pitcher_games = add_pitcher_team_info(pitcher_games, sc)
    team_offense = build_team_offense_k_table(sc)

    opp_features = team_offense.rename(columns={"batting_team": "opponent_team"})[
        [
            "game_date",
            "game_pk",
            "opponent_team",
            "opp_strikeouts_per_game_last10",
            "opp_k_rate_last10",
        ]
    ]

    pitcher_games = pitcher_games.merge(
        opp_features,
        on=["game_date", "game_pk", "opponent_team"],
        how="left",
    )

    return pitcher_games


def add_rolling_pitcher_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing 3-game and 10-game rolling averages using prior games only.
    """
    pitcher_games = pitcher_games.sort_values(["pitcher", "game_date"]).copy()

    rolling_cols = [
        "strikeouts",
        "pitches",
        "batters_faced",
        "whiff_per_pitch",
        "avg_velo",
        "avg_spin",
    ]

    for col in rolling_cols:
        pitcher_games[f"{col}_last3"] = (
            pitcher_games.groupby("pitcher")[col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

        pitcher_games[f"{col}_last10"] = (
            pitcher_games.groupby("pitcher")[col]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        )

    return pitcher_games


def add_rate_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived strikeout rate features from rolling windows.
    """
    pitcher_games = pitcher_games.copy()

    pitcher_games["k_per_pitch_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["pitches_last10"]
    )

    pitcher_games["k_rate_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["batters_faced_last10"]
    )

    pitcher_games["k_per_pitch_last10"] = pitcher_games["k_per_pitch_last10"].replace(
        [np.inf, -np.inf], np.nan
    )
    pitcher_games["k_rate_last10"] = pitcher_games["k_rate_last10"].replace(
        [np.inf, -np.inf], np.nan
    )

    return pitcher_games


def get_feature_columns() -> list[str]:
    """
    Return the current model feature list.
    """
    return [
        "pitches_last3",
        "pitches_last10",
        "whiff_per_pitch_last3",
        "avg_velo_last3",
        "avg_spin_last3",
        "k_per_pitch_last10",
        "k_rate_last10",
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
    ]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator

def build_tomorrow_features(
    slate_df: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    min_career_starts: int = 5,
) -> pd.DataFrame:
    """
    Build model-ready features for tomorrow's slate using historical pitcher-game data.

    Parameters
    ----------
    slate_df : pd.DataFrame
        Tomorrow slate with one row per probable starter.
        Expected columns:
            game_date, game_pk, pitcher, player_name, team, opponent,
            home_team, away_team, is_home, p_throws

    pitcher_games : pd.DataFrame
        Historical pitcher-game level dataframe.
        Expected columns:
            game_date, game_pk, pitcher, player_name, pitches, strikeouts,
            whiffs, avg_velo, avg_spin, batters_faced, home_team, away_team,
            p_throws, whiff_per_pitch

    min_career_starts : int
        Minimum number of prior starts required to keep a pitcher.

    Returns
    -------
    pd.DataFrame
        One row per tomorrow pitcher with engineered historical features.
    """

    slate_df = slate_df.copy()
    pitcher_games = pitcher_games.copy()

    slate_df["game_date"] = pd.to_datetime(slate_df["game_date"])
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])

    # Sort once so rolling windows are correct
    pitcher_games = pitcher_games.sort_values(["player_name", "game_date", "game_pk"])

    feature_rows = []

    for _, row in slate_df.iterrows():
        game_date = row["game_date"]
        pitcher_id = row.get("pitcher", np.nan)
        player_name = row["player_name"]

        # Prefer pitcher ID if available, otherwise fall back to player_name
        if pd.notna(pitcher_id) and str(pitcher_id).strip() != "":
            hist = pitcher_games[
                (pitcher_games["pitcher"] == pitcher_id)
                & (pitcher_games["game_date"] < game_date)
            ].copy()
        else:
            hist = pitcher_games[
                (pitcher_games["player_name"] == player_name)
                & (pitcher_games["game_date"] < game_date)
            ].copy()

        hist = hist.sort_values(["game_date", "game_pk"])

        # Skip rookies / tiny samples
        if len(hist) < min_career_starts:
            continue

        last3 = hist.tail(3)
        last5 = hist.tail(5)
        last10 = hist.tail(10)

        out = row.to_dict()

        # Core counts
        out["career_starts"] = len(hist)
        out["starts_last3_available"] = len(last3)
        out["starts_last5_available"] = len(last5)
        out["starts_last10_available"] = len(last10)

        # Season / career-to-date features
        out["strikeouts_per_game"] = hist["strikeouts"].mean()
        out["pitches_per_game"] = hist["pitches"].mean()
        out["batters_faced_per_game"] = hist["batters_faced"].mean()
        out["whiffs_per_game"] = hist["whiffs"].mean()

        out["k_per_pitch"] = _safe_div(hist["strikeouts"].sum(), hist["pitches"].sum())
        out["k_rate"] = _safe_div(hist["strikeouts"].sum(), hist["batters_faced"].sum())
        out["whiff_per_pitch_season"] = _safe_div(hist["whiffs"].sum(), hist["pitches"].sum())

        out["avg_velo_season"] = hist["avg_velo"].mean()
        out["avg_spin_season"] = hist["avg_spin"].mean()

        # Last 3
        out["strikeouts_last3"] = last3["strikeouts"].mean()
        out["pitches_last3"] = last3["pitches"].mean()
        out["batters_faced_last3"] = last3["batters_faced"].mean()
        out["whiffs_last3"] = last3["whiffs"].mean()

        out["k_per_pitch_last3"] = _safe_div(last3["strikeouts"].sum(), last3["pitches"].sum())
        out["k_rate_last3"] = _safe_div(last3["strikeouts"].sum(), last3["batters_faced"].sum())
        out["whiff_per_pitch_last3"] = _safe_div(last3["whiffs"].sum(), last3["pitches"].sum())

        out["avg_velo_last3"] = last3["avg_velo"].mean()
        out["avg_spin_last3"] = last3["avg_spin"].mean()

        # Last 5
        out["strikeouts_last5"] = last5["strikeouts"].mean()
        out["pitches_last5"] = last5["pitches"].mean()
        out["batters_faced_last5"] = last5["batters_faced"].mean()
        out["whiffs_last5"] = last5["whiffs"].mean()

        out["k_per_pitch_last5"] = _safe_div(last5["strikeouts"].sum(), last5["pitches"].sum())
        out["k_rate_last5"] = _safe_div(last5["strikeouts"].sum(), last5["batters_faced"].sum())
        out["whiff_per_pitch_last5"] = _safe_div(last5["whiffs"].sum(), last5["pitches"].sum())

        out["avg_velo_last5"] = last5["avg_velo"].mean()
        out["avg_spin_last5"] = last5["avg_spin"].mean()

        # Last 10
        out["strikeouts_last10"] = last10["strikeouts"].mean()
        out["pitches_last10"] = last10["pitches"].mean()
        out["batters_faced_last10"] = last10["batters_faced"].mean()
        out["whiffs_last10"] = last10["whiffs"].mean()

        out["k_per_pitch_last10"] = _safe_div(last10["strikeouts"].sum(), last10["pitches"].sum())
        out["k_rate_last10"] = _safe_div(last10["strikeouts"].sum(), last10["batters_faced"].sum())
        out["whiff_per_pitch_last10"] = _safe_div(last10["whiffs"].sum(), last10["pitches"].sum())

        out["avg_velo_last10"] = last10["avg_velo"].mean()
        out["avg_spin_last10"] = last10["avg_spin"].mean()

        # Trend features
        out["k_trend_last3_vs_last10"] = out["strikeouts_last3"] - out["strikeouts_last10"]
        out["velo_trend_last3_vs_last10"] = out["avg_velo_last3"] - out["avg_velo_last10"]
        out["whiff_trend_last3_vs_last10"] = out["whiff_per_pitch_last3"] - out["whiff_per_pitch_last10"]

        # Rest days
        last_game_date = hist["game_date"].max()
        out["days_rest"] = (game_date - last_game_date).days if pd.notna(last_game_date) else np.nan

        # Home/away carry-through
        out["is_home"] = int(row["is_home"])

        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)

    if features_df.empty:
        return features_df

    # Optional cleanup / ordering
    base_cols = [
        "game_date", "game_pk", "pitcher", "player_name",
        "team", "opponent", "home_team", "away_team",
        "is_home", "p_throws"
    ]

    feature_cols = [c for c in features_df.columns if c not in base_cols]
    features_df = features_df[base_cols + feature_cols]

    return features_df


def build_model_df(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final model-ready dataframe from engineered pitcher-game features.
    """
    features = get_feature_columns()

    keep_cols = [
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "strikeouts",
    ] + features

    model_df = pitcher_games[keep_cols].copy()
    model_df["game_date"] = pd.to_datetime(model_df["game_date"])

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=["strikeouts"] + features)
    model_df = model_df.sort_values("game_date")

    return model_df