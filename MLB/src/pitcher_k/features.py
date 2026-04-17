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
    """
    Infer pitcher team and opponent for each pitcher-game from pitch-level rows.
    Requires 'inning_topbot' in the Statcast dataframe.
    """
    temp = sc.copy()

    temp["batting_team"] = temp["home_team"]
    temp.loc[temp["inning_topbot"] == "Top", "batting_team"] = temp["away_team"]
    temp.loc[temp["inning_topbot"] == "Bot", "batting_team"] = temp["home_team"]

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
    """
    Build team offensive strikeout tendencies by game, then rolling opponent features.
    """
    temp = sc.copy()

    temp["batting_team"] = temp["home_team"]
    temp.loc[temp["inning_topbot"] == "Top", "batting_team"] = temp["away_team"]
    temp.loc[temp["inning_topbot"] == "Bot", "batting_team"] = temp["home_team"]

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