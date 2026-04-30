# MLB/src/pitcher_k/feature_engineering.py

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from common.contracts import (
    assert_no_duplicate_keys,
    assert_non_empty,
    assert_non_null_columns,
    require_columns,
    validate_pitcher_games_contract,
)
from .config import STARTER_LIKE_MIN_BATTERS_FACED, STARTER_LIKE_MIN_PITCHES


def build_pitcher_game_table(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Statcast data into pitcher-game level stats.
    """
    require_columns(
        sc,
        [
            "game_date",
            "game_pk",
            "pitcher",
            "player_name",
            "pitch_type",
            "is_k",
            "is_whiff",
            "release_speed",
            "release_spin_rate",
            "batter",
            "home_team",
            "away_team",
            "p_throws",
        ],
        "statcast_df",
    )
    assert_non_empty(sc, "statcast_df")

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
    pitcher_games["whiff_per_pitch"] = pitcher_games["whiff_per_pitch"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    validate_pitcher_games_contract(
        pitcher_games.assign(
            pitching_team="TEMP",
            opponent_team="TEMP",
        )
    )
    assert_no_duplicate_keys(
        pitcher_games,
        ["game_date", "game_pk", "pitcher"],
        "pitcher_games",
    )

    return pitcher_games


def build_pitcher_team_lookup(sc: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        sc,
        [
            "game_date",
            "game_pk",
            "pitcher",
            "home_team",
            "away_team",
            "inning_topbot",
        ],
        "statcast_df",
    )
    assert_non_empty(sc, "statcast_df")

    df = sc[
        [
            "game_date",
            "game_pk",
            "pitcher",
            "home_team",
            "away_team",
            "inning_topbot",
        ]
    ].copy()

    df["pitching_team"] = df.apply(
        lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"],
        axis=1,
    )

    df["opponent_team"] = df.apply(
        lambda row: row["away_team"] if row["pitching_team"] == row["home_team"] else row["home_team"],
        axis=1,
    )

    team_lookup = (
        df.groupby(["game_date", "game_pk", "pitcher"], as_index=False)
        .agg(
            pitching_team=("pitching_team", "first"),
            opponent_team=("opponent_team", "first"),
        )
    )

    require_columns(
        team_lookup,
        ["game_date", "game_pk", "pitcher", "pitching_team", "opponent_team"],
        "team_lookup",
    )
    assert_non_empty(team_lookup, "team_lookup")
    assert_no_duplicate_keys(
        team_lookup,
        ["game_date", "game_pk", "pitcher"],
        "team_lookup",
    )
    assert_non_null_columns(
        team_lookup,
        ["game_date", "game_pk", "pitcher", "pitching_team", "opponent_team"],
        "team_lookup",
    )

    return team_lookup


def add_pitcher_team_info(pitcher_games: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    """
    Merge inferred pitcher team/opponent team info onto pitcher-game table.
    """
    require_columns(
        pitcher_games,
        ["game_date", "game_pk", "pitcher", "player_name"],
        "pitcher_games",
    )
    assert_non_empty(pitcher_games, "pitcher_games")

    team_lookup = build_pitcher_team_lookup(sc)

    pitcher_games = pitcher_games.merge(
        team_lookup,
        on=["game_date", "game_pk", "pitcher"],
        how="left",
    )

    validate_pitcher_games_contract(pitcher_games)
    assert_non_null_columns(
        pitcher_games,
        ["pitching_team", "opponent_team"],
        "pitcher_games",
    )

    return pitcher_games


def build_team_offense_k_table(sc: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        sc,
        ["game_date", "game_pk", "inning_topbot", "away_team", "home_team", "is_k", "batter"],
        "statcast_df",
    )
    assert_non_empty(sc, "statcast_df")

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
    team_offense["team_k_rate"] = team_offense["team_k_rate"].replace([np.inf, -np.inf], np.nan)

    team_offense = team_offense.sort_values(["batting_team", "game_date"]).copy()

    team_offense["opp_strikeouts_per_game_last10"] = (
        team_offense.groupby("batting_team")["team_batter_strikeouts"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

    team_offense["opp_k_rate_last10"] = (
        team_offense.groupby("batting_team")["team_k_rate"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

    require_columns(
        team_offense,
        [
            "game_date",
            "game_pk",
            "batting_team",
            "opp_strikeouts_per_game_last10",
            "opp_k_rate_last10",
        ],
        "team_offense",
    )
    assert_no_duplicate_keys(
        team_offense,
        ["game_date", "game_pk", "batting_team"],
        "team_offense",
    )

    return team_offense


def add_opponent_k_features(pitcher_games: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        pitcher_games,
        ["game_date", "game_pk", "opponent_team"],
        "pitcher_games",
    )
    assert_non_empty(pitcher_games, "pitcher_games")

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

    require_columns(
        pitcher_games,
        ["opp_strikeouts_per_game_last10", "opp_k_rate_last10"],
        "pitcher_games",
    )

    return pitcher_games


def add_rolling_pitcher_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing rolling features using prior games only.
    """
    require_columns(
        pitcher_games,
        [
            "pitcher",
            "game_date",
            "strikeouts",
            "pitches",
            "batters_faced",
            "whiff_per_pitch",
            "avg_velo",
            "avg_spin",
        ],
        "pitcher_games",
    )
    assert_non_empty(pitcher_games, "pitcher_games")

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

    strikeouts_history = pitcher_games.groupby("pitcher")["strikeouts"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std(ddof=0)
    )
    pitcher_games["strikeouts_stddev_last10"] = strikeouts_history
    pitcher_games["strikeouts_p25_last10"] = pitcher_games.groupby("pitcher")["strikeouts"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).quantile(0.25)
    )
    pitcher_games["strikeouts_p75_last10"] = pitcher_games.groupby("pitcher")["strikeouts"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).quantile(0.75)
    )

    require_columns(
        pitcher_games,
        [
            "pitches_last3",
            "pitches_last10",
            "whiff_per_pitch_last3",
            "avg_velo_last3",
            "avg_spin_last3",
            "strikeouts_last10",
            "batters_faced_last10",
            "strikeouts_stddev_last10",
            "strikeouts_p25_last10",
            "strikeouts_p75_last10",
        ],
        "pitcher_games",
    )

    return pitcher_games


def add_rate_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived strikeout rate features from rolling windows.
    """
    require_columns(
        pitcher_games,
        ["strikeouts_last10", "pitches_last10", "batters_faced_last10"],
        "pitcher_games",
    )
    assert_non_empty(pitcher_games, "pitcher_games")

    pitcher_games = pitcher_games.copy()

    pitcher_games["k_per_pitch_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["pitches_last10"]
    )

    pitcher_games["k_rate_last10"] = (
        pitcher_games["strikeouts_last10"] / pitcher_games["batters_faced_last10"]
    )

    pitcher_games["k_per_pitch_last10"] = pitcher_games["k_per_pitch_last10"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    pitcher_games["k_rate_last10"] = pitcher_games["k_rate_last10"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    require_columns(
        pitcher_games,
        ["k_per_pitch_last10", "k_rate_last10"],
        "pitcher_games",
    )

    return pitcher_games


def filter_starter_like_appearances(
    pitcher_games: pd.DataFrame,
    *,
    min_pitches: int = STARTER_LIKE_MIN_PITCHES,
    min_batters_faced: int = STARTER_LIKE_MIN_BATTERS_FACED,
) -> pd.DataFrame:
    """
    Keep appearances with starter-like workload for starter projection workflows.
    """
    require_columns(
        pitcher_games,
        ["pitches", "batters_faced"],
        "pitcher_games",
    )

    starter_like_mask = (
        (pitcher_games["pitches"] >= min_pitches)
        | (pitcher_games["batters_faced"] >= min_batters_faced)
    )
    return pitcher_games.loc[starter_like_mask].copy()


def build_team_context(pitcher_games: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Build latest available opponent context for each team prior to as_of_date.
    """
    require_columns(
        pitcher_games,
        [
            "game_date",
            "game_pk",
            "opponent_team",
            "opp_strikeouts_per_game_last10",
            "opp_k_rate_last10",
        ],
        "pitcher_games",
    )
    assert_non_empty(pitcher_games, "pitcher_games")

    df = pitcher_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    as_of_date = pd.to_datetime(as_of_date)

    df = df[df["game_date"] < as_of_date].copy()
    df = df.sort_values(["opponent_team", "game_date", "game_pk"])

    team_context = (
        df.groupby("opponent_team", as_index=False)
        .tail(1)[
            [
                "opponent_team",
                "opp_strikeouts_per_game_last10",
                "opp_k_rate_last10",
            ]
        ]
        .drop_duplicates(subset=["opponent_team"])
        .reset_index(drop=True)
    )

    require_columns(
        team_context,
        ["opponent_team", "opp_strikeouts_per_game_last10", "opp_k_rate_last10"],
        "team_context",
    )

    return team_context


def normalize_player_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = str(name).strip()

    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}"

    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))

    name = " ".join(name.split()).lower()
    return name


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator
