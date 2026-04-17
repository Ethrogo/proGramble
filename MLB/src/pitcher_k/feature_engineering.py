import numpy as np
import pandas as pd
import unicodedata

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

def build_team_context(pitcher_games: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Build latest available opponent context for each team prior to as_of_date.

    Expects pitcher_games to already contain:
        opponent_team,
        opp_strikeouts_per_game_last10,
        opp_k_rate_last10
    """
    df = pitcher_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    as_of_date = pd.to_datetime(as_of_date)

    required = [
        "game_date",
        "opponent_team",
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pitcher_games is missing required opponent context columns: {missing}")

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

