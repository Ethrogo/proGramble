# src/pitcher_k/features.py

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