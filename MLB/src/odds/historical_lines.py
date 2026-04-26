from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.contracts import require_columns, validate_historical_lines_contract
from odds.normalize import normalize_player_name
from pitcher_k.config import PITCHER_K_PROP_MARKET

RAW_HISTORICAL_LINES_REQUIRED_COLUMNS = [
    "game_date",
    "player_name",
    "market_key",
    "bookmaker",
    "side",
    "line",
    "price",
]

HISTORICAL_LINES_COLUMNS = [
    "game_date",
    "player_name",
    "player_name_norm",
    "market_key",
    "bookmaker",
    "bookmaker_key",
    "side",
    "line",
    "price",
    "event_id",
    "commence_time",
    "selection_rule",
    "source",
    "pulled_at",
    "snapshot_type",
    "is_closing_line",
    "snapshot_rank",
]

DEFAULT_SELECTION_RULE = "latest_pregame_snapshot_per_game_player_book_side"
DEFAULT_SOURCE = "native_raw_historical_lines"


def empty_historical_lines_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORICAL_LINES_COLUMNS)


def _optional_series(df: pd.DataFrame, column: str, default) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_raw_historical_line_snapshots(raw_dir: Path) -> pd.DataFrame:
    """
    Load raw historical line snapshot CSVs from disk.
    """
    if not raw_dir.exists():
        return pd.DataFrame()

    csv_paths = sorted(raw_dir.rglob("*.csv"))
    if not csv_paths:
        return pd.DataFrame()

    frames = [pd.read_csv(path) for path in csv_paths]
    if not frames:
        return pd.DataFrame()

    loaded = pd.concat(frames, ignore_index=True)
    return loaded


def normalize_historical_line_snapshots(
    raw_df: pd.DataFrame,
    *,
    market_key: str = PITCHER_K_PROP_MARKET,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    default_source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """
    Normalize raw historical line snapshots into a canonical schema.
    """
    if raw_df.empty:
        return empty_historical_lines_df()

    require_columns(raw_df, RAW_HISTORICAL_LINES_REQUIRED_COLUMNS, "raw_historical_lines_df")

    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["market_key"] = df["market_key"].astype(str)
    df = df[df["market_key"] == market_key].copy()
    if df.empty:
        return empty_historical_lines_df()

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_norm"] = df["player_name"].apply(normalize_player_name)
    df["bookmaker"] = df["bookmaker"].astype(str).str.strip()
    df["bookmaker_key"] = _optional_series(df, "bookmaker_key", "").astype(str).str.strip()
    df.loc[df["bookmaker_key"] == "", "bookmaker_key"] = df["bookmaker"]
    df["side"] = df["side"].astype(str).str.strip()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["event_id"] = _optional_series(df, "event_id", None)
    df["commence_time"] = pd.to_datetime(
        _optional_series(df, "commence_time", None),
        errors="coerce",
        utc=True,
    )
    df["pulled_at"] = pd.to_datetime(
        _optional_series(df, "pulled_at", None),
        errors="coerce",
        utc=True,
    )
    df["snapshot_type"] = _optional_series(df, "snapshot_type", "raw_snapshot").fillna("raw_snapshot")
    df["source"] = _optional_series(df, "source", default_source).fillna(default_source)
    df["selection_rule"] = _optional_series(df, "selection_rule", selection_rule).fillna(selection_rule)

    df["is_closing_line"] = False
    if "is_closing_line" in df.columns:
        df["is_closing_line"] = df["is_closing_line"].fillna(False).astype(bool)

    df["snapshot_rank"] = pd.to_numeric(
        _optional_series(df, "snapshot_rank", None),
        errors="coerce",
    )

    return df[HISTORICAL_LINES_COLUMNS]


def curate_historical_lines(
    snapshots_df: pd.DataFrame,
    *,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> pd.DataFrame:
    """
    Select one deterministic line per game_date x player x sportsbook x side.
    Prefers the latest snapshot at or before commence_time; otherwise falls back
    to the latest available snapshot.
    """
    if snapshots_df.empty:
        return empty_historical_lines_df()

    df = snapshots_df.copy()
    require_columns(
        df,
        [
            "game_date",
            "player_name",
            "player_name_norm",
            "bookmaker",
            "side",
            "line",
            "price",
            "selection_rule",
            "source",
        ],
        "historical_line_snapshots_df",
    )

    df["selection_rule"] = selection_rule
    pregame_mask = (
        df["pulled_at"].notna()
        & df["commence_time"].notna()
        & (df["pulled_at"] <= df["commence_time"])
    )
    df["snapshot_sort_ts"] = df["pulled_at"].fillna(df["commence_time"])
    df["is_pregame_snapshot"] = pregame_mask
    df["snapshot_rank"] = pd.to_numeric(df["snapshot_rank"], errors="coerce")

    group_keys = ["game_date", "player_name_norm", "bookmaker", "side"]

    def select_group(group: pd.DataFrame) -> pd.Series:
        pregame = group[group["is_pregame_snapshot"]].copy()
        candidates = pregame if not pregame.empty else group.copy()
        candidates = candidates.sort_values(
            by=["snapshot_sort_ts", "snapshot_rank", "line", "price"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        selected = candidates.iloc[0].copy()
        selected["is_closing_line"] = bool(not pregame.empty)
        return selected

    curated = (
        df.groupby(group_keys, as_index=False, group_keys=False)
        .apply(select_group)
        .reset_index(drop=True)
    )

    curated = curated[HISTORICAL_LINES_COLUMNS].copy()
    validate_historical_lines_contract(curated)
    return curated


def build_historical_lines_artifact_df(
    raw_dir: Path,
    *,
    market_key: str = PITCHER_K_PROP_MARKET,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> pd.DataFrame:
    raw_df = load_raw_historical_line_snapshots(raw_dir)
    snapshots_df = normalize_historical_line_snapshots(
        raw_df,
        market_key=market_key,
        selection_rule=selection_rule,
    )
    curated = curate_historical_lines(
        snapshots_df,
        selection_rule=selection_rule,
    )
    return curated
