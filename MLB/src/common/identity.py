from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata

import pandas as pd

PARTICIPANT_NAME_COLUMN = "participant_name"
PARTICIPANT_NAME_NORM_COLUMN = "participant_name_norm"
PARTICIPANT_ID_COLUMN = "participant_id"
PARTICIPANT_SOURCE_ID_COLUMN = "participant_source_id"
PARTICIPANT_SOURCE_ID_TYPE_COLUMN = "participant_source_id_type"
PARTICIPANT_SOURCE_KEY_COLUMN = "participant_source_key"
PARTICIPANT_JOIN_KEY_COLUMN = "participant_join_key"

MARKET_FAMILY_COLUMN = "market_family"
SIDE_NORM_COLUMN = "side_norm"
MARKET_SELECTION_KEY_COLUMN = "market_selection_key"
MARKET_OFFER_KEY_COLUMN = "market_offer_key"


@dataclass(frozen=True)
class ParticipantIdentitySpec:
    canonical_id: str = PARTICIPANT_ID_COLUMN
    display_name: str = PARTICIPANT_NAME_COLUMN
    normalized_name: str = PARTICIPANT_NAME_NORM_COLUMN
    source_id: str = PARTICIPANT_SOURCE_ID_COLUMN
    source_id_type: str = PARTICIPANT_SOURCE_ID_TYPE_COLUMN
    source_key: str = PARTICIPANT_SOURCE_KEY_COLUMN
    join_key: str = PARTICIPANT_JOIN_KEY_COLUMN
    join_precedence: tuple[str, ...] = field(
        default_factory=lambda: (
            PARTICIPANT_ID_COLUMN,
            PARTICIPANT_SOURCE_KEY_COLUMN,
            PARTICIPANT_NAME_NORM_COLUMN,
        )
    )


@dataclass(frozen=True)
class MarketIdentitySpec:
    market_family: str = MARKET_FAMILY_COLUMN
    side_normalized: str = SIDE_NORM_COLUMN
    selection_key: str = MARKET_SELECTION_KEY_COLUMN
    offer_key: str = MARKET_OFFER_KEY_COLUMN


def normalize_participant_name(name: object) -> str:
    if pd.isna(name) or not isinstance(name, str):
        return ""

    text = name.strip()
    if not text:
        return ""

    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        text = f"{first} {last}"

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = (
        text.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )
    return " ".join(text.split())


def _string_series(df: pd.DataFrame, column: str | None, default: str = "") -> pd.Series:
    if not column or column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip()


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _build_source_key(source_id_type: str, source_id: str) -> str:
    if not source_id_type or not source_id:
        return ""
    return f"{source_id_type}:{source_id}"


def _build_join_key(participant_id: str, source_key: str, normalized_name: str) -> str:
    if participant_id:
        return participant_id
    if source_key:
        return source_key
    if normalized_name:
        return f"name:{normalized_name}"
    return ""


def _normalize_bookmaker_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return "".join(ch for ch in text if ch.isalnum() or ch == "_")


def _normalize_side(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _format_line_key(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value).strip()


def ensure_participant_identity(
    df: pd.DataFrame,
    *,
    display_name_col: str = "player_name",
    normalized_name_col: str | None = "player_name_norm",
    source_id_col: str | None = None,
    source_id_type: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    display_name = _string_series(out, PARTICIPANT_NAME_COLUMN)
    fallback_display_name = _string_series(out, display_name_col)
    out[PARTICIPANT_NAME_COLUMN] = display_name.where(display_name != "", fallback_display_name)

    normalized_name = _string_series(out, PARTICIPANT_NAME_NORM_COLUMN)
    if normalized_name_col and normalized_name_col in out.columns:
        fallback_norm = _string_series(out, normalized_name_col).map(normalize_participant_name)
    else:
        fallback_norm = out[PARTICIPANT_NAME_COLUMN].map(normalize_participant_name)
    out[PARTICIPANT_NAME_NORM_COLUMN] = normalized_name.where(normalized_name != "", fallback_norm)

    if normalized_name_col and normalized_name_col not in out.columns:
        out[normalized_name_col] = out[PARTICIPANT_NAME_NORM_COLUMN]

    source_id = _string_series(out, PARTICIPANT_SOURCE_ID_COLUMN)
    if source_id_col and source_id_col in out.columns:
        fallback_source_id = _string_series(out, source_id_col)
    else:
        fallback_source_id = pd.Series([""] * len(out), index=out.index, dtype="object")
    out[PARTICIPANT_SOURCE_ID_COLUMN] = source_id.where(source_id != "", fallback_source_id)

    source_id_type_series = _string_series(out, PARTICIPANT_SOURCE_ID_TYPE_COLUMN)
    default_source_type = source_id_type or ""
    source_id_type_fallback = pd.Series(
        [default_source_type] * len(out),
        index=out.index,
        dtype="object",
    )
    out[PARTICIPANT_SOURCE_ID_TYPE_COLUMN] = source_id_type_series.where(
        source_id_type_series != "",
        source_id_type_fallback,
    )
    out.loc[
        out[PARTICIPANT_SOURCE_ID_COLUMN] == "",
        PARTICIPANT_SOURCE_ID_TYPE_COLUMN,
    ] = ""

    source_key = _string_series(out, PARTICIPANT_SOURCE_KEY_COLUMN)
    out[PARTICIPANT_SOURCE_KEY_COLUMN] = [
        _first_non_empty(
            existing,
            _build_source_key(source_type, source_id_value),
        )
        for existing, source_type, source_id_value in zip(
            source_key,
            out[PARTICIPANT_SOURCE_ID_TYPE_COLUMN],
            out[PARTICIPANT_SOURCE_ID_COLUMN],
        )
    ]

    canonical_id = _string_series(out, PARTICIPANT_ID_COLUMN)
    out[PARTICIPANT_ID_COLUMN] = canonical_id.where(
        canonical_id != "",
        out[PARTICIPANT_SOURCE_KEY_COLUMN],
    )

    join_key = _string_series(out, PARTICIPANT_JOIN_KEY_COLUMN)
    out[PARTICIPANT_JOIN_KEY_COLUMN] = [
        _first_non_empty(
            existing,
            _build_join_key(participant_id, source_key_value, normalized_name_value),
        )
        for existing, participant_id, source_key_value, normalized_name_value in zip(
            join_key,
            out[PARTICIPANT_ID_COLUMN],
            out[PARTICIPANT_SOURCE_KEY_COLUMN],
            out[PARTICIPANT_NAME_NORM_COLUMN],
        )
    ]

    return out


def ensure_market_identity(
    df: pd.DataFrame,
    *,
    sport: str | None = None,
    market_key: str | None = None,
    market_family: str = "player_prop",
    side_col: str = "side",
    line_col: str = "line",
    bookmaker_col: str = "bookmaker",
    bookmaker_key_col: str = "bookmaker_key",
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if sport is not None and "sport" not in out.columns:
        out["sport"] = sport
    if market_key is not None and "market_key" not in out.columns:
        out["market_key"] = market_key
    if MARKET_FAMILY_COLUMN not in out.columns:
        out[MARKET_FAMILY_COLUMN] = market_family

    if bookmaker_key_col not in out.columns and bookmaker_col in out.columns:
        out[bookmaker_key_col] = out[bookmaker_col].map(_normalize_bookmaker_key)
    if bookmaker_key_col in out.columns:
        out[bookmaker_key_col] = out[bookmaker_key_col].map(_normalize_bookmaker_key)

    if side_col in out.columns and SIDE_NORM_COLUMN not in out.columns:
        out[SIDE_NORM_COLUMN] = out[side_col].map(_normalize_side)

    if (
        {"sport", "market_key", PARTICIPANT_JOIN_KEY_COLUMN, SIDE_NORM_COLUMN, line_col}.issubset(out.columns)
        and MARKET_SELECTION_KEY_COLUMN not in out.columns
    ):
        out[MARKET_SELECTION_KEY_COLUMN] = [
            "|".join(
                [
                    str(sport_value).strip(),
                    str(market_value).strip(),
                    str(participant_key).strip(),
                    str(side_value).strip(),
                    _format_line_key(line_value),
                ]
            )
            if all(
                [
                    str(sport_value).strip(),
                    str(market_value).strip(),
                    str(participant_key).strip(),
                    str(side_value).strip(),
                    _format_line_key(line_value),
                ]
            )
            else ""
            for sport_value, market_value, participant_key, side_value, line_value in zip(
                out["sport"],
                out["market_key"],
                out[PARTICIPANT_JOIN_KEY_COLUMN],
                out[SIDE_NORM_COLUMN],
                out[line_col],
            )
        ]

    if (
        {MARKET_SELECTION_KEY_COLUMN, bookmaker_key_col}.issubset(out.columns)
        and MARKET_OFFER_KEY_COLUMN not in out.columns
    ):
        out[MARKET_OFFER_KEY_COLUMN] = [
            f"{selection_key}|{book_key}"
            if selection_key and book_key
            else ""
            for selection_key, book_key in zip(
                out[MARKET_SELECTION_KEY_COLUMN],
                out[bookmaker_key_col],
            )
        ]

    if event_id_col in out.columns and "event_id" not in out.columns:
        out["event_id"] = out[event_id_col]

    return out
