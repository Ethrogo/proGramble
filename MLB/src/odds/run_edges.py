from __future__ import annotations

import pandas as pd
from common.identity import PARTICIPANT_JOIN_KEY_COLUMN

from .compare import join_projections_to_odds, best_over_edges, prepare_projection_df
from .odds_api import fetch_all_player_props, fetch_event_player_props
from .normalize import odds_json_to_dataframe


MLB_TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

MLB_TEAM_CODE_ALIASES = {
    "ATH": "ATH",
    "OAK": "ATH",
    "KC": "KC",
    "KCR": "KC",
    "SD": "SD",
    "SDP": "SD",
    "SF": "SF",
    "SFG": "SF",
    "TB": "TB",
    "TBR": "TB",
    "WSH": "WSH",
    "WSN": "WSH",
}


def _build_edge_pipeline_diagnostics(
    *,
    fetch_scope: str,
    raw_events: list[dict],
    odds_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    best_edges_df: pd.DataFrame,
    bookmaker_filter_applied: bool,
) -> dict[str, object]:
    return {
        "fetch_scope": fetch_scope,
        "bookmaker_filter_applied": bookmaker_filter_applied,
        "raw_event_count": int(len(raw_events)),
        "normalized_odds_rows": int(len(odds_df)),
        "joined_rows": int(len(joined_df)),
        "best_edge_rows": int(len(best_edges_df)),
    }


def _event_team_codes(event: dict) -> tuple[str, str]:
    home = str(event.get("home_team", "") or "").strip()
    away = str(event.get("away_team", "") or "").strip()
    return _normalize_team_code(home), _normalize_team_code(away)


def _normalize_team_code(team: str) -> str:
    normalized = MLB_TEAM_NAME_TO_ABBR.get(team, team.upper())
    return MLB_TEAM_CODE_ALIASES.get(normalized, normalized)


def _missing_projection_event_ids(
    projections: pd.DataFrame,
    joined_df: pd.DataFrame,
    raw_events: list[dict],
    *,
    projection_join_key: str,
    sport: str,
    market: str,
    participant_key: str,
    prediction_column: str,
) -> list[str]:
    if not raw_events:
        return []

    prepared = prepare_projection_df(
        projections,
        participant_key=participant_key,
        prediction_column=prediction_column,
        projection_join_key=projection_join_key,
        sport=sport,
        market_key=market,
    )
    joined_keys = set(joined_df.get(projection_join_key, pd.Series(dtype="object")).astype(str))
    missing = prepared[~prepared[projection_join_key].astype(str).isin(joined_keys)].copy()
    if missing.empty:
        return []

    event_ids: list[str] = []
    for _, row in missing.iterrows():
        home_team = _normalize_team_code(str(row.get("home_team", "") or "").strip())
        away_team = _normalize_team_code(str(row.get("away_team", "") or "").strip())
        team = _normalize_team_code(str(row.get("team", "") or "").strip())
        opponent = _normalize_team_code(str(row.get("opponent", "") or "").strip())
        for event in raw_events:
            event_id = str(event.get("id", "") or "").strip()
            if not event_id:
                continue
            event_home, event_away = _event_team_codes(event)
            if home_team and away_team and event_home == home_team and event_away == away_team:
                event_ids.append(event_id)
                break
            if team and opponent and {event_home, event_away} == {team, opponent}:
                event_ids.append(event_id)
                break
    return list(dict.fromkeys(event_ids))


def run_edge_pipeline(
    projections: pd.DataFrame,
    market: str,
    *,
    participant_key: str = "player_name",
    prediction_column: str = "predicted_value",
    projection_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    odds_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    sport: str = "MLB",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fetch_scope = "configured_books"
    bookmaker_filter_applied = True
    raw_events = fetch_all_player_props(market=market)
    odds_df = odds_json_to_dataframe(raw_events)
    initial_fetch_snapshot = {
        "raw_event_count": int(len(raw_events)),
        "normalized_odds_rows": int(len(odds_df)),
    }

    joined = join_projections_to_odds(
        projections,
        odds_df,
        participant_key=participant_key,
        prediction_column=prediction_column,
        projection_join_key=projection_join_key,
        odds_join_key=odds_join_key,
        sport=sport,
        market_key=market,
    )

    retry_event_ids = _missing_projection_event_ids(
        projections,
        joined,
        raw_events,
        projection_join_key=projection_join_key,
        sport=sport,
        market=market,
        participant_key=participant_key,
        prediction_column=prediction_column,
    )
    if retry_event_ids:
        retry_events: list[dict] = []
        for event_id in retry_event_ids:
            retry_event = fetch_event_player_props(
                event_id=event_id,
                market=market,
                sport=sport,
                bookmakers=None,
                use_configured_bookmakers=False,
            )
            if retry_event:
                retry_events.append(retry_event)

        retry_odds_df = odds_json_to_dataframe(retry_events)
        if not retry_odds_df.empty:
            odds_df = pd.concat([odds_df, retry_odds_df], ignore_index=True)
            if "event_id" in odds_df.columns and "bookmaker_key" in odds_df.columns and "market_key" in odds_df.columns and "player_name_norm" in odds_df.columns and "side" in odds_df.columns and "line" in odds_df.columns:
                odds_df = odds_df.drop_duplicates(
                    subset=["event_id", "bookmaker_key", "market_key", "player_name_norm", "side", "line"],
                    keep="last",
                ).reset_index(drop=True)
            joined = join_projections_to_odds(
                projections,
                odds_df,
                participant_key=participant_key,
                prediction_column=prediction_column,
                projection_join_key=projection_join_key,
                odds_join_key=odds_join_key,
                sport=sport,
                market_key=market,
            )
            fetch_scope = "targeted_all_region_books"
            bookmaker_filter_applied = False

    if joined.empty and market == "pitcher_walks":
        raw_events = fetch_all_player_props(
            market=market,
            bookmakers=None,
            use_configured_bookmakers=False,
        )
        odds_df = odds_json_to_dataframe(raw_events)
        joined = join_projections_to_odds(
            projections,
            odds_df,
            participant_key=participant_key,
            prediction_column=prediction_column,
            projection_join_key=projection_join_key,
            odds_join_key=odds_join_key,
            sport=sport,
            market_key=market,
        )
        fetch_scope = "all_region_books"
        bookmaker_filter_applied = False

    if joined.empty:
        diagnostics = _build_edge_pipeline_diagnostics(
            fetch_scope=fetch_scope,
            raw_events=raw_events,
            odds_df=odds_df,
            joined_df=joined,
            best_edges_df=pd.DataFrame(),
            bookmaker_filter_applied=bookmaker_filter_applied,
        )
        diagnostics["targeted_retry_event_ids"] = retry_event_ids
        diagnostics["initial_fetch"] = initial_fetch_snapshot
        return joined, pd.DataFrame(), diagnostics

    group_key = PARTICIPANT_JOIN_KEY_COLUMN if PARTICIPANT_JOIN_KEY_COLUMN in joined.columns else f"{participant_key}_proj"
    best_edges = best_over_edges(joined, group_key=group_key)
    diagnostics = _build_edge_pipeline_diagnostics(
        fetch_scope=fetch_scope,
        raw_events=raw_events,
        odds_df=odds_df,
        joined_df=joined,
        best_edges_df=best_edges,
        bookmaker_filter_applied=bookmaker_filter_applied,
    )
    diagnostics["targeted_retry_event_ids"] = retry_event_ids
    diagnostics["initial_fetch"] = initial_fetch_snapshot
    return joined, best_edges, diagnostics
