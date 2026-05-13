from __future__ import annotations

import pandas as pd
from common.identity import PARTICIPANT_JOIN_KEY_COLUMN

from .odds_api import fetch_all_player_props
from .normalize import odds_json_to_dataframe
from .compare import join_projections_to_odds, best_over_edges


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

    if odds_df.empty:
        retry_events = fetch_all_player_props(
            market=market,
            use_configured_bookmakers=False,
        )
        retry_odds_df = odds_json_to_dataframe(retry_events)
        if retry_odds_df.empty:
            diagnostics = _build_edge_pipeline_diagnostics(
                fetch_scope="all_region_books",
                raw_events=retry_events,
                odds_df=retry_odds_df,
                joined_df=pd.DataFrame(),
                best_edges_df=pd.DataFrame(),
                bookmaker_filter_applied=False,
            )
            diagnostics["initial_fetch"] = _build_edge_pipeline_diagnostics(
                fetch_scope="configured_books",
                raw_events=raw_events,
                odds_df=odds_df,
                joined_df=pd.DataFrame(),
                best_edges_df=pd.DataFrame(),
                bookmaker_filter_applied=True,
            )
            return pd.DataFrame(), pd.DataFrame(), diagnostics
        raw_events = retry_events
        odds_df = retry_odds_df
        fetch_scope = "all_region_books"
        bookmaker_filter_applied = False

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

    if joined.empty:
        diagnostics = _build_edge_pipeline_diagnostics(
            fetch_scope=fetch_scope,
            raw_events=raw_events,
            odds_df=odds_df,
            joined_df=joined,
            best_edges_df=pd.DataFrame(),
            bookmaker_filter_applied=bookmaker_filter_applied,
        )
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
    return joined, best_edges, diagnostics
