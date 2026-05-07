from __future__ import annotations

import pandas as pd
from common.identity import PARTICIPANT_JOIN_KEY_COLUMN

from .odds_api import fetch_all_player_props
from .normalize import odds_json_to_dataframe
from .compare import join_projections_to_odds, best_over_edges


def run_edge_pipeline(
    projections: pd.DataFrame,
    market: str,
    *,
    participant_key: str = "player_name",
    projection_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    odds_join_key: str = PARTICIPANT_JOIN_KEY_COLUMN,
    sport: str = "MLB",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_events = fetch_all_player_props(market=market)
    odds_df = odds_json_to_dataframe(raw_events)

    if odds_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    joined = join_projections_to_odds(
        projections,
        odds_df,
        participant_key=participant_key,
        projection_join_key=projection_join_key,
        odds_join_key=odds_join_key,
        sport=sport,
        market_key=market,
    )

    if joined.empty:
        return joined, pd.DataFrame()

    group_key = PARTICIPANT_JOIN_KEY_COLUMN if PARTICIPANT_JOIN_KEY_COLUMN in joined.columns else f"{participant_key}_proj"
    best_edges = best_over_edges(joined, group_key=group_key)
    return joined, best_edges
