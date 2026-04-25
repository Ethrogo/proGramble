from __future__ import annotations

import pandas as pd

from .odds_api import fetch_all_player_props 
from .normalize import odds_json_to_dataframe
from .compare import join_projections_to_odds, best_over_edges
from pitcher_k.config import PITCHER_K_PROP_MARKET

def run_edge_pipeline(projections: pd.DataFrame, market: str,) -> pd.DataFrame:
    raw_events = fetch_all_player_props(market=PITCHER_K_PROP_MARKET)
    odds_df = odds_json_to_dataframe(raw_events)

    if odds_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    joined = join_projections_to_odds(projections, odds_df)

    if joined.empty:
        return joined, pd.DataFrame()

    best_edges = best_over_edges(joined)
    return joined, best_edges