from __future__ import annotations

import pandas as pd
from .odds_api import fetch_all_pitcher_strikeout_props
from .normalize import odds_json_to_dataframe
from .compare import join_projections_to_odds, best_over_edges


def run_edge_pipeline(projections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_events = fetch_all_pitcher_strikeout_props()
    odds_df = odds_json_to_dataframe(raw_events)

    joined = join_projections_to_odds(projections, odds_df)
    best_edges = best_over_edges(joined)

    return joined, best_edges