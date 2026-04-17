# src/pitcher_k/preprocessing.py

import pandas as pd


def add_outcome_flags(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Add strikeout and whiff indicator columns to pitch-level Statcast data.
    """
    sc = sc.copy()

    sc["is_k"] = sc["events"].isin([
        "strikeout",
        "strikeout_double_play"
    ]).astype(int)

    sc["is_whiff"] = sc["description"].isin([
        "swinging_strike",
        "swinging_strike_blocked"
    ]).astype(int)

    return sc