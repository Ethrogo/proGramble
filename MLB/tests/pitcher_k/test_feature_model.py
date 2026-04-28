import pandas as pd

from pitcher_k.config import BASE_FEATURES
from pitcher_k.feature_model import build_model_df, get_feature_columns


def test_get_feature_columns_reads_from_config():
    assert get_feature_columns() == BASE_FEATURES


def test_build_model_df_uses_configured_feature_columns():
    pitcher_games = pd.DataFrame(
        [
            {
                "game_date": "2026-04-10",
                "game_pk": 1001,
                "pitcher": 111,
                "player_name": "Jacob deGrom",
                "strikeouts": 8,
                "pitches_last3": 95.0,
                "pitches_last10": 96.0,
                "whiff_per_pitch_last3": 0.14,
                "avg_velo_last3": 97.5,
                "avg_spin_last3": 2455.0,
                "k_per_pitch_last10": 0.08,
                "k_rate_last10": 0.30,
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
                "strikeouts_stddev_last10": 1.2,
                "strikeouts_p25_last10": 6.0,
                "strikeouts_p75_last10": 9.0,
            }
        ]
    )

    model_df = build_model_df(pitcher_games)

    expected_cols = [
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "strikeouts",
    ] + BASE_FEATURES + [
        "strikeouts_stddev_last10",
        "strikeouts_p25_last10",
        "strikeouts_p75_last10",
    ]
    assert list(model_df.columns) == expected_cols
