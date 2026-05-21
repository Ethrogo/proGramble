RAW_STATCAST_START = "2023-03-27"
RAW_STATCAST_END = "2025-09-30"
TRAIN_SPLIT_DATE = "2025-08-01"
PITCHER_K_PROP_MARKET = "pitcher_strikeouts"
STARTER_LIKE_MIN_PITCHES = 40
STARTER_LIKE_MIN_BATTERS_FACED = 12

BASE_FEATURES = [
    "pitches_last10",
    "whiff_per_pitch_last3",
    "pitches_trend_last3_vs_last10",
    "avg_velo_last3",
    "avg_spin_last3",
    "k_rate_last10",
    "pitches_per_batter_last10",
    "opp_strikeouts_per_game_last10",
]

TARGET_COL = "strikeouts"
TARGET_FORMULATION = "single_stage_direct_count_regression"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 8,
    "reg_lambda": 5.0,
    "reg_alpha": 0.5,
    "seed": 42,
    "eval_metric": "mae",
}

XGB_NUM_BOOST_ROUND = 200
XGB_EARLY_STOPPING_ROUNDS = 25
TRAIN_VALIDATION_FRACTION = 0.15
