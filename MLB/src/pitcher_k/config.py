RAW_STATCAST_START = "2025-03-27"
RAW_STATCAST_END = "2025-09-30"
TRAIN_SPLIT_DATE = "2025-08-01"

BASE_FEATURES = [
    "pitches_last3",
    "pitches_last10",
    "whiff_per_pitch_last3",
    "avg_velo_last3",
    "avg_spin_last3",
    "k_per_pitch_last10",
    "k_rate_last10",
    "opp_strikeouts_per_game_last10",
    "opp_k_rate_last10",

]

TARGET_COL = "strikeouts"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "eval_metric": "mae",
}