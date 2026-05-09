RAW_STATCAST_START = "2023-03-27"
RAW_STATCAST_END = "2025-09-30"
TRAIN_SPLIT_DATE = "2025-08-01"
PITCHER_BB_PROP_MARKET = "pitcher_walks"
STARTER_LIKE_MIN_PITCHES = 40
STARTER_LIKE_MIN_BATTERS_FACED = 12

# Build on the shared pitcher-game pipeline using walk-specific rolling and
# opponent-context features plus a small workload baseline.
BASE_FEATURES = [
    "pitches_last3",
    "pitches_last10",
    "batters_faced_last3",
    "batters_faced_last10",
    "walks_last3",
    "walks_last10",
    "avg_velo_last3",
    "avg_spin_last3",
    "bb_per_pitch_last10",
    "bb_rate_last10",
    "opp_walks_per_game_last10",
    "opp_bb_rate_last10",
]

TARGET_COL = "walks"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "eval_metric": "mae",
}
