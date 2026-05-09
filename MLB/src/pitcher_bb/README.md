# Pitcher Walks Scaffold

This package defines the initial market and model configuration scaffold for MLB pitcher walks props.

Current support level:

- `pitcher_walks` market key is reserved in code via `PITCHER_BB_PROP_MARKET`.
- `TARGET_COL` is defined as `walks`.
- `BASE_FEATURES` is intentionally conservative and limited to shared pitcher workload/context columns that already exist in the current feature pipeline.

Not wired yet:

- walk-specific feature engineering
- pitcher-walks training workflow and artifacts
- pitcher-walks daily-card workflow
- grading logic for pitcher-walk outcomes

This keeps the module truthful to the current repo shape while giving the project a stable place to grow the pitcher-walk workflow next.
