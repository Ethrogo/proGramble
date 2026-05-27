# Pitcher Walks

This package supports the current MLB pitcher walks workflow end to end.

Currently wired:

- `pitcher_walks` market key via `PITCHER_BB_PROP_MARKET`
- walk-specific feature engineering and tomorrow-feature generation
- XGBoost training artifacts with chronological validation-based early stopping
- reproducible holdout evaluation metadata, interval calibration, and historical-lines workflow backtests
- daily-card prediction, odds joining, pick creation, and outcome grading

Still missing relative to the more mature `pitcher_k` governance path:

- shadow challenger tracking and forward champion-vs-challenger reporting
- promotion policy and rollback guardrails
- a walk-specific pick-ranking policy instead of the shared strikeout policy

The current design keeps the model operational while leaving space for the next round of walk-specific evaluation and governance work.
