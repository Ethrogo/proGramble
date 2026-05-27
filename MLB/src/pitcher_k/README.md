# Pitcher K Governance

This package now supports a conservative `champion` vs `challenger` review loop for `pitcher_k`.

Current operating model:

- `xgboost_champion` remains the live production model
- `ridge_challenger` runs in shadow mode on the same daily candidate universe
- shadow predictions are graded later from realized Statcast outcomes
- a forward comparison report is rebuilt from overlapping tracked rows

Tracked shadow artifacts are written under `data/tracking/`:

- `pitcher_k_shadow_predictions.csv`
- `pitcher_k_shadow_overlap.csv`
- `pitcher_k_shadow_summary.json`
- `pitcher_k_shadow_regression.png`
- `pitcher_k_shadow_workflow.png`

## Promotion Policy

The promotion policy is implemented in `pitcher_k/promotion_policy.py` and summarized in `pitcher_k_shadow_summary.json`.

Current guardrails:

- no automatic promotion
- manual approval is always required before a production swap
- minimum forward evidence is required before any review is considered
- tiny metric differences are treated as ties
- mixed signals default to `hold`
- provisional or recently promoted champions can be flagged as `rollback_candidate`

Default review thresholds:

- minimum overlapping forward days: `21`
- minimum overlapping workflow picks per model: `30`
- minimum disagreement-slice rows: `10`
- challenger must lead on at least `2` material metrics
- challenger must lead on at least `1` regression metric
- challenger must lead on at least `1` workflow metric
- challenger cannot have a material workflow regression and still be promoted

Primary metrics considered:

- `MAE`
- `RMSE`
- calibration gap
- `ROI` per pick
- profit units
- pick volume
- disagreement-slice `ROI`

Possible review outcomes:

- `promote`
- `hold`
- `insufficient_evidence`
- `rollback_candidate`

Interpretation:

- `promote` means the challenger has cleared the forward-review policy and is eligible for manual promotion review
- `hold` means keep the current champion in place
- `insufficient_evidence` means there is not enough overlapping forward data yet
- `rollback_candidate` means a provisional or recently promoted champion is underperforming enough to justify manual rollback review

This policy is intentionally stricter than a normal offline model bakeoff because the offline `XGBoost` vs linear differences have been small, and the project is still building out richer overlapping live evaluation.
