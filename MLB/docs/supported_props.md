# Supported MLB Props

## Fully Wired Today

- `pitcher_strikeouts`
  - training artifacts
  - daily projections
  - live odds join
  - picks / postable picks
  - historical line artifact support
  - grading and tracking

## Scaffolded For Future Workflow Expansion

- `pitcher_walks`
  - configuration scaffold exists in [src/pitcher_bb/config.py](../src/pitcher_bb/config.py)
  - market key: `pitcher_walks`
  - target column: `walks`
  - initial base features are defined from shared pitcher workload/context columns

`pitcher_walks` is not yet a full workflow. The current training job, prediction workflow, and grading path remain strikeout-specific, so adding only the config/package is the correct implementation at this stage.
