# Supported MLB Props

## Fully Wired Today

- `pitcher_strikeouts`
  - training artifacts
  - daily projections
  - live odds join
  - picks / postable picks
  - historical line artifact support
  - grading and tracking

## Partially Wired

- `pitcher_walks`
  - configuration exists in [src/pitcher_bb/config.py](../src/pitcher_bb/config.py)
  - market key: `pitcher_walks`
  - target column: `walks`
  - training artifact workflow exists for pitcher-walk models
  - initial base features are defined from shared pitcher workload/context columns plus walk-specific rolling features
  - Needs specific pick policy to be implemented

`pitcher_walks` is not yet a full live-picks workflow. Training artifacts now have a dedicated module and artifact directory, but daily projections, live odds picks, and grading remain strikeout-specific.
