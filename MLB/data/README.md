# MLB Data Boundaries

This directory intentionally separates committed reference inputs from runtime outputs.

- `raw/` contains checked-in example source data that is useful for notebooks and repeatable local exploration.
- `artifacts/latest/`, `artifacts/previous/`, and `artifacts/_staging/` are operational model artifacts produced by jobs and promoted between runs.
- `inputs/starters/` contains runtime starter slates saved by the daily card job.
- `outputs/` contains runtime projections, joined edges, and pick exports from daily runs.

Only long-lived example/reference data under `raw/` should be committed by default. Operational artifacts and daily run outputs should be recreated by workflows or local jobs and shared via workflow artifacts when needed.
