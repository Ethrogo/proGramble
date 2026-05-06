# MLB Data Boundaries

This directory intentionally separates committed reference inputs from runtime outputs.

- `raw/` contains source-shaped inputs used for local exploration and job hydration.
- `raw/historical_lines/` is a local-only import directory for externally hydrated historical market snapshots.
- `artifacts/latest/`, `artifacts/previous/`, and `artifacts/_staging/` are operational model artifacts produced by jobs and promoted between runs.
- `inputs/starters/` contains runtime starter slates saved by the daily card job.
- `outputs/` contains runtime projections, joined edges, and pick exports from daily runs.

## Historical Lines Policy

`raw/historical_lines/` is the landing zone for native historical line snapshot CSVs that contributors populate locally from an external source. Those files are intentionally ignored by Git because they can be large, vendor-derived, or both.

What belongs in `raw/historical_lines/`:

- Raw CSV snapshots shaped like the native historical lines loader expects.
- Local or externally hydrated files used to build the curated `artifacts/latest/historical_lines.csv` artifact.
- Temporary developer data needed to reproduce or inspect the historical lines workflow locally.

What should never be committed:

- Bulk historical sportsbook datasets.
- Vendor-exported raw market archives.
- Developer-specific local hydrated inputs under `raw/historical_lines/`.

How to populate locally:

1. Place externally hydrated CSV snapshot files in a separate local folder.
2. Run `python src/jobs/populate_historical_lines_raw.py <path-to-file-or-directory>` from `MLB/`.
3. Build the curated artifact with `python src/jobs/build_historical_lines_artifact.py`.

The import job copies validated CSVs into `raw/historical_lines/` while preserving subdirectories for directory imports.

## Test And CI Fixtures

CI and unit tests do not depend on any developer-local files under `raw/historical_lines/`. Deterministic fixture inputs live under `tests/fixtures/historical_lines/`, and artifact-oriented tests copy those fixtures into explicit temporary directories before exercising the loader or jobs.

Only small, curated fixture data and reproducible artifacts should be committed by default. Operational artifacts and daily run outputs should be recreated by workflows or local jobs and shared via workflow artifacts when needed.
