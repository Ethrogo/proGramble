# Performance Reporting

The tracked performance ledger for the live MLB workflow is rebuilt from:

- `MLB/data/tracking/official_picks_history.csv`

The rebuild entrypoint is:

```powershell
cd MLB
$env:PYTHONPATH = "src"
python -c "from jobs.run_daily_card import persist_official_picks_profit_reports; persist_official_picks_profit_reports(allow_empty_replacement=True)"
```

## Published views

The repo publishes three summary artifacts from the same history ledger:

- `official_picks_profit_summary.json`: combined report with all summary views and segmented breakdowns
- `official_picks_profit_summary_all_time.json`: all tracked official picks to date
- `official_picks_profit_summary_current_regime.json`: current-regime performance for the active live workflow regime

The by-book companion CSV remains:

- `official_picks_profit_by_book.csv`

Its `summary_scope` column distinguishes `all_time` from `current_regime`.

## Current regime rule

`current_regime` is intentionally defined by start date:

- rule type: `start_date`
- start date: `2026-05-07`

This date-based rule is preferred over model-version or policy-version filtering because it stays reproducible from tracked history alone, including older manual seed rows that may not carry complete version metadata.

Any official pick with:

- `record_source == run_daily_card`, and
- `game_date >= 2026-05-07`

is treated as part of the current workflow regime when provenance needs to be inferred from the ledger.

## Why both views exist

- `all_time` is the long-term ledger and gives the broadest historical context.
- `current_regime` answers the more decision-relevant question: how the currently deployed live system is performing.

Reading only the all-time summary can hide meaningful changes after workflow, model, or policy updates. The current-regime view keeps that distinction explicit.
