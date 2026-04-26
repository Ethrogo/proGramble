# proGramble

proGramble is a sports modeling workflow repo focused on turning raw game data and live market data into model-driven daily prop cards.

Today the implemented system is centered on one production workflow under `MLB/`:

- train an MLB pitcher strikeout model from Statcast data
- pull today's probable starters from the MLB Stats API
- generate same-day pitcher strikeout projections
- fetch sportsbook player prop markets from The Odds API
- compare projections to market lines and rank edges
- export a smaller postable card

The codebase is structured so the same pattern can later be extended to other MLB props and other sports, but those extensions are not implemented yet.

## Current capabilities

The current MLB pipeline already includes:

- historical Statcast ingestion via `pybaseball`
- pitch-level preprocessing and pitcher-game feature engineering
- XGBoost training with saved model artifacts and metadata
- artifact promotion from `_staging/` to `latest/` with fallback to `previous/`
- starter-slate ingestion and normalization for today's MLB games
- odds ingestion, normalization, and bookmaker comparison for pitcher strikeout props
- policy-based pick classification into `official`, `lean`, and `pass`
- postable pick filtering with configurable limits
- contract validation for key intermediate dataframes
- automated tests for jobs, starters, pitcher-k features, odds, and Discord notifications
- a scheduled GitHub Actions workflow for daily training, card generation, artifact upload, and Discord notification

## Repository layout

```text
proGramble/
|-- .github/
|   `-- workflows/
|       `-- daily-mlb-card.yml
|-- MLB/
|   |-- data/
|   |   |-- raw/
|   |   |-- artifacts/
|   |   |   |-- _staging/
|   |   |   |-- latest/
|   |   |   `-- previous/
|   |   |-- inputs/
|   |   |   `-- starters/
|   |   `-- outputs/
|   |       |-- projections/
|   |       |-- edges/
|   |       `-- picks/
|   |-- notebooks/
|   |-- src/
|   |   |-- common/
|   |   |-- jobs/
|   |   |-- notifications/
|   |   |-- odds/
|   |   |-- pitcher_k/
|   |   `-- starters/
|   |-- tests/
|   |-- pytest.ini
|   `-- requirements.txt
|-- LICENSE
`-- README.md
```

## Architecture

### 1. Training artifacts

`MLB/src/jobs/build_training_artifacts.py` builds the model inputs and saved artifacts:

1. load raw Statcast data
2. add pitch-level outcome flags
3. aggregate pitcher-game level history
4. engineer rolling pitcher and opponent strikeout features
5. build a model-ready dataframe
6. time-split train and test sets
7. train an XGBoost regressor
8. save:
   - `model.ubj`
   - `pitcher_games.csv`
   - `model_df.csv`
   - `metadata.json`

Artifacts are written to `MLB/data/artifacts/_staging/`, validated, then promoted into `latest/`. If a previous `latest/` artifact set exists, it is copied into `previous/` first.

### 2. Daily card workflow

`MLB/src/jobs/run_daily_card.py` is the end-to-end daily execution path:

1. pull today's probable starters from the MLB Stats API
2. validate and normalize the starter slate
3. load the most recent saved model and pitcher-game history
4. resolve the active workflow spec for the prop
5. build today features for the current slate
6. generate pitcher strikeout projections
7. fetch market data from The Odds API
8. join projections to normalized odds rows
9. choose the best market per pitcher according to the configured pick policy
10. classify picks and export a smaller postable subset

Outputs are written to:

- `MLB/data/inputs/starters/today_starters.csv`
- `MLB/data/outputs/projections/today_projections.csv`
- `MLB/data/outputs/edges/today_joined_edges.csv`
- `MLB/data/outputs/picks/today_all_picks.csv`
- `MLB/data/outputs/picks/today_postable_picks.csv`

### 3. Contracts and validation

`MLB/src/common/contracts.py` defines reusable dataframe contracts for:

- starter slates
- historical pitcher-game tables
- joined projection/odds data
- final picks output

These checks enforce required columns, non-null expectations, uniqueness, and boolean-like fields so pipeline failures happen early and loudly.

### 3.5 Workflow spec

`MLB/src/common/workflows.py` defines the first lightweight workflow spec for MLB pitcher strikeouts. It describes:

- sport
- participant key
- market key
- feature builder
- predictor
- projection/odds join keys
- pick-ranking policy

The daily card job consumes that spec directly, which keeps the current behavior intact while giving future props a repeatable orchestration shape without introducing a large framework.

### 4. Odds and pick policy

The odds workflow in `MLB/src/odds/` currently supports the MLB `pitcher_strikeouts` market and bookmaker normalization for:

- DraftKings
- FanDuel
- BetMGM
- Caesars

Pick selection is policy-driven. The default policy:

- chooses the best over and under market per pitcher
- compares both directions against the model projection
- ranks picks by type and value score
- labels picks with confidence tiers
- caps the number of postable `official` and `lean` plays

## Local setup

From the repo root:

```powershell
cd MLB
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set environment variables as needed:

- `ODDS_API_KEY` for The Odds API
- `DISCORD_WEBHOOK_URL` for workflow notifications

The test configuration expects `PYTHONPATH=src`. In PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

## Running the workflow locally

Build or refresh training artifacts:

```powershell
cd MLB
$env:PYTHONPATH = "src"
python -m jobs.build_training_artifacts
```

Run the daily MLB card:

```powershell
cd MLB
$env:PYTHONPATH = "src"
$env:ODDS_API_KEY = "your_key_here"
python -m jobs.run_daily_card
```

## Testing

Run the full MLB test suite:

```powershell
cd MLB
$env:PYTHONPATH = "src"
pytest
```

The suite includes:

- unit tests for feature engineering, projections, odds normalization, value logic, and pick policy
- job-level tests for training artifact creation and daily card generation
- integration-marked tests for live API behavior where applicable

## Automation

`.github/workflows/daily-mlb-card.yml` runs the current production workflow on a schedule and on manual dispatch. It:

1. installs MLB dependencies
2. rebuilds training artifacts
3. runs the daily card job
4. sends a Discord status notification
5. uploads generated artifacts and daily outputs

## Current boundaries

What is implemented now:

- MLB pitcher strikeout modeling
- same-day probable starter ingestion
- same-day odds comparison and card generation
- reproducible saved artifacts for model reuse

What is not implemented yet:

- other MLB prop markets
- bet tracking / result grading
- multi-sport shared abstractions beyond the current folder structure
- fully packaged CLI or deployment layer beyond the GitHub Actions workflow
