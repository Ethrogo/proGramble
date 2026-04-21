# proGramble

proGramble is a sports modeling and betting workflow project focused on turning raw game and market data into daily, model-driven prop analysis.

The current implementation is centered on **MLB pitcher strikeout props**, but the repo is designed to grow into:
- additional MLB props
- additional sports
- more robust daily automation
- broader odds and market comparison workflows

## Current scope

Today, the main working project lives under `MLB/` and includes:
- pitcher strikeout modeling
- starter slate ingestion and validation
- sportsbook odds ingestion and normalization
- daily card generation jobs
- automated tests for jobs, odds, and starters

## Project goals

The long-term goal is to build a reusable workflow for sports props that can:

1. ingest tomorrow’s slate
2. collect probable starters / participants
3. build model-ready features
4. generate predictions
5. compare predictions to sportsbook lines
6. rank value spots
7. support repeatable daily execution

## Repository structure

```text
proGramble/
├── .github/
│   └── workflows/
├── MLB/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   │   ├── jobs/
│   │   ├── odds/
│   │   ├── pitcher_k/
│   │   └── starters/
│   ├── tests/
│   ├── pytest.ini
│   └── requirements.txt
├── LICENSE
└── README.md