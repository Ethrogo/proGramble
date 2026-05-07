# Canonical Identity Standard

## Participant Identity

Workflow-facing dataframes should carry these canonical participant columns:

- `participant_id`: canonical participant identifier when available. Today MLB uses the native MLBAM pitcher id encoded as `mlbam_player:<id>`.
- `participant_source_id`: source-native participant id for the row.
- `participant_source_id_type`: the namespace for `participant_source_id`, such as `mlbam_player`.
- `participant_source_key`: stable source-qualified key, `<participant_source_id_type>:<participant_source_id>`.
- `participant_name`: display name to show to users.
- `participant_name_norm`: normalized fallback name for unresolved joins.
- `participant_join_key`: canonical join key used across workflows.

Join precedence is:

1. `participant_id`
2. `participant_source_key`
3. `participant_name_norm`

`participant_name_norm` remains a fallback, not the primary identity layer.

## Market Identity

Rows that represent an offered or graded market should carry:

- `sport`
- `market_key`
- `market_family`
- `side`
- `side_norm`
- `line`
- `bookmaker`
- `bookmaker_key`
- `event_id` when available
- `market_selection_key`: `sport|market_key|participant_join_key|side_norm|line`
- `market_offer_key`: `market_selection_key|bookmaker_key`

`market_selection_key` identifies the participant-market-side-line combination independent of book. `market_offer_key` identifies the sportsbook-specific offer.

## Join Rules

- Projections to live odds: join on `participant_join_key`, and include `sport` and `market_key` when both sides provide them.
- Projections to historical lines: join on `participant_join_key` plus `game_date`, and include `sport` and `market_key` when both sides provide them.
- Picks: preserve participant and market identity columns from the joined rows.
- Tracking / grading rows: persist the participant columns, market columns, and a `pick_key` built from `game_date|market_offer_key` when available, with legacy name-based fallback only when canonical offer identity is absent.

## Current MLB Notes

- MLB pitcher strikeout projections now derive canonical participant identity from `pitcher` / MLBAM ids when present.
- Odds rows still usually lack participant ids, so they join through `participant_name_norm` today, but through the shared precedence logic instead of workflow-specific normalization code.
- The same standard is intended to support future props, books, sports, and graded-result workflows without reintroducing ad hoc name matching.
