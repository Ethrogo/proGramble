# MLB/tests/pitcher_k/test_feature_tomorrow.py

import pandas as pd
import pytest

from pitcher_k.feature_tomorrow import build_tomorrow_features


def _valid_slate_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2026-04-19",
                "game_pk": 123456,
                "pitcher": 111,
                "player_name": "Jacob deGrom",
                "team": "TEX",
                "opponent": "SEA",
                "home_team": "TEX",
                "away_team": "SEA",
                "is_home": 1,
                "p_throws": "R",
            }
        ]
    )


def _valid_pitcher_games_df() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "game_date": f"2026-04-{10 + i:02d}",
                "game_pk": 200000 + i,
                "pitcher": 111,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "strikeouts": 7 + (i % 2),
                "pitches": 95 + i,
                "batters_faced": 25 + i,
                "whiffs": 14 + i,
                "avg_velo": 97.0 + (i * 0.1),
                "avg_spin": 2450 + i,
            }
        )
    return pd.DataFrame(rows)


def _valid_team_context_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "opponent_team": "SEA",
                "opp_strikeouts_per_game_last10": 9.4,
                "opp_k_rate_last10": 0.255,
            }
        ]
    )


def test_build_tomorrow_features_raises_on_missing_required_slate_columns():
    slate_df = _valid_slate_df().drop(columns=["team"])
    pitcher_games = _valid_pitcher_games_df()

    with pytest.raises(ValueError, match="starters_df is missing required columns"):
        build_tomorrow_features(slate_df, pitcher_games)


def test_build_tomorrow_features_raises_on_missing_required_pitcher_games_columns():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df().drop(columns=["strikeouts"])

    with pytest.raises(ValueError, match="pitcher_games is missing required columns"):
        build_tomorrow_features(slate_df, pitcher_games)


def test_build_tomorrow_features_returns_empty_when_too_few_career_starts():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df().head(3)

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        min_career_starts=5,
    )

    assert isinstance(features, pd.DataFrame)
    assert features.empty
    assert features.attrs["skipped_pitchers"] == 1


def test_build_tomorrow_features_merges_team_context():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df()
    team_context = _valid_team_context_df()

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        team_context=team_context,
        min_career_starts=5,
    )

    assert len(features) == 1
    assert features.loc[0, "opp_strikeouts_per_game_last10"] == 9.4
    assert features.loc[0, "opp_k_rate_last10"] == 0.255


def test_build_tomorrow_features_creates_expected_output_columns():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df()
    team_context = _valid_team_context_df()

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        team_context=team_context,
        min_career_starts=5,
    )

    expected_cols = {
        "game_date",
        "game_pk",
        "pitcher",
        "player_name",
        "team",
        "opponent",
        "is_home",
        "pitches_last3",
        "pitches_last10",
        "whiff_per_pitch_last3",
        "avg_velo_last3",
        "avg_spin_last3",
        "k_per_pitch_last10",
        "k_rate_last10",
        "opp_strikeouts_per_game_last10",
        "opp_k_rate_last10",
        "strikeouts_stddev_last10",
        "strikeouts_p25_last10",
        "strikeouts_p75_last10",
    }

    assert expected_cols.issubset(features.columns)
    assert features.attrs["skipped_pitchers"] == 0


def test_build_tomorrow_features_computes_recent_strikeout_uncertainty():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df()
    team_context = _valid_team_context_df()

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        team_context=team_context,
        min_career_starts=5,
    )

    recent_strikeouts = pitcher_games["strikeouts"]

    assert features.loc[0, "strikeouts_stddev_last10"] == pytest.approx(
        recent_strikeouts.std(ddof=0)
    )
    assert features.loc[0, "strikeouts_p25_last10"] == pytest.approx(
        recent_strikeouts.quantile(0.25)
    )
    assert features.loc[0, "strikeouts_p75_last10"] == pytest.approx(
        recent_strikeouts.quantile(0.75)
    )


def test_build_tomorrow_features_falls_back_to_name_match_when_pitcher_id_has_no_history():
    slate_df = _valid_slate_df()
    slate_df.loc[0, "pitcher"] = 999999

    pitcher_games = _valid_pitcher_games_df()
    team_context = _valid_team_context_df()

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        team_context=team_context,
        min_career_starts=5,
    )

    assert len(features) == 1
    assert features.loc[0, "player_name"] == "Jacob deGrom"


def test_build_tomorrow_features_uses_only_starter_like_history():
    slate_df = _valid_slate_df()
    pitcher_games = _valid_pitcher_games_df()
    team_context = _valid_team_context_df()

    relief_rows = pd.DataFrame(
        [
            {
                "game_date": "2026-04-16",
                "game_pk": 300001,
                "pitcher": 111,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "strikeouts": 1,
                "pitches": 18,
                "batters_faced": 5,
                "whiffs": 2,
                "avg_velo": 96.2,
                "avg_spin": 2440,
            },
            {
                "game_date": "2026-04-17",
                "game_pk": 300002,
                "pitcher": 111,
                "player_name": "Jacob deGrom",
                "pitching_team": "TEX",
                "opponent_team": "SEA",
                "strikeouts": 2,
                "pitches": 22,
                "batters_faced": 6,
                "whiffs": 3,
                "avg_velo": 96.4,
                "avg_spin": 2442,
            },
        ]
    )
    pitcher_games = pd.concat([pitcher_games, relief_rows], ignore_index=True)

    features = build_tomorrow_features(
        slate_df,
        pitcher_games,
        team_context=team_context,
        min_career_starts=5,
    )

    starter_only = _valid_pitcher_games_df()
    expected_last3 = starter_only.tail(3)
    expected_last10 = starter_only.tail(6)

    assert len(features) == 1
    assert features.loc[0, "pitches_last3"] == pytest.approx(expected_last3["pitches"].mean())
    assert features.loc[0, "pitches_last10"] == pytest.approx(expected_last10["pitches"].mean())
    assert features.loc[0, "k_rate_last10"] == pytest.approx(
        expected_last10["strikeouts"].sum() / expected_last10["batters_faced"].sum()
    )
