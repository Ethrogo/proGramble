from pitcher_bb.config import BASE_FEATURES, PITCHER_BB_PROP_MARKET, TARGET_COL


def test_pitcher_bb_config_defines_expected_market_and_target():
    assert PITCHER_BB_PROP_MARKET == "pitcher_walks"
    assert TARGET_COL == "walks"


def test_pitcher_bb_config_starts_with_non_empty_base_features():
    assert BASE_FEATURES
    assert "pitches_last3" in BASE_FEATURES
    assert "batters_faced_last10" in BASE_FEATURES
