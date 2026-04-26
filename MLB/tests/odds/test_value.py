import pytest

from odds.value import american_to_implied_probability


def test_american_to_implied_probability_negative_price():
    result = american_to_implied_probability(-120)

    assert result == pytest.approx(120 / 220)


def test_american_to_implied_probability_even_money():
    result = american_to_implied_probability(100)

    assert result == pytest.approx(0.5)


def test_american_to_implied_probability_positive_price():
    result = american_to_implied_probability(150)

    assert result == pytest.approx(100 / 250)