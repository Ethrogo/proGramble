# MLB/src/odds/value.py

from __future__ import annotations

import math


def american_to_implied_probability(price: int | float) -> float:
    """
    Convert American odds to implied probability.

    Examples:
    -120 -> 0.5455
    +100 -> 0.5000
    +150 -> 0.4000
    """
    price = float(price)

    if price < 0:
        return abs(price) / (abs(price) + 100)

    return 100 / (price + 100)


def american_to_profit_per_unit(price: int | float) -> float:
    """
    Convert American odds to profit per 1.0 unit risked.

    Examples:
    -120 -> 0.8333
    +100 -> 1.0000
    +150 -> 1.5000
    """
    price = float(price)

    if price < 0:
        return 100 / abs(price)

    return price / 100


def normal_cdf(value: float, *, mean: float = 0.0, std_dev: float = 1.0) -> float:
    if std_dev <= 0:
        return 1.0 if value >= mean else 0.0

    z_score = (value - mean) / (std_dev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z_score))
