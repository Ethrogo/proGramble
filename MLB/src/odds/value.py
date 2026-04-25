# MLB/src/odds/value.py

from __future__ import annotations


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