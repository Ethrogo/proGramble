#MLB/src/odds/config.py
from __future__ import annotations

import os

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

ODDS_SPORT = "baseball_mlb"
EVENT_DISCOVERY_MARKET = "h2h"


BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "williamhill_us"]

BOOKMAKER_NOTES = {
    # The Odds API documents this key as Caesars in the US region and notes that
    # it is only available on paid subscriptions.
    "williamhill_us": "Caesars (paid subscriptions only on The Odds API)",
}

BOOK_DISPLAY_NAMES = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "williamhill_us": "Caesars",
}
