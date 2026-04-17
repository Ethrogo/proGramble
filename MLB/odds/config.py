from __future__ import annotations

import os

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

ODDS_SPORT = "baseball_mlb"
ODDS_MARKET = "pitcher_strikeouts"
BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "williamhill_us"]

BOOK_DISPLAY_NAMES = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "williamhill_us": "Caesars",
}