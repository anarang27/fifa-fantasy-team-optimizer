"""Canonical per-player-season stats schema.

All ingestion sources (FBref scrape, synthetic history) normalize into this flat
schema so the projection model has a single, stable contract to consume. Totals
are season totals; the projection layer derives per-90 rates from them.
"""

from __future__ import annotations

PLAYER_SEASON_COLUMNS = [
    "player_id",      # canonical id (FBref id or generated)
    "player",         # display name
    "country",        # national team / nation
    "position",       # GK / DEF / MID / FWD
    "season",         # e.g. "2022-2023" or "2022" (Qatar WC)
    "competition",    # e.g. "ENG-Premier League", "FIFA World Cup"
    # playing time
    "minutes",
    "nineties",       # minutes / 90
    "games",
    # universal actions (season totals)
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "pens_won",
    "pens_conceded",
    # position-specific raw counts
    "shots_on_target",   # forwards
    "key_passes",        # midfielders (chances created proxy)
    "tackles",           # midfielders
    "saves",             # goalkeepers
    "penalty_saves",     # goalkeepers
    "goals_against",     # goalkeepers / team
    "clean_sheets",      # goalkeepers / defenders
]

NUMERIC_COLUMNS = PLAYER_SEASON_COLUMNS[6:]
