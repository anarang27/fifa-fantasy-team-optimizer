"""Game rules and constants for FIFA World Cup 2026 Fantasy.

Single source of truth for budget, squad composition, country limits,
formations, transfers, and scoring constants. Every other module imports
these so the rules live in exactly one place.
"""

from __future__ import annotations

from enum import Enum


class Position(str, Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class Stage(str, Enum):
    GROUP = "group"
    R32 = "r32"
    R16 = "r16"
    QF = "qf"
    SF = "sf"
    FINAL = "final"


# --- Budget ---------------------------------------------------------------
BASE_BUDGET = 100.0  # $m, group stage
KNOCKOUT_BUDGET_BONUS = 5.0  # +$5m once Round of 32 opens


def budget_for_stage(stage: Stage) -> float:
    """Group stages use the base budget; knockouts get the +$5m bump."""
    if stage == Stage.GROUP:
        return BASE_BUDGET
    return BASE_BUDGET + KNOCKOUT_BUDGET_BONUS


# --- Squad composition ----------------------------------------------------
SQUAD_SIZE = 15
SQUAD_COMPOSITION: dict[Position, int] = {
    Position.GK: 2,
    Position.DEF: 5,
    Position.MID: 5,
    Position.FWD: 3,
}

# --- Starting XI / formation bounds --------------------------------------
# A valid starting XI has exactly 1 GK and 10 outfield players. Constraining
# the per-line counts to these ranges reproduces *exactly* the 7 legal
# formations (4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2), so no extra
# formation variables are needed in the optimizer.
XI_SIZE = 11
XI_GK = 1
XI_DEF_RANGE = (3, 5)
XI_MID_RANGE = (3, 5)
XI_FWD_RANGE = (1, 3)

VALID_FORMATIONS: set[tuple[int, int, int]] = {
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (3, 4, 3),
    (3, 5, 2),
    (5, 4, 1),
    (5, 3, 2),
}

# --- Country limits per stage --------------------------------------------
COUNTRY_LIMIT: dict[Stage, int] = {
    Stage.GROUP: 3,
    Stage.R32: 3,
    Stage.R16: 4,
    Stage.QF: 5,
    Stage.SF: 6,
    Stage.FINAL: 8,
}

# --- Transfers per stage --------------------------------------------------
# None == unlimited. Extra transfers beyond the allocation cost EXTRA_TRANSFER_PENALTY.
TRANSFER_ALLOCATION: dict[str, int | None] = {
    "pre_tournament": None,
    "matchday_2": 2,
    "matchday_3": 2,
    "r32": None,
    "r16": 4,
    "qf": 4,
    "sf": 5,
    "final": 6,
}
EXTRA_TRANSFER_PENALTY = 3  # points deducted per extra transfer

# --- Captain --------------------------------------------------------------
CAPTAIN_MULTIPLIER = 2  # captain scores double

# --- Scouting / scarcity bonus -------------------------------------------
SCOUTING_BONUS = 2
SCOUTING_POINTS_THRESHOLD = 4  # must score MORE than 4 pts
SCOUTING_OWNERSHIP_THRESHOLD = 0.05  # in fewer than 5% of teams
