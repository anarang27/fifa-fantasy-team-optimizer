"""Fantasy scoring engine.

Pure functions that map a single player's raw match statistics to fantasy
points, following the FIFA World Cup 2026 Fantasy rules exactly. This module is
the single source of truth for the scoring tables; the projection model and the
backtests both call into it.

Captain doubling, auto-subs, and the scouting bonus are deliberately NOT handled
here -- this returns the raw points a player earns from their own performance.
Captaincy/ownership effects are applied at the squad level by the optimizer and
the live scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy.rules import Position

CLEAN_SHEET_MIN_MINUTES = 60
LONG_APPEARANCE_MINUTES = 60

# Points awarded for a goal, by the scorer's position.
GOAL_POINTS: dict[Position, int] = {
    Position.GK: 9,
    Position.DEF: 7,
    Position.MID: 6,
    Position.FWD: 5,
}

# Clean-sheet points, by position. Forwards get nothing.
CLEAN_SHEET_POINTS: dict[Position, int] = {
    Position.GK: 5,
    Position.DEF: 5,
    Position.MID: 1,
    Position.FWD: 0,
}


@dataclass
class MatchStats:
    """A single player's raw stats for one match.

    `team_goals_conceded` is the number of goals the player's team conceded,
    used to derive clean sheets and the GK/DEF concede penalty. A clean sheet is
    derived (team conceded 0 AND player played 60+ minutes), so callers do not
    pass it directly.
    """

    position: Position
    minutes: int = 0

    # Universal actions
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    penalties_won: int = 0
    penalties_conceded: int = 0
    direct_free_kick_goals: int = 0  # subset of `goals`, adds a +1 bonus each

    # Team defensive context
    team_goals_conceded: int = 0

    # Goalkeeper-specific
    saves: int = 0
    penalty_saves: int = 0  # excludes shootouts

    # Midfielder-specific
    tackles: int = 0
    chances_created: int = 0

    # Forward-specific
    shots_on_target: int = 0


def _appearance_points(minutes: int) -> int:
    if minutes <= 0:
        return 0
    if minutes >= LONG_APPEARANCE_MINUTES:
        return 2  # +1 for appearing, +1 for 60+ minutes
    return 1


def _has_clean_sheet(s: MatchStats) -> bool:
    return s.team_goals_conceded == 0 and s.minutes >= CLEAN_SHEET_MIN_MINUTES


def _universal_points(s: MatchStats) -> int:
    pts = _appearance_points(s.minutes)
    pts += s.goals * GOAL_POINTS[s.position]
    pts += s.assists * 3
    pts -= s.yellow_cards * 1
    pts -= s.red_cards * 2
    pts -= s.own_goals * 2
    pts += s.penalties_won * 2
    pts -= s.penalties_conceded * 1
    pts += s.direct_free_kick_goals  # +1 bonus per direct free-kick goal
    return pts


def _concede_penalty(s: MatchStats) -> int:
    """First goal conceded is 0, each additional goal is -1 (GK and DEF)."""
    return -max(0, s.team_goals_conceded - 1)


def _gk_points(s: MatchStats) -> int:
    pts = 0
    if _has_clean_sheet(s):
        pts += CLEAN_SHEET_POINTS[Position.GK]
    pts += _concede_penalty(s)
    pts += s.penalty_saves * 3
    pts += s.saves // 3  # +1 for every 3 saves
    return pts


def _def_points(s: MatchStats) -> int:
    pts = 0
    if _has_clean_sheet(s):
        pts += CLEAN_SHEET_POINTS[Position.DEF]
    pts += _concede_penalty(s)
    return pts


def _mid_points(s: MatchStats) -> int:
    pts = 0
    if _has_clean_sheet(s):
        pts += CLEAN_SHEET_POINTS[Position.MID]
    pts += s.tackles // 3  # +1 for every 3 tackles
    pts += s.chances_created // 2  # +1 for every 2 chances created
    return pts


def _fwd_points(s: MatchStats) -> int:
    return s.shots_on_target // 2  # +1 for every 2 shots on target


_POSITION_SCORERS = {
    Position.GK: _gk_points,
    Position.DEF: _def_points,
    Position.MID: _mid_points,
    Position.FWD: _fwd_points,
}


def score_match(s: MatchStats) -> int:
    """Total fantasy points a player earns from one match (excludes captaincy)."""
    return _universal_points(s) + _POSITION_SCORERS[s.position](s)
