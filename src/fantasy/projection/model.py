"""Expected-points projection model.

Transparent, defensible baseline: aggregate each player's per-90 rates from
history (weighting recent seasons and international competition more), estimate
expected minutes, then convert rates into an expected fantasy-points-per-match
value using the SAME scoring constants as the scoring engine.

Threshold rewards (every 3 saves, every 2 shots on target, etc.) are handled in
expectation by dividing the expected count by the threshold -- the correct
linear expectation for a per-unit reward. Clean sheets and the goalkeeper/
defender concede penalty use a Poisson model on team goals-against.

This is intentionally interpretable. The component expectations are exactly the
features one would feed a LightGBM/XGBoost upgrade later, so swapping in a
learned model is localized to this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fantasy.optimize.squad import Player
from fantasy.projection.minutes import expected_minutes
from fantasy.rules import Position
from fantasy.scoring.engine import CLEAN_SHEET_POINTS, GOAL_POINTS

RATE_COLUMNS = [
    "goals", "assists", "yellow_cards", "red_cards", "own_goals",
    "pens_won", "pens_conceded", "shots_on_target", "key_passes",
    "tackles", "saves", "penalty_saves",
]


@dataclass
class ProjectionConfig:
    # Per-season weight (more recent = higher). Defaults cover the current
    # single-season scope plus older seasons for multi-season backtests.
    season_weights: dict[str, float] = field(default_factory=lambda: {
        "2526": 1.0, "2425": 1.0, "2324": 0.8, "2223": 0.6,
        "2024": 1.0, "2022": 1.0,
    })
    # International football is more representative of the World Cup than club.
    world_cup_weight: float = 2.0
    default_season_weight: float = 0.5
    # Team goals-against per match for clean-sheet / concede modelling.
    default_team_ga: float = 1.2
    team_ga_override: dict[str, float] = field(default_factory=dict)


def _is_international(competition: str) -> bool:
    c = competition.lower()
    return (
        competition.upper().startswith("INT-")
        or "world cup" in c
        or "european championship" in c
        or "nations league" in c
        or "copa america" in c
        or "international" in c
    )


def _row_weight(row, cfg: ProjectionConfig) -> float:
    w = cfg.season_weights.get(str(row["season"]), cfg.default_season_weight)
    if _is_international(str(row["competition"])):
        w *= cfg.world_cup_weight
    return w


def aggregate_rates(history: pd.DataFrame, cfg: ProjectionConfig | None = None) -> pd.DataFrame:
    """Collapse multi-season history into one weighted per-90 rate row per player."""
    cfg = cfg or ProjectionConfig()
    h = history.copy()
    h["_w"] = h.apply(lambda r: _row_weight(r, cfg), axis=1)

    # Identity: pick the most-played appearance of each player as the label.
    # Player identity across seasons is keyed on (player, country).
    h["_key"] = h["player"].astype(str) + "|" + h["country"].astype(str)

    rows = []
    for key, grp in h.groupby("_key"):
        w = grp["_w"].to_numpy()
        wsum_nineties = float((w * grp["nineties"]).sum())
        wsum_games = float((w * grp["games"]).sum())
        wsum_minutes = float((w * grp["minutes"]).sum())
        label = grp.sort_values("minutes", ascending=False).iloc[0]

        rec = {
            "player_id": label["player_id"],
            "player": label["player"],
            "country": label["country"],
            "position": label["position"],
            "minutes": wsum_minutes,
            "games": wsum_games,
            "nineties": wsum_nineties,
        }
        denom = wsum_nineties if wsum_nineties > 0 else np.nan
        for col in RATE_COLUMNS:
            rec[f"{col}_p90"] = float((w * grp[col]).sum()) / denom if denom else 0.0
        # Team goals-against per game (for keepers/defenders this is meaningful).
        rec["ga_per_game"] = (
            float((w * grp["goals_against"]).sum()) / wsum_games if wsum_games > 0 else 0.0
        )
        rows.append(rec)

    agg = pd.DataFrame(rows).fillna(0.0)
    return expected_minutes(agg)


def _team_ga(country: str, hist_ga: float, cfg: ProjectionConfig) -> float:
    if country in cfg.team_ga_override:
        return cfg.team_ga_override[country]
    return hist_ga if hist_ga > 0 else cfg.default_team_ga


def project_points(history: pd.DataFrame, cfg: ProjectionConfig | None = None) -> pd.DataFrame:
    """Return per-player expected fantasy points for one match (`ep` column)."""
    cfg = cfg or ProjectionConfig()
    agg = aggregate_rates(history, cfg)

    p_play = agg["p_play"].to_numpy()
    exp_min = agg["exp_minutes"].to_numpy()
    match_frac = (exp_min / 90.0) * p_play  # expected playing fraction of a match

    def exp_count(col):
        return agg[f"{col}_p90"].to_numpy() * match_frac

    # Appearance: +1 to appear, +1 more for 60+.
    long_app = (exp_min >= 60).astype(float)
    appearance = p_play * (1.0 + long_app)
    # Smooth probability of reaching 60' for clean-sheet eligibility.
    p60 = p_play * np.clip((exp_min - 30.0) / 30.0, 0.0, 1.0)

    lam = np.array([
        _team_ga(c, g, cfg) for c, g in zip(agg["country"], agg["ga_per_game"])
    ])
    p_clean = np.exp(-lam)
    # E[max(0, GA-1)] for a Poisson(lam): lam - (1 - e^-lam).
    concede_pen = -(lam - (1.0 - np.exp(-lam))) * p_play

    pos = agg["position"].to_numpy()
    is_gk = pos == "GK"
    is_def = pos == "DEF"
    is_mid = pos == "MID"
    is_fwd = pos == "FWD"

    goal_pts = np.select(
        [is_gk, is_def, is_mid, is_fwd],
        [GOAL_POINTS[Position.GK], GOAL_POINTS[Position.DEF],
         GOAL_POINTS[Position.MID], GOAL_POINTS[Position.FWD]],
        default=GOAL_POINTS[Position.MID],
    )

    ep = appearance.copy()
    ep += exp_count("goals") * goal_pts
    ep += exp_count("assists") * 3
    ep -= exp_count("yellow_cards") * 1
    ep -= exp_count("red_cards") * 2
    ep -= exp_count("own_goals") * 2
    ep += exp_count("pens_won") * 2
    ep -= exp_count("pens_conceded") * 1

    # Goalkeeper extras
    ep += is_gk * (p60 * p_clean * CLEAN_SHEET_POINTS[Position.GK]
                   + concede_pen
                   + exp_count("saves") / 3.0
                   + exp_count("penalty_saves") * 3.0)
    # Defender extras
    ep += is_def * (p60 * p_clean * CLEAN_SHEET_POINTS[Position.DEF] + concede_pen)
    # Midfielder extras
    ep += is_mid * (p60 * p_clean * CLEAN_SHEET_POINTS[Position.MID]
                    + exp_count("tackles") / 3.0
                    + exp_count("key_passes") / 2.0)
    # Forward extras
    ep += is_fwd * (exp_count("shots_on_target") / 2.0)

    agg["ep"] = np.round(ep, 3)
    return agg


def players_from_projection(
    projection: pd.DataFrame,
    matched_prices: pd.DataFrame,
) -> list[Player]:
    """Join projected EP to the matched price list and build optimizer Players.

    `matched_prices` must contain: player_id, name, country, position, price,
    matched_id (from the entity matcher). Players without a projection get EP 0.
    """
    ep_by_id = dict(zip(projection["player_id"], projection["ep"]))
    players: list[Player] = []
    for _, row in matched_prices.iterrows():
        ep = float(ep_by_id.get(row.get("matched_id"), 0.0))
        players.append(
            Player(
                player_id=str(row["player_id"]),
                name=str(row["name"]),
                country=str(row["country"]),
                position=Position(row["position"]),
                price=float(row["price"]),
                ep=ep,
            )
        )
    return players
