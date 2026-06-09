"""Live recompute loop.

After each matchday the history table is refreshed (re-scraped from FBref), EP is
re-projected, and the optimizer recommends transfers for the next round under the
stage's transfer allowance (with the -3 hit for extra moves) plus a booster
recommendation. This is the function a scheduler calls once per round.
"""

from __future__ import annotations

import pandas as pd

from fantasy.optimize.transfers import BoosterPlan, TransferPlan, plan_boosters, plan_transfers
from fantasy.pipeline.run import build_players
from fantasy.projection.model import ProjectionConfig
from fantasy.rules import Stage


def recommend_update(
    current_squad_ids: set[str],
    price_df: pd.DataFrame,
    history: pd.DataFrame,
    stage: Stage,
    free_transfers: int | None,
    cfg: ProjectionConfig | None = None,
    overrides: dict[str, str] | None = None,
    progression_prob: dict[str, float] | None = None,
) -> dict:
    """Recommend transfers + booster for the next round given refreshed history."""
    players, report = build_players(price_df, history, cfg, overrides)
    by_id = {p.player_id: p for p in players}

    missing = current_squad_ids - by_id.keys()
    if missing:
        raise ValueError(f"Current squad ids not found in player pool: {sorted(missing)}")
    current = [by_id[i] for i in current_squad_ids]

    transfer_plan: TransferPlan = plan_transfers(current, players, stage, free_transfers)
    boosters: list[BoosterPlan] = plan_boosters(
        current, players, stage, free_transfers, progression_prob
    )
    return {
        "match_report": report,
        "transfer_plan": transfer_plan,
        "boosters": boosters,
        "players": players,
    }


def describe_update(update: dict) -> str:
    """Human-readable summary of a live update recommendation."""
    plan: TransferPlan = update["transfer_plan"]
    lines = [
        f"Transfers: {plan.num_transfers} "
        f"(free: {plan.free_transfers if plan.free_transfers is not None else 'unlimited'}, "
        f"hit: -{plan.point_hit})",
    ]
    for out_p, in_p in zip(plan.transfers_out, plan.transfers_in):
        lines.append(f"  OUT {out_p.name:<20} -> IN {in_p.name:<20} (EP {in_p.ep:.2f})")
    lines.append(f"Net expected points: {plan.net_expected_points:.2f}")
    lines.append("Booster ranking:")
    for b in update["boosters"]:
        gain = "n/a" if b.estimated_gain is None else f"{b.estimated_gain:+.2f}"
        lines.append(f"  {b.booster:<13} gain {gain:>6}  {b.detail}")
    return "\n".join(lines)
