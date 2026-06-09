"""Round-to-round transfer planning and booster evaluation.

`plan_transfers` re-optimizes the squad for the next round given the current
squad and the free-transfer allocation, charging the -3 point hit for any extra
transfers (handled inside the squad ILP).

`plan_boosters` estimates the expected-points gain of each booster for a round.
Some boosters (Wildcard, 12th Man) are computed exactly from EP/prices; others
(Max Captain, Qualification, Mystery) depend on outcome distributions or
progression probabilities we may not have, so they are reported with the method
and any inputs that are still required.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy.optimize.squad import Player, SquadSolution, optimize_squad
from fantasy.rules import EXTRA_TRANSFER_PENALTY, Stage


@dataclass
class TransferPlan:
    solution: SquadSolution
    transfers_in: list[Player]
    transfers_out: list[Player]
    num_transfers: int
    free_transfers: int | None
    point_hit: int
    net_expected_points: float


def plan_transfers(
    current_squad: list[Player],
    pool: list[Player],
    stage: Stage,
    free_transfers: int | None,
    budget: float | None = None,
    force_exclude: set[str] = frozenset(),
) -> TransferPlan:
    """Recommend the best squad for the next round under the transfer allowance."""
    current_ids = {p.player_id for p in current_squad}
    solution = optimize_squad(
        pool,
        stage=stage,
        budget=budget,
        current_squad_ids=current_ids,
        free_transfers=free_transfers,
        force_exclude=force_exclude,
    )

    new_ids = {p.player_id for p in solution.squad}
    transfers_in = [p for p in solution.squad if p.player_id not in current_ids]
    transfers_out = [p for p in current_squad if p.player_id not in new_ids]
    num_transfers = len(transfers_in)

    if free_transfers is None:
        point_hit = 0
    else:
        point_hit = max(0, num_transfers - free_transfers) * EXTRA_TRANSFER_PENALTY

    return TransferPlan(
        solution=solution,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        num_transfers=num_transfers,
        free_transfers=free_transfers,
        point_hit=point_hit,
        net_expected_points=solution.expected_points - point_hit,
    )


@dataclass
class BoosterPlan:
    booster: str
    estimated_gain: float | None  # None == needs data we don't have yet
    detail: str


def plan_boosters(
    current_squad: list[Player],
    pool: list[Player],
    stage: Stage,
    free_transfers: int | None,
    progression_prob: dict[str, float] | None = None,
) -> list[BoosterPlan]:
    """Estimate each booster's expected-points gain for the round, best first."""
    base = plan_transfers(current_squad, pool, stage, free_transfers)
    base_net = base.net_expected_points
    plans: list[BoosterPlan] = []

    # Wildcard: unlimited transfers this round (no point hit).
    wild = plan_transfers(current_squad, pool, stage, free_transfers=None)
    plans.append(
        BoosterPlan(
            "wildcard",
            round(wild.net_expected_points - base_net, 2),
            f"Unlimited transfers -> {wild.num_transfers} moves, "
            f"net EP {wild.net_expected_points:.2f} vs base {base_net:.2f}.",
        )
    )

    # 12th Man: one extra scorer (any player not already in the squad), no budget/cap.
    squad_ids = {p.player_id for p in base.solution.squad}
    candidates = [p for p in pool if p.player_id not in squad_ids]
    if candidates:
        best = max(candidates, key=lambda p: p.ep)
        plans.append(
            BoosterPlan("12th_man", round(best.ep, 2), f"Add {best.name} (EP {best.ep:.2f}) as a non-sub, non-captain extra scorer.")
        )

    # Max Captain: doubles the highest-scoring starter. In expectation we already
    # captain the max-EP starter, so the *expected* gain over normal captaincy is
    # ~0; its real value is variance reduction. Needs a points distribution to
    # quantify, so we flag rather than fabricate a number.
    plans.append(
        BoosterPlan(
            "max_captain",
            None,
            "Guarantees the captaincy lands on the top scorer. Expected gain over "
            "normal captaincy is ~0; benefit is variance reduction (needs a points "
            "distribution to quantify).",
        )
    )

    # Qualification Booster: +2 per starter that progresses (R32+). Computable if
    # progression probabilities are supplied.
    if stage != Stage.GROUP:
        if progression_prob is not None:
            gain = 2 * sum(progression_prob.get(p.player_id, 0.0) for p in base.solution.starting_xi)
            detail = "Sum of 2 x progression probability over the starting XI."
        else:
            gain = None
            detail = "Provide progression_prob per player to estimate (+2 per starter that advances)."
        plans.append(BoosterPlan("qualification", round(gain, 2) if gain is not None else None, detail))

    plans.append(BoosterPlan("mystery", None, "Revealed when Round of 32 opens; cannot be modelled until known."))

    plans.sort(key=lambda b: (b.estimated_gain is not None, b.estimated_gain or 0), reverse=True)
    return plans
