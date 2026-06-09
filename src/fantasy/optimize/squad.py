"""Squad selection via Integer Linear Programming.

Given a pool of players with prices and expected points (EP), pick the optimal
15-man squad, the starting XI, and the captain that maximize expected points
subject to budget, squad composition, per-country, and formation rules.

This is an exact optimization (PuLP + CBC), not a heuristic: the returned squad
is provably optimal for the given EP/prices/constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from fantasy.rules import (
    CAPTAIN_MULTIPLIER,
    COUNTRY_LIMIT,
    EXTRA_TRANSFER_PENALTY,
    SQUAD_COMPOSITION,
    XI_DEF_RANGE,
    XI_FWD_RANGE,
    XI_GK,
    XI_MID_RANGE,
    XI_SIZE,
    Position,
    Stage,
    budget_for_stage,
)

# Weight on bench EP in the objective. Small and positive so the optimizer
# prefers a stronger bench only as a tie-breaker, never at the expense of the XI.
BENCH_WEIGHT = 0.1


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    country: str
    position: Position
    price: float
    ep: float  # expected fantasy points for the round being optimized


@dataclass
class SquadSolution:
    squad: list[Player]
    starting_xi: list[Player]
    bench: list[Player]
    captain: Player
    vice_captain: Player
    formation: tuple[int, int, int]
    total_cost: float
    expected_points: float  # XI + captain bonus (bench excluded), the live score
    status: str

    def summary(self) -> str:
        lines = [
            f"Formation: {self.formation[0]}-{self.formation[1]}-{self.formation[2]}",
            f"Cost: ${self.total_cost:.1f}m   Expected points (XI + captain): {self.expected_points:.2f}",
            f"Captain: {self.captain.name}   Vice: {self.vice_captain.name}",
            "Starting XI:",
        ]
        for p in self.starting_xi:
            tag = " (C)" if p is self.captain else (" (V)" if p is self.vice_captain else "")
            lines.append(f"  {p.position.value:3} {p.name:<22} {p.country:<14} ${p.price:>4.1f}m  EP {p.ep:>5.2f}{tag}")
        lines.append("Bench:")
        for p in self.bench:
            lines.append(f"  {p.position.value:3} {p.name:<22} {p.country:<14} ${p.price:>4.1f}m  EP {p.ep:>5.2f}")
        return "\n".join(lines)


def optimize_squad(
    players: list[Player],
    stage: Stage = Stage.GROUP,
    budget: float | None = None,
    force_include: set[str] = frozenset(),
    force_exclude: set[str] = frozenset(),
    current_squad_ids: set[str] | None = None,
    free_transfers: int | None = None,
    extra_squad_constraints=None,
) -> SquadSolution:
    """Solve for the optimal squad/XI/captain.

    When `current_squad_ids` is given together with a finite `free_transfers`,
    each transfer beyond the free allocation costs `EXTRA_TRANSFER_PENALTY`
    points (modelled directly in the objective). `free_transfers=None` means
    unlimited (no penalty).

    `extra_squad_constraints` is an optional callable receiving
    (problem, squad_vars_by_id) so callers can add constraints without this
    function needing to know about them.
    """
    if budget is None:
        budget = budget_for_stage(stage)
    country_limit = COUNTRY_LIMIT[stage]

    pool = [p for p in players if p.player_id not in force_exclude]
    ids = [p.player_id for p in pool]
    by_id = {p.player_id: p for p in pool}
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate player_id values in pool")

    prob = pulp.LpProblem("squad", pulp.LpMaximize)

    s = {i: pulp.LpVariable(f"s_{i}", cat="Binary") for i in ids}  # in 15-squad
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}  # in starting XI
    c = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in ids}  # captain

    # Optional transfer penalty: charge for transfers beyond the free allocation.
    transfer_penalty_term = 0
    if current_squad_ids is not None and free_transfers is not None:
        kept = pulp.lpSum(s[i] for i in ids if i in current_squad_ids)
        transfers_made = sum(SQUAD_COMPOSITION.values()) - kept  # 15 - kept
        paid = pulp.LpVariable("paid_transfers", lowBound=0, cat="Integer")
        prob += paid >= transfers_made - free_transfers
        transfer_penalty_term = EXTRA_TRANSFER_PENALTY * paid

    # Objective: XI points + captain bonus + small weight on bench depth.
    prob += (
        pulp.lpSum(by_id[i].ep * x[i] for i in ids)
        + pulp.lpSum(by_id[i].ep * (CAPTAIN_MULTIPLIER - 1) * c[i] for i in ids)
        + BENCH_WEIGHT * pulp.lpSum(by_id[i].ep * (s[i] - x[i]) for i in ids)
        - transfer_penalty_term
    )

    # Squad size and exact composition.
    prob += pulp.lpSum(s[i] for i in ids) == sum(SQUAD_COMPOSITION.values())
    for pos, count in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(s[i] for i in ids if by_id[i].position == pos) == count

    # Budget.
    prob += pulp.lpSum(by_id[i].price * s[i] for i in ids) <= budget

    # Per-country limit.
    countries = {by_id[i].country for i in ids}
    for ctry in countries:
        prob += pulp.lpSum(s[i] for i in ids if by_id[i].country == ctry) <= country_limit

    # Starting XI: 11 players, each must be in the squad.
    for i in ids:
        prob += x[i] <= s[i]
    prob += pulp.lpSum(x[i] for i in ids) == XI_SIZE

    # Formation: GK fixed, outfield lines bounded (reproduces the 7 formations).
    prob += pulp.lpSum(x[i] for i in ids if by_id[i].position == Position.GK) == XI_GK
    _bound_line(prob, x, by_id, ids, Position.DEF, XI_DEF_RANGE)
    _bound_line(prob, x, by_id, ids, Position.MID, XI_MID_RANGE)
    _bound_line(prob, x, by_id, ids, Position.FWD, XI_FWD_RANGE)

    # Captain: exactly one, must be a starter.
    prob += pulp.lpSum(c[i] for i in ids) == 1
    for i in ids:
        prob += c[i] <= x[i]

    # Forced inclusions.
    for i in force_include:
        if i in s:
            prob += s[i] == 1

    if extra_squad_constraints is not None:
        extra_squad_constraints(prob, s)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"No optimal solution found (status: {status}). Check budget/constraints feasibility.")

    return _build_solution(pool, by_id, s, x, c, status)


def _bound_line(prob, x, by_id, ids, pos: Position, bounds: tuple[int, int]):
    line = pulp.lpSum(x[i] for i in ids if by_id[i].position == pos)
    prob += line >= bounds[0]
    prob += line <= bounds[1]


def _build_solution(pool, by_id, s, x, c, status) -> SquadSolution:
    squad = [by_id[i] for i in by_id if s[i].value() == 1]
    xi = [by_id[i] for i in by_id if x[i].value() == 1]
    bench = [p for p in squad if p not in xi]
    captain = next(by_id[i] for i in by_id if c[i].value() == 1)

    # Vice-captain: highest-EP starter that is not the captain.
    vice = max((p for p in xi if p is not captain), key=lambda p: p.ep)

    # Order for readability.
    order = {Position.GK: 0, Position.DEF: 1, Position.MID: 2, Position.FWD: 3}
    xi.sort(key=lambda p: (order[p.position], -p.ep))
    bench.sort(key=lambda p: (order[p.position], -p.ep))

    formation = (
        sum(1 for p in xi if p.position == Position.DEF),
        sum(1 for p in xi if p.position == Position.MID),
        sum(1 for p in xi if p.position == Position.FWD),
    )
    total_cost = sum(p.price for p in squad)
    expected_points = sum(p.ep for p in xi) + captain.ep * (CAPTAIN_MULTIPLIER - 1)

    return SquadSolution(
        squad=squad,
        starting_xi=xi,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        formation=formation,
        total_cost=total_cost,
        expected_points=expected_points,
        status=status,
    )
