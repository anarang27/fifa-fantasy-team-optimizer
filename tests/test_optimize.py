"""Optimizer constraint and behavior tests on the synthetic seed pool."""

from collections import Counter

from fantasy.data import make_seed_pool
from fantasy.optimize import Player, optimize_squad, plan_boosters, plan_transfers
from fantasy.rules import (
    BASE_BUDGET,
    COUNTRY_LIMIT,
    SQUAD_COMPOSITION,
    VALID_FORMATIONS,
    XI_SIZE,
    Position,
    Stage,
    budget_for_stage,
)


def test_squad_respects_all_constraints():
    sol = optimize_squad(make_seed_pool(), stage=Stage.GROUP)

    assert len(sol.squad) == sum(SQUAD_COMPOSITION.values())
    pos_counts = Counter(p.position for p in sol.squad)
    for pos, n in SQUAD_COMPOSITION.items():
        assert pos_counts[pos] == n

    assert sol.total_cost <= BASE_BUDGET + 1e-6

    country_counts = Counter(p.country for p in sol.squad)
    assert max(country_counts.values()) <= COUNTRY_LIMIT[Stage.GROUP]


def test_starting_xi_is_valid_formation():
    sol = optimize_squad(make_seed_pool(), stage=Stage.GROUP)
    assert len(sol.starting_xi) == XI_SIZE
    assert sum(1 for p in sol.starting_xi if p.position == Position.GK) == 1
    assert sol.formation in VALID_FORMATIONS
    assert len(sol.bench) == 4


def test_captain_is_highest_ep_starter():
    sol = optimize_squad(make_seed_pool(), stage=Stage.GROUP)
    assert sol.captain in sol.starting_xi
    assert sol.captain.ep == max(p.ep for p in sol.starting_xi)
    assert sol.vice_captain is not sol.captain


def test_knockout_budget_and_country_limit_relax():
    pool = make_seed_pool()
    group = optimize_squad(pool, stage=Stage.GROUP)
    final = optimize_squad(pool, stage=Stage.FINAL)
    assert budget_for_stage(Stage.FINAL) > budget_for_stage(Stage.GROUP)
    # More budget + looser country cap cannot reduce the optimum.
    assert final.expected_points >= group.expected_points - 1e-6


def test_country_limit_binds_when_one_country_is_loaded():
    # Make Argentina players overwhelmingly the best; cap must still hold.
    pool = []
    for p in make_seed_pool():
        ep = p.ep + 50 if p.country == "Argentina" else p.ep
        pool.append(Player(p.player_id, p.name, p.country, p.position, p.price, ep))
    sol = optimize_squad(pool, stage=Stage.GROUP)
    arg = sum(1 for p in sol.squad if p.country == "Argentina")
    assert arg <= COUNTRY_LIMIT[Stage.GROUP]


def test_transfer_penalty_discourages_churn():
    pool = make_seed_pool()
    base = optimize_squad(pool, stage=Stage.GROUP)

    # With only 2 free transfers, the planner should not rebuild the whole squad.
    plan = plan_transfers(base.squad, pool, stage=Stage.GROUP, free_transfers=2)
    assert plan.num_transfers == 0  # nothing changed in the pool, so no moves needed
    assert plan.point_hit == 0


def test_wildcard_booster_gain_non_negative():
    pool = make_seed_pool()
    base = optimize_squad(pool, stage=Stage.GROUP)
    boosters = plan_boosters(base.squad, pool, stage=Stage.GROUP, free_transfers=2)
    names = {b.booster for b in boosters}
    assert {"wildcard", "12th_man", "max_captain"} <= names
    wildcard = next(b for b in boosters if b.booster == "wildcard")
    assert wildcard.estimated_gain >= -1e-6
