"""End-to-end and live-loop integration tests on synthetic data."""

from collections import Counter

from fantasy.data import make_seed_history, make_seed_price_list
from fantasy.pipeline.run import build_players
from fantasy.pipeline.live import describe_update, recommend_update
from fantasy.optimize.squad import optimize_squad
from fantasy.rules import COUNTRY_LIMIT, SQUAD_COMPOSITION, Stage


def test_positions_recovered_from_fbref_when_absent():
    history = make_seed_history()
    prices = make_seed_price_list(history)
    assert "position" not in prices.columns  # mirrors players.csv (no position)
    players, report = build_players(prices, history)
    # Positions came from the matched FBref data.
    assert all(p.position.value in {"GK", "DEF", "MID", "FWD"} for p in players)
    assert "dropped_no_position" in report
    assert report["usable"] == len(players)


def test_end_to_end_demo_produces_valid_squad():
    history = make_seed_history()
    prices = make_seed_price_list(history)
    players, report = build_players(prices, history)
    assert report["match_rate"] >= 0.9

    sol = optimize_squad(players, stage=Stage.GROUP)
    assert len(sol.squad) == sum(SQUAD_COMPOSITION.values())
    assert sol.total_cost <= 100.0 + 1e-6
    counts = Counter(p.country for p in sol.squad)
    assert max(counts.values()) <= COUNTRY_LIMIT[Stage.GROUP]
    # The optimizer should prefer players that actually project points.
    assert sol.expected_points > 0


def test_live_update_recommends_under_transfer_limit():
    history = make_seed_history()
    prices = make_seed_price_list(history)
    players, _ = build_players(prices, history)
    base = optimize_squad(players, stage=Stage.GROUP)
    squad_ids = {p.player_id for p in base.squad}

    # Re-run with same data: no transfers should be needed.
    update = recommend_update(squad_ids, prices, history, Stage.GROUP, free_transfers=2)
    assert update["transfer_plan"].num_transfers == 0
    assert update["transfer_plan"].point_hit == 0
    # describe_update should render without error.
    assert "Booster ranking" in describe_update(update)
