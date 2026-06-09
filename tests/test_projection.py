"""Projection model and backtest tests."""

import numpy as np

from fantasy.data import make_seed_history
from fantasy.projection import project_points
from fantasy.projection.backtest import backtest_world_cup


def test_projection_produces_clean_ep():
    history = make_seed_history()
    proj = project_points(history)
    assert "ep" in proj.columns
    assert not proj["ep"].isna().any()
    assert np.isfinite(proj["ep"]).all()
    # Each player should appear once.
    assert proj["player_id"].is_unique


def test_minutes_gate_zeroes_non_players():
    history = make_seed_history()
    proj = project_points(history)
    # Everyone in seed plays, so all p_play > 0 and EP is mostly positive.
    assert (proj["p_play"] > 0).mean() > 0.95
    assert proj["ep"].mean() > 0


def test_backtest_is_predictive_on_seed():
    history = make_seed_history()
    result = backtest_world_cup(history)
    # Synthetic ability drives both train rates and WC actuals, so the
    # projection must rank players positively against actuals.
    assert result["spearman"] > 0.3
    assert result["top20pct_overlap"] > 0.3
    assert result["n_players"] > 30
