"""End-to-end pipeline: history + prices -> matched -> projected -> optimal squad.

`run_pipeline` is the real entry point (loads a price CSV and a history table).
`run_demo` runs the whole thing on synthetic seed data with no network, so the
system is verifiable offline and serves as living documentation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from fantasy.data import make_seed_history, make_seed_price_list
from fantasy.ingest.matching import match_players, match_report
from fantasy.ingest.prices import load_price_list
from fantasy.optimize.squad import Player, SquadSolution, optimize_squad
from fantasy.projection.model import ProjectionConfig, players_from_projection, project_points
from fantasy.rules import Stage

# PuLP/CBC emit noisy deprecation warnings; quiet them for CLI/demo runs.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")


def build_players(
    price_df: pd.DataFrame,
    history: pd.DataFrame,
    cfg: ProjectionConfig | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[list[Player], dict]:
    """Match prices to history, resolve positions, project EP, build Players.

    Position comes from the price list when present, otherwise from the matched
    FBref record. Players with no resolvable position cannot be fielded and are
    dropped (counted in the report).
    """
    if "player_id" not in price_df.columns:
        price_df = price_df.assign(player_id=[f"u{i}" for i in range(len(price_df))])
    matched = match_players(price_df, history, overrides=overrides)
    report = match_report(matched)

    # Resolve position: explicit price-list position wins, else FBref's.
    price_pos = matched["position"] if "position" in matched.columns else None
    resolved = []
    for i, row in matched.iterrows():
        pos = row["position"] if price_pos is not None and pd.notna(row.get("position")) else None
        resolved.append(pos or row.get("matched_position"))
    matched = matched.assign(position=resolved)

    dropped = int(matched["position"].isna().sum())
    matched = matched[matched["position"].notna()].copy()
    report["dropped_no_position"] = dropped
    report["usable"] = int(len(matched))

    projection = project_points(history, cfg)
    players = players_from_projection(projection, matched)
    return players, report


def run_pipeline(
    price_path: str | Path,
    history: pd.DataFrame,
    stage: Stage = Stage.GROUP,
    cfg: ProjectionConfig | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[SquadSolution, dict]:
    """Load a real price list + history table and return the optimal squad."""
    price_df = load_price_list(price_path)
    players, report = build_players(price_df, history, cfg, overrides)
    solution = optimize_squad(players, stage=stage)
    return solution, report


def run_demo(stage: Stage = Stage.GROUP) -> SquadSolution:
    """Full offline run on synthetic data. Returns (and prints) the optimal squad."""
    history = make_seed_history()
    price_df = make_seed_price_list(history)
    players, report = build_players(price_df, history)
    solution = optimize_squad(players, stage=stage)
    print(f"Entity match: {report['matched']}/{report['total']} ({report['match_rate']:.0%})")
    print(solution.summary())
    return solution


if __name__ == "__main__":
    run_demo()
