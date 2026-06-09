"""Backtest the projection against Qatar 2022 actuals.

Train rates on pre-World-Cup history only, project expected points, then compare
to actual average fantasy points per match at the 2022 World Cup (computed with
the real scoring engine). Reports MAE, rank correlation, and top-N overlap so we
can tell whether the projection is actually predictive before trusting it live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.projection.model import ProjectionConfig, project_points
from fantasy.rules import Position
from fantasy.scoring.engine import MatchStats, score_match

WC_SEASON = "2022"


def _actual_points_per_match(row: pd.Series) -> float:
    games = max(1, int(row["games"]))
    per_min = min(90.0, row["minutes"] / games)

    def pg(col):  # per-game rounded count
        return int(round(row[col] / games))

    stats = MatchStats(
        position=Position(row["position"]),
        minutes=int(round(per_min)),
        goals=pg("goals"),
        assists=pg("assists"),
        yellow_cards=pg("yellow_cards"),
        red_cards=pg("red_cards"),
        own_goals=pg("own_goals"),
        penalties_won=pg("pens_won"),
        penalties_conceded=pg("pens_conceded"),
        team_goals_conceded=pg("goals_against"),
        saves=pg("saves"),
        penalty_saves=pg("penalty_saves"),
        tackles=pg("tackles"),
        chances_created=pg("key_passes"),
        shots_on_target=pg("shots_on_target"),
    )
    return float(score_match(stats))


def backtest_world_cup(history: pd.DataFrame, cfg: ProjectionConfig | None = None) -> dict:
    """Return metrics + merged frame comparing projected EP to WC 2022 actuals."""
    train = history[history["season"] != WC_SEASON]
    holdout = history[history["season"] == WC_SEASON].copy()
    if holdout.empty:
        raise ValueError("No 2022 World Cup rows in history to backtest against.")

    projection = project_points(train, cfg)
    projection["_key"] = projection["player"].astype(str) + "|" + projection["country"].astype(str)

    holdout["_key"] = holdout["player"].astype(str) + "|" + holdout["country"].astype(str)
    holdout["actual"] = holdout.apply(_actual_points_per_match, axis=1)

    merged = projection.merge(
        holdout[["_key", "actual"]], on="_key", how="inner"
    )
    if merged.empty:
        raise ValueError("No overlap between trained players and WC participants.")

    err = merged["ep"] - merged["actual"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(merged["ep"].corr(merged["actual"], method="spearman"))

    n = max(1, len(merged) // 5)  # top 20%
    top_pred = set(merged.nlargest(n, "ep")["_key"])
    top_actual = set(merged.nlargest(n, "actual")["_key"])
    overlap = len(top_pred & top_actual) / n

    return {
        "n_players": int(len(merged)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "spearman": round(corr, 3),
        "top20pct_overlap": round(overlap, 3),
        "merged": merged,
    }
