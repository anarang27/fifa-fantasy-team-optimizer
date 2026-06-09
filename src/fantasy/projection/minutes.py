"""Minutes model: expected minutes and participation probability per match.

Fantasy points are gated on appearances, so a minutes estimate is essential. We
keep a transparent proxy: minutes-per-game-played drives both the expected
minutes and a monotonic participation probability (more minutes -> more nailed).
This is the obvious place to later swap in a learned model trained on
starts/rotation; the rest of the projection only depends on the two outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PLAY_PROB = 0.30
MAX_PLAY_PROB = 0.98


def expected_minutes(agg: pd.DataFrame) -> pd.DataFrame:
    """Add `exp_minutes` and `p_play` columns from aggregated playing time.

    Expects columns `minutes` (weighted total) and `games` (weighted total).
    """
    out = agg.copy()
    games = out["games"].replace(0, np.nan)
    mpg = (out["minutes"] / games).clip(0, 90).fillna(0.0)
    out["exp_minutes"] = mpg
    out["p_play"] = (mpg / 90.0 + 0.2).clip(MIN_PLAY_PROB, MAX_PLAY_PROB)
    # Players with no history at all should not be assumed to play.
    out.loc[out["games"] <= 0, "p_play"] = 0.0
    return out
