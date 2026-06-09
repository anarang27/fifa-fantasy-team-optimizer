"""Entity matching between the user price list and scraped FBref history.

The two sources spell names differently ("Bruno Fernandes" vs
"Bruno Miguel Borges Fernandes"), so we fuzzy-match on normalized names,
restricted to the same country when available, with an optional manual-override
map for the stubborn cases. This join is on the critical path: bad matches mean
the optimizer scores the wrong player.
"""

from __future__ import annotations

import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from fantasy.ingest.countries import normalize_country

DEFAULT_THRESHOLD = 85.0


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().replace(".", " ").replace("-", " ").split())


def match_players(
    price_df: pd.DataFrame,
    history_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    overrides: dict[str, str] | None = None,
    restrict_by_country: bool = True,
) -> pd.DataFrame:
    """Return price_df augmented with matched history player_id and score.

    `overrides` maps a price-list player_id to a history player_id and bypasses
    fuzzy matching. Unmatched rows get matched_id=None and need manual review.
    """
    overrides = overrides or {}
    price_df = price_df.copy()
    if "player_id" not in price_df.columns:
        price_df["player_id"] = [f"u{i}" for i in range(len(price_df))]
    hist = history_df.copy()
    hist["_norm"] = hist["player"].map(_normalize_name)
    hist["_country"] = hist["country"].map(normalize_country)

    # One row per history player (most recent/most-played season wins).
    hist = hist.sort_values("minutes", ascending=False).drop_duplicates("player_id")
    pos_by_id = dict(zip(hist["player_id"], hist["position"]))

    results = []
    for _, row in price_df.iterrows():
        pid = row["player_id"]
        if pid in overrides:
            results.append((overrides[pid], 100.0, "override"))
            continue

        query = _normalize_name(row["name"])
        candidates = hist
        if restrict_by_country and "country" in price_df.columns:
            same = hist[hist["_country"] == normalize_country(row["country"])]
            if not same.empty:
                candidates = same

        choices = dict(zip(candidates["player_id"], candidates["_norm"]))
        if not choices:
            results.append((None, 0.0, "no_candidates"))
            continue

        # token_set_ratio handles sub/superset names well:
        # "Bruno Fernandes" vs "Bruno Miguel Borges Fernandes" -> 100.
        best = process.extractOne(
            query, choices, scorer=fuzz.token_set_ratio
        )
        if best is None:
            results.append((None, 0.0, "no_match"))
            continue
        _, score, matched_id = best
        if score >= threshold:
            results.append((matched_id, float(score), "fuzzy"))
        else:
            results.append((None, float(score), "below_threshold"))

    out = price_df.copy()
    out["matched_id"] = [r[0] for r in results]
    out["match_score"] = [r[1] for r in results]
    out["match_method"] = [r[2] for r in results]
    out["matched_position"] = [pos_by_id.get(r[0]) for r in results]
    return out


def match_report(matched_df: pd.DataFrame) -> dict:
    """Summary stats for a match run."""
    total = len(matched_df)
    matched = matched_df["matched_id"].notna().sum()
    return {
        "total": int(total),
        "matched": int(matched),
        "unmatched": int(total - matched),
        "match_rate": round(matched / total, 4) if total else 0.0,
    }
