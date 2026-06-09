"""Deterministic synthetic player pool for tests and demos.

Generates a plausible pool with enough players per position and several
countries so the optimizer and transfer planner can be exercised without any
network access or scraped data.
"""

from __future__ import annotations

import random

import pandas as pd

from fantasy.ingest.schema import PLAYER_SEASON_COLUMNS
from fantasy.optimize.squad import Player
from fantasy.rules import Position

COUNTRIES = [
    "Argentina", "Brazil", "France", "England", "Spain",
    "Portugal", "Germany", "Netherlands", "Croatia", "Morocco",
]

_POOL_PER_POSITION = {
    Position.GK: 8,
    Position.DEF: 24,
    Position.MID: 24,
    Position.FWD: 16,
}

_PRICE_RANGE = {
    Position.GK: (4.0, 6.0),
    Position.DEF: (4.0, 7.5),
    Position.MID: (5.0, 12.0),
    Position.FWD: (6.0, 13.0),
}


def make_seed_pool(seed: int = 42) -> list[Player]:
    """Build a deterministic synthetic pool. EP is loosely tied to price plus noise."""
    rng = random.Random(seed)
    players: list[Player] = []
    idx = 0
    for pos, n in _POOL_PER_POSITION.items():
        lo, hi = _PRICE_RANGE[pos]
        for _ in range(n):
            country = COUNTRIES[idx % len(COUNTRIES)]
            price = round(rng.uniform(lo, hi), 1)
            # Pricier players tend to score more, with noise so value picks exist.
            base = (price - lo) / (hi - lo)
            ep = round(max(0.0, base * 8 + rng.gauss(2.5, 1.5)), 2)
            players.append(
                Player(
                    player_id=f"p{idx}",
                    name=f"{pos.value}_{idx}",
                    country=country,
                    position=pos,
                    price=price,
                    ep=ep,
                )
            )
            idx += 1
    return players


# --- Synthetic history (for offline projection + backtest) ----------------

_SQUAD_PER_COUNTRY = {Position.GK: 3, Position.DEF: 8, Position.MID: 8, Position.FWD: 5}
_CLUB_SEASONS = ["2223", "2324", "2425"]
_WC_SEASON = "2022"

# Per-90 rate ranges by position (loosely realistic).
_RATE_PROFILE = {
    Position.GK: {"saves": (1.5, 4.0), "penalty_saves": (0.0, 0.15)},
    Position.DEF: {"goals": (0.02, 0.18), "assists": (0.03, 0.2), "tackles": (1.0, 3.0), "key_passes": (0.3, 1.2)},
    Position.MID: {"goals": (0.08, 0.45), "assists": (0.1, 0.45), "tackles": (0.8, 3.0), "key_passes": (1.0, 3.0)},
    Position.FWD: {"goals": (0.25, 0.9), "assists": (0.05, 0.35), "shots_on_target": (1.0, 3.2), "key_passes": (0.5, 2.0)},
}


def _zero_stats() -> dict:
    return {c: 0.0 for c in PLAYER_SEASON_COLUMNS[6:]}


def _season_row(rng, pid, name, country, pos, season, competition, ability, games):
    mpg = rng.uniform(55, 90) * (0.6 + 0.4 * ability)
    mpg = min(90.0, mpg)
    minutes = mpg * games
    nineties = minutes / 90.0
    stats = _zero_stats()
    stats["minutes"] = round(minutes)
    stats["nineties"] = round(nineties, 2)
    stats["games"] = games

    for stat, (lo, hi) in _RATE_PROFILE[pos].items():
        rate = (lo + (hi - lo) * ability) * rng.uniform(0.7, 1.3)
        stats[stat] = round(rate * nineties)

    stats["yellow_cards"] = round(rng.uniform(0.05, 0.25) * nineties)
    stats["red_cards"] = 1 if rng.random() < 0.03 else 0
    stats["pens_won"] = round(rng.uniform(0.0, 0.08) * nineties)
    stats["pens_conceded"] = round(rng.uniform(0.0, 0.05) * nineties)

    if pos in (Position.GK, Position.DEF):
        ga_per_game = rng.uniform(0.6, 1.6) * (1.4 - 0.6 * ability)
        stats["goals_against"] = round(ga_per_game * games)
        if pos == Position.GK:
            stats["clean_sheets"] = round(max(0, games * (0.5 - 0.25 * (ga_per_game))))

    return {
        "player_id": pid,
        "player": name,
        "country": country,
        "position": pos.value,
        "season": season,
        "competition": competition,
        **stats,
    }


def make_seed_history(seed: int = 7) -> pd.DataFrame:
    """Synthetic multi-season history in the canonical schema, incl. Qatar 2022."""
    rng = random.Random(seed)
    rows = []
    for country in COUNTRIES:
        pidx = 0
        for pos, n in _SQUAD_PER_COUNTRY.items():
            for _ in range(n):
                ability = rng.betavariate(2, 3)  # skewed toward average
                name = f"{country[:3]}_{pos.value}_{pidx}"
                pid = f"{country}|{name}"
                for season in _CLUB_SEASONS:
                    rows.append(_season_row(
                        rng, pid, name, country, pos, season,
                        "ENG-Premier League", ability, games=rng.randint(24, 34),
                    ))
                # Qatar 2022 appearance for ~70% of players.
                if rng.random() < 0.7:
                    rows.append(_season_row(
                        rng, pid, name, country, pos, _WC_SEASON,
                        "FIFA World Cup", ability, games=rng.randint(3, 7),
                    ))
                pidx += 1
    return pd.DataFrame(rows)[PLAYER_SEASON_COLUMNS]


_NAME_NOISE = ["", " Jr", " Silva", " dos Santos"]


def make_seed_price_list(history: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Build a user-style price list from history.

    Mirrors the real players.csv: name, country, price only -- NO position
    column, so the pipeline must recover positions from the matched FBref data.
    Names get suffix noise to exercise fuzzy matching.
    """
    rng = random.Random(seed)
    latest = history[history["season"] == "2425"].copy()
    rows = []
    for _, r in latest.iterrows():
        nineties = max(0.1, r["nineties"])
        scorish = (r["goals"] + r["assists"]) / nineties
        price = round(4.0 + min(9.0, scorish * 12) + rng.uniform(-0.5, 0.5), 1)
        price = max(4.0, price)
        noisy_name = r["player"] + rng.choice(_NAME_NOISE)
        rows.append({
            "name": noisy_name,
            "country": r["country"],
            "price": price,
        })
    return pd.DataFrame(rows)

