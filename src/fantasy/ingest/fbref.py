"""FBref league-level scraping via the `soccerdata` library (no paid API).

Strategy (Option A): pull league-wide player tables, one request per stat
category per competition (each request returns ALL players in that competition).
We combine 6 stat categories -- standard, shooting, passing, defense, misc,
keeper -- which together cover every stat the scoring engine needs (minutes,
goals, assists, cards, shots on target, key passes, tackles, saves, goals
against, clean sheets, penalties).

This is a handful of throttled requests covering thousands of players, not a
per-player crawl, so it stays well under FBref's rate limit. Coverage is limited
to soccerdata-supported competitions; add more to `club_leagues` to widen it.

Network notes:
- FBref sits behind a Cloudflare challenge. soccerdata generally passes it from a
  residential IP. If you get a 403 "Just a moment..." page, run from a home
  connection or back soccerdata with a Cloudflare-aware session.
- soccerdata caches to disk (SOCCERDATA_DIR) and throttles automatically, so
  re-runs do not re-hit the site.
"""

from __future__ import annotations

import os

import pandas as pd

from fantasy.ingest.countries import normalize_country
from fantasy.ingest.schema import NUMERIC_COLUMNS, PLAYER_SEASON_COLUMNS
from fantasy.ingest.storage import save_table

# Most recent completed season only (2025/26), per the requirement.
DEFAULT_SEASON = "2526"
DEFAULT_CLUB_LEAGUE = "Big 5 European Leagues Combined"

# Most recent international tournaments soccerdata/FBref exposes. NOTE: FBref via
# soccerdata does not provide this-season WC qualifiers / Nations League /
# friendlies, so "international games" is limited to these tournaments.
DEFAULT_INTERNATIONAL = [("INT-European Championship", "2024")]

STAT_TYPES = ["standard", "shooting", "passing", "defense", "misc", "keeper"]

# The join keys soccerdata returns as the player-season index.
KEYS = ["league", "season", "team", "player"]

# Map an FBref position token to our four position groups.
_POSITION_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}

# Canonical stat -> the FBref column's last-level name.
CANONICAL_FROM_FBREF = {
    "minutes": "Min",
    "nineties": "90s",
    "games": "MP",
    "goals": "Gls",
    "assists": "Ast",
    "yellow_cards": "CrdY",
    "red_cards": "CrdR",
    "own_goals": "OG",
    "pens_won": "PKwon",
    "pens_conceded": "PKcon",
    "shots_on_target": "SoT",
    "key_passes": "KP",
    "tackles": "Tkl",
    "saves": "Saves",
    "penalty_saves": "PKsv",
    "goals_against": "GA",
    "clean_sheets": "CS",
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse FBref MultiIndex columns to their last NON-EMPTY level.

    soccerdata's meta columns (nation, pos, age) carry the name in level 0 with
    an empty level 1, while stat columns (Performance/Gls) carry it in level 1.
    Taking the last non-empty token handles both.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        flat = []
        for col in df.columns:
            if isinstance(col, tuple):
                parts = [str(x) for x in col if str(x) and not str(x).startswith("Unnamed")]
                flat.append(parts[-1] if parts else "")
            else:
                flat.append(col)
        df.columns = flat
    return df


def _normalize_position(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "MID"
    primary = value.replace(",", " ").split()[0][:2].upper()
    return _POSITION_MAP.get(primary, "MID")


def _nation_to_country(value) -> str:
    """FBref 'nation' is like 'ar ARG' or 'eng ENG'; map the code to a name."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return normalize_country(value.split()[-1])


def combine_stat_frames(
    frames: dict[str, pd.DataFrame], competition: str, season: str
) -> pd.DataFrame:
    """Combine per-stat-type FBref tables into the canonical schema.

    Pure function (no network): takes raw soccerdata-style frames keyed by stat
    type and returns one canonical row per player for this competition. Stats
    are pulled from whichever table provides them, joined on player keys, so the
    duplicate column names across tables (Min, MP, Gls, ...) don't collide.
    """
    base: pd.DataFrame | None = None
    for stat, raw in frames.items():
        if raw is None or len(raw) == 0:
            continue
        df = _flatten(raw).reset_index()
        df = df.loc[:, ~df.columns.duplicated()]
        keys = [k for k in KEYS if k in df.columns]
        if not keys:
            continue

        sub = df[keys].copy()
        if stat == "standard":
            sub["nation"] = df["nation"] if "nation" in df.columns else ""
            sub["pos"] = df["pos"] if "pos" in df.columns else ""
        for canonical, fb_name in CANONICAL_FROM_FBREF.items():
            if fb_name in df.columns:
                sub[canonical] = pd.to_numeric(df[fb_name], errors="coerce")

        if base is None:
            base = sub
        else:
            new_cols = [c for c in sub.columns if c not in base.columns]
            if new_cols:
                base = base.merge(sub[keys + new_cols], on=keys, how="outer")

    if base is None:
        raise RuntimeError(f"No usable stat tables for {competition} {season}")
    for meta in ("nation", "pos"):
        if meta not in base.columns:
            base[meta] = ""

    out = pd.DataFrame()
    out["player"] = base["player"].astype(str)
    # Prefer player nationality; for international comps it's blank, so fall back
    # to the team (which is the national side, i.e. the country).
    from_nation = base["nation"].map(_nation_to_country)
    from_team = base["team"].map(normalize_country) if "team" in base.columns else ""
    out["country"] = from_nation.where(from_nation.str.len() > 0, from_team)
    out["position"] = base["pos"].map(_normalize_position)
    out["season"] = season
    out["competition"] = competition
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(base[col], errors="coerce") if col in base.columns else 0.0
    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].fillna(0.0)

    # Stable per (player, country) so the projection aggregates all competitions.
    out["player_id"] = out["player"] + "|" + out["country"]
    return out[PLAYER_SEASON_COLUMNS]


def _read_fbref(league: str, season: str) -> dict[str, pd.DataFrame]:
    """Fetch each stat-type table for one league-season from FBref."""
    os.environ.setdefault("SOCCERDATA_DIR", os.path.join(os.getcwd(), ".soccerdata"))
    import soccerdata as sd  # lazy: only needed when actually scraping

    fb = sd.FBref(leagues=league, seasons=[season])
    frames: dict[str, pd.DataFrame] = {}
    for stat in STAT_TYPES:
        try:
            frames[stat] = fb.read_player_season_stats(stat_type=stat)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"FBref returned no data for {league} {season}")
    return frames


def build_history(
    season: str = DEFAULT_SEASON,
    club_leagues: list[str] | None = None,
    international: list[tuple[str, str]] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Scrape ONE season across all configured competitions + internationals.

    Each competition contributes one canonical row per player; the projection
    layer sums these per player, giving an "all competitions" view across the
    competitions soccerdata supports. Add more entries to `club_leagues` (any
    soccerdata-supported FBref league) to widen coverage.
    """
    club_leagues = club_leagues or [DEFAULT_CLUB_LEAGUE]
    international = DEFAULT_INTERNATIONAL if international is None else international

    frames: list[pd.DataFrame] = []
    for league in club_leagues:
        frames.append(combine_stat_frames(_read_fbref(league, season), league, season))
    for comp, comp_season in international:
        frames.append(combine_stat_frames(_read_fbref(comp, comp_season), comp, comp_season))

    history = pd.concat(frames, ignore_index=True)
    if save:
        save_table(history, "player_history")
    return history
