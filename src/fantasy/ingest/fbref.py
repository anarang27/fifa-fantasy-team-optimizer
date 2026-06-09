"""FBref scraping via the `soccerdata` library (no paid API).

Pulls player-season stats across multiple stat types (standard, shooting,
passing, defense, misc, keeper), normalizes the messy MultiIndex columns into
the canonical schema, and writes the result to storage.

Network note: FBref must be scraped politely. `soccerdata` caches responses on
disk by default; we keep a single league/season loop with conservative behavior.
Because scraping requires outbound network to FBref, `build_history` is meant to
be run in an environment with internet access; the rest of the pipeline consumes
the cached/processed output.
"""

from __future__ import annotations

import pandas as pd

from fantasy.ingest.schema import NUMERIC_COLUMNS, PLAYER_SEASON_COLUMNS
from fantasy.ingest.storage import save_table

# Most recent completed season only (2025/26), per the updated requirement.
DEFAULT_SEASON = "2526"
DEFAULT_CLUB_LEAGUE = "Big 5 European Leagues Combined"

# Most recent international tournaments soccerdata/FBref exposes. NOTE: FBref via
# soccerdata does not provide this-season WC qualifiers / Nations League /
# friendlies, so "international games" is limited to these tournaments.
DEFAULT_INTERNATIONAL = [("INT-European Championship", "2024")]

# Map an FBref position string to our four position groups.
_POSITION_MAP = {
    "GK": "GK",
    "DF": "DEF",
    "MF": "MID",
    "FW": "FWD",
}

# Canonical target -> the FBref column's last-level name to look for.
_STAT_COLUMN_ALIASES = {
    "minutes": ["Min"],
    "nineties": ["90s"],
    "games": ["MP"],
    "goals": ["Gls"],
    "assists": ["Ast"],
    "yellow_cards": ["CrdY"],
    "red_cards": ["CrdR"],
    "own_goals": ["OG"],
    "pens_won": ["PKwon"],
    "pens_conceded": ["PKcon"],
    "shots_on_target": ["SoT"],
    "key_passes": ["KP"],
    "tackles": ["Tkl"],
    "saves": ["Saves"],
    "penalty_saves": ["PKsv"],
    "goals_against": ["GA"],
    "clean_sheets": ["CS"],
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse FBref MultiIndex columns to their most specific (last) level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _first_present(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            col = df[n]
            # Duplicate column names can yield a DataFrame; take the first.
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col
    return None


def _normalize_position(value) -> str:
    if not isinstance(value, str) or not value:
        return "MID"
    primary = value.split(",")[0].strip()[:2]
    return _POSITION_MAP.get(primary, "MID")


def normalize_player_stats(raw: pd.DataFrame, season: str, competition: str) -> pd.DataFrame:
    """Map a flattened FBref player-stats table to the canonical schema."""
    df = _flatten(raw).reset_index()

    out = pd.DataFrame()
    name = _first_present(df, ["player", "Player"])
    nation = _first_present(df, ["nation", "Nation", "team", "Team", "squad", "Squad"])
    pos = _first_present(df, ["pos", "Pos"])

    out["player"] = name if name is not None else ""
    out["country"] = nation if nation is not None else ""
    out["position"] = (pos if pos is not None else pd.Series([""] * len(df))).map(_normalize_position)
    out["season"] = season
    out["competition"] = competition

    for canonical, aliases in _STAT_COLUMN_ALIASES.items():
        col = _first_present(df, aliases)
        out[canonical] = pd.to_numeric(col, errors="coerce") if col is not None else 0.0

    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].fillna(0.0)
    out["player_id"] = (
        out["player"].astype(str) + "|" + out["country"].astype(str) + "|" + season
    )
    return out[PLAYER_SEASON_COLUMNS]


def _read_fbref(league: str, seasons, stat_types):
    """Pull and merge several stat types from FBref into one flat table."""
    import soccerdata as sd  # lazy: only needed when actually scraping

    fb = sd.FBref(leagues=league, seasons=seasons)
    merged: pd.DataFrame | None = None
    for stat in stat_types:
        try:
            part = _flatten(fb.read_player_season_stats(stat_type=stat)).reset_index()
        except Exception:
            continue
        if merged is None:
            merged = part
        else:
            new_cols = [c for c in part.columns if c not in merged.columns]
            key = [c for c in ["player", "Player"] if c in merged.columns and c in part.columns]
            if key and new_cols:
                merged = merged.merge(part[key + new_cols], on=key, how="left")
    if merged is None:
        raise RuntimeError(f"FBref returned no data for {league} {seasons}")
    return merged


def build_history(
    season: str = DEFAULT_SEASON,
    club_leagues: list[str] | None = None,
    international: list[tuple[str, str]] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Scrape ONE season across all configured competitions + internationals.

    Each (competition) contributes one normalized row per player; the projection
    layer sums these per player, giving an "all competitions" view. Add more
    entries to `club_leagues` (any soccerdata-supported FBref league) to widen
    coverage beyond the big-5 European leagues.
    """
    club_leagues = club_leagues or [DEFAULT_CLUB_LEAGUE]
    international = DEFAULT_INTERNATIONAL if international is None else international
    stat_types = ["standard", "shooting", "passing", "defense", "misc", "keeper"]

    frames: list[pd.DataFrame] = []
    for league in club_leagues:
        raw = _read_fbref(league, [season], stat_types)
        frames.append(normalize_player_stats(raw, season, league))

    for comp, comp_season in international:
        raw = _read_fbref(comp, [comp_season], stat_types)
        frames.append(normalize_player_stats(raw, comp_season, comp))

    history = pd.concat(frames, ignore_index=True)
    if save:
        save_table(history, "player_history")
    return history
