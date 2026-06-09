"""Load the user-provided player pool + fixed prices.

This is the canonical player list for the optimizer. Prices are NOT scraped.

Handles the real `players.csv` shape: columns Name, Country (FIFA 3-letter
code), Price (e.g. "$10.5m"). Position is OPTIONAL -- if the file has no
position column, positions are filled later from the matched FBref data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fantasy.ingest.countries import normalize_country
from fantasy.rules import Position

REQUIRED = {"name", "country", "price"}

_POSITION_ALIASES = {
    "gk": "GK", "goalkeeper": "GK", "g": "GK",
    "def": "DEF", "defender": "DEF", "d": "DEF",
    "mid": "MID", "midfielder": "MID", "m": "MID",
    "fwd": "FWD", "forward": "FWD", "f": "FWD", "att": "FWD", "attacker": "FWD",
}

_PRICE_RE = re.compile(r"[^0-9.]+")


def _normalize_position(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key.upper() in Position.__members__:
        return key.upper()
    if key in _POSITION_ALIASES:
        return _POSITION_ALIASES[key]
    raise ValueError(f"Unrecognized position '{value}'. Use GK/DEF/MID/FWD.")


def _parse_price(value) -> float:
    """Parse prices like '$10.5m', '10m', '8' -> 10.5 / 10.0 / 8.0."""
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    cleaned = _PRICE_RE.sub("", str(value))
    if not cleaned:
        raise ValueError(f"Could not parse price '{value}'.")
    return float(cleaned)


def load_price_list(path: str | Path) -> pd.DataFrame:
    """Load and validate the price list into a normalized DataFrame.

    Output columns: player_id, name, country (canonical name), country_code,
    price, and position (may be None where the file omitted it).
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Price list missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["name"] = df["name"].astype(str).str.strip()
    out["country_code"] = df["country"].astype(str).str.strip()
    out["country"] = out["country_code"].map(normalize_country)
    out["price"] = df["price"].map(_parse_price)
    if out["price"].isna().any():
        raise ValueError("Some prices could not be parsed as numbers.")

    if "position" in df.columns:
        out["position"] = df["position"].map(_normalize_position)
    else:
        out["position"] = None

    if "player_id" in df.columns:
        out["player_id"] = df["player_id"].astype(str)
    else:
        out["player_id"] = [f"u{i}" for i in range(len(df))]

    return out.reset_index(drop=True)
