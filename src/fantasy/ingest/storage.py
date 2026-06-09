"""Lightweight persistence: Parquet for analytics, SQLite for queryable storage.

Parquet is the primary format (columnar, fast, schema-preserving). A SQLite
mirror is offered for ad-hoc querying. Both live under data/processed/.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SQLITE_PATH = PROCESSED_DIR / "fantasy.db"


def _ensure_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str, to_sqlite: bool = True) -> Path:
    """Persist a DataFrame as Parquet (and optionally a SQLite table)."""
    _ensure_dir()
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    if to_sqlite:
        with sqlite3.connect(SQLITE_PATH) as con:
            df.to_sql(name, con, if_exists="replace", index=False)
    return path


def load_table(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed table '{name}' at {path}. Run ingestion first.")
    return pd.read_parquet(path)
