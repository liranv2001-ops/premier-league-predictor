"""Persist collected and derived data to SQLite.

Two databases, deliberately separate:

* ``data/raw/premier_league_raw.db`` - the landing zone, faithful to what the sources
  returned. Written by ``src/data_collection``.
* ``data/processed/pl.db`` - the cleaned, feature-ready database built by
  ``src/features``.

Both use these helpers; the caller picks the path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.data_collection.config import RAW_DB_PATH

logger = logging.getLogger(__name__)

MATCHES_TABLE = "matches"
PLAYER_STATS_TABLE = "player_season_stats"
TEAM_MATCH_STATS_TABLE = "team_match_stats"
FIXTURES_TABLE = "understat_fixtures"
PLAYER_MATCH_STATS_TABLE = "player_match_stats"


def write_table(df: pd.DataFrame, table: str, db_path: Path = RAW_DB_PATH) -> int:
    """Write a DataFrame to a SQLite database, replacing the table.

    Replacing wholesale keeps re-runs idempotent: collection is cached and feature
    building is deterministic, so there is no benefit to incremental appends and no
    risk of duplicate rows.

    Args:
        df: Data to store. An empty frame is skipped.
        table: Destination table name.
        db_path: Target database. Defaults to the raw database.

    Returns:
        The number of rows written.
    """
    if df.empty:
        logger.warning("Nothing to write to %r - skipping.", table)
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        df.to_sql(table, engine, if_exists="replace", index=False)
    finally:
        engine.dispose()

    logger.info("Wrote %d rows to %s::%s", len(df), db_path.name, table)
    return len(df)


def read_table(table: str, db_path: Path = RAW_DB_PATH) -> pd.DataFrame:
    """Read a table back from a SQLite database.

    Args:
        table: Table name.
        db_path: Source database. Defaults to the raw database.

    Returns:
        The table contents as a DataFrame.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        return pd.read_sql_table(table, engine)
    finally:
        engine.dispose()
