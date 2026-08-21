"""Paths and table names for the processed feature database."""

from __future__ import annotations

from src.data_collection.config import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DB_PATH = PROCESSED_DIR / "pl.db"

MATCH_FEATURES_TABLE = "match_features"
PLAYER_SEASON_FEATURES_TABLE = "player_season_features"
PLAYER_MATCH_FEATURES_TABLE = "player_match_features"
