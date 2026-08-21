"""Feature engineering: raw database -> model-ready tables in ``data/processed/pl.db``.

Reads only from ``data/raw/premier_league_raw.db``. Never touches the network - that is
``src/data_collection``'s job, per the layer boundary in CLAUDE.md.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_collection.config import RAW_DB_PATH
from src.data_collection.storage import (
    MATCHES_TABLE,
    PLAYER_MATCH_STATS_TABLE,
    PLAYER_STATS_TABLE,
    TEAM_MATCH_STATS_TABLE,
    read_table,
    write_table,
)
from src.features.config import (
    MATCH_FEATURES_TABLE,
    PLAYER_MATCH_FEATURES_TABLE,
    PLAYER_SEASON_FEATURES_TABLE,
    PROCESSED_DB_PATH,
)
from src.features.player_features import (
    build_player_match_features,
    build_player_season_features,
)
from src.features.team_features import build_match_features

logger = logging.getLogger(__name__)

__all__ = [
    "build_all",
    "build_match_features",
    "build_player_match_features",
    "build_player_season_features",
]


def _earliest_season(matches: pd.DataFrame) -> str:
    """Return the earliest season present, which is history-only.

    The first season in the raw data has no predecessor, so its ``prev_season_rank``
    would be a sentinel for all 20 clubs. It is collected purely so the *second*
    season has real previous-season features, and is not emitted itself.

    Args:
        matches: Raw match rows.

    Returns:
        The earliest season label.
    """
    return str(sorted(matches["season"].unique())[0])


def build_all(*, only: str = "all") -> dict[str, int]:
    """Build the feature tables and write them to ``data/processed/pl.db``.

    Args:
        only: ``"all"``, ``"team"`` or ``"player"``.

    Returns:
        Rows written per table.
    """
    written: dict[str, int] = {}

    if only in ("all", "team"):
        matches = read_table(MATCHES_TABLE, RAW_DB_PATH)
        team_stats = read_table(TEAM_MATCH_STATS_TABLE, RAW_DB_PATH)
        logger.info("Building team features from %d matches", len(matches))
        features = build_match_features(
            matches, team_stats, drop_seasons=(_earliest_season(matches),)
        )
        written[MATCH_FEATURES_TABLE] = write_table(
            features, MATCH_FEATURES_TABLE, PROCESSED_DB_PATH
        )

    if only in ("all", "player"):
        player_seasons = read_table(PLAYER_STATS_TABLE, RAW_DB_PATH)
        logger.info("Building player-season features from %d rows", len(player_seasons))
        season_features = build_player_season_features(player_seasons)
        written[PLAYER_SEASON_FEATURES_TABLE] = write_table(
            season_features, PLAYER_SEASON_FEATURES_TABLE, PROCESSED_DB_PATH
        )

        try:
            player_matches = read_table(PLAYER_MATCH_STATS_TABLE, RAW_DB_PATH)
        except ValueError:
            logger.warning(
                "No %s table in the raw database - skipping player form features. "
                "Run: python -m src.data_collection.cli --source player_matches",
                PLAYER_MATCH_STATS_TABLE,
            )
        else:
            logger.info("Building player-match features from %d rows", len(player_matches))
            match_features = build_player_match_features(player_matches)
            written[PLAYER_MATCH_FEATURES_TABLE] = write_table(
                match_features, PLAYER_MATCH_FEATURES_TABLE, PROCESSED_DB_PATH
            )

    return written
