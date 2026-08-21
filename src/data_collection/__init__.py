"""Data collection: match results and player statistics into ``data/raw``.

Sources are key-free. Historical seasons are cached permanently, so repeated runs cost
almost no network traffic - with one exception, ``player_match_stats``, which needs one
request per match and is therefore opt-in.
"""

from __future__ import annotations

import logging

from src.data_collection.config import DEFAULT_MIN_INTERVAL_SECONDS, season_start_years
from src.data_collection.football_data_uk import collect_matches
from src.data_collection.http_client import CachedSession
from src.data_collection.storage import (
    FIXTURES_TABLE,
    MATCHES_TABLE,
    PLAYER_MATCH_STATS_TABLE,
    PLAYER_STATS_TABLE,
    TEAM_MATCH_STATS_TABLE,
    write_table,
)
from src.data_collection.understat import collect_league_data
from src.data_collection.understat_matches import collect_player_match_stats

logger = logging.getLogger(__name__)

DEFAULT_COMPLETED_SEASONS = 5

#: Everything except ``player_matches``, which costs ~1,900 requests.
CHEAP_SOURCES = ("matches", "players")
ALL_SOURCES = ("matches", "players", "player_matches")

__all__ = [
    "ALL_SOURCES",
    "CHEAP_SOURCES",
    "CachedSession",
    "collect_all",
    "collect_league_data",
    "collect_matches",
    "collect_player_match_stats",
]


def collect_all(
    n_completed_seasons: int = DEFAULT_COMPLETED_SEASONS,
    *,
    sources: tuple[str, ...] = CHEAP_SOURCES,
    force_refresh: bool = False,
    min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
) -> dict[str, int]:
    """Run the collectors and persist everything under ``data/raw``.

    Args:
        n_completed_seasons: How many finished seasons to fetch, in addition to the
            season currently in progress.
        sources: Which collectors to run. ``"player_matches"`` is excluded by default
            because it costs one request per match.
        force_refresh: Ignore cached responses and re-fetch.
        min_interval: Minimum seconds between requests to the same host.

    Returns:
        Rows written per table.
    """
    years = season_start_years(n_completed_seasons)
    logger.info("Seasons: %s", ", ".join(str(y) for y in years))

    written: dict[str, int] = {}
    with CachedSession(min_interval=min_interval) as session:
        if "matches" in sources:
            logger.info("Collecting match results from football-data.co.uk")
            matches = collect_matches(session, years, force_refresh=force_refresh)
            written[MATCHES_TABLE] = write_table(matches, MATCHES_TABLE)

        needs_league = {"players", "player_matches"} & set(sources)
        if needs_league:
            logger.info("Collecting league data from Understat")
            league = collect_league_data(session, years, force_refresh=force_refresh)

            if "players" in sources:
                written[PLAYER_STATS_TABLE] = write_table(league.players, PLAYER_STATS_TABLE)
                written[TEAM_MATCH_STATS_TABLE] = write_table(
                    league.team_history, TEAM_MATCH_STATS_TABLE
                )
                written[FIXTURES_TABLE] = write_table(league.fixtures, FIXTURES_TABLE)

            if "player_matches" in sources:
                logger.info("Collecting per-match player statistics (one request per match)")
                player_matches = collect_player_match_stats(
                    session, league.fixtures, force_refresh=force_refresh
                )
                written[PLAYER_MATCH_STATS_TABLE] = write_table(
                    player_matches, PLAYER_MATCH_STATS_TABLE
                )

    return written
