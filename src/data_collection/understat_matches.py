"""Collect per-player, per-match statistics from Understat.

This is the only expensive collector in the project: one request per match, so roughly
1,900 requests for five seasons. It exists because the league endpoint returns season
*totals* only, and player form ("last 5 appearances") cannot be derived from those.

The cost is one-time and the run is resumable - completed seasons cache with no expiry,
so an interrupted run re-reads from disk and only fetches what is still missing.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.data_collection.config import (
    CURRENT_SEASON_TTL_SECONDS,
    UNDERSTAT_MATCH_URL,
    current_season_start_year,
)
from src.data_collection.http_client import CachedSession, NotFoundError
from src.data_collection.understat import UnderstatFormatError

logger = logging.getLogger(__name__)

#: Source field -> stored column, from the ``rosters`` block.
ROSTER_FIELDS = {
    "player_id": "player_id",
    "player": "player_name",
    "position": "position",
    "h_a": "h_a",
    "time": "minutes",
    "goals": "goals",
    "own_goals": "own_goals",
    "assists": "assists",
    "shots": "shots",
    "key_passes": "key_passes",
    "yellow_card": "yellow_cards",
    "red_card": "red_cards",
    "xG": "xg",
    "xA": "xa",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
}

INT_COLUMNS = [
    "minutes",
    "goals",
    "own_goals",
    "assists",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
]
FLOAT_COLUMNS = ["xg", "xa", "xg_chain", "xg_buildup"]

#: How often to log progress during the long run.
PROGRESS_EVERY = 50


def parse_match_rosters(payload: str, match_id: str) -> pd.DataFrame:
    """Parse one ``getMatchData`` response into per-player rows.

    Args:
        payload: Raw JSON response body.
        match_id: Understat match ID, stored on every row.

    Returns:
        One row per player who appeared, both sides combined.

    Raises:
        UnderstatFormatError: If the payload is not JSON or has no ``rosters`` block.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UnderstatFormatError(
            f"Match {match_id}: response was not valid JSON. The endpoint contract has "
            f"probably changed - re-inspect understat.com/getMatchData/{match_id}."
        ) from exc

    rosters = data.get("rosters")
    if not isinstance(rosters, dict) or not rosters:
        raise UnderstatFormatError(
            f"Match {match_id}: no 'rosters' block; keys were {sorted(data)}."
        )

    rows: list[dict[str, object]] = []
    for side in rosters.values():
        if not isinstance(side, dict):
            continue
        for entry in side.values():
            rows.append({dst: entry.get(src) for src, dst in ROSTER_FIELDS.items()})

    if not rows:
        raise UnderstatFormatError(f"Match {match_id}: rosters block contained no players.")

    df = pd.DataFrame(rows)
    df["match_id"] = match_id

    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("Int64")
    for column in FLOAT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)

    return df


def collect_player_match_stats(
    session: CachedSession,
    fixtures: pd.DataFrame,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch per-player statistics for every fixture.

    Args:
        session: HTTP client providing retry, rate limiting and caching.
        fixtures: Output of ``understat.parse_fixtures`` - needs ``match_id``,
            ``home_slug``, ``away_slug``, ``datetime``, ``season``,
            ``season_start_year``.
        force_refresh: Ignore cached responses. Expensive; think before using it.

    Returns:
        One row per player-appearance across all fixtures.
    """
    if fixtures.empty:
        logger.warning("No fixtures supplied - nothing to collect.")
        return pd.DataFrame()

    current = current_season_start_year()
    total = len(fixtures)
    logger.info("Fetching player rosters for %d matches", total)

    frames: list[pd.DataFrame] = []
    for position, fixture in enumerate(fixtures.itertuples(index=False), start=1):
        url = UNDERSTAT_MATCH_URL.format(match_id=fixture.match_id)
        ttl = CURRENT_SEASON_TTL_SECONDS if fixture.season_start_year >= current else None

        try:
            payload = session.get_text(
                url,
                ttl=ttl,
                force_refresh=force_refresh,
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        except NotFoundError:
            logger.warning("Match %s returned no data - skipping.", fixture.match_id)
            continue

        match_df = parse_match_rosters(payload, fixture.match_id)

        # Understat labels each player "h" or "a"; turn that into an actual club.
        match_df["team_slug"] = match_df["h_a"].map(
            {"h": fixture.home_slug, "a": fixture.away_slug}
        )
        match_df["opponent_slug"] = match_df["h_a"].map(
            {"h": fixture.away_slug, "a": fixture.home_slug}
        )
        match_df["date"] = fixture.datetime
        match_df["season"] = fixture.season
        match_df["season_start_year"] = fixture.season_start_year
        frames.append(match_df)

        if position % PROGRESS_EVERY == 0 or position == total:
            logger.info("  %d/%d matches (%.0f%%)", position, total, 100 * position / total)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
