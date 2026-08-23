"""Collect Premier League match results from football-data.co.uk.

One CSV per season (``E0.csv``), free and key-free. Each row is a single match with
full-time and half-time results plus shots, corners and cards.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from requests import RequestException

from src.data_collection.config import (
    CURRENT_SEASON_TTL_SECONDS,
    FOOTBALL_DATA_UK_DIR,
    FOOTBALL_DATA_UK_URL,
    current_season_start_year,
    normalise_team,
    season_code,
    season_label,
)
from src.data_collection.http_client import CachedSession, NotFoundError

logger = logging.getLogger(__name__)

#: Source columns we keep, mapped to snake_case. Anything else (betting odds, referee)
#: is left in the raw CSV on disk rather than in the database.
COLUMN_MAP = {
    "Date": "date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

INT_COLUMNS = [
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
]


def _parse_season_csv(text: str, start_year: int) -> pd.DataFrame:
    """Parse one season's raw CSV into a tidy match table.

    Args:
        text: Raw CSV content.
        start_year: Season start year, used for the ``season`` column.

    Returns:
        One row per played match, with canonical team slugs.
    """
    raw = pd.read_csv(io.StringIO(text))

    present = {src: dst for src, dst in COLUMN_MAP.items() if src in raw.columns}
    df = raw[list(present)].rename(columns=present)

    # Rows past the last played match are blank padding in some season files.
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    for column in INT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    df["season"] = season_label(start_year)
    df["season_start_year"] = start_year
    df["home_slug"] = df["home_team"].map(normalise_team)
    df["away_slug"] = df["away_team"].map(normalise_team)

    return df.reset_index(drop=True)


def collect_matches(
    session: CachedSession,
    start_years: list[int],
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and parse match results for the given seasons.

    The raw CSV for each season is written verbatim to ``data/raw/football_data_uk/``
    before parsing, so the untouched source is always recoverable.

    A missing file for the season in progress is expected rather than fatal - the file
    only appears once the first matches have been played.

    Args:
        session: HTTP client providing retry, rate limiting and caching.
        start_years: Season start years to fetch, ascending.
        force_refresh: Ignore cached responses.

    Returns:
        All matches concatenated, or an empty frame if nothing could be fetched.
    """
    current = current_season_start_year()
    FOOTBALL_DATA_UK_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for year in start_years:
        code = season_code(year)
        url = FOOTBALL_DATA_UK_URL.format(code=code)
        ttl = CURRENT_SEASON_TTL_SECONDS if year >= current else None

        try:
            text = session.get_text(url, ttl=ttl, force_refresh=force_refresh)
        except NotFoundError:
            if year >= current:
                logger.warning(
                    "No results file yet for %s - the season has not started or no "
                    "matches have been played.",
                    season_label(year),
                )
                continue
            raise
        except RequestException as exc:
            # Completed seasons never expire from the cache, so only the in-progress
            # season needs the network. A transient blip there must not throw away a run
            # that is otherwise fully cached.
            if year >= current:
                logger.warning(
                    "Could not reach football-data.co.uk for %s (%s) - continuing without it.",
                    season_label(year),
                    exc.__class__.__name__,
                )
                continue
            raise

        (FOOTBALL_DATA_UK_DIR / f"E0_{code}.csv").write_text(text, encoding="utf-8")

        season_df = _parse_season_csv(text, year)
        logger.info("  %s: %d matches", season_label(year), len(season_df))
        frames.append(season_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
