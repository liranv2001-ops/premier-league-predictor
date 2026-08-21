"""Collect per-player season statistics from Understat.

Understat's league page is a client-side app: the data is not in the HTML. Its own
``league.min.js`` fetches ``/getLeagueData/{league}/{season}``, which returns JSON with
``teams``, ``players`` and ``dates`` keys. That endpoint is what this module calls -
one request per season.

Historically the data was embedded in the page as ``var playersData = JSON.parse(...)``.
That pattern no longer exists as of August 2026.
"""

from __future__ import annotations

import json
import logging
from typing import NamedTuple

import pandas as pd

from src.data_collection.config import (
    CURRENT_SEASON_TTL_SECONDS,
    UNDERSTAT_DIR,
    UNDERSTAT_URL,
    current_season_start_year,
    normalise_teams,
    season_label,
)
from src.data_collection.http_client import CachedSession, NotFoundError

logger = logging.getLogger(__name__)

#: Source field -> stored column. ``time`` is minutes played, which the awards models
#: need in order to ignore low-minute outliers.
FIELD_MAP = {
    "id": "player_id",
    "player_name": "player_name",
    "team_title": "team_name",
    "position": "position",
    "games": "games",
    "time": "minutes",
    "goals": "goals",
    "assists": "assists",
    "shots": "shots",
    "key_passes": "key_passes",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "npg": "non_penalty_goals",
    "xG": "xg",
    "xA": "xa",
    "npxG": "non_penalty_xg",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
}

INT_COLUMNS = [
    "games",
    "minutes",
    "goals",
    "assists",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "non_penalty_goals",
]

FLOAT_COLUMNS = ["xg", "xa", "non_penalty_xg", "xg_chain", "xg_buildup"]

#: Per-match team fields from ``teams[].history``. These arrive in the same response as
#: the player totals, so extracting them costs no extra requests.
HISTORY_FLOAT_FIELDS = {
    "xG": "xg",
    "xGA": "xga",
    "npxG": "npxg",
    "npxGA": "npxga",
    "xpts": "xpts",
    "npxGD": "npxgd",
}
#: Last-resort PPDA when a payload has no usable defensive actions anywhere. Premier
#: League PPDA sits around 10-13; this is a neutral mid-table value, not a real
#: measurement, and only ever applies when the season median is unavailable too.
DEFAULT_PPDA = 11.0

HISTORY_INT_FIELDS = {
    "scored": "scored",
    "missed": "missed",
    "deep": "deep",
    "deep_allowed": "deep_allowed",
    "pts": "points",
}


class UnderstatFormatError(RuntimeError):
    """Raised when the endpoint's response is not in the expected shape.

    Understat is an undocumented endpoint with no stability guarantee. Failing loudly
    here is deliberate: a silent empty table would poison the awards models without any
    obvious symptom.
    """


def parse_players(payload: str, start_year: int) -> pd.DataFrame:
    """Parse a ``getLeagueData`` response into a player statistics table.

    Args:
        payload: Raw JSON response body.
        start_year: Season start year, used for the ``season`` column.

    Returns:
        One row per player, with canonical team slugs and numeric dtypes.

    Raises:
        UnderstatFormatError: If the payload is not JSON or has no ``players`` list.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UnderstatFormatError(
            "Understat response was not valid JSON. The endpoint contract has probably "
            "changed - re-inspect https://understat.com/js/league.min.js."
        ) from exc

    players = data.get("players")
    if not isinstance(players, list) or not players:
        raise UnderstatFormatError(
            f"Understat returned no 'players' list for {season_label(start_year)}. "
            f"Top-level keys were: {sorted(data)}."
        )

    df = pd.DataFrame(players)

    missing = set(FIELD_MAP) - set(df.columns)
    if missing:
        raise UnderstatFormatError(
            f"Understat response is missing expected fields: {sorted(missing)}."
        )

    df = df[list(FIELD_MAP)].rename(columns=FIELD_MAP)

    # Every numeric arrives as a string; without the cast, "29" > "3" is False and the
    # top-scorer ranking silently sorts lexicographically.
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    df["season"] = season_label(start_year)
    df["season_start_year"] = start_year

    # A player transferred mid-season appears once, with season totals and every club
    # listed. The list is alphabetical, not chronological, so it cannot tell us which
    # club is the most recent - `team_slug` is therefore left null for these players
    # rather than guessing wrong. `team_slugs` keeps the full list and `n_teams` flags
    # them, so downstream code can decide explicitly.
    slug_lists = df["team_name"].map(normalise_teams)
    df["team_slugs"] = slug_lists.map("|".join)
    df["n_teams"] = slug_lists.map(len).astype("Int64")
    df["team_slug"] = slug_lists.map(lambda slugs: slugs[0] if len(slugs) == 1 else None).astype(
        "string"
    )
    df["player_slug"] = (
        df["player_name"]
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
    )

    return df.reset_index(drop=True)


def parse_team_history(payload: str, start_year: int) -> pd.DataFrame:
    """Parse per-match team statistics from a ``getLeagueData`` response.

    ``teams[].history`` carries xG, xGA, PPDA and expected points for every match a
    team played. It arrives in the same response as the player totals, so this is free.

    Args:
        payload: Raw JSON response body.
        start_year: Season start year.

    Returns:
        One row per team-match (roughly 760 per season).

    Raises:
        UnderstatFormatError: If the payload has no usable ``teams`` block.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UnderstatFormatError("Understat response was not valid JSON.") from exc

    teams = data.get("teams")
    if not isinstance(teams, dict) or not teams:
        raise UnderstatFormatError(
            f"No 'teams' block for {season_label(start_year)}; keys were {sorted(data)}."
        )

    rows: list[dict[str, object]] = []
    for team in teams.values():
        slug = normalise_teams(team["title"])[0]
        for entry in team.get("history", []):
            row: dict[str, object] = {
                "team_slug": slug,
                "team_name": team["title"],
                "date": entry["date"],
                "h_a": entry["h_a"],
                "result": entry["result"],
            }
            for src, dst in HISTORY_FLOAT_FIELDS.items():
                row[dst] = float(entry[src])
            for src, dst in HISTORY_INT_FIELDS.items():
                row[dst] = int(entry[src])
            # PPDA arrives as {"att": passes, "def": defensive actions}; the metric is
            # the ratio. A team with no defensive actions would divide by zero.
            for src, dst in (("ppda", "ppda"), ("ppda_allowed", "ppda_allowed")):
                pair = entry.get(src) or {}
                defence = float(pair.get("def", 0) or 0)
                row[dst] = float(pair.get("att", 0)) / defence if defence else float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = season_label(start_year)
    df["season_start_year"] = start_year

    # A handful of matches genuinely have zero defensive actions logged. Fill with the
    # season median rather than leaving a NaN to propagate into the feature tables.
    #
    # The median itself is NaN when *every* row is affected, so fall back to a constant
    # after it. Without that second step a single bad payload puts NaN into
    # `match_features`, which is documented as NaN-free by construction.
    for column in ("ppda", "ppda_allowed"):
        df[column] = df[column].fillna(df[column].median()).fillna(DEFAULT_PPDA)

    return df.reset_index(drop=True)


def parse_fixtures(payload: str, start_year: int) -> pd.DataFrame:
    """Parse the fixture list, including Understat's match IDs.

    The IDs are what ``understat_matches`` needs to fetch per-match player rosters.

    Args:
        payload: Raw JSON response body.
        start_year: Season start year.

    Returns:
        One row per *played* match. Unplayed fixtures are excluded.

    Raises:
        UnderstatFormatError: If the payload has no ``dates`` list.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UnderstatFormatError("Understat response was not valid JSON.") from exc

    dates = data.get("dates")
    if not isinstance(dates, list) or not dates:
        raise UnderstatFormatError(
            f"No 'dates' list for {season_label(start_year)}; keys were {sorted(data)}."
        )

    rows = [
        {
            "match_id": entry["id"],
            "home_slug": normalise_teams(entry["h"]["title"])[0],
            "away_slug": normalise_teams(entry["a"]["title"])[0],
            "datetime": entry["datetime"],
            "home_goals": int(entry["goals"]["h"]),
            "away_goals": int(entry["goals"]["a"]),
        }
        for entry in dates
        if entry.get("isResult")
    ]
    if not rows:
        raise UnderstatFormatError(f"No played fixtures for {season_label(start_year)}.")

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["season"] = season_label(start_year)
    df["season_start_year"] = start_year
    return df.reset_index(drop=True)


class LeagueData(NamedTuple):
    """The three tables extracted from the league endpoint."""

    players: pd.DataFrame
    team_history: pd.DataFrame
    fixtures: pd.DataFrame


def collect_league_data(
    session: CachedSession,
    start_years: list[int],
    *,
    force_refresh: bool = False,
) -> LeagueData:
    """Download the league endpoint once per season and parse all three tables from it.

    Args:
        session: HTTP client providing retry, rate limiting and caching.
        start_years: Season start years to fetch, ascending.
        force_refresh: Ignore cached responses.

    Returns:
        Player totals, per-match team statistics and the fixture list.
    """
    current = current_season_start_year()
    UNDERSTAT_DIR.mkdir(parents=True, exist_ok=True)
    players: list[pd.DataFrame] = []
    history: list[pd.DataFrame] = []
    fixtures: list[pd.DataFrame] = []

    for year in start_years:
        url = UNDERSTAT_URL.format(year=year)
        ttl = CURRENT_SEASON_TTL_SECONDS if year >= current else None
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/EPL/{year}",
        }

        try:
            payload = session.get_text(url, ttl=ttl, force_refresh=force_refresh, headers=headers)
        except NotFoundError:
            if year >= current:
                logger.warning(
                    "No Understat data yet for %s - the season has barely started.",
                    season_label(year),
                )
                continue
            raise

        try:
            season_players = parse_players(payload, year)
            season_history = parse_team_history(payload, year)
            season_fixtures = parse_fixtures(payload, year)
        except UnderstatFormatError:
            if year >= current:
                logger.warning("Understat has no data for %s yet - skipping.", season_label(year))
                continue
            raise

        season_players.to_csv(UNDERSTAT_DIR / f"players_{year}.csv", index=False)
        logger.info(
            "  %s: %d players, %d team-matches, %d fixtures",
            season_label(year),
            len(season_players),
            len(season_history),
            len(season_fixtures),
        )
        players.append(season_players)
        history.append(season_history)
        fixtures.append(season_fixtures)

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return LeagueData(_concat(players), _concat(history), _concat(fixtures))
