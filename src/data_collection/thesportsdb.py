"""Fetch club badges and player photos from TheSportsDB.

Best-effort by design. The dashboard renders a monogram fallback for anything missing,
so a club or player that cannot be resolved is a logged skip rather than a failure -
much better than putting the wrong face on an award card.

Two disambiguation guards, both found the hard way:

* Searching "Nottingham Forest" returns a **netball** club first, so results are
  filtered to ``strSport == "Soccer"``.
* Searching "Mohamed Salah" returns a player at Trabzonspor. Player names are not
  unique, so the returned club is cross-checked against the club we expect.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_collection.config import (
    PROJECT_ROOT,
    UnknownTeamError,
    get_optional_token,
    normalise_team,
)
from src.data_collection.http_client import CachedSession, NotFoundError

logger = logging.getLogger(__name__)

#: TheSportsDB's public test key. Documented for open use, so the project needs no
#: registration; a real key can be supplied via THESPORTSDB_KEY in .env.
PUBLIC_TEST_KEY = "3"

API_ROOT = "https://www.thesportsdb.com/api/v1/json/{key}"
TEAM_SEARCH = API_ROOT + "/searchteams.php?t={query}"
PLAYER_SEARCH = API_ROOT + "/searchplayers.php?p={query}"

#: Only football. The database covers every sport, and several clubs share a name with
#: a netball or rugby side.
SPORT = "Soccer"

#: Free tier allows 30 requests a minute; 2s between calls stays comfortably inside it.
MIN_INTERVAL_SECONDS = 2.0

#: Search terms for clubs whose display name is not what the database indexes.
#:
#: The search returns a single result and it is often the wrong sport: "Leeds" finds a
#: basketball team, "Ipswich" a rugby side, and "Nottingham Forest" a netball club. The
#: full club name disambiguates where the short one cannot.
SEARCH_ALIASES = {
    "leeds": "Leeds United",
    "ipswich": "Ipswich Town",
    "nottingham-forest": "Nottingham Forest FC",
    "brighton": "Brighton & Hove Albion",
    "wolves": "Wolverhampton Wanderers",
    "west-ham": "West Ham United",
    "newcastle-united": "Newcastle United",
    "tottenham": "Tottenham Hotspur",
    "bournemouth": "AFC Bournemouth",
}

LOGO_DIR = PROJECT_ROOT / "frontend" / "public" / "logos"
PLAYER_DIR = PROJECT_ROOT / "frontend" / "public" / "players"


def api_key() -> str:
    """Return the configured key, or the public test key.

    Returns:
        The API key to use.
    """
    return get_optional_token("THESPORTSDB_KEY") or PUBLIC_TEST_KEY


def _search(session: CachedSession, url: str) -> dict[str, object]:
    """Run a search and parse the response.

    Args:
        session: HTTP client.
        url: Fully formed search URL.

    Returns:
        The decoded payload, or an empty dict if the response was unusable.
    """
    try:
        payload = session.get_text(url, ttl=None)
    except NotFoundError, OSError:
        return {}
    try:
        return dict(json.loads(payload))
    except json.JSONDecodeError:
        logger.warning("Unparseable response from %s", url)
        return {}


def find_team_badge(session: CachedSession, team_name: str, slug: str | None = None) -> str | None:
    """Find a club's badge URL.

    Args:
        session: HTTP client.
        team_name: Club name as the dashboard displays it.
        slug: Club slug, used to look up a better search term.

    Returns:
        The badge URL, or ``None`` if no football club matched.
    """
    queries = [team_name]
    alias = SEARCH_ALIASES.get(slug or "")
    if alias and alias != team_name:
        queries.insert(0, alias)

    for query in queries:
        data = _search(session, TEAM_SEARCH.format(key=api_key(), query=query))
        teams = data.get("teams") or []
        if not isinstance(teams, list):
            continue

        for entry in teams:
            if not isinstance(entry, dict) or entry.get("strSport") != SPORT:
                continue
            badge = entry.get("strBadge") or entry.get("strTeamBadge")
            if badge:
                return str(badge)

    logger.warning("No football badge found for %r (tried %s)", team_name, queries)
    return None


def find_player_photo(
    session: CachedSession, player_name: str, expected_team_slug: str
) -> str | None:
    """Find a player's cutout photo, verifying it is the right player.

    Args:
        session: HTTP client.
        player_name: Player name to search for.
        expected_team_slug: The club we believe the player belongs to.

    Returns:
        The photo URL, or ``None`` if nothing matched confidently.
    """
    data = _search(session, PLAYER_SEARCH.format(key=api_key(), query=player_name))
    players = data.get("player") or []
    if not isinstance(players, list):
        return None

    footballers = [
        entry for entry in players if isinstance(entry, dict) and entry.get("strSport") == SPORT
    ]

    for entry in footballers:
        team = entry.get("strTeam")
        if not team:
            continue
        try:
            if normalise_team(str(team)) != expected_team_slug:
                continue
        except UnknownTeamError:
            # The club is outside the Premier League, so it is not our player.
            continue
        photo = entry.get("strCutout") or entry.get("strThumb")
        if photo:
            return str(photo)

    if footballers:
        # A name match exists but plays elsewhere - almost certainly a different person.
        logger.info(
            "Skipping %r: found at %r, expected %r",
            player_name,
            footballers[0].get("strTeam"),
            expected_team_slug,
        )
    else:
        logger.info("No football player found for %r", player_name)
    return None


def download_image(session: CachedSession, url: str, destination: Path) -> bool:
    """Download an image to disk.

    Args:
        session: HTTP client, used only for its rate limiter and retry policy.
        url: Image URL.
        destination: Where to write it.

    Returns:
        Whether the file was written.
    """
    if destination.exists():
        logger.debug("Already have %s", destination.name)
        return True

    session.limiter.wait("r2.thesportsdb.com")
    try:
        response = session.session.get(url, timeout=30)
        response.raise_for_status()
    except OSError as exc:
        logger.warning("Could not download %s: %s", url, exc)
        return False

    if not response.content:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    logger.info("Saved %s (%d KB)", destination.name, len(response.content) // 1024)
    return True


def collect_assets(predictions: dict[str, object]) -> dict[str, int]:
    """Fetch every badge and player photo the dashboard will ask for.

    Args:
        predictions: The payload from ``data/processed/predictions.json``.

    Returns:
        Counts of badges and photos saved.
    """
    table = predictions.get("table") or []
    assert isinstance(table, list)

    #: Every player the dashboard shows: the leaders and their shortlists.
    wanted_players: dict[tuple[str, str], str] = {}
    for award in ("top_scorer", "top_assists", "player_of_the_season"):
        block = predictions.get(award) or {}
        if not isinstance(block, dict):
            continue
        for row in block.get("candidates") or []:
            if isinstance(row, dict):
                wanted_players[(str(row["slug"]), str(row["team_slug"]))] = str(row["player"])

    saved = {"logos": 0, "players": 0}
    with CachedSession(min_interval=MIN_INTERVAL_SECONDS) as session:
        logger.info("Fetching %d club badges", len(table))
        for row in table:
            assert isinstance(row, dict)
            slug, name = str(row["slug"]), str(row["team"])
            badge = find_team_badge(session, name, slug)
            if badge and download_image(session, badge, LOGO_DIR / f"{slug}.png"):
                saved["logos"] += 1

        logger.info("Fetching %d player photos", len(wanted_players))
        for (slug, team_slug), name in wanted_players.items():
            photo = find_player_photo(session, name, team_slug)
            if photo and download_image(session, photo, PLAYER_DIR / f"{slug}.png"):
                saved["players"] += 1

    logger.info(
        "Saved %d/%d badges and %d/%d player photos - anything missing falls back to a "
        "monogram in the dashboard",
        saved["logos"],
        len(table),
        saved["players"],
        len(wanted_players),
    )
    return saved
