"""Configuration, paths, season arithmetic and team-name normalisation.

Secrets are read from the environment via ``python-dotenv``. No source used today
requires a key; the helpers exist for future keyed sources (e.g. TheSportsDB for
club crests and player photos).
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / ".cache"
FOOTBALL_DATA_UK_DIR = RAW_DIR / "football_data_uk"
UNDERSTAT_DIR = RAW_DIR / "understat"
RAW_DB_PATH = RAW_DIR / "premier_league_raw.db"

# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------

FOOTBALL_DATA_UK_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Understat serves the league page as a client-side app; this is the XHR endpoint its
# own league.min.js calls. Returns {"teams": ..., "players": ..., "dates": ...}.
UNDERSTAT_URL = "https://understat.com/getLeagueData/EPL/{year}"

# Per-match player rosters. One request per match, so ~1,900 for five seasons - the
# only expensive endpoint in the project. Match IDs come from the league endpoint.
UNDERSTAT_MATCH_URL = "https://understat.com/getMatchData/{match_id}"

USER_AGENT = "premier-league-predictor/0.1 (personal ML project; contact via GitHub)"

#: Minimum seconds between requests to the same host. Understat is an undocumented
#: endpoint, so we stay well below anything that could look like hammering.
DEFAULT_MIN_INTERVAL_SECONDS = 3.0

#: Completed seasons never change, so their responses are cached forever. Only the
#: in-progress season needs re-fetching.
CURRENT_SEASON_TTL_SECONDS = 12 * 60 * 60

#: A Premier League season starts in August.
SEASON_START_MONTH = 8


def get_optional_token(name: str) -> str | None:
    """Read an optional API token from the environment.

    Args:
        name: Environment variable name, e.g. ``"THESPORTSDB_KEY"``.

    Returns:
        The token, or ``None`` if it is unset or blank. Never returns a literal
        default - secrets belong in ``.env``, which is git-ignored.
    """
    value = os.getenv(name)
    return value.strip() or None if value else None


def current_season_start_year(today: date | None = None) -> int:
    """Return the start year of the season in progress.

    Seasons are labelled by the calendar year they start in, so 2025/26 is "2025".
    Between January and July the season in progress started the *previous* year.

    Args:
        today: Date to evaluate. Defaults to the real current date.

    Returns:
        The four-digit start year of the current season.
    """
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def season_start_years(n_completed: int, today: date | None = None) -> list[int]:
    """Return the last ``n_completed`` finished seasons plus the current one.

    Args:
        n_completed: How many completed seasons to include.
        today: Date to evaluate. Defaults to the real current date.

    Returns:
        Ascending start years, ending with the season currently in progress.
    """
    current = current_season_start_year(today)
    return list(range(current - n_completed, current + 1))


def season_code(start_year: int) -> str:
    """Convert a start year to football-data.co.uk's ``YYZZ`` code.

    Args:
        start_year: Season start year, e.g. ``2025``.

    Returns:
        The four-character code, e.g. ``"2526"``.
    """
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Return the human-readable season label, e.g. ``"2025/26"``.

    Args:
        start_year: Season start year.

    Returns:
        The label used in the ``season`` column of every stored table.
    """
    return f"{start_year}/{(start_year + 1) % 100:02d}"


# --------------------------------------------------------------------------------------
# Team names
# --------------------------------------------------------------------------------------

#: The two sources spell the same clubs differently ("Man United" vs "Manchester
#: United"). Every alias maps to one canonical slug so the match and player tables can
#: be joined, and so asset filenames line up with `assets/logos/{slug}.png`.
TEAM_SLUGS: dict[str, str] = {
    "arsenal": "arsenal",
    "aston villa": "aston-villa",
    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",
    "brentford": "brentford",
    "brighton": "brighton",
    "brighton & hove albion": "brighton",
    "burnley": "burnley",
    "chelsea": "chelsea",
    "coventry": "coventry",
    "coventry city": "coventry",
    "crystal palace": "crystal-palace",
    "everton": "everton",
    "fulham": "fulham",
    "hull": "hull",
    "hull city": "hull",
    "ipswich": "ipswich",
    "ipswich town": "ipswich",
    "leeds": "leeds",
    "leeds united": "leeds",
    "leicester": "leicester",
    "leicester city": "leicester",
    "liverpool": "liverpool",
    "luton": "luton",
    "luton town": "luton",
    "man city": "manchester-city",
    "manchester city": "manchester-city",
    "man united": "manchester-united",
    "manchester united": "manchester-united",
    "newcastle": "newcastle-united",
    "newcastle united": "newcastle-united",
    "norwich": "norwich",
    "norwich city": "norwich",
    "nott'm forest": "nottingham-forest",
    "nottingham forest": "nottingham-forest",
    "sheffield united": "sheffield-united",
    "southampton": "southampton",
    "sunderland": "sunderland",
    "tottenham": "tottenham",
    "tottenham hotspur": "tottenham",
    "watford": "watford",
    "west brom": "west-brom",
    "west bromwich albion": "west-brom",
    "west ham": "west-ham",
    "west ham united": "west-ham",
    "wolves": "wolves",
    "wolverhampton wanderers": "wolves",
}


class UnknownTeamError(KeyError):
    """Raised when a club name has no slug mapping.

    Promoted clubs appear every season, and silently slugifying an unknown name would
    split one club across two identities and break the match/player join. Failing here
    forces the mapping to be updated.
    """


def normalise_team(name: str) -> str:
    """Map a club name from either source to its canonical slug.

    Args:
        name: Club name as it appears in the source data.

    Returns:
        The canonical slug, e.g. ``"manchester-united"``.

    Raises:
        UnknownTeamError: If the name is not in :data:`TEAM_SLUGS`.
    """
    key = re.sub(r"\s+", " ", name.strip().lower())
    try:
        return TEAM_SLUGS[key]
    except KeyError:
        raise UnknownTeamError(
            f"No slug for club {name!r}. Add it to TEAM_SLUGS in "
            f"src/data_collection/config.py (likely a newly promoted club)."
        ) from None


def normalise_teams(name: str) -> list[str]:
    """Map a possibly multi-club field to slugs, in order.

    Understat reports every club a player represented during the season as one
    comma-separated string (``"Burnley,Newcastle United"`` for a January transfer), and
    the statistics are season totals across all of them.

    Args:
        name: One or more club names, comma-separated.

    Returns:
        Slugs in source order; the last entry is the player's most recent club.

    Raises:
        UnknownTeamError: If any club name is unmapped.
    """
    parts = [part for part in name.split(",") if part.strip()]
    if not parts:
        raise UnknownTeamError(f"Empty club field: {name!r}")
    return [normalise_team(part) for part in parts]
