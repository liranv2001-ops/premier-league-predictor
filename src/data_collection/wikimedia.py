"""Fetch player photos from Wikimedia Commons under verified free licences.

Two rules shape this module.

**Identify the person, then take their image - never search for images by text.**
Searching Commons for "Mohamed Salah" returns *Mohamed Salah (football manager)*, a
different man, and misses Bukayo Saka entirely. Resolving the player's English Wikipedia
article and taking its lead image gets the right person every time, because the article
is *about* them. This is the same failure the TheSportsDB collector hit with a
Trabzonspor namesake, and the same principle fixes it.

**The licence check is an allowlist that fails closed.** An unrecognised or missing
licence is a rejection, not a shrug. Wikipedia hosts non-free images under fair use, so
"it was on Wikipedia" is not evidence that something may be redistributed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from src.data_collection.club_badges import PLACEHOLDER_NAME, PLAYER_DIR
from src.data_collection.http_client import CachedSession, NotFoundError

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

PLAYER_MAPPING = PLAYER_DIR / "mapping.json"

#: Wikimedia's user-agent policy requires something descriptive and contactable.
USER_AGENT = (
    "premier-league-predictor/0.1 (personal ML learning project; "
    "https://github.com/ - contact via repository) python-requests"
)

#: Wikimedia is generous but shared infrastructure; one request per second is polite.
MIN_INTERVAL_SECONDS = 1.0

#: Width to request from Commons' thumbnailer.
#:
#: The originals are full-resolution press photographs - Haaland's is 2.9 MB, and the
#: eleven candidates together came to 16.5 MB for images the dashboard renders at 72px.
#: 400px still looks sharp on a 3x display and cuts the payload by roughly 95%.
THUMBNAIL_WIDTH = 400

#: Licence prefixes that permit redistribution. Matched case-insensitively against
#: Commons' ``LicenseShortName``.
#:
#: An allowlist, deliberately. A blocklist would let an unrecognised or newly-invented
#: licence string through by default, and the failure mode there is republishing
#: someone's copyrighted photograph.
FREE_LICENCE_PREFIXES = (
    "cc0",
    "cc by",
    "cc-by",
    "public domain",
    "pd-",
    "no restrictions",
)

#: Creative Commons modifiers that make a licence **not** free, even though the string
#: still opens with "CC BY". NonCommercial forbids commercial reuse and NoDerivatives
#: forbids adaptation, so neither qualifies as open - a prefix check alone would wave
#: "CC BY-NC 4.0" straight through.
NON_FREE_MODIFIERS = frozenset({"nc", "nd"})

#: Article title patterns tried in order. Footballers frequently share a name with
#: someone more famous, so the disambiguated forms matter.
TITLE_PATTERNS = (
    "{name}",
    "{name} (footballer)",
    "{name} (football forward)",
    "{name} (football midfielder)",
)


class Photo(NamedTuple):
    """A verified, free-licensed photograph."""

    filename: str
    url: str
    licence: str
    licence_url: str
    author: str
    article: str


def is_free_licence(licence: str | None) -> bool:
    """Whether a Commons licence string permits redistribution.

    Args:
        licence: The ``LicenseShortName`` value, or ``None``.

    Returns:
        ``True`` only for a recognised free licence.
    """
    if not licence:
        return False
    normalised = licence.strip().lower()

    # Reject NC/ND before the prefix check: "CC BY-NC 4.0" opens exactly like a free
    # licence and is not one.
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalised) if token}
    if tokens & NON_FREE_MODIFIERS:
        return False

    return any(normalised.startswith(prefix) for prefix in FREE_LICENCE_PREFIXES)


def _strip_html(value: str) -> str:
    """Reduce Commons' HTML author field to plain text.

    Args:
        value: Raw HTML, often an anchor tag.

    Returns:
        The visible text.
    """
    return re.sub(r"<[^>]+>", "", value).strip()


def _query(session: CachedSession, api: str, params: dict[str, str]) -> dict[str, Any]:
    """Run a MediaWiki API query.

    Args:
        session: HTTP client.
        api: API endpoint.
        params: Query parameters.

    Returns:
        The decoded payload, or an empty dict if unusable.
    """
    query = "&".join(f"{key}={value}" for key, value in params.items())
    try:
        body = session.get_text(f"{api}?{query}", ttl=None, headers={"User-Agent": USER_AGENT})
    except NotFoundError, OSError:
        return {}
    try:
        return dict(json.loads(body))
    except json.JSONDecodeError:
        logger.warning("Unparseable response from %s", api)
        return {}


def _article(session: CachedSession, title: str) -> tuple[str, str, str] | None:
    """Fetch one article's lead image and intro text.

    Args:
        session: HTTP client.
        title: Article title to look up; redirects are followed.

    Returns:
        ``(resolved title, image filename, intro text)``, or ``None`` if the article is
        missing or has no lead image.
    """
    from urllib.parse import quote

    data = _query(
        session,
        WIKIPEDIA_API,
        {
            "action": "query",
            "format": "json",
            "titles": quote(title),
            "prop": "pageimages|extracts",
            "piprop": "name",
            "explaintext": "1",
            "exintro": "1",
            "redirects": "1",
        },
    )
    pages = (data.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return None

    for page in pages.values():
        if not isinstance(page, dict) or "missing" in page:
            continue
        image = page.get("pageimage")
        if not image:
            continue
        return str(page.get("title", title)), str(image), str(page.get("extract", ""))
    return None


def _search_titles(session: CachedSession, query: str, limit: int = 4) -> list[str]:
    """Search Wikipedia for candidate article titles.

    Args:
        session: HTTP client.
        query: Search text.
        limit: How many titles to return.

    Returns:
        Article titles, best match first.
    """
    from urllib.parse import quote

    data = _query(
        session,
        WIKIPEDIA_API,
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": quote(query),
            "srlimit": str(limit),
        },
    )
    results = (data.get("query") or {}).get("search") or []
    return [str(row["title"]) for row in results if isinstance(row, dict) and "title" in row]


def find_article_image(
    session: CachedSession, player_name: str, club: str
) -> tuple[str, str] | None:
    """Resolve a player's Wikipedia article and take its lead image.

    The club is used to **verify** the article is about the right person, not merely to
    find it. A mononym is the dangerous case: plain "Thiago" redirects to Thiago
    Alcantara, a retired midfielder who never played for Brentford. Requiring the club to
    appear in the article's own introduction rejects that, at the cost of one extra
    cached request.

    Args:
        session: HTTP client.
        player_name: The player's name.
        club: Club the player is expected to play for.

    Returns:
        ``(image filename, article title)``, or ``None`` if nothing could be verified.
    """
    # The club's distinctive word - "Manchester United" verifies on "Manchester", which
    # would also match Manchester City, so prefer the most specific token.
    club_tokens = [token for token in re.split(r"[\s\-']+", club) if len(token) > 3]
    needle = club_tokens[-1].lower() if club_tokens else club.lower()

    candidates = [pattern.format(name=player_name) for pattern in TITLE_PATTERNS]
    candidates += _search_titles(session, f"{player_name} {club} footballer")

    seen: set[str] = set()
    for title in candidates:
        if title in seen:
            continue
        seen.add(title)

        article = _article(session, title)
        if not article:
            continue
        resolved, image, intro = article

        if needle not in intro.lower():
            logger.info(
                "Rejecting article %r for %s: intro does not mention %s",
                resolved,
                player_name,
                club,
            )
            continue
        return image, resolved

    logger.info("No verified Wikipedia article with a lead image for %r (%s)", player_name, club)
    return None


def describe_file(session: CachedSession, filename: str) -> tuple[str, str, str, str] | None:
    """Look up a Commons file's URL, licence and author.

    Args:
        session: HTTP client.
        filename: File name without the ``File:`` prefix.

    Returns:
        ``(url, licence, licence_url, author)``, or ``None`` if unavailable.
    """
    from urllib.parse import quote

    data = _query(
        session,
        COMMONS_API,
        {
            "action": "query",
            "format": "json",
            "titles": quote(f"File:{filename}"),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": str(THUMBNAIL_WIDTH),
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        },
    )
    pages = (data.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return None

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = (meta.get("LicenseShortName") or {}).get("value")
        # Prefer the thumbnail; fall back to the original if the thumbnailer declined.
        source_url = info.get("thumburl") or info.get("url")
        if not source_url:
            continue
        return (
            str(source_url),
            str(licence or ""),
            str((meta.get("LicenseUrl") or {}).get("value", "")),
            _strip_html(str((meta.get("Artist") or {}).get("value", ""))) or "Unknown",
        )
    return None


def find_free_photo(session: CachedSession, player_name: str, club: str) -> Photo | None:
    """Find a free-licensed photograph of a player.

    Two independent gates: the article must be verifiably about this player at this
    club, and the image must carry a licence on the free allowlist. Either one failing
    sends the player to the placeholder.

    Args:
        session: HTTP client.
        player_name: The player's name.
        club: Club the player is expected to play for.

    Returns:
        The verified photo, or ``None`` if none qualified.
    """
    resolved = find_article_image(session, player_name, club)
    if not resolved:
        return None
    filename, article = resolved

    described = describe_file(session, filename)
    if not described:
        logger.info("Could not read licence metadata for %r", filename)
        return None
    url, licence, licence_url, author = described

    if not is_free_licence(licence):
        # Wikipedia hosts non-free images under fair use; that is not redistributable.
        logger.info(
            "Rejecting %r for %s: licence %r is not on the free allowlist",
            filename,
            player_name,
            licence or "(none)",
        )
        return None

    return Photo(
        filename=filename,
        url=url,
        licence=licence,
        licence_url=licence_url,
        author=author,
        article=f"https://en.wikipedia.org/wiki/{article.replace(' ', '_')}",
    )


def download(session: CachedSession, photo: Photo, destination: Path) -> bool:
    """Download a photo to disk.

    Args:
        session: HTTP client, used for its rate limiter.
        photo: The verified photo.
        destination: Where to write it.

    Returns:
        Whether the file was written.
    """
    session.limiter.wait("upload.wikimedia.org")
    try:
        response = session.session.get(photo.url, timeout=30, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except OSError as exc:
        logger.warning("Could not download %s: %s", photo.url, exc)
        return False

    if not response.content:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    logger.info(
        "Saved %s (%d KB, %s)", destination.name, len(response.content) // 1024, photo.licence
    )
    return True


def collect_player_photos(candidates: dict[str, tuple[str, str]]) -> dict[str, object]:
    """Fetch a free-licensed photo for each candidate and record attribution.

    Args:
        candidates: Player slug -> ``(display name, club)``.

    Returns:
        The mapping payload, including the ``placeholders`` list.
    """
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)
    players: dict[str, dict[str, str]] = {}
    placeholders: list[dict[str, str]] = []

    with CachedSession(min_interval=MIN_INTERVAL_SECONDS) as session:
        for slug, (name, club) in candidates.items():
            photo = find_free_photo(session, name, club)
            if photo is None:
                placeholders.append(
                    {
                        "player": name,
                        "slug": slug,
                        "club": club,
                        "path": f"assets/players/{PLACEHOLDER_NAME}",
                        "reason": (
                            "No photograph could be confirmed as this player under a free licence."
                        ),
                    }
                )
                continue

            extension = Path(photo.filename).suffix.lower() or ".jpg"
            destination = PLAYER_DIR / f"{slug}{extension}"
            if not download(session, photo, destination):
                placeholders.append(
                    {
                        "player": name,
                        "slug": slug,
                        "club": club,
                        "path": f"assets/players/{PLACEHOLDER_NAME}",
                        "reason": f"Download failed for {photo.filename}.",
                    }
                )
                continue

            players[slug] = {
                "player": name,
                "club": club,
                "path": f"assets/players/{destination.name}",
                "licence": photo.licence,
                "licence_url": photo.licence_url,
                "author": photo.author,
                "source": (
                    "https://commons.wikimedia.org/wiki/File:" + photo.filename.replace(" ", "_")
                ),
                "article": photo.article,
            }

    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": (
            "Photographs from Wikimedia Commons under free licences. Attribution is "
            "required by CC BY and CC BY-SA and is displayed in the dashboard footer."
        ),
        "players": players,
        "placeholders": placeholders,
    }
    PLAYER_MAPPING.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "Verified free photos for %d/%d candidates; %d using the placeholder",
        len(players),
        len(candidates),
        len(placeholders),
    )
    for entry in placeholders:
        logger.info("  placeholder: %s - %s", entry["player"], entry["reason"])

    return payload
