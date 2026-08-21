"""HTTP client with retries, per-host rate limiting and an on-disk response cache.

The cache is what keeps this project inside any source's tolerance: completed seasons
are immutable, so their responses are stored once and never re-fetched. A full cold run
is ~12 requests; a warm run is 1 (the season in progress).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_collection.config import (
    CACHE_DIR,
    DEFAULT_MIN_INTERVAL_SECONDS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)

#: Transient statuses worth retrying. 429 is included because football-data.co.uk
#: really does return it under repeated access.
RETRY_STATUSES = (429, 500, 502, 503, 504)


class NotFoundError(Exception):
    """Raised on a 404, so callers can treat a missing file as an expected outcome."""


@dataclass
class RateLimiter:
    """Enforces a minimum interval between requests to the same host.

    Attributes:
        min_interval: Minimum seconds between two requests to one host.
    """

    min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS
    _last_call: dict[str, float] = field(default_factory=dict)

    def wait(self, host: str) -> None:
        """Sleep as long as needed before the next request to ``host``.

        Args:
            host: Hostname being requested.
        """
        previous = self._last_call.get(host)
        now = time.monotonic()
        if previous is not None:
            elapsed = now - previous
            if elapsed < self.min_interval:
                delay = self.min_interval - elapsed
                logger.debug("Rate limit: sleeping %.2fs before %s", delay, host)
                time.sleep(delay)
        self._last_call[host] = time.monotonic()


class CachedSession:
    """A ``requests`` session with retry, rate limiting and an on-disk cache.

    Args:
        cache_dir: Directory for cached response bodies and metadata.
        min_interval: Minimum seconds between requests to the same host.
        user_agent: Value sent as ``User-Agent``; identifies this project.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(min_interval=min_interval)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=5,
            connect=3,
            read=3,
            backoff_factor=1.5,
            status_forcelist=RETRY_STATUSES,
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ----------------------------------------------------------------------------------
    # Cache plumbing
    # ----------------------------------------------------------------------------------

    def _paths_for(self, url: str) -> tuple[Path, Path]:
        """Return the body and metadata paths for a URL.

        Args:
            url: Request URL.

        Returns:
            A ``(body_path, meta_path)`` tuple.
        """
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.cache", self.cache_dir / f"{digest}.meta.json"

    def _read_cache(self, url: str, ttl: float | None) -> str | None:
        """Return the cached body for ``url`` if it is still valid.

        Args:
            url: Request URL.
            ttl: Seconds the entry stays fresh, or ``None`` to never expire.

        Returns:
            The cached body, or ``None`` on a miss or expiry.
        """
        body_path, meta_path = self._paths_for(url)
        if not (body_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            return None
        if ttl is not None and time.time() - float(meta.get("fetched_at", 0)) > ttl:
            logger.debug("Cache expired for %s", url)
            return None
        return body_path.read_text(encoding="utf-8")

    def _write_cache(self, url: str, body: str, etag: str | None) -> None:
        """Persist a response body and its metadata.

        Args:
            url: Request URL.
            body: Response text.
            etag: ``ETag`` header, if the server sent one.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        body_path, meta_path = self._paths_for(url)
        body_path.write_text(body, encoding="utf-8")
        meta_path.write_text(
            json.dumps({"url": url, "fetched_at": time.time(), "etag": etag}),
            encoding="utf-8",
        )

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------

    def get_text(
        self,
        url: str,
        *,
        ttl: float | None = None,
        force_refresh: bool = False,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Fetch ``url`` as text, using the cache when possible.

        Args:
            url: Request URL.
            ttl: Cache lifetime in seconds. ``None`` means never expire, which is
                correct for completed seasons.
            force_refresh: Bypass the cache and re-fetch.
            headers: Extra request headers.

        Returns:
            The response body.

        Raises:
            NotFoundError: If the server returned 404.
            requests.HTTPError: For any other unsuccessful status.
        """
        if not force_refresh:
            cached = self._read_cache(url, ttl)
            if cached is not None:
                logger.info("CACHE HIT  %s", url)
                return cached

        host = urlparse(url).netloc
        self.limiter.wait(host)
        logger.info("FETCH      %s", url)
        response = self.session.get(url, timeout=30, headers=headers)

        # football-data.co.uk answers a missing file with "300 Multiple Choices" and an
        # HTML page of near-matches, not a 404. requests follows real redirects itself,
        # so any remaining 3xx here means there is no document to read.
        if response.status_code == 404 or 300 <= response.status_code < 400:
            raise NotFoundError(f"{response.status_code} - no document at {url}")
        response.raise_for_status()

        self._write_cache(url, response.text, response.headers.get("ETag"))
        return response.text

    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()

    def __enter__(self) -> CachedSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
