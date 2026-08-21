"""Tests for src/data_collection. Everything here is offline - no test hits the network."""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data_collection import config, football_data_uk, understat
from src.data_collection.config import (
    UnknownTeamError,
    current_season_start_year,
    normalise_team,
    normalise_teams,
    season_code,
    season_label,
    season_start_years,
)
from src.data_collection.http_client import CachedSession, NotFoundError, RateLimiter
from src.data_collection.understat import UnderstatFormatError, parse_players

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------------
# Season arithmetic
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 1, 15), 2025),  # mid-season, calendar year already rolled over
        (date(2026, 7, 31), 2025),  # summer break, still the old season
        (date(2026, 8, 1), 2026),  # new season starts in August
        (date(2026, 12, 31), 2026),
    ],
)
def test_current_season_start_year(today, expected):
    assert current_season_start_year(today) == expected


def test_season_start_years_includes_current_season():
    years = season_start_years(5, today=date(2026, 8, 20))
    assert years == [2021, 2022, 2023, 2024, 2025, 2026]


@pytest.mark.parametrize(
    ("year", "code", "label"),
    [(2025, "2526", "2025/26"), (2021, "2122", "2021/22"), (1999, "9900", "1999/00")],
)
def test_season_code_and_label(year, code, label):
    assert season_code(year) == code
    assert season_label(year) == label


# ----------------------------------------------------------------------------------
# Team normalisation - the join between the two sources depends on this
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("football_data_uk_name", "understat_name", "slug"),
    [
        ("Man United", "Manchester United", "manchester-united"),
        ("Man City", "Manchester City", "manchester-city"),
        ("Nott'm Forest", "Nottingham Forest", "nottingham-forest"),
        ("Newcastle", "Newcastle United", "newcastle-united"),
        ("Wolves", "Wolverhampton Wanderers", "wolves"),
    ],
)
def test_both_sources_normalise_to_the_same_slug(football_data_uk_name, understat_name, slug):
    assert normalise_team(football_data_uk_name) == slug
    assert normalise_team(understat_name) == slug


def test_normalise_team_is_case_and_whitespace_insensitive():
    assert normalise_team("  MAN   UNITED  ") == "manchester-united"


def test_unknown_team_raises_rather_than_guessing():
    # A newly promoted club must fail loudly - silently slugifying would split one
    # club across two identities and quietly break the match/player join.
    with pytest.raises(UnknownTeamError, match="No slug for club"):
        normalise_team("Wrexham")


def test_normalise_teams_splits_multi_club_field():
    assert normalise_teams("Burnley,Newcastle United") == ["burnley", "newcastle-united"]


def test_normalise_teams_rejects_unknown_member():
    with pytest.raises(UnknownTeamError):
        normalise_teams("Chelsea,Wrexham")


# ----------------------------------------------------------------------------------
# Understat parsing
# ----------------------------------------------------------------------------------


@pytest.fixture
def understat_payload():
    return (FIXTURES / "understat_epl_2024.json").read_text(encoding="utf-8")


def test_parse_players_casts_numerics(understat_payload):
    df = parse_players(understat_payload, 2024)
    salah = df.loc[df["player_name"] == "Mohamed Salah"].iloc[0]

    # The endpoint returns every number as a string. Without the cast, ranking by
    # goals would sort lexicographically and "29" would lose to "6".
    assert salah["goals"] == 29
    assert salah["assists"] == 18
    assert salah["minutes"] == 3392
    assert df["goals"].dtype == "Int64"
    assert df["xg"].dtype == float
    assert df["goals"].idxmax() == salah.name


def test_parse_players_adds_season_and_slugs(understat_payload):
    df = parse_players(understat_payload, 2024)
    salah = df.loc[df["player_name"] == "Mohamed Salah"].iloc[0]

    assert salah["season"] == "2024/25"
    assert salah["season_start_year"] == 2024
    assert salah["team_slug"] == "liverpool"
    assert salah["player_slug"] == "mohamed-salah"


def test_parse_players_leaves_team_slug_null_for_multi_club_players(understat_payload):
    df = parse_players(understat_payload, 2024)
    rashford = df.loc[df["player_name"] == "Marcus Rashford"].iloc[0]

    # Understat lists clubs alphabetically, not chronologically, so the most recent
    # club is unknowable from this endpoint. Guessing would mislabel him.
    assert rashford["team_slugs"] == "aston-villa|manchester-united"
    assert rashford["n_teams"] == 2
    assert pd.isna(rashford["team_slug"])


def test_parse_players_rejects_non_json():
    with pytest.raises(UnderstatFormatError, match="not valid JSON"):
        parse_players("<html>Not the endpoint you were looking for</html>", 2024)


def test_parse_players_rejects_empty_player_list():
    with pytest.raises(UnderstatFormatError, match="no 'players' list"):
        parse_players(json.dumps({"teams": {}, "players": [], "dates": []}), 2024)


def test_parse_players_rejects_missing_fields():
    payload = json.dumps({"players": [{"id": "1", "player_name": "Someone"}]})
    with pytest.raises(UnderstatFormatError, match="missing expected fields"):
        parse_players(payload, 2024)


# ----------------------------------------------------------------------------------
# football-data.co.uk parsing
# ----------------------------------------------------------------------------------

SAMPLE_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,B365H\n"
    "E0,16/08/2025,12:30,Man United,Nott'm Forest,2,1,H,1,0,H,14,8,1.85\n"
    "E0,16/08/2025,15:00,Wolves,Man City,0,3,A,0,1,A,6,19,7.50\n"
    ",,,,,,,,,,,,,\n"  # trailing blank padding, present in real season files
)


def test_parse_season_csv_maps_columns_and_slugs():
    df = football_data_uk._parse_season_csv(SAMPLE_CSV, 2025)

    assert len(df) == 2  # blank padding row dropped
    assert df.loc[0, "home_slug"] == "manchester-united"
    assert df.loc[0, "away_slug"] == "nottingham-forest"
    assert df.loc[0, "home_goals"] == 2
    assert df.loc[0, "season"] == "2025/26"
    assert df.loc[1, "away_slug"] == "manchester-city"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert "B365H" not in df.columns  # betting odds stay in the raw CSV only


def test_collect_matches_tolerates_missing_current_season(monkeypatch, tmp_path):
    """A missing current-season file is expected before the season starts."""
    monkeypatch.setattr(football_data_uk, "FOOTBALL_DATA_UK_DIR", tmp_path)
    monkeypatch.setattr(football_data_uk, "current_season_start_year", lambda: 2026)

    class StubSession:
        def get_text(self, url, **kwargs):
            if "2627" in url:
                raise NotFoundError("300 - no document")
            return SAMPLE_CSV

    df = football_data_uk.collect_matches(StubSession(), [2025, 2026])

    assert len(df) == 2
    assert df["season"].unique().tolist() == ["2025/26"]


def test_collect_matches_still_raises_for_a_missing_past_season(monkeypatch, tmp_path):
    monkeypatch.setattr(football_data_uk, "FOOTBALL_DATA_UK_DIR", tmp_path)
    monkeypatch.setattr(football_data_uk, "current_season_start_year", lambda: 2026)

    class StubSession:
        def get_text(self, url, **kwargs):
            raise NotFoundError("300 - no document")

    with pytest.raises(NotFoundError):
        football_data_uk.collect_matches(StubSession(), [2022])


# ----------------------------------------------------------------------------------
# HTTP client: caching and rate limiting
# ----------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, text="payload", status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {"ETag": "abc123"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("unexpected error status")


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return CachedSession(cache_dir=tmp_path / "cache", min_interval=0)


def test_cache_miss_fetches_then_hit_avoids_network(session, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse("first")

    monkeypatch.setattr(session.session, "get", fake_get)

    assert session.get_text("https://example.test/a") == "first"
    assert session.get_text("https://example.test/a") == "first"
    assert len(calls) == 1, "second call should be served from cache"


def test_expired_ttl_refetches(session, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(f"body-{len(calls)}")

    monkeypatch.setattr(session.session, "get", fake_get)

    session.get_text("https://example.test/b", ttl=100)
    # Backdate the cache entry past its TTL.
    _, meta_path = session._paths_for("https://example.test/b")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - 1000
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert session.get_text("https://example.test/b", ttl=100) == "body-2"
    assert len(calls) == 2


def test_ttl_none_never_expires(session, monkeypatch):
    """Completed seasons are immutable, so their cache entries must never expire."""
    calls = []
    monkeypatch.setattr(
        session.session, "get", lambda url, **kw: (calls.append(url), FakeResponse())[1]
    )

    session.get_text("https://example.test/c", ttl=None)
    _, meta_path = session._paths_for("https://example.test/c")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = 0  # 1970
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    session.get_text("https://example.test/c", ttl=None)
    assert len(calls) == 1


def test_force_refresh_bypasses_cache(session, monkeypatch):
    calls = []
    monkeypatch.setattr(
        session.session, "get", lambda url, **kw: (calls.append(url), FakeResponse())[1]
    )

    session.get_text("https://example.test/d")
    session.get_text("https://example.test/d", force_refresh=True)
    assert len(calls) == 2


@pytest.mark.parametrize("status", [300, 404])
def test_not_found_statuses_raise_not_found_error(session, monkeypatch, status):
    # football-data.co.uk answers a missing file with 300, not 404.
    monkeypatch.setattr(session.session, "get", lambda url, **kw: FakeResponse("<html>", status))
    with pytest.raises(NotFoundError):
        session.get_text("https://example.test/missing")


def test_rate_limiter_sleeps_between_calls_to_same_host(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    limiter = RateLimiter(min_interval=3.0)
    limiter.wait("example.test")
    limiter.wait("example.test")

    assert len(slept) == 1
    assert 0 < slept[0] <= 3.0


def test_rate_limiter_does_not_sleep_across_different_hosts(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    limiter = RateLimiter(min_interval=3.0)
    limiter.wait("a.test")
    limiter.wait("b.test")

    assert slept == []


# ----------------------------------------------------------------------------------
# Secrets
# ----------------------------------------------------------------------------------


def test_get_optional_token_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("SOME_TOKEN", raising=False)
    assert config.get_optional_token("SOME_TOKEN") is None


def test_get_optional_token_treats_blank_as_unset(monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "   ")
    assert config.get_optional_token("SOME_TOKEN") is None


def test_get_optional_token_reads_and_strips(monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "  secret  ")
    assert config.get_optional_token("SOME_TOKEN") == "secret"


def test_no_module_hardcodes_a_credential():
    """Guard the "keys live in .env" rule against future edits."""
    src_dir = Path(understat.__file__).parent
    for path in src_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for marker in ("api_key =", "apikey =", 'token = "', 'secret = "'):
            assert marker not in text, f"{path.name} looks like it hardcodes a secret"
