"""Tests for the collector modules that had no direct coverage.

Everything here is offline. Network calls go through a stub session exposing the same
`get_text` surface as `CachedSession`, matching the pattern in `test_data_collection.py`.

The disambiguation guards get the most attention, because they are the ones whose
failure mode is silent: a wrong badge or the wrong player's face still renders perfectly.
"""

import json

import pandas as pd
import pytest
from requests import ConnectTimeout

from src.data_collection import football_data_uk, thesportsdb, understat, wikimedia
from src.data_collection.storage import read_table, write_table
from src.data_collection.understat import (
    UnderstatFormatError,
    parse_fixtures,
    parse_team_history,
)
from src.data_collection.understat_matches import parse_match_rosters


class StubSession:
    """A `CachedSession` stand-in that replays canned responses.

    Args:
        responses: Substring of the URL -> response body. The first substring found in
            a requested URL wins, so tests can key on the distinctive part of an
            endpoint rather than reconstructing the whole query string.
    """

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def get_text(self, url: str, **_kwargs: object) -> str:
        self.requested.append(url)
        for fragment, body in self.responses.items():
            if fragment in url:
                return body
        return "{}"


# ----------------------------------------------------------------------------------
# understat_matches.parse_match_rosters
# ----------------------------------------------------------------------------------


def _match_payload(**overrides: object) -> str:
    """A getMatchData response with two players per side."""

    def player(pid: str, side: str, minutes: int, goals: int) -> dict[str, object]:
        return {
            "id": pid,
            "player_id": pid,
            "player": f"Player {pid}",
            "position": "FW",
            "h_a": side,
            "time": str(minutes),
            "goals": str(goals),
            "own_goals": "0",
            "assists": "0",
            "shots": "3",
            "key_passes": "1",
            "yellow_card": "0",
            "red_card": "0",
            "xG": "0.42",
            "xA": "0.11",
            "xGChain": "0.5",
            "xGBuildup": "0.2",
        }

    payload: dict[str, object] = {
        "rosters": {
            "h": {"1": player("1", "h", 90, 2), "2": player("2", "h", 90, 0)},
            "a": {"3": player("3", "a", 90, 1), "4": player("4", "a", 45, 0)},
        },
        "shots": {},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_match_rosters_returns_both_sides():
    df = parse_match_rosters(_match_payload(), "999")

    assert len(df) == 4
    assert set(df["h_a"]) == {"h", "a"}
    assert (df["match_id"] == "999").all()


def test_parse_match_rosters_casts_numerics():
    """Understat sends every number as a string; uncast, sums and ranks are nonsense."""
    df = parse_match_rosters(_match_payload(), "999")

    assert df["goals"].dtype == "Int64"
    assert df["xg"].dtype == float
    assert df["goals"].sum() == 3
    assert df["minutes"].sum() == 315


@pytest.mark.parametrize(
    "payload",
    ["not json at all", json.dumps({"shots": {}}), json.dumps({"rosters": {}})],
)
def test_parse_match_rosters_rejects_unusable_payloads(payload):
    """Failing loudly beats writing an empty table nobody notices."""
    with pytest.raises(UnderstatFormatError):
        parse_match_rosters(payload, "999")


def test_parse_match_rosters_fills_missing_fields_with_zero():
    payload = json.loads(_match_payload())
    del payload["rosters"]["h"]["1"]["key_passes"]
    df = parse_match_rosters(json.dumps(payload), "999")

    assert df["key_passes"].notna().all()


# ----------------------------------------------------------------------------------
# understat.parse_team_history and parse_fixtures
# ----------------------------------------------------------------------------------


def _league_payload(ppda_def: int = 24, is_result: bool = True, with_players: bool = False) -> str:
    players = (
        [
            {
                "id": "1",
                "player_name": "Someone",
                "team_title": "Arsenal",
                "position": "FW",
                "games": "10",
                "time": "900",
                "goals": "5",
                "assists": "2",
                "shots": "20",
                "key_passes": "8",
                "yellow_cards": "1",
                "red_cards": "0",
                "npg": "4",
                "xG": "4.5",
                "xA": "1.8",
                "npxG": "3.9",
                "xGChain": "6.0",
                "xGBuildup": "2.0",
            }
        ]
        if with_players
        else []
    )
    history_entry = {
        "h_a": "h",
        "xG": "1.8",
        "xGA": "0.9",
        "npxG": "1.6",
        "npxGA": "0.9",
        "npxGD": "0.7",
        "xpts": "2.1",
        "ppda": {"att": 240, "def": ppda_def},
        "ppda_allowed": {"att": 200, "def": 20},
        "deep": "7",
        "deep_allowed": "4",
        "scored": "2",
        "missed": "1",
        "pts": "3",
        "result": "w",
        "date": "2024-08-17 15:00:00",
    }
    return json.dumps(
        {
            "teams": {
                "1": {"id": "1", "title": "Arsenal", "history": [history_entry]},
                "2": {"id": "2", "title": "Chelsea", "history": [history_entry]},
            },
            "players": players,
            "dates": [
                {
                    "id": "26602",
                    "isResult": is_result,
                    "h": {"id": "1", "title": "Arsenal"},
                    "a": {"id": "2", "title": "Chelsea"},
                    "goals": {"h": "2", "a": "1"},
                    "xG": {"h": "1.8", "a": "0.9"},
                    "datetime": "2024-08-17 15:00:00",
                }
            ],
        }
    )


def test_parse_team_history_extracts_xg_and_ppda():
    df = parse_team_history(_league_payload(), 2024)

    assert len(df) == 2
    row = df.iloc[0]
    assert row["xg"] == pytest.approx(1.8)
    assert row["season"] == "2024/25"
    # PPDA is passes divided by defensive actions, not the raw pair.
    assert row["ppda"] == pytest.approx(240 / 24)


def test_parse_team_history_survives_zero_defensive_actions():
    """A match with no defensive actions logged would otherwise divide by zero."""
    df = parse_team_history(_league_payload(ppda_def=0), 2024)

    assert df["ppda"].notna().all()
    assert (df["ppda"] > 0).all()


def test_parse_team_history_maps_club_names_to_slugs():
    df = parse_team_history(_league_payload(), 2024)
    assert set(df["team_slug"]) == {"arsenal", "chelsea"}


def test_parse_fixtures_returns_match_ids():
    df = parse_fixtures(_league_payload(), 2024)

    assert list(df["match_id"]) == ["26602"]
    assert df.iloc[0]["home_slug"] == "arsenal"
    assert df.iloc[0]["away_slug"] == "chelsea"
    assert df.iloc[0]["home_goals"] == 2


def test_parse_fixtures_excludes_unplayed_matches():
    """A fixture with no result has nothing to collect - asking for it wastes a request."""
    with pytest.raises(UnderstatFormatError, match="No played fixtures"):
        parse_fixtures(_league_payload(is_result=False), 2024)


@pytest.mark.parametrize("parser", [parse_team_history, parse_fixtures])
def test_league_parsers_reject_non_json(parser):
    with pytest.raises(UnderstatFormatError):
        parser("<html>maintenance</html>", 2024)


# ----------------------------------------------------------------------------------
# thesportsdb: the sport and club guards
# ----------------------------------------------------------------------------------


SAMPLE_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS\n"
    "E0,16/08/2025,12:30,Man United,Nott'm Forest,2,1,H,1,0,H,14,8\n"
    "E0,16/08/2025,15:00,Wolves,Man City,0,3,A,0,1,A,6,19\n"
)


def _teams_response(*teams: dict[str, str]) -> str:
    return json.dumps({"teams": list(teams)})


def test_badge_search_skips_the_wrong_sport():
    """Searching "Nottingham Forest" really does return a netball club first."""
    session = StubSession(
        {
            "searchteams": _teams_response(
                {"strTeam": "Nottingham Forest", "strSport": "Netball", "strBadge": "netball.png"},
                {"strTeam": "Nottingham Forest", "strSport": "Soccer", "strBadge": "football.png"},
            )
        }
    )
    assert thesportsdb.find_team_badge(session, "Nottingham Forest") == "football.png"


def test_badge_search_returns_none_when_only_other_sports_match():
    session = StubSession(
        {
            "searchteams": _teams_response(
                {"strTeam": "Leeds Force", "strSport": "Basketball", "strBadge": "hoops.png"}
            )
        }
    )
    assert thesportsdb.find_team_badge(session, "Leeds") is None


def test_badge_search_tries_the_alias_first():
    """ "Leeds" finds a basketball team; "Leeds United" finds the football club."""
    session = StubSession({"searchteams": _teams_response()})
    thesportsdb.find_team_badge(session, "Leeds", "leeds")

    assert "Leeds United" in session.requested[0], session.requested


def test_player_photo_rejects_a_namesake_at_another_club():
    """The Trabzonspor "Mohamed Salah" must not end up on Liverpool's award card."""
    session = StubSession(
        {
            "searchplayers": json.dumps(
                {
                    "player": [
                        {
                            "strPlayer": "Mohamed Salah",
                            "strSport": "Soccer",
                            "strTeam": "Trabzonspor",
                            "strCutout": "wrong-man.png",
                        }
                    ]
                }
            )
        }
    )
    assert thesportsdb.find_player_photo(session, "Mohamed Salah", "liverpool") is None


def test_player_photo_accepts_a_match_at_the_expected_club():
    session = StubSession(
        {
            "searchplayers": json.dumps(
                {
                    "player": [
                        {
                            "strPlayer": "Mohamed Salah",
                            "strSport": "Soccer",
                            "strTeam": "Liverpool",
                            "strCutout": "right-man.png",
                        }
                    ]
                }
            )
        }
    )
    assert thesportsdb.find_player_photo(session, "Mohamed Salah", "liverpool") == "right-man.png"


# ----------------------------------------------------------------------------------
# wikimedia: the club verification guard
# ----------------------------------------------------------------------------------


def _article_response(title: str, image: str, intro: str) -> str:
    return json.dumps(
        {"query": {"pages": {"1": {"title": title, "pageimage": image, "extract": intro}}}}
    )


def test_article_lookup_rejects_a_namesake_whose_intro_omits_the_club():
    """Plain "Thiago" redirects to Thiago Alcantara, who never played for Brentford."""
    session = StubSession(
        {
            "titles=Thiago&": _article_response(
                "Thiago Alcantara",
                "alcantara.jpg",
                "Thiago Alcantara is a retired midfielder who played for Liverpool.",
            ),
            "list=search": json.dumps({"query": {"search": []}}),
        }
    )
    assert wikimedia.find_article_image(session, "Thiago", "Brentford") is None


def test_article_lookup_accepts_when_the_intro_names_the_club():
    session = StubSession(
        {
            "titles": _article_response(
                "Igor Thiago",
                "igor.jpg",
                "Igor Thiago is a Brazilian forward who plays for Brentford.",
            )
        }
    )
    assert wikimedia.find_article_image(session, "Thiago", "Brentford") == (
        "igor.jpg",
        "Igor Thiago",
    )


def test_article_lookup_uses_the_most_specific_club_token():
    """ "Manchester" would match Manchester City too, so verification uses "United"."""
    session = StubSession(
        {
            "titles": _article_response(
                "Someone Else",
                "someone.jpg",
                "Someone Else plays for Manchester City.",
            ),
            "list=search": json.dumps({"query": {"search": []}}),
        }
    )
    assert wikimedia.find_article_image(session, "Someone", "Manchester United") is None


def test_article_lookup_returns_none_when_there_is_no_lead_image():
    session = StubSession(
        {"titles": json.dumps({"query": {"pages": {"1": {"title": "Someone", "extract": "x"}}}})}
    )
    assert wikimedia.find_article_image(session, "Someone", "Arsenal") is None


# ----------------------------------------------------------------------------------
# Network tolerance on the in-progress season
# ----------------------------------------------------------------------------------


class FlakySession:
    """Serves cached bodies but raises a network error for one season."""

    def __init__(self, fails_on: str, body: str) -> None:
        self.fails_on = fails_on
        self.body = body

    def get_text(self, url: str, **_kwargs: object) -> str:
        if self.fails_on in url:
            raise ConnectTimeout("connection timed out")
        return self.body


def test_understat_survives_a_network_failure_on_the_current_season(monkeypatch):
    """Completed seasons never expire from cache, so only the live season needs the net.

    Losing an otherwise fully cached run because of one blip on a season that is
    expected to be empty anyway is the wrong trade - and it would fail the weekly
    workflow for no reason.
    """
    monkeypatch.setattr(understat, "current_season_start_year", lambda: 2026)
    session = FlakySession("EPL/2026", _league_payload(with_players=True))

    league = understat.collect_league_data(session, [2024, 2026])

    assert not league.players.empty, "the cached season should still have been parsed"
    assert set(league.players["season"]) == {"2024/25"}


def test_understat_still_fails_on_a_completed_season(monkeypatch):
    """A historical season must be in cache; a network error there is a real failure."""
    monkeypatch.setattr(understat, "current_season_start_year", lambda: 2026)
    session = FlakySession("EPL/2024", _league_payload(with_players=True))

    with pytest.raises(ConnectTimeout):
        understat.collect_league_data(session, [2024])


def test_football_data_survives_a_network_failure_on_the_current_season(monkeypatch, tmp_path):
    monkeypatch.setattr(football_data_uk, "FOOTBALL_DATA_UK_DIR", tmp_path)
    monkeypatch.setattr(football_data_uk, "current_season_start_year", lambda: 2026)

    session = FlakySession("2627", SAMPLE_CSV)
    df = football_data_uk.collect_matches(session, [2025, 2026])

    assert len(df) == 2
    assert set(df["season"]) == {"2025/26"}


def test_football_data_still_fails_on_a_completed_season(monkeypatch, tmp_path):
    monkeypatch.setattr(football_data_uk, "FOOTBALL_DATA_UK_DIR", tmp_path)
    monkeypatch.setattr(football_data_uk, "current_season_start_year", lambda: 2026)

    session = FlakySession("2223", SAMPLE_CSV)
    with pytest.raises(ConnectTimeout):
        football_data_uk.collect_matches(session, [2022])


# ----------------------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"slug": ["arsenal", "chelsea"], "points": [85, 60]})


def test_write_and_read_round_trip(tmp_path):
    db = tmp_path / "test.db"
    assert write_table(_frame(), "standings", db) == 2

    restored = read_table("standings", db)
    pd.testing.assert_frame_equal(restored, _frame())


def test_writing_twice_replaces_rather_than_appends(tmp_path):
    """Re-runs must be idempotent; appending would silently double every table."""
    db = tmp_path / "test.db"
    write_table(_frame(), "standings", db)
    write_table(_frame(), "standings", db)

    assert len(read_table("standings", db)) == 2


def test_empty_frame_is_skipped(tmp_path):
    db = tmp_path / "test.db"
    assert write_table(pd.DataFrame(), "standings", db) == 0
    assert not db.exists()


def test_write_creates_the_parent_directory(tmp_path):
    db = tmp_path / "nested" / "deeper" / "test.db"
    assert write_table(_frame(), "standings", db) == 2
    assert db.exists()
