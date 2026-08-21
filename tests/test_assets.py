"""Tests for generated club badges and free-licensed player photos.

The licence tests here are compliance checks, not style checks. If
`test_every_photo_carries_a_free_licence` fails, the project is redistributing an image
it may not be allowed to redistribute.
"""

import json
import xml.etree.ElementTree as ElementTree

import pytest

from src.data_collection.club_badges import (
    CLUB_COLOURS,
    LOGO_DIR,
    LOGO_MAPPING,
    PLACEHOLDER_NAME,
    PLAYER_DIR,
    badge_svg,
    club_initials,
    placeholder_svg,
)
from src.data_collection.wikimedia import FREE_LICENCE_PREFIXES, PLAYER_MAPPING, is_free_licence
from src.models.predictions import PREDICTIONS_PATH

# ----------------------------------------------------------------------------------
# Club codes
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Manchester United", "MU"),
        ("Manchester City", "MC"),
        ("Nottingham Forest", "NF"),
        ("Crystal Palace", "CP"),
        ("Arsenal", "ARS"),
        ("Liverpool", "LIV"),
    ],
)
def test_club_initials(name, expected):
    assert club_initials(name) == expected


def test_single_word_clubs_get_three_letters_so_they_do_not_collide():
    """Two letters would make Brentford and Brighton both "BR"."""
    assert club_initials("Brentford") != club_initials("Brighton")


def test_club_initials_handles_an_empty_name():
    assert club_initials("") == "?"


# ----------------------------------------------------------------------------------
# Badge SVGs
# ----------------------------------------------------------------------------------


def test_badge_is_well_formed_xml():
    root = ElementTree.fromstring(badge_svg("arsenal", "Arsenal"))
    assert root.tag.endswith("svg")


def test_badge_contains_the_club_code_and_colours():
    svg = badge_svg("manchester-united", "Manchester United")
    primary, secondary = CLUB_COLOURS["manchester-united"]
    assert ">MU<" in svg
    assert primary in svg
    assert secondary in svg


def test_badge_states_it_is_not_the_official_crest():
    """Someone will eventually mistake these for real crests. The file says otherwise."""
    assert "NOT the club's official crest" in badge_svg("chelsea", "Chelsea")


def test_badge_generation_is_deterministic():
    """Regenerating must not churn git."""
    assert badge_svg("everton", "Everton") == badge_svg("everton", "Everton")


def test_placeholder_is_well_formed_xml():
    root = ElementTree.fromstring(placeholder_svg())
    assert root.tag.endswith("svg")


def test_every_club_in_the_colour_table_produces_a_valid_badge():
    for slug in CLUB_COLOURS:
        ElementTree.fromstring(badge_svg(slug, slug.replace("-", " ").title()))


# ----------------------------------------------------------------------------------
# The licence allowlist
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "licence",
    ["CC0", "CC BY 4.0", "CC BY-SA 4.0", "CC-BY-SA-3.0", "Public domain", "PD-old"],
)
def test_free_licences_are_accepted(licence):
    assert is_free_licence(licence)


@pytest.mark.parametrize(
    "licence",
    ["Fair use", "All rights reserved", "Non-free logo", "Copyrighted", "", None],
)
def test_non_free_licences_are_rejected(licence):
    """The allowlist must fail closed - an unknown licence is a rejection."""
    assert not is_free_licence(licence)


def test_non_commercial_licences_are_rejected():
    """CC BY-NC and CC BY-ND are not free licences despite the "CC BY" prefix."""
    for licence in ("CC BY-NC 4.0", "CC BY-NC-SA 3.0", "CC BY-ND 4.0"):
        assert not is_free_licence(licence), f"{licence} must not be treated as free"


def test_allowlist_is_not_empty():
    assert FREE_LICENCE_PREFIXES


# ----------------------------------------------------------------------------------
# The generated artefacts on disk
# ----------------------------------------------------------------------------------


def _predictions():
    if not PREDICTIONS_PATH.exists():
        pytest.skip("predictions.json not built")
    return json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def logo_mapping():
    if not LOGO_MAPPING.exists():
        pytest.skip("logo mapping not generated - run --source badges")
    return json.loads(LOGO_MAPPING.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def player_mapping():
    if not PLAYER_MAPPING.exists():
        pytest.skip("player mapping not generated - run --source photos")
    return json.loads(PLAYER_MAPPING.read_text(encoding="utf-8"))


def test_every_club_has_a_mapping_entry(logo_mapping):
    for row in _predictions()["table"]:
        assert row["slug"] in logo_mapping["logos"], f"{row['slug']} missing from the mapping"


def test_every_mapped_badge_exists_on_disk(logo_mapping):
    """The badges are git-ignored because they regenerate, so skip on a clean checkout."""
    if not any(LOGO_DIR.glob("*.svg")):
        pytest.skip("badges not generated - run --source badges")

    for row in _predictions()["table"]:
        slug = row["slug"]
        assert (LOGO_DIR / f"{slug}.svg").exists(), f"{slug}.svg not on disk"


def test_logo_mapping_records_it_is_not_official_crests(logo_mapping):
    assert "NOT official club crests" in logo_mapping["note"]


def test_placeholder_exists_on_disk():
    assert (PLAYER_DIR / PLACEHOLDER_NAME).exists()


def _candidate_slugs(predictions):
    slugs = {}
    for award in ("top_scorer", "top_assists", "player_of_the_season"):
        for row in predictions[award]["candidates"]:
            slugs[row["slug"]] = row["player"]
    return slugs


def test_every_candidate_is_either_credited_or_on_the_placeholder_list(player_mapping):
    """Never both, never neither - each player has exactly one resolution."""
    credited = set(player_mapping["players"])
    placeheld = {entry["slug"] for entry in player_mapping["placeholders"]}

    assert not (credited & placeheld), "a player cannot be both credited and a placeholder"

    for slug, name in _candidate_slugs(_predictions()).items():
        assert slug in credited or slug in placeheld, f"{name} has no photo resolution"


def test_every_photo_carries_a_free_licence(player_mapping):
    """The compliance test. A failure here means redistributing an image we may not."""
    for slug, entry in player_mapping["players"].items():
        assert is_free_licence(entry["licence"]), f"{slug}: {entry['licence']} is not free"


def test_every_photo_names_an_author_and_a_source(player_mapping):
    """CC BY and CC BY-SA require crediting the author and pointing at the original."""
    for slug, entry in player_mapping["players"].items():
        assert entry["author"].strip(), f"{slug} has no author"
        assert entry["source"].startswith("https://commons.wikimedia.org/"), slug
        assert entry["licence_url"].startswith("http"), f"{slug} has no licence URL"


def test_every_photo_file_exists(player_mapping):
    """Photos are git-ignored - the mapping is the committed record, not the images."""
    if not any(PLAYER_DIR.glob("*.jpg")):
        pytest.skip("photos not fetched - run --source photos")

    for slug, entry in player_mapping["players"].items():
        filename = entry["path"].split("/")[-1]
        assert (PLAYER_DIR / filename).exists(), f"{slug}: {filename} not on disk"


def test_placeholder_entries_explain_themselves(player_mapping):
    """The separate documented list - each entry must say why."""
    for entry in player_mapping["placeholders"]:
        assert entry["reason"].strip()
        assert entry["path"].endswith(PLACEHOLDER_NAME)
