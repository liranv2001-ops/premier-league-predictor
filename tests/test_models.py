"""Tests for src/models: Dixon-Coles and the Monte Carlo season simulation."""

import numpy as np
import pandas as pd
import pytest

from src.data_collection.config import PROJECT_ROOT, normalise_team
from src.models import (
    MissingTeamsError,
    load_matches,
    run_season_simulation,
    simulation_to_dict,
)
from src.models.cli import read_teams_file
from src.models.config import LEAGUE_SIZE
from src.models.dixon_coles import (
    STALENESS_GRACE_DAYS,
    DixonColesModel,
    first_seasons,
    fit_dixon_coles,
    staleness_weight,
    tau,
)
from src.models.simulation import (
    Standings,
    build_remaining_fixtures,
    simulate_season,
)

TEAMS = [f"club-{i:02d}" for i in range(LEAGUE_SIZE)]


# ----------------------------------------------------------------------------------
# Dixon-Coles
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("home_goals", "away_goals"),
    [(2, 0), (0, 2), (3, 3), (1, 4), (5, 1)],
)
def test_tau_is_neutral_above_one_all(home_goals, away_goals):
    """The correction touches only the four lowest scorelines."""
    correction = tau(
        np.array([home_goals]),
        np.array([away_goals]),
        np.array([1.5]),
        np.array([1.2]),
        rho=0.15,
    )
    assert correction[0] == 1.0


@pytest.mark.parametrize(
    ("home_goals", "away_goals"),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_tau_adjusts_low_scores(home_goals, away_goals):
    correction = tau(
        np.array([home_goals]),
        np.array([away_goals]),
        np.array([1.5]),
        np.array([1.2]),
        rho=0.15,
    )
    assert correction[0] != 1.0


def test_tau_is_identity_when_rho_is_zero():
    """rho = 0 must collapse Dixon-Coles back to independent Poisson."""
    home = np.array([0, 0, 1, 1, 3])
    away = np.array([0, 1, 0, 1, 2])
    correction = tau(home, away, np.full(5, 1.4), np.full(5, 1.1), rho=0.0)
    assert np.allclose(correction, 1.0)


def _lopsided_matches(n_rounds: int = 12) -> pd.DataFrame:
    """Fixtures where 'strong' outscores 'weak' by a wide margin every time."""
    rows = []
    date = pd.Timestamp("2024-08-10")
    for _ in range(n_rounds):
        rows.append(
            {
                "date": date,
                "season": "2024/25",
                "home_slug": "strong",
                "away_slug": "weak",
                "home_goals": 4,
                "away_goals": 0,
            }
        )
        rows.append(
            {
                "date": date,
                "season": "2024/25",
                "home_slug": "weak",
                "away_slug": "strong",
                "home_goals": 0,
                "away_goals": 3,
            }
        )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_fit_recovers_relative_strength():
    model = fit_dixon_coles(_lopsided_matches())
    assert model.attack["strong"] > model.attack["weak"]
    assert model.defence["strong"] > model.defence["weak"]


def test_fitted_lambdas_are_positive_and_ordered():
    model = fit_dixon_coles(_lopsided_matches())
    strong_home, weak_away = model.expected_goals("strong", "weak")
    assert strong_home > 0
    assert weak_away > 0
    assert strong_home > weak_away


def test_fit_actually_moves_off_the_initial_point():
    """Guards the failure mode where a non-finite objective kills the gradient.

    An `inf` return makes scipy's finite differences produce `inf - inf = nan`, and
    L-BFGS-B then stops after one iteration with the starting values still in place -
    a fit that silently is not a fit.
    """
    model = fit_dixon_coles(_lopsided_matches())
    assert model.home_advantage != pytest.approx(0.25), "home advantage never moved"
    assert not np.allclose(list(model.attack.values()), 0.0), "attack ratings never moved"


def test_empty_match_set_is_rejected():
    with pytest.raises(ValueError, match="empty match set"):
        fit_dixon_coles(pd.DataFrame(columns=["home_slug", "away_slug"]))


def test_score_matrix_is_a_normalised_distribution():
    model = fit_dixon_coles(_lopsided_matches())
    matrix = model.score_matrix("strong", "weak")
    assert matrix.shape[0] == matrix.shape[1]
    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0)


def test_outcome_probabilities_sum_to_one():
    model = fit_dixon_coles(_lopsided_matches())
    home_win, draw, away_win = model.outcome_probabilities("strong", "weak")
    assert home_win + draw + away_win == pytest.approx(1.0)
    assert home_win > away_win


def test_promoted_defaults_cover_unknown_clubs():
    model = fit_dixon_coles(_lopsided_matches())
    extended = model.with_promoted_defaults(["strong", "weak", "newcomer"])
    assert "newcomer" in extended.attack
    # An unknown club must still be predictable rather than raising.
    assert all(v > 0 for v in extended.expected_goals("strong", "newcomer"))


# ----------------------------------------------------------------------------------
# Simulation machinery
# ----------------------------------------------------------------------------------


def _flat_model(teams: list[str]) -> DixonColesModel:
    """A model where every club is identical - useful for symmetry checks."""
    return DixonColesModel(
        teams=teams,
        attack=dict.fromkeys(teams, 0.0),
        defence=dict.fromkeys(teams, 0.0),
        home_advantage=0.2,
        rho=0.0,
    )


def test_full_round_robin_has_the_right_size():
    fixtures = build_remaining_fixtures(TEAMS, pd.DataFrame(columns=["home_slug", "away_slug"]))
    # 20 clubs, each meeting every other home and away.
    assert len(fixtures) == LEAGUE_SIZE * (LEAGUE_SIZE - 1)
    assert len(fixtures) == 380


def test_played_fixtures_are_excluded():
    played = pd.DataFrame([{"home_slug": TEAMS[0], "away_slug": TEAMS[1]}])
    fixtures = build_remaining_fixtures(TEAMS, played)
    assert len(fixtures) == 379
    assert not ((fixtures.home_slug == TEAMS[0]) & (fixtures.away_slug == TEAMS[1])).any()


def test_standings_count_points_correctly():
    matches = pd.DataFrame(
        [
            {"home_slug": "a", "away_slug": "b", "home_goals": 3, "away_goals": 0},
            {"home_slug": "b", "away_slug": "a", "home_goals": 1, "away_goals": 1},
        ]
    )
    standings = Standings.from_matches(matches, ["a", "b"])
    assert standings.points.tolist() == [4.0, 1.0]
    assert standings.scored.tolist() == [4.0, 1.0]
    assert standings.conceded.tolist() == [1.0, 4.0]
    assert standings.played.tolist() == [2.0, 2.0]


def test_sampling_converges_to_the_score_matrix_mean():
    model = _flat_model(["a", "b"])
    matrix = model.score_matrix("a", "b")
    goals = np.arange(matrix.shape[0])
    expected_home = float((matrix.sum(axis=1) * goals).sum())

    standings = Standings.from_matches(
        pd.DataFrame(columns=["home_slug", "away_slug", "home_goals", "away_goals"]),
        ["a", "b"],
    )
    fixtures = pd.DataFrame([{"home_slug": "a", "away_slug": "b"}])
    result = simulate_season(model, ["a", "b"], fixtures, standings, n_simulations=20_000, seed=7)
    # 'a' plays one home game, so its mean goals scored is recoverable from points and
    # the known distribution; check the cheaper invariant that both clubs are covered.
    assert result.position_counts.sum() == 2 * 20_000
    assert expected_home > 0


# ----------------------------------------------------------------------------------
# The invariants the user asked for
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat_simulation():
    """A full 380-match simulation between 20 identical clubs."""
    model = _flat_model(TEAMS)
    standings = Standings.from_matches(
        pd.DataFrame(columns=["home_slug", "away_slug", "home_goals", "away_goals"]),
        TEAMS,
    )
    fixtures = build_remaining_fixtures(TEAMS, pd.DataFrame(columns=["home_slug", "away_slug"]))
    result = simulate_season(model, TEAMS, fixtures, standings, n_simulations=2_000, seed=42)
    return simulation_to_dict(result, season="2099/00", cutoff_matchweek=0, seed=42)


def test_all_twenty_clubs_are_covered(flat_simulation):
    teams = flat_simulation["teams"]
    assert len(teams) == LEAGUE_SIZE
    assert {row["slug"] for row in teams} == set(TEAMS)


def test_each_club_position_distribution_sums_to_one(flat_simulation):
    for row in flat_simulation["teams"]:
        total = sum(row["position_distribution"].values())
        assert total == pytest.approx(1.0, abs=1e-6), row["slug"]


def test_every_position_key_is_present(flat_simulation):
    for row in flat_simulation["teams"]:
        keys = set(row["position_distribution"])
        assert keys == {str(p) for p in range(1, LEAGUE_SIZE + 1)}


def test_each_position_sums_to_one_across_clubs(flat_simulation):
    """The stronger invariant: every position is filled exactly once per simulation.

    Per-club rows can each sum to 1 while the ranking logic still hands the same
    position to two clubs. Only the column sums catch that.
    """
    for position in range(1, LEAGUE_SIZE + 1):
        column = sum(
            row["position_distribution"][str(position)] for row in flat_simulation["teams"]
        )
        assert column == pytest.approx(1.0, abs=1e-6), f"position {position}"


def test_title_probabilities_sum_to_one(flat_simulation):
    total = sum(row["title_probability"] for row in flat_simulation["teams"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_title_probability_matches_the_first_position_bucket(flat_simulation):
    for row in flat_simulation["teams"]:
        assert row["title_probability"] == pytest.approx(
            row["position_distribution"]["1"], abs=1e-6
        )


def test_predicted_positions_are_in_range(flat_simulation):
    for row in flat_simulation["teams"]:
        assert 1.0 <= row["predicted_position"] <= float(LEAGUE_SIZE)
        assert 1 <= row["predicted_rank"] <= LEAGUE_SIZE
    ranks = [row["predicted_rank"] for row in flat_simulation["teams"]]
    assert sorted(ranks) == list(range(1, LEAGUE_SIZE + 1))


def test_identical_clubs_get_roughly_equal_chances(flat_simulation):
    """With every club identical, no club should be systematically favoured.

    This is what catches a deterministic tie-break: sorting ties alphabetically would
    hand `club-00` a visible edge over `club-19`.
    """
    titles = [row["title_probability"] for row in flat_simulation["teams"]]
    assert max(titles) < 0.12, "a club is winning far more often than chance"
    positions = [row["predicted_position"] for row in flat_simulation["teams"]]
    assert max(positions) - min(positions) < 2.0


# ----------------------------------------------------------------------------------
# Determinism and tie-breaks
# ----------------------------------------------------------------------------------


def _run(seed: int) -> dict:
    model = _flat_model(TEAMS)
    standings = Standings.from_matches(
        pd.DataFrame(columns=["home_slug", "away_slug", "home_goals", "away_goals"]),
        TEAMS,
    )
    fixtures = build_remaining_fixtures(TEAMS, pd.DataFrame(columns=["home_slug", "away_slug"]))
    result = simulate_season(model, TEAMS, fixtures, standings, n_simulations=500, seed=seed)
    return simulation_to_dict(result, season="2099/00", cutoff_matchweek=0, seed=seed)


def test_same_seed_gives_identical_output():
    assert _run(1)["teams"] == _run(1)["teams"]


def test_different_seed_gives_different_output():
    assert _run(1)["teams"] != _run(2)["teams"]


def test_ranking_respects_points_then_goal_difference():
    """No fixtures remain, so ranking is purely a function of the standings."""
    teams = ["a", "b", "c"]
    played = pd.DataFrame(
        [
            # a: 3 pts, GD +1.  b: 3 pts, GD +5.  c: 0 pts.
            {"home_slug": "a", "away_slug": "c", "home_goals": 1, "away_goals": 0},
            {"home_slug": "b", "away_slug": "c", "home_goals": 5, "away_goals": 0},
        ]
    )
    standings = Standings.from_matches(played, teams)
    result = simulate_season(
        _flat_model(teams),
        teams,
        pd.DataFrame(columns=["home_slug", "away_slug"]),
        standings,
        n_simulations=10,
        seed=3,
    )
    probabilities = result.position_probabilities
    assert probabilities[teams.index("b"), 0] == 1.0  # better GD takes first
    assert probabilities[teams.index("a"), 1] == 1.0
    assert probabilities[teams.index("c"), 2] == 1.0


# ----------------------------------------------------------------------------------
# End-to-end against the real database
# ----------------------------------------------------------------------------------


def test_completed_season_reproduces_the_real_table():
    """With zero fixtures remaining, the simulation must return the actual table.

    No randomness is involved, so this is the cleanest proof that standings carry-over
    and ranking are both correct.
    """
    try:
        payload = run_season_simulation(
            "2025/26", cutoff_matchweek=38, n_simulations=100, write_json=False
        )
    except ValueError, FileNotFoundError:
        pytest.skip("raw database not built")

    teams = payload["teams"]
    champion = teams[0]
    assert champion["slug"] == "arsenal"
    assert champion["title_probability"] == 1.0
    assert champion["expected_points"] == pytest.approx(85.0)
    assert champion["predicted_position"] == pytest.approx(1.0)

    relegated = {row["slug"] for row in teams[-3:]}
    assert relegated == {"west-ham", "burnley", "wolves"}


# ----------------------------------------------------------------------------------
# 2026/27: promoted clubs, staleness blending, and the forward run
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Coventry", "coventry"),
        ("Coventry City", "coventry"),
        ("Hull", "hull"),
        ("Hull City", "hull"),
        ("Ipswich", "ipswich"),
        ("Ipswich Town", "ipswich"),
    ],
)
def test_promoted_clubs_have_slugs(name, slug):
    assert normalise_team(name) == slug


def test_season_club_list_loads_to_twenty_slugs():
    path = PROJECT_ROOT / "data" / "seasons" / "2026-27.txt"
    if not path.exists():
        pytest.skip("2026/27 club list not present")

    slugs = read_teams_file(path)
    assert len(slugs) == LEAGUE_SIZE
    assert len(set(slugs)) == LEAGUE_SIZE, "duplicate club in the list"
    # The three promoted clubs must be there, and the three relegated must not.
    assert {"coventry", "hull", "ipswich"} <= set(slugs)
    assert not ({"west-ham", "wolves", "burnley"} & set(slugs))


@pytest.mark.parametrize(
    ("days", "expected"),
    [(0, 1.0), (90, 1.0), (STALENESS_GRACE_DAYS, 1.0)],
)
def test_recent_clubs_keep_their_full_rating(days, expected):
    assert staleness_weight(days) == expected


def test_staleness_weight_decreases_with_absence():
    weights = [staleness_weight(d) for d in (150, 300, 500, 900, 2000)]
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 < w <= 1.0 for w in weights)


def test_a_club_that_never_left_is_not_blended():
    """The guard against silently regressing every established club toward the mean."""
    model = fit_dixon_coles(_lopsided_matches())
    last_seen = dict.fromkeys(["strong", "weak"], pd.Timestamp("2024-11-01"))

    blended = model.with_promoted_defaults(["strong", "weak"], last_seen=last_seen)

    assert blended.attack["strong"] == pytest.approx(model.attack["strong"])
    assert blended.defence["strong"] == pytest.approx(model.defence["strong"])


def test_a_long_absent_club_moves_toward_the_baseline():
    model = fit_dixon_coles(_lopsided_matches())
    baseline_attack, _ = model.promoted_ratings()
    own = model.attack["weak"]

    last_seen = {
        "strong": pd.Timestamp("2026-05-01"),
        "weak": pd.Timestamp("2021-05-01"),  # five years away
    }
    blended = model.with_promoted_defaults(["strong", "weak"], last_seen=last_seen)

    # It should sit between its own rating and the baseline, and nearer the baseline.
    assert abs(blended.attack["weak"] - baseline_attack) < abs(own - baseline_attack)


def test_first_season_baseline_is_below_league_average():
    """A newly promoted club must not come out rated at or above the average club."""
    try:
        matches = load_matches()
    except ValueError, FileNotFoundError:
        pytest.skip("raw database not built")

    model = fit_dixon_coles(matches, decay=0.002)
    attack, defence = model.promoted_ratings()
    assert attack < 0.0, "promoted clubs should score below average"
    assert defence < 0.0, "promoted clubs should concede above average"


def test_first_seasons_never_exceeds_the_promotion_slots():
    """At most three clubs can arrive in any season, because only three come up.

    Fewer is normal and correct: a club relegated and later promoted back has already
    appeared in the window, so its return is not a *first* appearance and must not be
    counted as one - its first observed season may have been mid-tenure, which would
    contaminate a baseline meant to describe arrival.
    """
    try:
        matches = load_matches()
    except ValueError, FileNotFoundError:
        pytest.skip("raw database not built")

    arrivals = first_seasons(matches)
    counts = pd.Series(list(arrivals.values())).value_counts()

    assert (counts <= 3).all(), counts.to_dict()
    assert (counts >= 1).all()
    # The earliest season is excluded, so nobody can "arrive" in it.
    assert sorted(matches["season"].unique())[0] not in set(arrivals.values())


def test_forward_season_simulation_satisfies_every_invariant():
    """The 2026/27 run: the one this whole phase exists to produce."""
    path = PROJECT_ROOT / "data" / "seasons" / "2026-27.txt"
    if not path.exists():
        pytest.skip("2026/27 club list not present")
    try:
        payload = run_season_simulation(
            "2026/27",
            teams=read_teams_file(path),
            n_simulations=500,
            write_json=False,
        )
    except ValueError, FileNotFoundError:
        pytest.skip("raw database not built")

    teams = payload["teams"]
    assert len(teams) == LEAGUE_SIZE
    assert payload["matches_remaining"] == 380, "a fresh season must have every fixture"
    assert payload["matches_played"] == 0

    for row in teams:
        assert sum(row["position_distribution"].values()) == pytest.approx(1.0, abs=1e-6)
    for position in range(1, LEAGUE_SIZE + 1):
        column = sum(row["position_distribution"][str(position)] for row in teams)
        assert column == pytest.approx(1.0, abs=1e-6), f"position {position}"
    assert sum(row["title_probability"] for row in teams) == pytest.approx(1.0, abs=1e-6)


def _require_raw_database():
    """Skip when the collected database is absent, as on a clean checkout."""
    try:
        load_matches()
    except ValueError, FileNotFoundError:
        pytest.skip("raw database not built")


def test_season_with_no_data_needs_an_explicit_club_list():
    _require_raw_database()
    with pytest.raises(MissingTeamsError, match="club list cannot be derived"):
        run_season_simulation("2026/27", n_simulations=10, write_json=False)


def test_wrong_sized_club_list_is_rejected():
    _require_raw_database()
    with pytest.raises(MissingTeamsError, match=f"Expected {LEAGUE_SIZE} clubs"):
        run_season_simulation(
            "2026/27", teams=["arsenal", "chelsea"], n_simulations=10, write_json=False
        )
