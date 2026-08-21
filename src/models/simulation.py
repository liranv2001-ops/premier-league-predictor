"""Monte Carlo simulation of a season's remaining fixtures.

Every remaining match is sampled from its Dixon-Coles joint score matrix, points are
added to whatever each club has already earned, and the resulting table is ranked. Run
10,000 times, the spread of finishing positions is the answer.

Sampling is vectorised per fixture rather than per simulation: one fixture's 10,000
outcomes come from a single `searchsorted` against the flattened score matrix. Looping
over simulations in Python would turn seconds into minutes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.config import (
    DRAW_POINTS,
    LEAGUE_SIZE,
    N_SIMULATIONS,
    RANDOM_SEED,
    WIN_POINTS,
)
from src.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)


@dataclass
class Standings:
    """Points, goals scored and goals conceded already banked before the cutoff.

    Attributes:
        teams: Club slugs, in a fixed order used by every array here.
        points: Points per club.
        scored: Goals scored per club.
        conceded: Goals conceded per club.
        played: Matches played per club.
    """

    teams: list[str]
    points: np.ndarray
    scored: np.ndarray
    conceded: np.ndarray
    played: np.ndarray

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, teams: list[str]) -> Standings:
        """Build standings from played matches.

        Args:
            matches: Played matches, possibly empty.
            teams: Every club in the season, including any with no matches yet.

        Returns:
            The standings implied by those matches.
        """
        index = {team: i for i, team in enumerate(teams)}
        points = np.zeros(len(teams))
        scored = np.zeros(len(teams))
        conceded = np.zeros(len(teams))
        played = np.zeros(len(teams))

        for match in matches.itertuples(index=False):
            home, away = index[match.home_slug], index[match.away_slug]
            home_goals, away_goals = int(match.home_goals), int(match.away_goals)

            scored[home] += home_goals
            conceded[home] += away_goals
            scored[away] += away_goals
            conceded[away] += home_goals
            played[home] += 1
            played[away] += 1

            if home_goals > away_goals:
                points[home] += WIN_POINTS
            elif home_goals < away_goals:
                points[away] += WIN_POINTS
            else:
                points[home] += DRAW_POINTS
                points[away] += DRAW_POINTS

        return cls(teams, points, scored, conceded, played)


@dataclass
class SimulationResult:
    """Aggregated output of a Monte Carlo run.

    Attributes:
        teams: Club slugs, in array order.
        position_counts: ``(n_teams, LEAGUE_SIZE)`` count of finishes per position.
        mean_points: Average final points per club.
        n_simulations: How many runs were aggregated.
    """

    teams: list[str]
    position_counts: np.ndarray
    mean_points: np.ndarray
    n_simulations: int

    @property
    def position_probabilities(self) -> np.ndarray:
        """Return the position distribution per club, each row summing to 1."""
        return self.position_counts / self.n_simulations

    @property
    def title_probability(self) -> np.ndarray:
        """Return each club's probability of finishing first."""
        return self.position_probabilities[:, 0]

    @property
    def mean_position(self) -> np.ndarray:
        """Return each club's average finishing position."""
        positions = np.arange(1, LEAGUE_SIZE + 1)
        return self.position_probabilities @ positions


def build_remaining_fixtures(teams: list[str], played: pd.DataFrame) -> pd.DataFrame:
    """Return the fixtures still to be played in a double round-robin.

    Args:
        teams: Every club in the season.
        played: Matches already played, with ``home_slug`` and ``away_slug``.

    Returns:
        One row per outstanding fixture.

    Note:
        When nothing has been played this is the complete 380-match schedule. Its
        *order* is irrelevant to the final table - each club meets each other club home
        and away regardless - which is why a real fixture list is not needed to simulate
        a season from scratch.
    """
    already = set(zip(played["home_slug"], played["away_slug"], strict=True))
    rows = [
        {"home_slug": home, "away_slug": away}
        for home in teams
        for away in teams
        if home != away and (home, away) not in already
    ]
    return pd.DataFrame(rows, columns=["home_slug", "away_slug"])


def _sample_fixture(
    model: DixonColesModel,
    home: str,
    away: str,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw scorelines for one fixture across every simulation.

    Args:
        model: The fitted model.
        home: Home club slug.
        away: Away club slug.
        n_simulations: Number of draws.
        rng: Seeded random generator.

    Returns:
        ``(home_goals, away_goals)``, each of length ``n_simulations``.
    """
    matrix = model.score_matrix(home, away)
    size = matrix.shape[0]

    # Flatten to a 1-D categorical distribution, then invert its CDF in one shot. This
    # is the vectorisation that keeps 380 x 10,000 draws in the seconds range.
    cdf = np.cumsum(matrix.ravel())
    draws = rng.random(n_simulations)
    flat = np.searchsorted(cdf, draws, side="right")
    np.clip(flat, 0, matrix.size - 1, out=flat)

    return flat // size, flat % size


def simulate_season(
    model: DixonColesModel,
    teams: list[str],
    remaining: pd.DataFrame,
    standings: Standings,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> SimulationResult:
    """Simulate the rest of a season many times over.

    Args:
        model: Fitted Dixon-Coles model covering every club in ``teams``.
        teams: Every club in the season.
        remaining: Fixtures still to play.
        standings: Points and goals already banked.
        n_simulations: Number of Monte Carlo runs.
        seed: Random seed, so runs are reproducible.

    Returns:
        The aggregated result.
    """
    rng = np.random.default_rng(seed)
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    points = np.tile(standings.points, (n_simulations, 1))
    scored = np.tile(standings.scored, (n_simulations, 1))
    conceded = np.tile(standings.conceded, (n_simulations, 1))

    for fixture in remaining.itertuples(index=False):
        home, away = index[fixture.home_slug], index[fixture.away_slug]
        home_goals, away_goals = _sample_fixture(
            model, fixture.home_slug, fixture.away_slug, n_simulations, rng
        )

        scored[:, home] += home_goals
        conceded[:, home] += away_goals
        scored[:, away] += away_goals
        conceded[:, away] += home_goals

        home_win = home_goals > away_goals
        away_win = away_goals > home_goals
        draw = ~home_win & ~away_win

        points[:, home] += home_win * WIN_POINTS + draw * DRAW_POINTS
        points[:, away] += away_win * WIN_POINTS + draw * DRAW_POINTS

    goal_difference = scored - conceded

    # Premier League tie-breaks: points, then goal difference, then goals scored. Ties
    # surviving all three are broken by a per-simulation coin flip - closer to the truth
    # (a playoff) than sorting alphabetically, which would hand clubs early in the
    # alphabet a systematic edge in every tied season.
    #
    # lexsort takes its *last* key as primary and is exact. Packing the criteria into a
    # single scaled float instead would put the tie-break term down at the limit of
    # float64 precision once points reach three digits.
    jitter = rng.random((n_simulations, n_teams))
    order = np.lexsort((jitter, -scored, -goal_difference, -points), axis=1)

    # `order` says which club sits in each rank slot; argsort of that inverts the
    # mapping into each club's own position.
    positions = np.argsort(order, axis=1)

    position_counts = np.zeros((n_teams, LEAGUE_SIZE), dtype=np.int64)
    for team_index in range(n_teams):
        position_counts[team_index] = np.bincount(positions[:, team_index], minlength=LEAGUE_SIZE)[
            :LEAGUE_SIZE
        ]

    logger.info("Simulated %d seasons over %d remaining fixtures", n_simulations, len(remaining))
    return SimulationResult(
        teams=teams,
        position_counts=position_counts,
        mean_points=points.mean(axis=0),
        n_simulations=n_simulations,
    )
