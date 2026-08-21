"""ELO ratings for Premier League clubs.

Kept in its own module because it is the only piece of feature engineering with real
carried state, and because that makes it testable in isolation.

The ratings this produces are *pre-match* by construction: :meth:`EloEngine.rate_match`
returns the two ratings as they stood before kickoff and only then applies the update.
Nothing downstream can accidentally read a post-match rating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Rating every club starts from, and the mean that ratings regress toward.
INITIAL_RATING = 1500.0

#: Clubs entering the division start below the mean. Promoted sides are, on average,
#: materially weaker than the league; seeding them at 1500 hands them a rating they
#: have not earned and drags down whoever beats them.
PROMOTED_RATING = 1350.0

#: Home advantage in rating points, added to the home side's rating when computing the
#: expected result. ~65 corresponds to the long-run home win rate in the Premier League.
HOME_ADVANTAGE = 65.0

#: Base sensitivity of a rating update.
K_FACTOR = 20.0

#: Between seasons, ratings move back toward the mean by this proportion - squads
#: change, and last May's form is weaker evidence in August.
SEASON_REGRESSION = 0.75


def expected_score(rating: float, opponent_rating: float, home_advantage: float = 0.0) -> float:
    """Probability-like expected result for one side.

    Args:
        rating: The side's rating.
        opponent_rating: The opponent's rating.
        home_advantage: Points added to ``rating`` before comparing. Pass
            :data:`HOME_ADVANTAGE` for the home side, 0 for the away side.

    Returns:
        Expected score in ``[0, 1]``, where 1 is a certain win.
    """
    exponent = (opponent_rating - rating - home_advantage) / 400.0
    return 1.0 / (1.0 + float(10.0**exponent))


def margin_multiplier(goal_difference: int) -> float:
    """Scale the update by the margin of victory.

    A flat K treats a 5-0 exactly like a 1-0, which throws away most of what a football
    result tells us. This is the standard logarithmic damping: bigger wins move ratings
    further, with diminishing returns.

    Args:
        goal_difference: Absolute goal difference of the match.

    Returns:
        A multiplier of at least 1.0.
    """
    return 1.0 + float(math.log1p(abs(goal_difference)))


@dataclass
class EloEngine:
    """Maintains a rating per club, updated match by match in date order.

    Attributes:
        k_factor: Base update sensitivity.
        home_advantage: Rating points granted to the home side.
        initial_rating: Starting rating for a club with no history.
        promoted_rating: Starting rating for a club entering the division mid-history.
        season_regression: Proportion of a rating retained across a season boundary.
    """

    k_factor: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    initial_rating: float = INITIAL_RATING
    promoted_rating: float = PROMOTED_RATING
    season_regression: float = SEASON_REGRESSION
    ratings: dict[str, float] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)

    def rating_of(self, team: str) -> float:
        """Return a club's current rating, seeding it if this is its first appearance.

        The first clubs ever seen start at :attr:`initial_rating`; any club appearing
        later is by definition newly promoted and starts at :attr:`promoted_rating`.

        Args:
            team: Club slug.

        Returns:
            The club's current rating.
        """
        if team not in self.ratings:
            # An empty book means we are seeding the very first season, where nobody
            # has history and everyone deserves the same start.
            self.ratings[team] = self.initial_rating if not self._seen else self.promoted_rating
        return self.ratings[team]

    def start_season(self, teams: list[str]) -> None:
        """Regress existing ratings toward the mean at a season boundary.

        Args:
            teams: Club slugs taking part in the new season.
        """
        for team in list(self.ratings):
            self.ratings[team] = INITIAL_RATING + self.season_regression * (
                self.ratings[team] - INITIAL_RATING
            )
        for team in teams:
            self.rating_of(team)
        self._seen.update(teams)

    def rate_match(
        self, home: str, away: str, home_goals: int, away_goals: int
    ) -> tuple[float, float]:
        """Return pre-match ratings, then apply the result.

        Args:
            home: Home club slug.
            away: Away club slug.
            home_goals: Goals scored by the home side.
            away_goals: Goals scored by the away side.

        Returns:
            The ``(home, away)`` ratings **as they stood before this match**.
        """
        home_pre = self.rating_of(home)
        away_pre = self.rating_of(away)
        self._seen.update((home, away))

        expected_home = expected_score(home_pre, away_pre, self.home_advantage)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        change = self.k_factor * margin_multiplier(home_goals - away_goals)
        delta = change * (actual_home - expected_home)

        # Zero-sum: whatever the home side gains, the away side loses.
        self.ratings[home] = home_pre + delta
        self.ratings[away] = away_pre - delta

        return home_pre, away_pre
