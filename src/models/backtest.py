"""Backtest the season model: train on everything before season X, predict X cold.

This is deliberately harder than the mid-season check in ``score_backtest``. There, half
the season had already been played and the standings were most of the answer. Here the
model sees nothing of season X at all - only prior seasons and the list of who is in it.

**The decay hyperparameter is selected inside the training window only.**
``run_season_simulation`` selects it against ``actual_remaining``, which at a matchweek-0
cutoff is the whole season being predicted. That is unreachable in production - you never
hold the results of a season still being played - but running it as a backtest would tune
on the answer and flatter every metric. This module holds out the last training season
instead, so season X is never scored against while fitting.

Every metric is reported beside a **carry-forward baseline**: last season's table with
the promoted clubs dropped into the relegated clubs' places. An error of "3.5 places" is
meaningless on its own; against a baseline it is either a result or an embarrassment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.config import LEAGUE_SIZE, N_SIMULATIONS, RANDOM_SEED
from src.models.dixon_coles import fit_dixon_coles
from src.models.evaluate import select_decay
from src.models.simulation import (
    Standings,
    build_remaining_fixtures,
    simulate_season,
)

logger = logging.getLogger(__name__)

#: A season needs at least this many prior seasons to be worth predicting - one season of
#: history cannot separate a good club from a lucky one.
MIN_TRAINING_SEASONS = 2

#: Places counted as the Champions League and relegation bands.
UCL_PLACES = 4
RELEGATION_PLACES = 3

#: The credible interval whose coverage is checked. If the model is honest, roughly this
#: share of clubs should finish inside their own band.
INTERVAL = 0.8


def final_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute the real final table for a set of matches.

    Args:
        matches: Every match of one season.

    Returns:
        One row per club with ``points``, ``goal_difference`` and ``rank``, best first.
    """
    rows: dict[str, dict[str, int]] = {}
    for match in matches.itertuples(index=False):
        for team, scored, conceded in (
            (match.home_slug, match.home_goals, match.away_goals),
            (match.away_slug, match.away_goals, match.home_goals),
        ):
            entry = rows.setdefault(team, {"points": 0, "scored": 0, "conceded": 0})
            entry["scored"] += int(scored)
            entry["conceded"] += int(conceded)
            if scored > conceded:
                entry["points"] += 3
            elif scored == conceded:
                entry["points"] += 1

    table = pd.DataFrame(
        [
            {
                "slug": slug,
                "points": entry["points"],
                "goal_difference": entry["scored"] - entry["conceded"],
                "scored": entry["scored"],
            }
            for slug, entry in rows.items()
        ]
    )
    table = table.sort_values(["points", "goal_difference", "scored"], ascending=False).reset_index(
        drop=True
    )
    table["rank"] = range(1, len(table) + 1)
    return table


def carry_forward_baseline(previous: pd.DataFrame, season_clubs: list[str]) -> dict[str, int]:
    """Predict a season by repeating the previous one's table.

    Clubs that survived keep their order; promoted clubs fill the vacated places at the
    bottom, which is where promoted clubs usually finish. This is the bar the model has
    to clear to have been worth building.

    Args:
        previous: Previous season's final table.
        season_clubs: Clubs in the season being predicted.

    Returns:
        Club slug -> predicted rank.
    """
    survivors = [slug for slug in previous.sort_values("rank")["slug"] if slug in season_clubs]
    promoted = sorted(set(season_clubs) - set(survivors))
    return {slug: rank for rank, slug in enumerate([*survivors, *promoted], start=1)}


@dataclass
class SeasonBacktest:
    """Metrics for one backtested season.

    Attributes:
        season: The season predicted.
        training_seasons: Seasons the model learned from.
        actual_champion: Who actually won.
        predicted_champion: The model's top pick by mean finishing position.
        champion_probability: Probability the model gave the actual champion.
        mean_position_error: Mean absolute difference between predicted and actual rank.
        baseline_position_error: The same for the carry-forward baseline.
        spearman: Rank correlation between predicted and actual order.
        ucl_overlap: How many of the real top four the model had in its top four.
        relegation_overlap: How many of the real bottom three it had in its bottom three.
        mean_points_error: Mean absolute error in expected points.
        interval_coverage: Share of clubs finishing inside their 80% band.
        decay: The decay chosen from the training window.
    """

    season: str
    training_seasons: list[str]
    actual_champion: str
    predicted_champion: str
    champion_probability: float
    mean_position_error: float
    baseline_position_error: float
    spearman: float
    ucl_overlap: int
    relegation_overlap: int
    mean_points_error: float
    interval_coverage: float
    decay: float
    per_club: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def champion_correct(self) -> bool:
        """Whether the model's top pick actually won."""
        return self.actual_champion == self.predicted_champion

    @property
    def beats_baseline(self) -> bool:
        """Whether the model ordered the table better than carrying last year forward."""
        return self.mean_position_error < self.baseline_position_error


def _select_decay_without_peeking(history: pd.DataFrame) -> float:
    """Choose a decay rate using only the training window.

    The last season of the training data becomes the validation split, so the season
    being predicted is never scored against.

    Args:
        history: Matches strictly before the season being predicted.

    Returns:
        The chosen decay rate.
    """
    seasons = sorted(history["season"].unique())
    if len(seasons) < 2:
        return 0.002

    validation_season = seasons[-1]
    train = history[history["season"] != validation_season]
    validation = history[history["season"] == validation_season]

    try:
        decay, _ = select_decay(train, validation)
    except ValueError:
        logger.warning("Could not select a decay rate; falling back to the default.")
        return 0.002
    return decay


def backtest_season(
    matches: pd.DataFrame,
    season: str,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> SeasonBacktest:
    """Train on everything before ``season`` and predict it from matchweek 0.

    Args:
        matches: Every match available, across all seasons.
        season: The season to predict.
        n_simulations: Monte Carlo runs.
        seed: Random seed.

    Returns:
        The season's metrics.

    Raises:
        ValueError: If there is too little history to train on.
    """
    seasons = sorted(matches["season"].unique())
    if season not in seasons:
        raise ValueError(f"{season} is not in the data.")

    training_seasons = [s for s in seasons if s < season]
    if len(training_seasons) < MIN_TRAINING_SEASONS:
        raise ValueError(
            f"{season} has only {len(training_seasons)} prior season(s); "
            f"at least {MIN_TRAINING_SEASONS} are needed."
        )

    history = matches[matches["season"].isin(training_seasons)]
    actual_matches = matches[matches["season"] == season]
    actual = final_table(actual_matches)
    season_clubs = sorted(actual["slug"])

    decay = _select_decay_without_peeking(history)
    model = fit_dixon_coles(history, decay=decay)
    last_seen = {
        str(slug): pd.to_datetime(group["date"]).max()
        for slug, group in pd.concat(
            [
                history[["home_slug", "date"]].rename(columns={"home_slug": "slug"}),
                history[["away_slug", "date"]].rename(columns={"away_slug": "slug"}),
            ]
        ).groupby("slug")
    }
    model = model.with_promoted_defaults(season_clubs, last_seen=last_seen)

    empty = matches.iloc[0:0]
    standings = Standings.from_matches(empty, season_clubs)
    fixtures = build_remaining_fixtures(season_clubs, empty)
    result = simulate_season(
        model, season_clubs, fixtures, standings, n_simulations=n_simulations, seed=seed
    )

    probabilities = result.position_probabilities
    mean_position = result.mean_position
    order = mean_position.argsort()
    predicted_rank = {result.teams[i]: rank for rank, i in enumerate(order, start=1)}

    actual_rank = dict(zip(actual["slug"], actual["rank"], strict=True))
    actual_points = dict(zip(actual["slug"], actual["points"], strict=True))

    previous = final_table(matches[matches["season"] == training_seasons[-1]])
    baseline_rank = carry_forward_baseline(previous, season_clubs)

    rows = []
    inside = 0
    for index, slug in enumerate(result.teams):
        cumulative = 0.0
        low = high = LEAGUE_SIZE
        seen_low = False
        for position in range(1, LEAGUE_SIZE + 1):
            cumulative += probabilities[index, position - 1]
            if not seen_low and cumulative >= (1 - INTERVAL) / 2:
                low = position
                seen_low = True
            if cumulative >= 1 - (1 - INTERVAL) / 2:
                high = position
                break

        finished = actual_rank[slug]
        if low <= finished <= high:
            inside += 1

        rows.append(
            {
                "slug": slug,
                "predicted_rank": predicted_rank[slug],
                "actual_rank": finished,
                "baseline_rank": baseline_rank[slug],
                "predicted_points": float(result.mean_points[index]),
                "actual_points": actual_points[slug],
                "title_probability": float(result.title_probability[index]),
                "interval_low": low,
                "interval_high": high,
            }
        )

    per_club = pd.DataFrame(rows).sort_values("actual_rank").reset_index(drop=True)

    champion = str(actual.iloc[0]["slug"])
    predicted_champion = str(result.teams[int(order[0])])
    champion_index = result.teams.index(champion)

    predicted_top4 = {slug for slug, rank in predicted_rank.items() if rank <= UCL_PLACES}
    actual_top4 = set(actual.head(UCL_PLACES)["slug"])
    predicted_bottom = {
        slug for slug, rank in predicted_rank.items() if rank > LEAGUE_SIZE - RELEGATION_PLACES
    }
    actual_bottom = set(actual.tail(RELEGATION_PLACES)["slug"])

    correlation = spearmanr(per_club["predicted_rank"], per_club["actual_rank"]).statistic

    return SeasonBacktest(
        season=season,
        training_seasons=training_seasons,
        actual_champion=champion,
        predicted_champion=predicted_champion,
        champion_probability=float(result.title_probability[champion_index]),
        mean_position_error=float(
            (per_club["predicted_rank"] - per_club["actual_rank"]).abs().mean()
        ),
        baseline_position_error=float(
            (per_club["baseline_rank"] - per_club["actual_rank"]).abs().mean()
        ),
        spearman=float(correlation),
        ucl_overlap=len(predicted_top4 & actual_top4),
        relegation_overlap=len(predicted_bottom & actual_bottom),
        mean_points_error=float(
            (per_club["predicted_points"] - per_club["actual_points"]).abs().mean()
        ),
        interval_coverage=inside / len(per_club),
        decay=decay,
        per_club=per_club,
    )


def backtestable_seasons(matches: pd.DataFrame) -> list[str]:
    """Seasons with enough history behind them to predict.

    Args:
        matches: Every match available.

    Returns:
        Season labels, oldest first.
    """
    seasons = sorted(matches["season"].unique())
    return seasons[MIN_TRAINING_SEASONS:]


def run_backtest(
    matches: pd.DataFrame,
    *,
    seasons: list[str] | None = None,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> list[SeasonBacktest]:
    """Backtest every eligible season.

    Args:
        matches: Every match available.
        seasons: Seasons to test. Defaults to every eligible one.
        n_simulations: Monte Carlo runs per season.
        seed: Random seed.

    Returns:
        One result per season.
    """
    targets = seasons or backtestable_seasons(matches)
    results = []
    for season in targets:
        logger.info("Backtesting %s", season)
        results.append(backtest_season(matches, season, n_simulations=n_simulations, seed=seed))
    return results


def summarise(results: list[SeasonBacktest]) -> dict[str, float]:
    """Aggregate metrics across backtested seasons.

    Args:
        results: Per-season results.

    Returns:
        Averages, plus the champion hit rate.
    """
    if not results:
        return {}
    return {
        "seasons": float(len(results)),
        "mean_position_error": float(np.mean([r.mean_position_error for r in results])),
        "baseline_position_error": float(np.mean([r.baseline_position_error for r in results])),
        "champion_hit_rate": float(np.mean([r.champion_correct for r in results])),
        "mean_champion_probability": float(np.mean([r.champion_probability for r in results])),
        "spearman": float(np.mean([r.spearman for r in results])),
        "ucl_overlap": float(np.mean([r.ucl_overlap for r in results])),
        "relegation_overlap": float(np.mean([r.relegation_overlap for r in results])),
        "mean_points_error": float(np.mean([r.mean_points_error for r in results])),
        "interval_coverage": float(np.mean([r.interval_coverage for r in results])),
    }
