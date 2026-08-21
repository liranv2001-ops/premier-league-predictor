"""Model evaluation: held-out likelihood, outcome scoring and decay selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.config import DECAY_GRID
from src.models.dixon_coles import DixonColesModel, fit_dixon_coles

logger = logging.getLogger(__name__)


@dataclass
class Scores:
    """Held-out scores for one fitted model.

    Attributes:
        log_likelihood: Mean per-match log-likelihood of the observed scoreline.
        log_loss: Mean negative log probability of the observed home/draw/away outcome.
        brier: Mean multi-class Brier score over the same three outcomes.
        n_matches: Matches scored.
    """

    log_likelihood: float
    log_loss: float
    brier: float
    n_matches: int


def score_model(model: DixonColesModel, matches: pd.DataFrame) -> Scores:
    """Score a fitted model against matches it was not trained on.

    Args:
        model: The fitted model.
        matches: Held-out matches.

    Returns:
        Log-likelihood, log-loss and Brier score.
    """
    log_likelihoods: list[float] = []
    log_losses: list[float] = []
    briers: list[float] = []

    for match in matches.itertuples(index=False):
        if match.home_slug not in model.attack or match.away_slug not in model.attack:
            continue

        matrix = model.score_matrix(match.home_slug, match.away_slug)
        home_goals = min(int(match.home_goals), matrix.shape[0] - 1)
        away_goals = min(int(match.away_goals), matrix.shape[1] - 1)
        log_likelihoods.append(float(np.log(matrix[home_goals, away_goals])))

        probabilities = np.array(model.outcome_probabilities(match.home_slug, match.away_slug))
        if match.home_goals > match.away_goals:
            actual = np.array([1.0, 0.0, 0.0])
        elif match.home_goals == match.away_goals:
            actual = np.array([0.0, 1.0, 0.0])
        else:
            actual = np.array([0.0, 0.0, 1.0])

        log_losses.append(float(-np.log(probabilities[actual.argmax()])))
        briers.append(float(np.sum((probabilities - actual) ** 2)))

    if not log_likelihoods:
        raise ValueError("No held-out matches could be scored - no club overlap.")

    return Scores(
        log_likelihood=float(np.mean(log_likelihoods)),
        log_loss=float(np.mean(log_losses)),
        brier=float(np.mean(briers)),
        n_matches=len(log_likelihoods),
    )


def select_decay(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    grid: tuple[float, ...] = DECAY_GRID,
) -> tuple[float, dict[float, Scores]]:
    """Pick the time-decay rate that scores best on held-out matches.

    Args:
        train: Matches to fit on.
        validation: Matches to score against.
        grid: Candidate decay rates.

    Returns:
        The best decay rate and every candidate's scores.
    """
    scores: dict[float, Scores] = {}
    for decay in grid:
        model = fit_dixon_coles(train, decay=decay)
        try:
            scores[decay] = score_model(model, validation)
        except ValueError:
            logger.warning("Decay %.4f could not be scored - skipping.", decay)

    if not scores:
        raise ValueError("No decay candidate could be scored.")

    best = max(scores, key=lambda d: scores[d].log_likelihood)
    logger.info(
        "Decay selection: best=%.4f (held-out log-likelihood %.4f)",
        best,
        scores[best].log_likelihood,
    )
    return best, scores
