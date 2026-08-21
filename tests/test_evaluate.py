"""Tests for src/models/evaluate.py - held-out scoring and decay selection."""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel, fit_dixon_coles
from src.models.evaluate import Scores, score_model, select_decay


def _matches(n_rounds: int, home_goals: int, away_goals: int, start: str) -> pd.DataFrame:
    """A run of identical results between two clubs."""
    rows = []
    date = pd.Timestamp(start)
    for _ in range(n_rounds):
        rows.append(
            {
                "date": date,
                "season": "2024/25",
                "home_slug": "alpha",
                "away_slug": "bravo",
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _flat_model() -> DixonColesModel:
    return DixonColesModel(
        teams=["alpha", "bravo"],
        attack={"alpha": 0.0, "bravo": 0.0},
        defence={"alpha": 0.0, "bravo": 0.0},
        home_advantage=0.2,
        rho=0.0,
    )


# ----------------------------------------------------------------------------------
# score_model
# ----------------------------------------------------------------------------------


def test_score_model_returns_all_three_metrics():
    scores = score_model(_flat_model(), _matches(5, 2, 1, "2024-08-17"))

    assert isinstance(scores, Scores)
    assert scores.n_matches == 5
    assert scores.log_likelihood < 0  # a log probability
    assert scores.log_loss > 0
    assert 0 <= scores.brier <= 2


def test_a_model_that_expects_the_actual_results_scores_better():
    """The whole point of the metric: a better-fitting model must score higher."""
    observed = _matches(12, 3, 0, "2024-08-17")
    fitted = fit_dixon_coles(observed)

    good = score_model(fitted, observed)
    bad = score_model(_flat_model(), observed)

    assert good.log_likelihood > bad.log_likelihood
    assert good.log_loss < bad.log_loss
    assert good.brier < bad.brier


def test_score_model_skips_clubs_the_model_never_saw():
    """A promoted club in the held-out set has no rating; it is skipped, not crashed on."""
    held_out = pd.concat(
        [
            _matches(2, 1, 1, "2024-08-17"),
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2024-09-01"),
                        "season": "2024/25",
                        "home_slug": "newcomer",
                        "away_slug": "alpha",
                        "home_goals": 0,
                        "away_goals": 2,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    scores = score_model(_flat_model(), held_out)
    assert scores.n_matches == 2


def test_score_model_raises_when_nothing_can_be_scored():
    unknown = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-09-01"),
                "season": "2024/25",
                "home_slug": "nobody",
                "away_slug": "nobody-else",
                "home_goals": 1,
                "away_goals": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="No held-out matches"):
        score_model(_flat_model(), unknown)


def test_scores_are_averages_not_sums():
    """Otherwise a larger held-out set would look worse purely for being larger."""
    small = score_model(_flat_model(), _matches(3, 1, 1, "2024-08-17"))
    large = score_model(_flat_model(), _matches(30, 1, 1, "2024-08-17"))

    assert small.log_likelihood == pytest.approx(large.log_likelihood, abs=1e-9)
    assert small.n_matches == 3
    assert large.n_matches == 30


def test_high_scoring_matches_are_less_likely_than_typical_ones():
    model = _flat_model()
    typical = score_model(model, _matches(4, 1, 1, "2024-08-17"))
    absurd = score_model(model, _matches(4, 7, 6, "2024-08-17"))

    assert absurd.log_likelihood < typical.log_likelihood


# ----------------------------------------------------------------------------------
# select_decay
# ----------------------------------------------------------------------------------


def test_select_decay_returns_a_candidate_from_the_grid():
    train = pd.concat(
        [_matches(10, 3, 0, "2023-08-17"), _matches(10, 0, 3, "2024-01-10")], ignore_index=True
    )
    validation = _matches(6, 0, 3, "2024-05-01")
    grid = (0.0, 0.002, 0.01)

    best, scores = select_decay(train, validation, grid=grid)

    assert best in grid
    assert set(scores) == set(grid)


def test_select_decay_picks_the_highest_log_likelihood():
    train = pd.concat(
        [_matches(10, 3, 0, "2023-08-17"), _matches(10, 0, 3, "2024-01-10")], ignore_index=True
    )
    validation = _matches(6, 0, 3, "2024-05-01")
    grid = (0.0, 0.002, 0.01)

    best, scores = select_decay(train, validation, grid=grid)

    assert scores[best].log_likelihood == max(s.log_likelihood for s in scores.values())


def test_heavier_decay_wins_when_recent_form_reversed():
    """Old results say alpha dominates; recent ones say bravo does. Decay should help."""
    train = pd.concat(
        [_matches(20, 4, 0, "2022-08-17"), _matches(20, 0, 4, "2024-02-01")], ignore_index=True
    )
    validation = _matches(8, 0, 4, "2024-05-01")

    best, _ = select_decay(train, validation, grid=(0.0, 0.005))
    assert best > 0.0, "ignoring stale evidence should have scored better"


def test_select_decay_rejects_an_unscoreable_validation_set():
    train = _matches(5, 1, 0, "2024-08-17")
    unrelated = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01"),
                "season": "2024/25",
                "home_slug": "nobody",
                "away_slug": "nobody-else",
                "home_goals": 1,
                "away_goals": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="No decay candidate"):
        select_decay(train, unrelated, grid=(0.0,))


def test_select_decay_scores_every_candidate():
    train = _matches(10, 2, 1, "2024-08-17")
    validation = _matches(4, 2, 1, "2025-01-01")
    grid = (0.0, 0.001, 0.002, 0.005)

    _, scores = select_decay(train, validation, grid=grid)

    assert len(scores) == len(grid)
    assert all(np.isfinite(s.log_likelihood) for s in scores.values())
