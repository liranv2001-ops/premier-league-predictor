"""Modelling: Dixon-Coles scorelines and Monte Carlo season simulation.

Reads ``data/processed/pl.db`` and writes a simulation JSON alongside it. Never touches
the network.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data_collection.config import RAW_DB_PATH
from src.data_collection.storage import MATCHES_TABLE, read_table
from src.features.config import MATCH_FEATURES_TABLE, PROCESSED_DB_PATH
from src.models.config import (
    LEAGUE_SIZE,
    N_SIMULATIONS,
    RANDOM_SEED,
    simulation_json_path,
)
from src.models.dixon_coles import (
    DEFAULT_COVARIATES,
    DixonColesModel,
    fit_dixon_coles,
)
from src.models.evaluate import score_model, select_decay
from src.models.simulation import (
    SimulationResult,
    Standings,
    build_remaining_fixtures,
    simulate_season,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DixonColesModel",
    "build_remaining_fixtures",
    "compare_variants",
    "fit_dixon_coles",
    "run_season_simulation",
    "simulate_season",
    "simulation_to_dict",
]

#: Matches per matchweek in a 20-club league.
MATCHES_PER_ROUND = LEAGUE_SIZE // 2

#: Matches held out for decay selection when the season has no remaining fixtures to
#: score against - one full season's worth.
HOLDOUT_MATCHES = 380

#: Used only when there is too little history to hold anything out.
DEFAULT_DECAY = 0.002


class MissingTeamsError(ValueError):
    """Raised when a season's club list is incomplete.

    Simulating a 17-club league would produce a table that looks plausible and is
    meaningless, so this is fatal rather than a warning.
    """


def display_name(slug: str) -> str:
    """Turn a club slug into a readable name.

    Args:
        slug: Club slug such as ``"manchester-united"``.

    Returns:
        A title-cased name.
    """
    return slug.replace("-", " ").title()


def load_matches(db_path: Path = RAW_DB_PATH, *, with_features: bool = False) -> pd.DataFrame:
    """Load all matches, with dates parsed and optionally joined to the feature table.

    Args:
        db_path: Raw database to read from.
        with_features: Join ``match_features`` so covariate columns are available.
            Matches with no feature row - the earliest season, which is history-only -
            are dropped, since a covariate model cannot use them.

    Returns:
        Matches ordered by date.
    """
    matches = read_table(MATCHES_TABLE, db_path)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    if with_features:
        features = read_table(MATCH_FEATURES_TABLE, PROCESSED_DB_PATH)
        features["date"] = pd.to_datetime(features["date"])
        keep = ["date", "home_slug", "away_slug"] + [
            column for pair in DEFAULT_COVARIATES for column in pair if column in features.columns
        ]
        matches = matches.merge(features[keep], on=["date", "home_slug", "away_slug"], how="inner")

    return matches.reset_index(drop=True)


def compare_variants(
    train: pd.DataFrame, validation: pd.DataFrame, decay: float
) -> dict[str, object]:
    """Fit classic and covariate-augmented Dixon-Coles and score both held out.

    Args:
        train: Matches to fit on, joined to ``match_features``.
        validation: Held-out matches.
        decay: Time-decay rate.

    Returns:
        Both variants' held-out scores and which one won.
    """
    classic = fit_dixon_coles(train, decay=decay)
    augmented = fit_dixon_coles(train, decay=decay, covariates=DEFAULT_COVARIATES)

    classic_scores = score_model(classic, validation)
    augmented_scores = score_model(augmented, validation)

    winner = (
        "with-covariates"
        if augmented_scores.log_likelihood > classic_scores.log_likelihood
        else "classic"
    )
    return {
        "classic": classic_scores,
        "with_covariates": augmented_scores,
        "winner": winner,
        "betas": dict(
            zip(augmented.covariate_names, augmented.betas.round(4).tolist(), strict=True)
        ),
    }


def last_top_flight_match(matches: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Find when each club most recently played in the division.

    Feeds the staleness blend: a club relegated two seasons ago should not carry its
    old rating forward at full strength.

    Args:
        matches: Matches with ``home_slug``, ``away_slug`` and ``date``.

    Returns:
        Club slug -> date of its latest match.
    """
    dates = pd.to_datetime(matches["date"])
    home = dates.groupby(matches["home_slug"]).max()
    away = dates.groupby(matches["away_slug"]).max()
    combined = pd.concat([home, away]).groupby(level=0).max()
    return {str(slug): timestamp for slug, timestamp in combined.items()}


def split_season(
    matches: pd.DataFrame, season: str, cutoff_matchweek: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into training history, matches played before the cutoff, and the rest.

    Args:
        matches: Every match available.
        season: Season to simulate.
        cutoff_matchweek: How many matchweeks of the season are treated as played.

    Returns:
        ``(history, played, actual_remaining)``. ``history`` is every earlier season
        plus the played part of this one - what the model may learn from.
    """
    season_matches = matches[matches["season"] == season].sort_values("date")
    earlier = matches[matches["date"] < season_matches["date"].min()]

    n_played = min(cutoff_matchweek * MATCHES_PER_ROUND, len(season_matches))
    played = season_matches.iloc[:n_played]
    actual_remaining = season_matches.iloc[n_played:]

    history = pd.concat([earlier, played], ignore_index=True)
    return history, played, actual_remaining


def simulation_to_dict(
    result: SimulationResult,
    *,
    season: str,
    cutoff_matchweek: int,
    seed: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convert a simulation result into the JSON payload.

    Args:
        result: The aggregated simulation.
        season: Season label.
        cutoff_matchweek: Matchweek the simulation started from.
        seed: Seed used.
        extra: Additional metadata to merge in.

    Returns:
        A JSON-serialisable dictionary.
    """
    probabilities = result.position_probabilities
    mean_position = result.mean_position
    title = result.title_probability

    # Rank by mean finishing position, which is the stable summary; ranking by title
    # probability alone would tie every club outside the title race at zero.
    order = mean_position.argsort()
    rank_of = {int(team_index): rank + 1 for rank, team_index in enumerate(order)}

    teams: list[dict[str, object]] = []
    for i, slug in enumerate(result.teams):
        teams.append(
            {
                "team": display_name(slug),
                "slug": slug,
                "predicted_position": round(float(mean_position[i]), 3),
                "predicted_rank": rank_of[i],
                "title_probability": round(float(title[i]), 5),
                "expected_points": round(float(result.mean_points[i]), 2),
                # Every position 1-20 is present, including zeros, so a consumer never
                # has to handle a missing bucket.
                "position_distribution": {
                    str(position + 1): round(float(probabilities[i, position]), 5)
                    for position in range(LEAGUE_SIZE)
                },
            }
        )

    teams.sort(key=lambda row: int(str(row["predicted_rank"])))

    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "season": season,
        "cutoff_matchweek": cutoff_matchweek,
        "n_simulations": result.n_simulations,
        "seed": seed,
        "model": "dixon-coles",
        "teams": teams,
    }
    if extra:
        payload.update(extra)
    return payload


def run_season_simulation(
    season: str,
    *,
    cutoff_matchweek: int = 38,
    teams: list[str] | None = None,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
    db_path: Path = RAW_DB_PATH,
    write_json: bool = True,
) -> dict[str, object]:
    """Fit the model and simulate a season's remaining fixtures.

    Args:
        season: Season to simulate, e.g. ``"2025/26"``.
        cutoff_matchweek: Matchweeks treated as already played.
        teams: Club list. Required for a season with no matches in the database.
        n_simulations: Monte Carlo runs.
        seed: Random seed.
        db_path: Raw database to read.
        write_json: Whether to write the result to ``data/processed``.

    Returns:
        The JSON payload.

    Raises:
        MissingTeamsError: If the club list cannot be determined or is the wrong size.
    """
    matches = load_matches(db_path)

    if season in set(matches["season"]):
        history, played, actual_remaining = split_season(matches, season, cutoff_matchweek)
        season_teams = teams or sorted(set(matches.loc[matches["season"] == season, "home_slug"]))
    else:
        # A season with no data at all - the club list has to come from the caller.
        if not teams:
            raise MissingTeamsError(
                f"{season} has no matches in the database, so its club list cannot be "
                f"derived. Pass --teams-file with all {LEAGUE_SIZE} clubs."
            )
        history = matches
        played = matches.iloc[0:0]
        actual_remaining = matches.iloc[0:0]
        season_teams = sorted(teams)

    if len(season_teams) != LEAGUE_SIZE:
        raise MissingTeamsError(
            f"Expected {LEAGUE_SIZE} clubs for {season}, got {len(season_teams)}: "
            f"{sorted(season_teams)}"
        )

    # Choose the decay rate on held-out data rather than guessing it. The training and
    # validation sets must be disjoint, or the search just rewards overfitting.
    if not actual_remaining.empty:
        # The natural split: fit on everything up to the cutoff, score on what follows.
        decay, decay_scores = select_decay(history, actual_remaining)
    elif len(history) > HOLDOUT_MATCHES:
        # Nothing remains to score against, so hold out the most recent matches instead.
        train = history.iloc[:-HOLDOUT_MATCHES]
        decay, decay_scores = select_decay(train, history.iloc[-HOLDOUT_MATCHES:])
    else:
        decay, decay_scores = DEFAULT_DECAY, {}

    model = fit_dixon_coles(history, decay=decay)
    model = model.with_promoted_defaults(season_teams, last_seen=last_top_flight_match(history))

    standings = Standings.from_matches(played, season_teams)
    remaining = build_remaining_fixtures(season_teams, played)
    logger.info(
        "%s: %d played, %d remaining, %d clubs",
        season,
        len(played),
        len(remaining),
        len(season_teams),
    )

    result = simulate_season(
        model,
        season_teams,
        remaining,
        standings,
        n_simulations=n_simulations,
        seed=seed,
    )

    extra: dict[str, object] = {
        "decay": decay,
        "home_advantage": round(model.home_advantage, 4),
        "rho": round(model.rho, 4),
        "matches_played": len(played),
        "matches_remaining": len(remaining),
    }
    if decay_scores:
        extra["held_out_log_likelihood"] = round(decay_scores[decay].log_likelihood, 4)

    payload = simulation_to_dict(
        result,
        season=season,
        cutoff_matchweek=cutoff_matchweek,
        seed=seed,
        extra=extra,
    )

    if write_json:
        path = simulation_json_path(season)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", path)

    return payload


def score_backtest(payload: dict[str, object], actual: pd.DataFrame) -> dict[str, object]:
    """Compare a simulation against what actually happened.

    Args:
        payload: Output of :func:`run_season_simulation`.
        actual: Every match of the season, including those after the cutoff.

    Returns:
        The real champion, the probability the model gave them, and rank errors.
    """
    points: dict[str, int] = {}
    goal_difference: dict[str, int] = {}
    for match in actual.itertuples(index=False):
        for team, scored, conceded in (
            (match.home_slug, match.home_goals, match.away_goals),
            (match.away_slug, match.away_goals, match.home_goals),
        ):
            points.setdefault(team, 0)
            goal_difference.setdefault(team, 0)
            goal_difference[team] += scored - conceded
            if scored > conceded:
                points[team] += 3
            elif scored == conceded:
                points[team] += 1

    final = sorted(points, key=lambda t: (-points[t], -goal_difference[t]))
    actual_rank = {team: i + 1 for i, team in enumerate(final)}

    teams = payload["teams"]
    assert isinstance(teams, list)
    rows = {str(row["slug"]): row for row in teams}

    champion = final[0]
    errors = [
        abs(int(rows[slug]["predicted_rank"]) - rank)
        for slug, rank in actual_rank.items()
        if slug in rows
    ]

    return {
        "actual_champion": champion,
        "actual_champion_points": points[champion],
        "title_probability_given": rows[champion]["title_probability"],
        "predicted_rank_of_champion": rows[champion]["predicted_rank"],
        "mean_absolute_rank_error": round(float(sum(errors) / len(errors)), 2),
    }
