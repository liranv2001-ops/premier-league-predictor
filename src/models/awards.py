"""Individual award models: top scorer, top assists, player of the season.

Two XGBoost regressors predict how many goals and assists each player will add between
a cutoff and the end of the season. Those predictions become Poisson means, and a Monte
Carlo run over them turns point estimates into the probabilities the awards actually
need - "how likely is this player to finish top" is not answerable from a single number.

Everything a training row sees is state *at* the cutoff. The targets are what happened
after it. `tests/test_awards.py` pins that down with the same tampering test used for
`src/features`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.player_features import QUALIFYING_MINUTES
from src.models.config import N_SIMULATIONS, RANDOM_SEED

logger = logging.getLogger(__name__)

#: Matches in a full Premier League season, per club.
MATCHES_PER_CLUB = 38

#: Cutoffs used to expand each season into several training rows. Spreading them out
#: teaches the model what a partial season looks like at every stage, rather than only
#: at the halfway point.
TRAINING_CUTOFFS = (5, 10, 15, 20, 25, 30)

#: Player of the Season weights. Must sum to 1 - asserted at import, because a silent
#: re-weighting would change every ranking with nothing in the output to show for it.
POTS_WEIGHTS = {
    "attacking": 0.50,
    "team": 0.30,
    "minutes": 0.20,
}
assert abs(sum(POTS_WEIGHTS.values()) - 1.0) < 1e-9, "PotS weights must sum to 1"

#: How many candidates each award reports.
TOP_N = 5

FEATURE_COLUMNS = [
    "goals_to_date",
    "assists_to_date",
    "minutes_to_date",
    "goals_per90_to_date",
    "assists_per90_to_date",
    "xg_per90_to_date",
    "xa_per90_to_date",
    "goals_per90_last5",
    "assists_per90_last5",
    "trend_goals",
    "trend_assists",
    "appearances_to_date",
    "minutes_per_appearance",
    "expected_remaining_minutes",
    "matches_remaining",
    "team_elo",
    "team_attack",
    "team_defence",
    "is_forward",
    "is_midfield",
    "is_defence",
]

#: Understat position strings mapped to coarse buckets. Its codes combine a line and a
#: side (``AMR``, ``DMC``), so a prefix match is what distinguishes them.
FORWARD_PREFIXES = ("F", "S")
MIDFIELD_PREFIXES = ("A", "M")
DEFENCE_PREFIXES = ("D",)


def _position_flags(position: pd.Series) -> pd.DataFrame:
    """Turn Understat position codes into coarse one-hot columns.

    Args:
        position: Raw position strings.

    Returns:
        Columns ``is_forward``, ``is_midfield``, ``is_defence``. A goalkeeper is all
        zeros, which is the correct encoding for a fourth category.
    """
    codes = position.fillna("").astype(str).str.upper()
    return pd.DataFrame(
        {
            "is_forward": codes.str.startswith(FORWARD_PREFIXES).astype(int),
            "is_midfield": codes.str.startswith(MIDFIELD_PREFIXES).astype(int),
            "is_defence": codes.str.startswith(DEFENCE_PREFIXES).astype(int),
        },
        index=position.index,
    )


def build_player_state(
    player_matches: pd.DataFrame,
    season: str,
    cutoff_matchweek: int,
    team_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise every player's season up to a cutoff, with their targets after it.

    Args:
        player_matches: ``player_match_features`` rows.
        season: Season to summarise.
        cutoff_matchweek: Matchweeks treated as played.
        team_ratings: One row per club with ``team_slug``, ``team_elo``,
            ``team_attack``, ``team_defence``.

    Returns:
        One row per player active before the cutoff, carrying the feature columns plus
        ``goals_remaining`` and ``assists_remaining``.
    """
    season_rows = player_matches[player_matches["season"] == season].copy()
    season_rows["date"] = pd.to_datetime(season_rows["date"])

    # Matchweek is derived per club: a club's Nth match of the season. Calendar dates
    # cannot be used directly because clubs play rearranged fixtures.
    season_rows = season_rows.sort_values(["team_slug", "date"])
    match_order = (
        season_rows.drop_duplicates(["team_slug", "match_id"]).groupby("team_slug").cumcount() + 1
    )
    order_lookup = dict(
        zip(
            season_rows.drop_duplicates(["team_slug", "match_id"])
            .set_index(["team_slug", "match_id"])
            .index,
            match_order,
            strict=True,
        )
    )
    season_rows["team_matchweek"] = [
        order_lookup[(team, match)]
        for team, match in zip(season_rows["team_slug"], season_rows["match_id"], strict=True)
    ]

    before = season_rows[season_rows["team_matchweek"] <= cutoff_matchweek]
    after = season_rows[season_rows["team_matchweek"] > cutoff_matchweek]

    if before.empty:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "goals_remaining", "assists_remaining"])

    aggregated = before.groupby("player_id").agg(
        player_name=("player_name", "last"),
        team_slug=("team_slug", "last"),
        position=("position", "last"),
        goals_to_date=("goals", "sum"),
        assists_to_date=("assists", "sum"),
        minutes_to_date=("minutes", "sum"),
        xg_to_date=("xg", "sum"),
        xa_to_date=("xa", "sum"),
        appearances_to_date=("match_id", "count"),
        goals_per90_last5=("goals_per90_last5", "last"),
        assists_per90_last5=("assists_per90_last5", "last"),
        trend_goals=("trend_goals", "last"),
        trend_assists=("trend_assists", "last"),
    )
    aggregated = aggregated.reset_index()

    minutes = aggregated["minutes_to_date"].astype(float)
    per90 = (90.0 / minutes).where(minutes > 0, 0.0)
    aggregated["goals_per90_to_date"] = aggregated["goals_to_date"] * per90
    aggregated["assists_per90_to_date"] = aggregated["assists_to_date"] * per90
    aggregated["xg_per90_to_date"] = aggregated["xg_to_date"] * per90
    aggregated["xa_per90_to_date"] = aggregated["xa_to_date"] * per90

    aggregated["minutes_per_appearance"] = (
        minutes / aggregated["appearances_to_date"].clip(lower=1)
    ).astype(float)

    # A club's remaining fixtures depend on how many it has already played, which is not
    # the same for every club when fixtures have been rearranged.
    played_per_team = before.drop_duplicates(["team_slug", "match_id"]).groupby("team_slug").size()
    aggregated["matches_remaining"] = (
        aggregated["team_slug"].map(played_per_team).fillna(0).rsub(MATCHES_PER_CLUB).clip(lower=0)
    )
    aggregated["expected_remaining_minutes"] = (
        aggregated["minutes_per_appearance"] * aggregated["matches_remaining"]
    )

    aggregated = aggregated.merge(team_ratings, on="team_slug", how="left")
    for column in ("team_elo", "team_attack", "team_defence"):
        aggregated[column] = aggregated[column].fillna(aggregated[column].median())

    aggregated = pd.concat([aggregated, _position_flags(aggregated["position"])], axis=1)

    remaining = after.groupby("player_id").agg(
        goals_remaining=("goals", "sum"),
        assists_remaining=("assists", "sum"),
        minutes_remaining=("minutes", "sum"),
    )
    aggregated = aggregated.merge(remaining, on="player_id", how="left")
    for column in ("goals_remaining", "assists_remaining", "minutes_remaining"):
        aggregated[column] = aggregated[column].fillna(0.0).astype(float)

    aggregated["season"] = season
    aggregated["cutoff_matchweek"] = cutoff_matchweek
    return aggregated


def build_preseason_state(
    player_matches: pd.DataFrame,
    prior_season: str,
    team_ratings: pd.DataFrame,
    target_season: str | None = None,
) -> pd.DataFrame:
    """Player state before a ball has been kicked, using last season's rates.

    A fresh season is a genuinely different prediction problem from a mid-season one:
    nothing has been scored, a full 38 matches remain, and the only evidence is the
    previous campaign. Training only on mid-season cutoffs and then serving this case
    is a train/serve mismatch - the model would be extrapolating outside every feature
    range it ever saw, and it visibly compresses predictions toward the mean.

    So this shape gets its own rows in the training set, and the same builder produces
    the prediction input. The two are identical by construction.

    Args:
        player_matches: ``player_match_features`` rows.
        prior_season: Season supplying the rates.
        team_ratings: Club ratings for the season being predicted.
        target_season: Season supplying the targets. ``None`` for prediction, where the
            targets are unknown.

    Returns:
        One row per player, with prior-season rates as features.
    """
    prior = player_matches[player_matches["season"] == prior_season]
    if prior.empty:
        return pd.DataFrame()

    aggregated = (
        prior.groupby("player_id")
        .agg(
            player_name=("player_name", "last"),
            team_slug=("team_slug", "last"),
            position=("position", "last"),
            prior_goals=("goals", "sum"),
            prior_assists=("assists", "sum"),
            prior_minutes=("minutes", "sum"),
            prior_xg=("xg", "sum"),
            prior_xa=("xa", "sum"),
            prior_appearances=("match_id", "count"),
        )
        .reset_index()
    )

    minutes = aggregated["prior_minutes"].astype(float)
    per90 = (90.0 / minutes).where(minutes > 0, 0.0)

    # Nothing has happened yet this season, so the counters are zero and the *rates*
    # carry the information.
    aggregated["goals_to_date"] = 0.0
    aggregated["assists_to_date"] = 0.0
    aggregated["minutes_to_date"] = 0.0
    aggregated["appearances_to_date"] = 0.0

    aggregated["goals_per90_to_date"] = aggregated["prior_goals"] * per90
    aggregated["assists_per90_to_date"] = aggregated["prior_assists"] * per90
    aggregated["xg_per90_to_date"] = aggregated["prior_xg"] * per90
    aggregated["xa_per90_to_date"] = aggregated["prior_xa"] * per90

    # With no in-season form, last-5 form is the prior-season rate and the trend is
    # flat by definition - there is nothing to be trending against.
    aggregated["goals_per90_last5"] = aggregated["goals_per90_to_date"]
    aggregated["assists_per90_last5"] = aggregated["assists_per90_to_date"]
    aggregated["trend_goals"] = 0.0
    aggregated["trend_assists"] = 0.0

    aggregated["minutes_per_appearance"] = (
        minutes / aggregated["prior_appearances"].clip(lower=1)
    ).astype(float)
    aggregated["matches_remaining"] = float(MATCHES_PER_CLUB)
    aggregated["expected_remaining_minutes"] = (
        aggregated["minutes_per_appearance"] * MATCHES_PER_CLUB
    )

    aggregated = aggregated.merge(team_ratings, on="team_slug", how="left")
    for column in ("team_elo", "team_attack", "team_defence"):
        if column not in aggregated.columns:
            aggregated[column] = np.nan
        aggregated[column] = aggregated[column].fillna(aggregated[column].median())

    aggregated = pd.concat([aggregated, _position_flags(aggregated["position"])], axis=1)

    if target_season is None:
        aggregated["goals_remaining"] = 0.0
        aggregated["assists_remaining"] = 0.0
        aggregated["minutes_remaining"] = aggregated["expected_remaining_minutes"]
        aggregated["season"] = prior_season
        aggregated["cutoff_matchweek"] = 0
        return aggregated

    # Targets are the player's full totals in the season that followed. Players who did
    # not appear are dropped rather than given a zero target: the question being asked
    # is "how will this player do next season", not "will they still be here". The
    # carry-forward assumption is disclosed in the output instead.
    target = player_matches[player_matches["season"] == target_season]
    totals = target.groupby("player_id").agg(
        goals_remaining=("goals", "sum"),
        assists_remaining=("assists", "sum"),
        minutes_remaining=("minutes", "sum"),
    )
    aggregated = aggregated.merge(totals, on="player_id", how="inner")
    aggregated["season"] = target_season
    aggregated["cutoff_matchweek"] = 0
    return aggregated


def build_training_set(
    player_matches: pd.DataFrame,
    team_ratings_by_season: dict[str, pd.DataFrame],
    *,
    cutoffs: tuple[int, ...] = TRAINING_CUTOFFS,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """Expand every season into rows at several cutoffs.

    Args:
        player_matches: ``player_match_features`` rows.
        team_ratings_by_season: Season -> club rating table.
        cutoffs: Matchweeks to snapshot.
        seasons: Seasons to include. Defaults to all present.

    Returns:
        The stacked training set.
    """
    seasons = seasons or sorted(player_matches["season"].unique())
    frames = [
        build_player_state(
            player_matches, season, cutoff, team_ratings_by_season.get(season, pd.DataFrame())
        )
        for season in seasons
        for cutoff in cutoffs
    ]

    # Pre-season rows, so the model is trained on the same shape it is later asked to
    # predict from. Without these it only ever sees partial seasons.
    ordered = sorted(player_matches["season"].unique())
    for prior, target in zip(ordered, ordered[1:], strict=False):
        if target not in seasons:
            continue
        frames.append(
            build_preseason_state(
                player_matches,
                prior,
                team_ratings_by_season.get(target, pd.DataFrame()),
                target_season=target,
            )
        )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    logger.info(
        "Built %d training rows from %d seasons x %d cutoffs",
        len(stacked),
        len(seasons),
        len(cutoffs),
    )
    return stacked


@dataclass
class AwardModels:
    """The two fitted regressors and how they scored on held-out data.

    Attributes:
        goals: Model predicting goals scored after the cutoff.
        assists: Model predicting assists after the cutoff.
        scores: Validation metrics, including the naive baseline for comparison.
    """

    goals: XGBRegressor
    assists: XGBRegressor
    scores: dict[str, float]

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict remaining goals and assists.

        Args:
            features: Rows carrying :data:`FEATURE_COLUMNS`.

        Returns:
            ``(goals_remaining, assists_remaining)``, both non-negative.
        """
        matrix = features[FEATURE_COLUMNS]
        goals = np.clip(self.goals.predict(matrix), 0.0, None)
        assists = np.clip(self.assists.predict(matrix), 0.0, None)
        return goals, assists


def _naive_baseline(rows: pd.DataFrame, stat: str) -> np.ndarray:
    """Predict by assuming the player's current rate simply continues.

    This is the bar any model has to clear. It is not a strawman - for counting stats
    over a partial season it is a genuinely strong predictor.

    Args:
        rows: Feature rows.
        stat: ``"goals"`` or ``"assists"``.

    Returns:
        Predicted remaining count per row.
    """
    rate = rows[f"{stat}_per90_to_date"].to_numpy(dtype=float)
    predicted: np.ndarray = rate * rows["expected_remaining_minutes"].to_numpy(dtype=float) / 90.0
    return predicted


def fit_award_models(
    training: pd.DataFrame,
    *,
    validation_season: str,
    seed: int = RANDOM_SEED,
) -> AwardModels:
    """Fit the goal and assist regressors, validating on a held-out season.

    The split is **by season, never random**. Each player-season appears at six
    cutoffs, so a random split would train on matchweek 25 and test on matchweek 10 of
    the same season - the same matches on both sides of the split.

    Args:
        training: Output of :func:`build_training_set`.
        validation_season: Season held out for scoring.
        seed: Random seed.

    Returns:
        The fitted models and their validation scores.

    Raises:
        ValueError: If either split ends up empty.
    """
    train = training[training["season"] != validation_season]
    validate = training[training["season"] == validation_season]
    if train.empty or validate.empty:
        raise ValueError(
            f"Cannot split on {validation_season!r}: "
            f"{len(train)} train rows, {len(validate)} validation rows."
        )

    scores: dict[str, float] = {}
    fitted: dict[str, XGBRegressor] = {}

    for stat in ("goals", "assists"):
        model = XGBRegressor(
            # Counts, not real numbers. Squared error would happily predict negative
            # goals and would chase the handful of 20-goal seasons instead of the mass
            # of players near zero.
            objective="count:poisson",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(train[FEATURE_COLUMNS], train[f"{stat}_remaining"])
        fitted[stat] = model

        actual = validate[f"{stat}_remaining"].to_numpy(dtype=float)
        predicted = np.clip(model.predict(validate[FEATURE_COLUMNS]), 0.0, None)
        baseline = _naive_baseline(validate, stat)

        scores[f"{stat}_mae"] = float(np.mean(np.abs(predicted - actual)))
        scores[f"{stat}_baseline_mae"] = float(np.mean(np.abs(baseline - actual)))

    scores["n_train"] = float(len(train))
    scores["n_validation"] = float(len(validate))

    for stat in ("goals", "assists"):
        model_mae = scores[f"{stat}_mae"]
        baseline_mae = scores[f"{stat}_baseline_mae"]
        verdict = "beats" if model_mae < baseline_mae else "LOSES TO"
        logger.info(
            "%s: MAE %.4f vs naive baseline %.4f - model %s the baseline",
            stat,
            model_mae,
            baseline_mae,
            verdict,
        )

    return AwardModels(goals=fitted["goals"], assists=fitted["assists"], scores=scores)


def award_probabilities(
    totals_so_far: np.ndarray,
    predicted_remaining: np.ndarray,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Probability that each player finishes the season top of a counting stat.

    A point prediction cannot answer this; only the spread can. Each player's remaining
    count is drawn from a Poisson with their predicted mean, added to what they already
    have, and the winner counted.

    Shared Golden Boots are real, so a ``k``-way tie awards ``1/k`` to each. That also
    makes the probabilities sum to exactly 1.

    Args:
        totals_so_far: Each player's count before the cutoff.
        predicted_remaining: Each player's predicted count after it.
        n_simulations: Monte Carlo runs.
        seed: Random seed.

    Returns:
        A probability per player.
    """
    if len(totals_so_far) == 0:
        return np.zeros(0)

    rng = np.random.default_rng(seed)
    draws = rng.poisson(
        np.clip(predicted_remaining, 0.0, None), size=(n_simulations, len(predicted_remaining))
    )
    finals = draws + totals_so_far

    best = finals.max(axis=1, keepdims=True)
    winners = finals == best
    # Split each simulation's credit among however many players tied for top.
    credit = winners / winners.sum(axis=1, keepdims=True)
    probabilities: np.ndarray = credit.sum(axis=0) / n_simulations
    return probabilities


def _normalise(values: pd.Series) -> pd.Series:
    """Min-max scale a column to ``[0, 1]``.

    Args:
        values: The column.

    Returns:
        The scaled column; all zeros if the column is constant.
    """
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def player_of_the_season(
    candidates: pd.DataFrame,
    team_points: dict[str, float],
    *,
    qualifying_minutes: int = QUALIFYING_MINUTES,
) -> pd.DataFrame:
    """Score players on attacking output, team performance and minutes.

    Args:
        candidates: Rows with ``predicted_goals``, ``predicted_assists``,
            ``predicted_minutes``, ``team_slug``.
        team_points: Club slug -> that club's expected points, so the awards and the
            predicted table cannot disagree about which club is good.
        qualifying_minutes: Minimum predicted minutes to be eligible.

    Returns:
        The eligible candidates with their component scores and total, best first.
    """
    eligible = candidates[candidates["predicted_minutes"] >= qualifying_minutes].copy()
    if eligible.empty:
        return eligible

    eligible["attacking_contribution"] = eligible["predicted_goals"] + eligible["predicted_assists"]
    eligible["team_points"] = eligible["team_slug"].map(team_points).astype(float)
    eligible["team_points"] = eligible["team_points"].fillna(eligible["team_points"].median())

    eligible["score_attacking"] = _normalise(eligible["attacking_contribution"])
    eligible["score_team"] = _normalise(eligible["team_points"])
    eligible["score_minutes"] = _normalise(eligible["predicted_minutes"])

    eligible["score"] = (
        POTS_WEIGHTS["attacking"] * eligible["score_attacking"]
        + POTS_WEIGHTS["team"] * eligible["score_team"]
        + POTS_WEIGHTS["minutes"] * eligible["score_minutes"]
    )
    return eligible.sort_values("score", ascending=False).reset_index(drop=True)
