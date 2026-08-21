"""Dixon-Coles bivariate Poisson model for match scorelines.

Each club gets an attack and a defence rating; the expected goals for a match are

    lambda_home = exp(gamma + attack[home] - defence[away])
    lambda_away = exp(        attack[away] - defence[home])

On top of independent Poisson, Dixon-Coles applies a correction ``tau`` to the four
lowest scorelines (0-0, 1-0, 0-1, 1-1). That correction is the entire reason to prefer
this over plain Poisson: independent Poisson systematically understates draws and 1-0s,
which is exactly where football results pile up.

Matches are weighted by an exponential time decay so that recent seasons dominate the
fit. The decay rate is chosen by held-out likelihood in :mod:`src.models.evaluate`,
not guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.config import MAX_GOALS

logger = logging.getLogger(__name__)

#: Bounds for the low-score correlation parameter. Outside roughly this range tau can
#: drive a scoreline probability negative, which makes the likelihood undefined.
RHO_BOUNDS = (-0.2, 0.2)

#: Bounds on attack/defence ratings, in log-goals. +-3 is far wider than any real club
#: reaches; they exist only to keep the optimiser from wandering into overflow.
RATING_BOUNDS = (-3.0, 3.0)

#: Bounds on the home-advantage term, in log-goals.
HOME_BOUNDS = (-1.0, 1.0)

#: Bounds on covariate coefficients, on the scaled covariates.
BETA_BOUNDS = (-2.0, 2.0)

#: Pseudo-club whose ratings stand in for an average newly promoted side.
PROMOTED_KEY = "__promoted__"

#: Days after which a club's own rating counts for only half, the rest coming from the
#: promoted-club baseline.
#:
#: Set to one season away from the division. A club relegated last May and promoted
#: straight back has had a full year to change its squad, so its old rating is worth
#: about as much as the generic estimate - no more, no less. A club gone for several
#: years converges on the baseline, which is the right limit.
STALENESS_HALF_LIFE_DAYS = 400.0

#: A club that played within this many days is treated as current and left untouched.
#: Covers a normal summer break, so established clubs are never blended.
STALENESS_GRACE_DAYS = 120.0

#: The log-mean is clipped to this range before exponentiating. exp(2.5) is 12 goals,
#: which no fit should ever want; the clip exists so a wild trial point during the line
#: search cannot produce a lambda large enough to drive the tau correction negative.
LOG_LAMBDA_BOUNDS = (-5.0, 2.5)

#: Floor for the tau correction, and the weight of the penalty for hitting it.
#:
#: The objective must stay *finite everywhere*. Returning inf for an infeasible trial
#: point breaks scipy's finite-difference gradient - it computes ``f(x+h) - f(x)``, and
#: ``inf - inf`` is nan, which makes L-BFGS-B give up after one iteration with the
#: initial values still in place. A smooth penalty steers the optimiser back instead.
MIN_CORRECTION = 1e-10
PENALTY_SCALE = 1e6


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles low-score correction.

    Adjusts only the four scorelines where the independence assumption fails; every
    other scoreline is left untouched (a factor of exactly 1).

    Args:
        home_goals: Home goals per match.
        away_goals: Away goals per match.
        lambda_home: Home expected goals per match.
        lambda_away: Away expected goals per match.
        rho: Correlation parameter.

    Returns:
        The multiplicative correction for each match.
    """
    correction = np.ones_like(lambda_home, dtype=float)

    both_nil = (home_goals == 0) & (away_goals == 0)
    home_nil_away_one = (home_goals == 0) & (away_goals == 1)
    home_one_away_nil = (home_goals == 1) & (away_goals == 0)
    one_all = (home_goals == 1) & (away_goals == 1)

    correction[both_nil] = 1.0 - lambda_home[both_nil] * lambda_away[both_nil] * rho
    correction[home_nil_away_one] = 1.0 + lambda_home[home_nil_away_one] * rho
    correction[home_one_away_nil] = 1.0 + lambda_away[home_one_away_nil] * rho
    correction[one_all] = 1.0 - rho

    return correction


def staleness_weight(days_since_last_match: float) -> float:
    """How much of a club's own rating still applies after an absence.

    Args:
        days_since_last_match: Days since the club last played in the division.

    Returns:
        A weight in ``(0, 1]``. 1.0 means "use the club's own rating unchanged";
        values below that blend toward the promoted-club baseline.
    """
    if days_since_last_match <= STALENESS_GRACE_DAYS:
        return 1.0
    elapsed = days_since_last_match - STALENESS_GRACE_DAYS
    return float(0.5 ** (elapsed / STALENESS_HALF_LIFE_DAYS))


def time_weights(dates: pd.Series, reference: pd.Timestamp, decay: float) -> np.ndarray:
    """Exponential decay weights, newest matches weighted highest.

    Args:
        dates: Match dates.
        reference: Date to measure age from - normally the cutoff.
        decay: Decay rate per day. ``0.0`` disables decay.

    Returns:
        A weight per match in ``(0, 1]``.
    """
    if decay <= 0:
        return np.ones(len(dates), dtype=float)
    age_days = (reference - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    weights: np.ndarray = np.exp(-decay * np.clip(age_days, 0, None))
    return weights


@dataclass
class DixonColesModel:
    """A fitted Dixon-Coles model.

    Attributes:
        teams: Club slugs in parameter order.
        attack: Attack rating per club, in log-goals.
        defence: Defence rating per club - higher means better defence.
        home_advantage: Home term in log-goals.
        rho: Low-score correction parameter.
        decay: Time-decay rate the model was fitted with.
    """

    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.0
    rho: float = 0.0
    decay: float = 0.0
    covariate_names: list[str] = field(default_factory=list)
    betas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Per-club covariate snapshot, used when predicting a fixture that has not been
    #: played and therefore has no feature row of its own.
    covariate_snapshot: dict[str, np.ndarray] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------------------

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for both sides.

        Args:
            home: Home club slug.
            away: Away club slug.

        Returns:
            ``(lambda_home, lambda_away)``.

        Raises:
            KeyError: If either club has no rating. Callers that may pass a newly
                promoted club should route through :meth:`with_promoted_defaults`.
        """
        log_home = self.home_advantage + self.attack[home] - self.defence[away]
        log_away = self.attack[away] - self.defence[home]

        if self.betas.size:
            log_home += float(self.betas @ self._snapshot_for(home))
            log_away += float(self.betas @ self._snapshot_for(away))

        log_home = float(np.clip(log_home, *LOG_LAMBDA_BOUNDS))
        log_away = float(np.clip(log_away, *LOG_LAMBDA_BOUNDS))
        return float(np.exp(log_home)), float(np.exp(log_away))

    def _snapshot_for(self, team: str) -> np.ndarray:
        """Return a club's covariate vector, or zeros if it has none.

        Args:
            team: Club slug.

        Returns:
            The covariate vector.
        """
        return self.covariate_snapshot.get(team, np.zeros(len(self.covariate_names)))

    def score_matrix(self, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Joint probability over scorelines.

        This - not two independent Poisson draws - is what the simulation samples from.
        Sampling the marginals independently would silently discard the low-score
        correction that is the whole point of the model.

        Args:
            home: Home club slug.
            away: Away club slug.
            max_goals: Highest scoreline represented per side.

        Returns:
            A ``(max_goals + 1, max_goals + 1)`` array summing to 1, where entry
            ``[i, j]`` is the probability of ``i`` home goals and ``j`` away goals.
        """
        lambda_home, lambda_away = self.expected_goals(home, away)
        goals = np.arange(max_goals + 1)

        home_pmf = poisson.pmf(goals, lambda_home)
        away_pmf = poisson.pmf(goals, lambda_away)
        matrix = np.outer(home_pmf, away_pmf)

        # Apply the correction to the 2x2 low-score corner.
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * self.rho
        matrix[0, 1] *= 1.0 + lambda_home * self.rho
        matrix[1, 0] *= 1.0 + lambda_away * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        # Truncation at max_goals and the correction both cost a little mass.
        matrix = np.clip(matrix, 0.0, None)
        normalised: np.ndarray = matrix / matrix.sum()
        return normalised

    def outcome_probabilities(self, home: str, away: str) -> tuple[float, float, float]:
        """Probability of a home win, draw and away win.

        Args:
            home: Home club slug.
            away: Away club slug.

        Returns:
            ``(home_win, draw, away_win)``, summing to 1.
        """
        matrix = self.score_matrix(home, away)
        draw = float(np.trace(matrix))
        home_win = float(np.tril(matrix, -1).sum())
        away_win = float(np.triu(matrix, 1).sum())
        return home_win, draw, away_win

    # ----------------------------------------------------------------------------------
    # Handling clubs with no history
    # ----------------------------------------------------------------------------------

    def promoted_ratings(self) -> tuple[float, float]:
        """Ratings to assign a club with no top-flight history.

        Returns:
            ``(attack, defence)`` for a newly promoted club.
        """
        return self.attack.get(PROMOTED_KEY, 0.0), self.defence.get(PROMOTED_KEY, 0.0)

    def with_promoted_defaults(
        self,
        teams: list[str],
        *,
        last_seen: dict[str, pd.Timestamp] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> DixonColesModel:
        """Seed unknown clubs, and pull stale ratings back toward the baseline.

        Two different situations, handled together:

        * A club with **no** history at all - a true newcomer - takes the promoted-club
          baseline outright.
        * A club that played in the division once but has been away since - relegated,
          then promoted again - keeps its own rating, blended toward the baseline in
          proportion to how stale that rating is. See :func:`staleness_weight`.

        Without the second case, having data makes a club look *worse* than being
        unknown: a side that was relegated with a poor record carries that record
        forward indefinitely, while the two clubs beside it that we know nothing about
        get the (higher) average. That asymmetry is not a judgement about football, it
        is an artefact of which clubs we happen to have observed.

        Args:
            teams: Clubs the caller intends to simulate.
            last_seen: Date of each club's most recent top-flight match. Omit to skip
                staleness blending entirely.
            as_of: Date to measure staleness from. Defaults to the latest ``last_seen``.

        Returns:
            A model that has a usable rating for every club in ``teams``.
        """
        attack_default, defence_default = self.promoted_ratings()
        attack = dict(self.attack)
        defence = dict(self.defence)

        if last_seen and as_of is None:
            as_of = max(last_seen.values())

        for team in teams:
            if team not in attack:
                logger.info(
                    "%s has no top-flight history - using the promoted-club baseline "
                    "(attack %+.3f, defence %+.3f)",
                    team,
                    attack_default,
                    defence_default,
                )
                attack[team] = attack_default
                defence[team] = defence_default
                continue

            if not last_seen or as_of is None or team not in last_seen:
                continue

            days = float((as_of - last_seen[team]).days)
            weight = staleness_weight(days)
            if weight >= 1.0 - 1e-9:
                continue

            blended_attack = weight * attack[team] + (1.0 - weight) * attack_default
            blended_defence = weight * defence[team] + (1.0 - weight) * defence_default
            logger.info(
                "%s last played %.0f days ago - blending %.0f%% own rating: "
                "attack %+.3f -> %+.3f, defence %+.3f -> %+.3f",
                team,
                days,
                100 * weight,
                attack[team],
                blended_attack,
                defence[team],
                blended_defence,
            )
            attack[team] = blended_attack
            defence[team] = blended_defence

        return DixonColesModel(
            teams=sorted(attack),
            attack=attack,
            defence=defence,
            home_advantage=self.home_advantage,
            rho=self.rho,
            decay=self.decay,
            covariate_names=self.covariate_names,
            betas=self.betas,
            covariate_snapshot=self.covariate_snapshot,
        )


#: Covariates drawn from ``match_features``, as (home column, away column) pairs.
#:
#: These are deliberately quantities that are **static for the rest of the season**.
#: Rolling form and rest days are richer, but they are undefined for a fixture that has
#: not been played - in a season simulation they would themselves have to be simulated,
#: compounding error at every matchweek. A cutoff-time snapshot is known for every
#: remaining fixture.
DEFAULT_COVARIATES: tuple[tuple[str, str], ...] = (
    ("home_elo_pre", "away_elo_pre"),
    ("home_prev_season_rank", "away_prev_season_rank"),
    ("home_was_promoted", "away_was_promoted"),
)

#: ELO lives around 1500 and rank around 10; left unscaled they would dominate the
#: gradient and force absurdly small betas.
COVARIATE_SCALES = {
    "elo_pre": (1500.0, 100.0),
    "prev_season_rank": (10.0, 5.0),
    "was_promoted": (0.0, 1.0),
}


def covariate_name(column: str) -> str:
    """Strip the ``home_``/``away_`` prefix from a feature column.

    Args:
        column: Column name such as ``"home_elo_pre"``.

    Returns:
        The bare feature name, e.g. ``"elo_pre"``.
    """
    return column.removeprefix("home_").removeprefix("away_")


def _scale(name: str, values: np.ndarray) -> np.ndarray:
    """Centre and scale a covariate so all betas live on a similar magnitude.

    Args:
        name: Bare covariate name.
        values: Raw values.

    Returns:
        The scaled values.
    """
    centre, spread = COVARIATE_SCALES.get(name, (0.0, 1.0))
    return (values.astype(float) - centre) / spread


def build_covariates(
    matches: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_COVARIATES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract scaled per-side covariate matrices from feature-joined matches.

    Args:
        matches: Matches joined to ``match_features``.
        pairs: (home column, away column) pairs to use.

    Returns:
        ``(home_matrix, away_matrix, names)``.

    Raises:
        KeyError: If a requested column is missing.
    """
    missing = [c for pair in pairs for c in pair if c not in matches.columns]
    if missing:
        raise KeyError(f"match_features is missing columns: {missing}")

    names = [covariate_name(home_col) for home_col, _ in pairs]
    home = np.column_stack([_scale(covariate_name(h), matches[h].to_numpy()) for h, _ in pairs])
    away = np.column_stack([_scale(covariate_name(a), matches[a].to_numpy()) for _, a in pairs])
    return home, away, names


def snapshot_covariates(
    matches: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_COVARIATES,
) -> dict[str, np.ndarray]:
    """Take each club's most recent covariate values.

    Args:
        matches: Matches joined to ``match_features``, ordered by date.
        pairs: The covariate pairs.

    Returns:
        Club slug -> scaled covariate vector.
    """
    snapshot: dict[str, np.ndarray] = {}
    ordered = matches.sort_values("date")
    for match in ordered.itertuples(index=False):
        for slug_attr, columns in (
            ("home_slug", [home for home, _ in pairs]),
            ("away_slug", [away for _, away in pairs]),
        ):
            team = getattr(match, slug_attr)
            snapshot[team] = np.array(
                [
                    _scale(covariate_name(column), np.array([getattr(match, column)]))[0]
                    for column in columns
                ]
            )
    return snapshot


def _negative_log_likelihood(
    params: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    home_cov: np.ndarray | None = None,
    away_cov: np.ndarray | None = None,
) -> float:
    """Weighted negative log-likelihood of the Dixon-Coles model.

    Args:
        params: Flat parameter vector - attack, defence, home advantage, rho.
        home_index: Index of the home club per match.
        away_index: Index of the away club per match.
        home_goals: Home goals per match.
        away_goals: Away goals per match.
        weights: Time-decay weight per match.
        n_teams: Number of clubs.

    Returns:
        The negative log-likelihood, plus a penalty if the parameters push the tau
        correction non-positive. Always finite - see :data:`MIN_CORRECTION`.
    """
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    n_cov = 0 if home_cov is None else home_cov.shape[1]
    betas = params[2 * n_teams : 2 * n_teams + n_cov]
    home_advantage = params[-2]
    rho = params[-1]

    # Identifiability: attack ratings are only defined up to a shift, so pin their mean
    # at zero. Without this the likelihood has a flat direction and the optimiser drifts.
    attack = attack - attack.mean()

    raw_home = home_advantage + attack[home_index] - defence[away_index]
    raw_away = attack[away_index] - defence[home_index]
    if n_cov and home_cov is not None and away_cov is not None:
        raw_home = raw_home + home_cov @ betas
        raw_away = raw_away + away_cov @ betas

    log_home = np.clip(raw_home, *LOG_LAMBDA_BOUNDS)
    log_away = np.clip(raw_away, *LOG_LAMBDA_BOUNDS)
    lambda_home = np.exp(log_home)
    lambda_away = np.exp(log_away)

    correction = tau(home_goals, away_goals, lambda_home, lambda_away, rho)

    # Penalise rather than reject, so the objective stays differentiable by finite
    # differences even where the parameters are infeasible.
    shortfall = np.clip(MIN_CORRECTION - correction, 0.0, None)
    penalty = PENALTY_SCALE * float(np.sum(shortfall**2))
    correction = np.maximum(correction, MIN_CORRECTION)

    log_likelihood = (
        np.log(correction)
        + home_goals * log_home
        - lambda_home
        + away_goals * log_away
        - lambda_away
    )
    return float(-np.sum(weights * log_likelihood)) + penalty


def fit_dixon_coles(
    matches: pd.DataFrame,
    *,
    decay: float = 0.0,
    reference_date: pd.Timestamp | None = None,
    covariates: tuple[tuple[str, str], ...] | None = None,
) -> DixonColesModel:
    """Fit attack, defence, home advantage, rho and optional covariate betas.

    Args:
        matches: Rows with ``home_slug``, ``away_slug``, ``home_goals``, ``away_goals``
            and ``date``. When ``covariates`` is given, must also carry the named
            ``match_features`` columns.
        decay: Time-decay rate per day.
        reference_date: Date to measure match age from. Defaults to the latest match.
        covariates: (home column, away column) pairs to include as a linear term in the
            log-mean. ``None`` fits classic Dixon-Coles.

    Returns:
        The fitted model.

    Raises:
        ValueError: If ``matches`` is empty.
    """
    if matches.empty:
        raise ValueError("Cannot fit Dixon-Coles on an empty match set.")

    teams = sorted(set(matches["home_slug"]) | set(matches["away_slug"]))
    index_of = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_index = matches["home_slug"].map(index_of).to_numpy()
    away_index = matches["away_slug"].map(index_of).to_numpy()
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)

    reference = reference_date or pd.to_datetime(matches["date"]).max()
    weights = time_weights(matches["date"], reference, decay)

    home_cov: np.ndarray | None = None
    away_cov: np.ndarray | None = None
    names: list[str] = []
    if covariates:
        home_cov, away_cov, names = build_covariates(matches, covariates)

    n_cov = len(names)
    initial = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), np.zeros(n_cov), [0.25], [0.0]])
    bounds = (
        [RATING_BOUNDS] * n_teams
        + [RATING_BOUNDS] * n_teams
        + [BETA_BOUNDS] * n_cov
        + [HOME_BOUNDS, RHO_BOUNDS]
    )

    result = minimize(
        _negative_log_likelihood,
        initial,
        args=(
            home_index,
            away_index,
            home_goals,
            away_goals,
            weights,
            n_teams,
            home_cov,
            away_cov,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        logger.warning("Optimiser reported: %s", result.message)

    attack = result.x[:n_teams] - result.x[:n_teams].mean()
    defence = result.x[n_teams : 2 * n_teams]
    betas = result.x[2 * n_teams : 2 * n_teams + n_cov]

    model = DixonColesModel(
        teams=teams,
        attack=dict(zip(teams, attack.tolist(), strict=True)),
        defence=dict(zip(teams, defence.tolist(), strict=True)),
        home_advantage=float(result.x[-2]),
        rho=float(result.x[-1]),
        decay=decay,
        covariate_names=names,
        betas=betas,
        covariate_snapshot=(snapshot_covariates(matches, covariates) if covariates else {}),
    )
    if names:
        logger.info(
            "Covariate betas: %s",
            ", ".join(f"{n}={b:+.4f}" for n, b in zip(names, betas.tolist(), strict=True)),
        )
    model.attack[PROMOTED_KEY], model.defence[PROMOTED_KEY] = _promoted_baseline(matches, model)

    logger.info(
        "Fitted Dixon-Coles on %d matches: home=%.3f rho=%.3f decay=%.4f",
        len(matches),
        model.home_advantage,
        model.rho,
        decay,
    )
    return model


def first_seasons(matches: pd.DataFrame) -> dict[str, str]:
    """Find each club that arrived mid-window, and the season it arrived in.

    Clubs present in the earliest season are excluded: we cannot tell whether they had
    just come up or had been in the division for a decade.

    Args:
        matches: The training matches, needing ``season``.

    Returns:
        Club slug -> the season label of its first appearance.
    """
    arrivals: dict[str, str] = {}
    seen: set[str] = set()
    for season in sorted(matches["season"].unique()):
        rows = matches[matches["season"] == season]
        clubs = set(rows["home_slug"]) | set(rows["away_slug"])
        if seen:
            for club in sorted(clubs - seen):
                arrivals[club] = str(season)
        seen |= clubs
    return arrivals


def _promoted_baseline(matches: pd.DataFrame, model: DixonColesModel) -> tuple[float, float]:
    """Measure what a newly promoted club does in its **first** season.

    A promoted club cannot be given the league average - promoted sides are
    systematically weaker, and seeding them at the mean would predict every newcomer to
    finish mid-table.

    The estimate deliberately uses each club's *first* season only, not its whole time
    in the division. Clubs that come up and then establish themselves would otherwise
    drag the baseline upward, and the clubs this baseline is for are, by definition,
    arriving rather than established.

    It is measured from goals rather than from the fitted ratings, which keeps it
    interpretable and independent of the optimiser: a club scoring 0.8 times the league
    average has an attack contribution of ``log(0.8)``.

    Args:
        matches: The training matches.
        model: Unused for the estimate itself; kept for signature stability.

    Returns:
        ``(attack, defence)`` for a club in its first season up.
    """
    del model  # the goals-based estimate does not need the fitted ratings

    if "season" not in matches.columns:
        return 0.0, 0.0

    arrivals = first_seasons(matches)
    if not arrivals:
        return 0.0, 0.0

    league_mean = float(
        (matches["home_goals"].sum() + matches["away_goals"].sum()) / (2 * len(matches))
    )
    if league_mean <= 0:
        return 0.0, 0.0

    attack_terms: list[float] = []
    defence_terms: list[float] = []

    for club, season in arrivals.items():
        rows = matches[matches["season"] == season]
        home = rows[rows["home_slug"] == club]
        away = rows[rows["away_slug"] == club]
        played = len(home) + len(away)
        if played == 0:
            continue

        scored = float(home["home_goals"].sum() + away["away_goals"].sum()) / played
        conceded = float(home["away_goals"].sum() + away["home_goals"].sum()) / played

        # Guard the log against a club that failed to score all season.
        attack_terms.append(float(np.log(max(scored, 0.05) / league_mean)))
        # Defence is oriented so that higher is better, hence the negation.
        defence_terms.append(-float(np.log(max(conceded, 0.05) / league_mean)))

    if not attack_terms:
        return 0.0, 0.0

    attack = float(np.mean(attack_terms))
    defence = float(np.mean(defence_terms))
    logger.info(
        "Promoted-club baseline from %d first seasons: attack=%+.3f defence=%+.3f",
        len(attack_terms),
        attack,
        defence,
    )
    return attack, defence
