"""Per-match team features.

Produces one row per match with ``home_*`` / ``away_*`` columns and a ``*_diff`` for
each pair, since the difference between the two sides is what most models actually use.

**Every feature is strictly pre-match.** The build walks matches in date order and
attaches each team's state *before* appending that match's outcome to the state. This
is the property that makes the whole table usable; ``tests/test_features.py`` pins it
down with a dedicated leakage test.

No column is ever NaN. That is achieved by construction rather than by a fillna at the
end - see the fallbacks in :func:`build_match_features`.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

import pandas as pd

from src.features.elo import EloEngine

logger = logging.getLogger(__name__)

#: Length of the rolling form window, in matches.
FORM_WINDOW = 5

#: Rest is capped here. The gap before a season opener is ~90 days, which is not
#: "rest" in any meaningful sense - it would just be a season-opener indicator wearing
#: a numeric disguise. `is_rest_capped` carries that signal honestly instead.
MAX_REST_DAYS = 14

#: Rank assigned to a club that was not in the division last season. Keeps the column
#: numeric and correctly ordered (worse than 20th) while `was_promoted` lets a model
#: treat the category separately rather than trusting the number.
PROMOTED_RANK = 21

#: Rolling statistics tracked per team, and the source columns they come from.
ROLLING_STATS = (
    "goals_scored",
    "goals_conceded",
    "xg",
    "xga",
    "npxgd",
    "ppda",
    "xpts",
)


def compute_season_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute the final league table for each season.

    Args:
        matches: Raw match rows with ``season``, ``home_slug``, ``away_slug``,
            ``home_goals``, ``away_goals``.

    Returns:
        One row per (season, team) with ``points``, ``goal_difference`` and ``rank``,
        plus the home/away points-per-game split used for the ``home_advantage``
        feature.
    """
    home = matches.rename(columns={"home_slug": "team", "away_slug": "opponent"}).assign(
        scored=lambda d: d["home_goals"],
        conceded=lambda d: d["away_goals"],
        venue="home",
    )
    away = matches.rename(columns={"away_slug": "team", "home_slug": "opponent"}).assign(
        scored=lambda d: d["away_goals"],
        conceded=lambda d: d["home_goals"],
        venue="away",
    )
    long = pd.concat(
        [
            home[["season", "team", "scored", "conceded", "venue"]],
            away[["season", "team", "scored", "conceded", "venue"]],
        ],
        ignore_index=True,
    )
    long["points"] = 3 * (long["scored"] > long["conceded"]) + (long["scored"] == long["conceded"])

    table = (
        long.groupby(["season", "team"], as_index=False)
        .agg(
            points=("points", "sum"),
            scored=("scored", "sum"),
            conceded=("conceded", "sum"),
            played=("points", "size"),
        )
        .assign(goal_difference=lambda d: d["scored"] - d["conceded"])
    )
    table["rank"] = (
        table.sort_values(["points", "goal_difference", "scored"], ascending=False)
        .groupby("season")
        .cumcount()
        + 1
    )

    ppg = (
        long.groupby(["season", "team", "venue"], as_index=False)["points"]
        .mean()
        .pivot(index=["season", "team"], columns="venue", values="points")
        .reset_index()
    )
    ppg["home_advantage"] = ppg["home"] - ppg["away"]

    return table.merge(ppg[["season", "team", "home_advantage"]], on=["season", "team"])


def _team_match_long(matches: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches into one row per team-match, joined to Understat xG data.

    Args:
        matches: Raw match rows.
        team_stats: ``team_match_stats`` rows from Understat.

    Returns:
        Two rows per match - one per side - with goals and xG statistics.
    """
    home = matches.assign(
        team=matches["home_slug"], opponent=matches["away_slug"], is_home=True
    ).rename(columns={"home_goals": "goals_scored", "away_goals": "goals_conceded"})
    away = matches.assign(
        team=matches["away_slug"], opponent=matches["home_slug"], is_home=False
    ).rename(columns={"away_goals": "goals_scored", "home_goals": "goals_conceded"})

    keep = [
        "date",
        "season",
        "season_start_year",
        "team",
        "opponent",
        "is_home",
        "goals_scored",
        "goals_conceded",
    ]
    long = pd.concat([home[keep], away[keep]], ignore_index=True)

    # Understat's per-match rows carry xG. Joining on (team, date) is safe because a
    # club plays at most one league match per day.
    stats = team_stats[["team_slug", "date", "xg", "xga", "npxgd", "ppda", "xpts"]].rename(
        columns={"team_slug": "team"}
    )
    stats = stats.assign(date=pd.to_datetime(stats["date"]).dt.normalize())
    long = long.assign(date=pd.to_datetime(long["date"]).dt.normalize())

    merged = long.merge(stats, on=["team", "date"], how="left")
    return merged.sort_values(["date", "team"]).reset_index(drop=True)


def build_match_features(
    matches: pd.DataFrame,
    team_stats: pd.DataFrame,
    *,
    drop_seasons: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the pre-match feature table.

    Walks every match in date order, reading each team's accumulated state to produce
    features and only then folding the result into that state.

    Args:
        matches: Raw match rows.
        team_stats: Understat ``team_match_stats`` rows.
        drop_seasons: Seasons to exclude from the output. Used to keep the earliest
            season as history for previous-season features without emitting it.

    Returns:
        One row per match, ordered by date, with no NaN values.
    """
    long = _team_match_long(matches, team_stats)

    # League-wide averages per season, the last-resort fallback for a club with no
    # prior matches at all (its very first appearance in the data).
    league_means = long.groupby("season")[["goals_scored", "goals_conceded"]].mean()
    league_xg_means = long.groupby("season")[["xg", "xga", "npxgd", "ppda", "xpts"]].mean()

    tables = compute_season_table(matches).set_index(["season", "team"])
    seasons = sorted(matches["season"].unique())
    previous_season = {season: seasons[i - 1] if i else None for i, season in enumerate(seasons)}

    engine = EloEngine()
    history: dict[str, dict[str, deque[float]]] = defaultdict(
        lambda: {stat: deque(maxlen=FORM_WINDOW) for stat in ROLLING_STATS}
    )
    last_played: dict[str, pd.Timestamp] = {}
    seasons_started: set[str] = set()

    def team_state(team: str, season: str, date: pd.Timestamp) -> dict[str, float]:
        """Read a team's pre-match state, falling back when history is thin."""
        book = history[team]
        state: dict[str, float] = {}
        n_prior = len(book["goals_scored"])

        for stat in ROLLING_STATS:
            window = book[stat]
            if window:
                state[f"{stat}_avg{FORM_WINDOW}"] = sum(window) / len(window)
            elif stat in ("goals_scored", "goals_conceded"):
                state[f"{stat}_avg{FORM_WINDOW}"] = float(league_means.loc[season, stat])
            else:
                state[f"{stat}_avg{FORM_WINDOW}"] = float(league_xg_means.loc[season, stat])

        state["n_prior_matches"] = float(n_prior)

        previous = last_played.get(team)
        rest = MAX_REST_DAYS if previous is None else (date - previous).days
        state["rest_days"] = float(min(max(rest, 1), MAX_REST_DAYS))

        prior_season = previous_season[season]
        key = (prior_season, team)
        if prior_season is not None and key in tables.index:
            state["prev_season_rank"] = float(tables.loc[key, "rank"])
            state["home_advantage"] = float(tables.loc[key, "home_advantage"])
            state["was_promoted"] = 0.0
        else:
            state["prev_season_rank"] = float(PROMOTED_RANK)
            state["home_advantage"] = 0.0
            state["was_promoted"] = 1.0

        return state

    rows: list[dict[str, Any]] = []
    ordered = matches.assign(date=pd.to_datetime(matches["date"])).sort_values(
        ["date", "home_slug"]
    )

    # (team, date) -> xG statistics, built once. A dict lookup keeps the main loop
    # readable and avoids repeated .loc calls that would silently return a frame if a
    # club somehow had two rows on one date.
    xg_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {
        (row.team, row.date): {
            stat: getattr(row, stat) for stat in ("xg", "xga", "npxgd", "ppda", "xpts")
        }
        for row in long.itertuples(index=False)
    }

    for match in ordered.itertuples(index=False):
        season = match.season
        if season not in seasons_started:
            engine.start_season(
                sorted(
                    set(matches.loc[matches["season"] == season, "home_slug"]),
                )
            )
            seasons_started.add(season)

        date = pd.Timestamp(match.date).normalize()
        home, away = match.home_slug, match.away_slug

        home_state = team_state(home, season, date)
        away_state = team_state(away, season, date)
        home_elo, away_elo = engine.rate_match(home, away, match.home_goals, match.away_goals)
        home_state["elo_pre"] = home_elo
        away_state["elo_pre"] = away_elo

        row: dict[str, Any] = {
            "date": date,
            "season": season,
            "season_start_year": match.season_start_year,
            "home_slug": home,
            "away_slug": away,
            "home_goals": int(match.home_goals),
            "away_goals": int(match.away_goals),
            "result": match.result,
            # True when either side's gap hit the cap. That is a season opener *or* an
            # international break - both mean "rest_days is a cap, not a measurement",
            # which is what a model needs to know.
            "is_rest_capped": int(
                home_state["rest_days"] == MAX_REST_DAYS or away_state["rest_days"] == MAX_REST_DAYS
            ),
        }
        for name, value in home_state.items():
            row[f"home_{name}"] = value
        for name, value in away_state.items():
            row[f"away_{name}"] = value
        for name in home_state:
            row[f"diff_{name}"] = home_state[name] - away_state[name]
        rows.append(row)

        # Only now, after the features are recorded, fold this match into the state.
        for team in (home, away):
            scored = match.home_goals if team == home else match.away_goals
            conceded = match.away_goals if team == home else match.home_goals
            book = history[team]
            book["goals_scored"].append(float(scored))
            book["goals_conceded"].append(float(conceded))
            for stat, value in xg_lookup.get((team, date), {}).items():
                if pd.notna(value):
                    book[stat].append(float(value))
            last_played[team] = date

    features = pd.DataFrame(rows)
    if drop_seasons:
        features = features.loc[~features["season"].isin(drop_seasons)].reset_index(drop=True)

    logger.info("Built %d match feature rows", len(features))
    return features
