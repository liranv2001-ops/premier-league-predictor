"""Player-level features: per-90 rates and form trends.

Two tables:

* ``player_season_features`` - one row per player-season, with per-90 rates and their
  shrunk counterparts.
* ``player_match_features`` - one row per appearance, with pre-match rolling form and
  the trend of recent form against the season so far.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Rolling form window, in appearances.
FORM_WINDOW = 5

#: Minutes below which a per-90 rate is not meaningful (five full matches).
QUALIFYING_MINUTES = 450

#: Shrinkage strength, in minutes. A player with exactly this many minutes is pulled
#: halfway to the league mean; someone with a full season barely moves.
#:
#: Without this, per-90 rates are unusable for ranking: minutes in this data run from 1
#: to 3,420, and a single goal in a single minute is a raw rate of 90 goals/90 - which
#: would top any Golden Boot ranking built on the raw column.
SHRINKAGE_MINUTES = 900

#: Counting stats that get a per-90 rate.
RATE_STATS = ("goals", "assists", "xg", "xa", "shots", "key_passes")

#: Stands in for ``team_slug`` when a player represented more than one club that
#: season. Not a real slug, and deliberately so - it must not join to a club.
MULTI_CLUB_SENTINEL = "multi-club"


def _per_90(counts: pd.Series, minutes: pd.Series) -> pd.Series:
    """Convert a counting stat to a per-90-minutes rate.

    Args:
        counts: The counting statistic.
        minutes: Minutes played.

    Returns:
        Rate per 90 minutes; 0.0 where no minutes were played.
    """
    rate = counts.astype(float) * 90.0 / minutes.astype(float)
    return rate.where(minutes.astype(float) > 0, 0.0)


def _shrink(rate: pd.Series, minutes: pd.Series, prior: float) -> pd.Series:
    """Pull a rate toward a prior, weighted by how much evidence backs it.

    Args:
        rate: Raw per-90 rate.
        minutes: Minutes played - the evidence behind the rate.
        prior: League-average rate to shrink toward.

    Returns:
        The shrunk rate.
    """
    weight = minutes.astype(float) / (minutes.astype(float) + SHRINKAGE_MINUTES)
    return weight * rate + (1.0 - weight) * prior


def build_player_season_features(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Build per-90 rates and shrunk rates for each player-season.

    Args:
        player_seasons: ``player_season_stats`` rows from the raw database.

    Returns:
        One row per player-season, with no NaN values.
    """
    df = player_seasons.copy()
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)

    for stat in RATE_STATS:
        df[stat] = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)
        df[f"{stat}_per_90"] = _per_90(df[stat], df["minutes"])

    df["is_qualified"] = (df["minutes"] >= QUALIFYING_MINUTES).astype(int)

    # The prior is the league's overall rate for that season - total events over total
    # minutes - not the mean of per-player rates, which would be dominated by cameos.
    for stat in RATE_STATS:
        totals = df.groupby("season").agg(events=(stat, "sum"), mins=("minutes", "sum"))
        prior = (totals["events"] * 90.0 / totals["mins"]).rename(f"{stat}_prior")
        df = df.merge(prior, left_on="season", right_index=True, how="left")
        df[f"{stat}_per_90_shrunk"] = _shrink(
            df[f"{stat}_per_90"], df["minutes"], df[f"{stat}_prior"]
        )
        df = df.drop(columns=[f"{stat}_prior"])

    df["minutes_per_game"] = (df["minutes"] / df["games"].clip(lower=1)).astype(float)

    # Players who moved mid-season carry a null team_slug from collection, because
    # Understat lists their clubs alphabetically and the current one is unknowable.
    # An explicit sentinel keeps the feature table NaN-free without inventing a club;
    # `n_teams` remains the column to filter on.
    df["team_slug"] = df["team_slug"].astype("string").fillna(MULTI_CLUB_SENTINEL)

    logger.info("Built %d player-season feature rows", len(df))
    return df.reset_index(drop=True)


def build_player_match_features(player_matches: pd.DataFrame) -> pd.DataFrame:
    """Build pre-match rolling form and trend features per appearance.

    For each player, in date order, the features describe the state *before* that
    match: the previous ``FORM_WINDOW`` appearances and the season up to but not
    including this one. The current match never contributes to its own features.

    Args:
        player_matches: ``player_match_stats`` rows from the raw database.

    Returns:
        One row per appearance, with no NaN values.
    """
    df = player_matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    for column in ("minutes", "goals", "assists"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float)

    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    grouped = df.groupby("player_id", sort=False)

    # shift(1) is what makes these pre-match: every window ends at the previous
    # appearance. Without it each row would see its own result.
    for stat in ("goals", "assists", "minutes"):
        df[f"{stat}_last{FORM_WINDOW}"] = (
            grouped[stat]
            .apply(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).sum())
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    recent_minutes = df[f"minutes_last{FORM_WINDOW}"]
    for stat in ("goals", "assists"):
        df[f"{stat}_per90_last{FORM_WINDOW}"] = (
            df[f"{stat}_last{FORM_WINDOW}"] * 90.0 / recent_minutes
        ).where(recent_minutes > 0, 0.0)

    # Season to date, again excluding the current match.
    season_group = df.groupby(["player_id", "season"], sort=False)
    for stat in ("goals", "assists", "minutes"):
        df[f"{stat}_season_to_date"] = (
            season_group[stat]
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )

    season_minutes = df["minutes_season_to_date"]
    for stat in ("goals", "assists"):
        df[f"{stat}_per90_season_to_date"] = (
            df[f"{stat}_season_to_date"] * 90.0 / season_minutes
        ).where(season_minutes > 0, 0.0)

        # A difference, not a ratio. A season-to-date rate of zero is common in the
        # opening weeks, and a ratio there is either undefined or an epsilon-driven
        # fiction. The difference is always defined and stays readable: positive means
        # the player is running hotter than their own season baseline.
        df[f"trend_{stat}"] = (
            df[f"{stat}_per90_last{FORM_WINDOW}"] - df[f"{stat}_per90_season_to_date"]
        )

    df["appearance_number"] = grouped.cumcount() + 1

    logger.info("Built %d player-match feature rows", len(df))
    return df.reset_index(drop=True)
