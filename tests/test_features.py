"""Tests for src/features.

Split into two layers:

* Property tests on synthetic fixtures - fast, deterministic, and the only place the
  leakage guarantee can actually be proven.
* Contract tests against the real built tables in ``data/processed/pl.db``, skipped
  when the database has not been built.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_collection.storage import read_table
from src.features.config import (
    MATCH_FEATURES_TABLE,
    PLAYER_MATCH_FEATURES_TABLE,
    PLAYER_SEASON_FEATURES_TABLE,
    PROCESSED_DB_PATH,
)
from src.features.elo import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    EloEngine,
    expected_score,
    margin_multiplier,
)
from src.features.player_features import (
    QUALIFYING_MINUTES,
    build_player_match_features,
    build_player_season_features,
)
from src.features.team_features import (
    MAX_REST_DAYS,
    PROMOTED_RANK,
    build_match_features,
    compute_season_table,
)

# ----------------------------------------------------------------------------------
# ELO
# ----------------------------------------------------------------------------------


def test_equal_ratings_favour_the_home_side():
    assert expected_score(1500, 1500, HOME_ADVANTAGE) > 0.5
    assert expected_score(1500, 1500, 0.0) == pytest.approx(0.5)


def test_expected_scores_are_complementary():
    home = expected_score(1600, 1400, HOME_ADVANTAGE)
    away = expected_score(1400, 1600 + HOME_ADVANTAGE, 0.0)
    assert home + away == pytest.approx(1.0)


def test_margin_multiplier_grows_with_scoreline():
    assert margin_multiplier(0) == pytest.approx(1.0)
    assert margin_multiplier(1) < margin_multiplier(3) < margin_multiplier(5)


def test_elo_updates_are_zero_sum():
    engine = EloEngine()
    engine.start_season(["a", "b"])
    before = sum(engine.ratings.values())
    engine.rate_match("a", "b", 3, 0)
    assert sum(engine.ratings.values()) == pytest.approx(before)


def test_rate_match_returns_pre_match_ratings():
    """The returned ratings must not include this match's own result."""
    engine = EloEngine()
    engine.start_season(["a", "b"])
    home_pre, away_pre = engine.rate_match("a", "b", 5, 0)

    assert home_pre == INITIAL_RATING
    assert away_pre == INITIAL_RATING
    assert engine.ratings["a"] > home_pre  # the update landed afterwards


def test_winning_team_overtakes_losing_team():
    engine = EloEngine()
    engine.start_season(["winner", "loser"])
    for _ in range(10):
        engine.rate_match("winner", "loser", 2, 0)
    assert engine.ratings["winner"] > engine.ratings["loser"]


def test_draw_between_equals_barely_moves_ratings():
    engine = EloEngine()
    engine.start_season(["a", "b"])
    engine.rate_match("a", "b", 1, 1)
    # The home side was favoured, so a draw costs it a little.
    assert engine.ratings["a"] < INITIAL_RATING
    assert abs(engine.ratings["a"] - INITIAL_RATING) < 15


def test_promoted_club_starts_below_the_mean():
    engine = EloEngine()
    engine.start_season(["a", "b"])
    assert engine.rating_of("a") == INITIAL_RATING
    # A club appearing only later has been promoted into the division.
    assert engine.rating_of("newcomer") < INITIAL_RATING


# ----------------------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------------------

TEAMS = ["alpha", "bravo", "charlie", "delta"]


def _synthetic_matches(n_rounds: int = 6, seasons=("2023/24", "2024/25")) -> pd.DataFrame:
    """Build a deterministic round-robin fixture list for testing."""
    rows = []
    for season_index, season in enumerate(seasons):
        date = pd.Timestamp(f"{2023 + season_index}-08-12")
        for rnd in range(n_rounds):
            for i in range(0, len(TEAMS), 2):
                home, away = TEAMS[i], TEAMS[(i + 1) % len(TEAMS)]
                if rnd % 2:
                    home, away = away, home
                rows.append(
                    {
                        "date": date,
                        "season": season,
                        "season_start_year": 2023 + season_index,
                        "home_slug": home,
                        "away_slug": away,
                        "home_goals": (rnd + i) % 4,
                        "away_goals": (rnd + i + 1) % 3,
                    }
                )
            date += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["result"] = np.where(
        df.home_goals > df.away_goals, "H", np.where(df.home_goals < df.away_goals, "A", "D")
    )
    return df


def _synthetic_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Minimal Understat-shaped xG rows matching the synthetic fixtures."""
    rows = []
    for match in matches.itertuples(index=False):
        for team, scored, conceded in (
            (match.home_slug, match.home_goals, match.away_goals),
            (match.away_slug, match.away_goals, match.home_goals),
        ):
            rows.append(
                {
                    "team_slug": team,
                    "date": match.date,
                    "xg": scored * 0.9 + 0.3,
                    "xga": conceded * 0.9 + 0.3,
                    "npxgd": (scored - conceded) * 0.8,
                    "ppda": 10.0 + scored,
                    "xpts": 1.0 + 0.3 * scored,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic():
    matches = _synthetic_matches()
    return matches, _synthetic_team_stats(matches)


# ----------------------------------------------------------------------------------
# The leakage guarantee
# ----------------------------------------------------------------------------------


def test_features_do_not_leak_future_results(synthetic):
    """Changing the LAST match must not change any earlier row.

    This is the test that matters. A leak - an off-by-one in a rolling window, a
    forgotten shift(1) - produces a model that validates beautifully and predicts
    nothing, and it is invisible in every other check.
    """
    matches, team_stats = synthetic
    baseline = build_match_features(matches, team_stats)

    tampered = matches.copy()
    last = tampered.index[-1]
    tampered.loc[last, "home_goals"] = 99
    tampered.loc[last, "away_goals"] = 0
    tampered.loc[last, "result"] = "H"
    tampered_stats = _synthetic_team_stats(tampered)

    after = build_match_features(tampered, tampered_stats)

    feature_columns = [c for c in baseline.columns if c.startswith(("home_", "away_", "diff_"))]
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1][feature_columns],
        after.iloc[:-1][feature_columns],
    )


def test_player_features_do_not_leak_future_appearances():
    """A player's features for match N must ignore match N and everything after it."""
    base = pd.DataFrame(
        {
            "player_id": ["1"] * 6,
            "player_name": ["Someone"] * 6,
            "season": ["2024/25"] * 6,
            "date": pd.date_range("2024-08-17", periods=6, freq="7D"),
            "minutes": [90] * 6,
            "goals": [0, 1, 0, 2, 1, 0],
            "assists": [1, 0, 0, 1, 0, 0],
        }
    )
    baseline = build_player_match_features(base)

    tampered = base.copy()
    tampered.loc[5, "goals"] = 50
    after = build_player_match_features(tampered)

    columns = [c for c in baseline.columns if c.startswith(("goals_", "assists_", "trend_"))]
    pd.testing.assert_frame_equal(baseline.iloc[:-1][columns], after.iloc[:-1][columns])


def test_first_appearance_has_zero_history():
    df = pd.DataFrame(
        {
            "player_id": ["1", "1"],
            "player_name": ["Someone"] * 2,
            "season": ["2024/25"] * 2,
            "date": [pd.Timestamp("2024-08-17"), pd.Timestamp("2024-08-24")],
            "minutes": [90, 90],
            "goals": [3, 0],
            "assists": [0, 0],
        }
    )
    out = build_player_match_features(df).sort_values("date").reset_index(drop=True)

    # Nothing precedes the first appearance, so every rolling column must be zero -
    # in particular it must not see its own 3 goals.
    assert out.loc[0, "goals_last5"] == 0
    assert out.loc[0, "goals_per90_last5"] == 0
    assert out.loc[1, "goals_last5"] == 3  # now it sees match 1, and only match 1


# ----------------------------------------------------------------------------------
# Determinism and shape
# ----------------------------------------------------------------------------------


def test_build_is_deterministic(synthetic):
    matches, team_stats = synthetic
    pd.testing.assert_frame_equal(
        build_match_features(matches, team_stats),
        build_match_features(matches, team_stats),
    )


def test_one_row_per_match(synthetic):
    matches, team_stats = synthetic
    out = build_match_features(matches, team_stats)
    assert len(out) == len(matches)
    assert not out.duplicated(["season", "date", "home_slug", "away_slug"]).any()


def test_drop_seasons_excludes_history_only_season(synthetic):
    matches, team_stats = synthetic
    out = build_match_features(matches, team_stats, drop_seasons=("2023/24",))
    assert set(out["season"]) == {"2024/25"}


def test_season_table_ranks_and_points(synthetic):
    matches, _ = synthetic
    table = compute_season_table(matches)
    for season, group in table.groupby("season"):
        assert sorted(group["rank"]) == list(range(1, len(group) + 1))
        # Every match awards 3 points in total, or 2 for a draw.
        played = matches[matches.season == season]
        draws = (played.home_goals == played.away_goals).sum()
        assert group["points"].sum() == 3 * len(played) - draws


def test_synthetic_features_have_no_nan(synthetic):
    matches, team_stats = synthetic
    out = build_match_features(matches, team_stats)
    missing = out.columns[out.isna().any()].tolist()
    assert missing == [], f"NaN in: {missing}"


# ----------------------------------------------------------------------------------
# Player rates
# ----------------------------------------------------------------------------------


def _player_seasons(n_filler: int = 40) -> pd.DataFrame:
    """Three interesting players plus a realistic supporting cast.

    The filler matters. Shrinkage pulls toward a prior estimated from the league, so a
    fixture of three players would derive that prior from the three players themselves
    and produce a nonsensically high league scoring rate. Real seasons have ~550
    players; the filler reproduces a believable population so the prior is meaningful.
    """
    interesting = {
        "player_id": ["1", "2", "3"],
        "player_name": ["Star", "Cameo", "Bench"],
        "season": ["2024/25"] * 3,
        "team_slug": ["alpha", None, "bravo"],
        "n_teams": [1, 2, 1],
        "games": [38, 1, 0],
        "minutes": [3420, 1, 0],
        "goals": [30, 1, 0],
        "assists": [10, 0, 0],
        "xg": [25.0, 0.1, 0.0],
        "xa": [9.0, 0.0, 0.0],
        "shots": [120, 1, 0],
        "key_passes": [70, 0, 0],
    }
    filler = {
        "player_id": [str(i + 10) for i in range(n_filler)],
        "player_name": [f"Squad {i}" for i in range(n_filler)],
        "season": ["2024/25"] * n_filler,
        "team_slug": ["charlie"] * n_filler,
        "n_teams": [1] * n_filler,
        "games": [30] * n_filler,
        "minutes": [2700] * n_filler,
        "goals": [3] * n_filler,
        "assists": [2] * n_filler,
        "xg": [3.0] * n_filler,
        "xa": [2.0] * n_filler,
        "shots": [20] * n_filler,
        "key_passes": [15] * n_filler,
    }
    return pd.concat([pd.DataFrame(interesting), pd.DataFrame(filler)], ignore_index=True)


def test_per_90_handles_zero_minutes():
    out = build_player_season_features(_player_seasons())
    bench = out[out.player_name == "Bench"].iloc[0]
    assert bench["goals_per_90"] == 0.0  # not a division-by-zero inf or NaN


def test_shrinkage_demotes_tiny_sample_rates():
    """The whole point: a one-minute goal must not outrank a 30-goal season."""
    out = build_player_season_features(_player_seasons())
    star = out[out.player_name == "Star"].iloc[0]
    cameo = out[out.player_name == "Cameo"].iloc[0]

    assert cameo["goals_per_90"] > star["goals_per_90"]  # raw rate is nonsense
    assert star["goals_per_90_shrunk"] > cameo["goals_per_90_shrunk"]  # shrunk is not


def test_is_qualified_uses_the_minutes_threshold():
    out = build_player_season_features(_player_seasons())
    assert out.set_index("player_name").loc["Star", "is_qualified"] == 1
    assert out.set_index("player_name").loc["Cameo", "is_qualified"] == 0
    assert QUALIFYING_MINUTES == 450


def test_multi_club_players_get_a_sentinel_not_a_nan():
    out = build_player_season_features(_player_seasons())
    cameo = out[out.player_name == "Cameo"].iloc[0]
    assert cameo["team_slug"] == "multi-club"
    assert out["team_slug"].isna().sum() == 0


def test_player_season_features_have_no_nan():
    out = build_player_season_features(_player_seasons())
    missing = out.columns[out.isna().any()].tolist()
    assert missing == [], f"NaN in: {missing}"


# ----------------------------------------------------------------------------------
# Contract tests against the real built tables
# ----------------------------------------------------------------------------------


def _load(table: str) -> pd.DataFrame:
    if not PROCESSED_DB_PATH.exists():
        pytest.skip("data/processed/pl.db not built - run python -m src.features.cli")
    try:
        return read_table(table, PROCESSED_DB_PATH)
    except ValueError:
        pytest.skip(f"{table} not built yet")


@pytest.fixture(scope="module")
def match_features():
    return _load(MATCH_FEATURES_TABLE)


@pytest.fixture(scope="module")
def season_features():
    return _load(PLAYER_SEASON_FEATURES_TABLE)


def test_real_match_features_have_no_nan(match_features):
    missing = match_features.columns[match_features.isna().any()].tolist()
    assert missing == [], f"NaN in: {missing}"


def test_real_player_season_features_have_no_nan(season_features):
    missing = season_features.columns[season_features.isna().any()].tolist()
    assert missing == [], f"NaN in: {missing}"


def test_real_player_match_features_have_no_nan():
    df = _load(PLAYER_MATCH_FEATURES_TABLE)
    missing = df.columns[df.isna().any()].tolist()
    assert missing == [], f"NaN in: {missing}"


@pytest.mark.parametrize("side", ["home", "away"])
def test_elo_stays_in_a_plausible_band(match_features, side):
    elo = match_features[f"{side}_elo_pre"]
    assert elo.between(1000, 2200).all(), (elo.min(), elo.max())


@pytest.mark.parametrize("side", ["home", "away"])
def test_rolling_goal_averages_are_plausible(match_features, side):
    for column in (f"{side}_goals_scored_avg5", f"{side}_goals_conceded_avg5"):
        assert match_features[column].between(0, 6).all()


@pytest.mark.parametrize("side", ["home", "away"])
def test_rest_days_are_bounded(match_features, side):
    rest = match_features[f"{side}_rest_days"]
    assert rest.between(1, MAX_REST_DAYS).all()


@pytest.mark.parametrize("side", ["home", "away"])
def test_previous_season_rank_is_a_valid_position(match_features, side):
    rank = match_features[f"{side}_prev_season_rank"]
    assert rank.between(1, PROMOTED_RANK).all()
    assert set(rank.unique()) <= set(range(1, PROMOTED_RANK + 1))


@pytest.mark.parametrize("side", ["home", "away"])
def test_xg_features_are_non_negative_and_ppda_is_positive(match_features, side):
    assert (match_features[f"{side}_xg_avg5"] >= 0).all()
    assert (match_features[f"{side}_xga_avg5"] >= 0).all()
    assert (match_features[f"{side}_ppda_avg5"] > 0).all()
    assert match_features[f"{side}_xpts_avg5"].between(0, 3).all()


def test_rolling_window_never_exceeds_its_length(match_features):
    for side in ("home", "away"):
        assert match_features[f"{side}_n_prior_matches"].between(0, 5).all()


def test_every_season_has_a_full_fixture_list(match_features):
    counts = match_features.groupby("season").size()
    assert (counts == 380).all(), counts.to_dict()


def test_exactly_three_promoted_clubs_per_season(match_features):
    """A sanity check that the sentinel is landing on real promotions, not on gaps."""
    for season, group in match_features.groupby("season"):
        promoted = set(group.loc[group["home_was_promoted"] == 1, "home_slug"])
        assert len(promoted) == 3, f"{season}: {promoted}"


def test_per_90_rates_are_non_negative(season_features):
    rate_columns = [c for c in season_features.columns if "_per_90" in c]
    assert rate_columns
    assert (season_features[rate_columns] >= 0).all().all()


def test_qualified_players_have_believable_scoring_rates(season_features):
    qualified = season_features[season_features["is_qualified"] == 1]
    assert qualified["goals_per_90"].max() < 3.0
    assert qualified["assists_per_90"].max() < 2.0


def test_shrunk_rates_are_never_more_extreme_than_the_league_leader(season_features):
    """Shrinkage pulls toward the mean, so it can only ever reduce the maximum."""
    for stat in ("goals", "assists"):
        assert (
            season_features[f"{stat}_per_90_shrunk"].max()
            <= season_features[f"{stat}_per_90"].max()
        )
