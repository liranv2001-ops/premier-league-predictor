"""Tests for the backtest harness.

The leakage test is the one that matters. A backtest that quietly trains on the season
it is scoring produces excellent numbers and tells you nothing, and nothing else in the
output would look wrong.
"""

import pandas as pd
import pytest

from src.models.backtest import (
    MIN_TRAINING_SEASONS,
    _select_decay_without_peeking,
    backtest_season,
    backtestable_seasons,
    carry_forward_baseline,
    final_table,
    run_backtest,
    summarise,
)

CLUBS = [f"club-{i:02d}" for i in range(20)]


def _season(season: str, start: str, strength: list[str] | None = None) -> pd.DataFrame:
    """A full double round-robin where earlier clubs in `strength` beat later ones."""
    order = strength or CLUBS
    rank_of = {slug: i for i, slug in enumerate(order)}
    rows = []
    date = pd.Timestamp(start)
    for home in order:
        for away in order:
            if home == away:
                continue
            # The stronger club (lower index) wins by a margin that grows with the gap.
            gap = rank_of[away] - rank_of[home]
            home_goals = max(0, 1 + gap // 5)
            away_goals = max(0, 1 - gap // 5)
            rows.append(
                {
                    "date": date,
                    "season": season,
                    "season_start_year": int(season[:4]),
                    "home_slug": home,
                    "away_slug": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )
            date += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_matches():
    return pd.concat(
        [
            _season("2021/22", "2021-08-14"),
            _season("2022/23", "2022-08-13"),
            _season("2023/24", "2023-08-12"),
        ],
        ignore_index=True,
    )


# ----------------------------------------------------------------------------------
# final_table
# ----------------------------------------------------------------------------------


def test_final_table_ranks_by_points_then_goal_difference():
    matches = pd.DataFrame(
        [
            # a: 3 pts GD +1.  b: 3 pts GD +5.  c: 0 pts.
            {"home_slug": "a", "away_slug": "c", "home_goals": 1, "away_goals": 0},
            {"home_slug": "b", "away_slug": "c", "home_goals": 5, "away_goals": 0},
        ]
    )
    table = final_table(matches)
    assert list(table["slug"]) == ["b", "a", "c"]
    assert list(table["rank"]) == [1, 2, 3]


def test_final_table_awards_a_point_each_for_a_draw():
    matches = pd.DataFrame([{"home_slug": "a", "away_slug": "b", "home_goals": 2, "away_goals": 2}])
    table = final_table(matches).set_index("slug")
    assert table.loc["a", "points"] == 1
    assert table.loc["b", "points"] == 1


def test_final_table_totals_match_the_matches_played(synthetic_matches):
    season = synthetic_matches[synthetic_matches["season"] == "2021/22"]
    table = final_table(season)
    draws = int((season["home_goals"] == season["away_goals"]).sum())
    assert table["points"].sum() == 3 * len(season) - draws


# ----------------------------------------------------------------------------------
# The baseline
# ----------------------------------------------------------------------------------


def test_baseline_keeps_survivors_in_order_and_puts_newcomers_last():
    previous = pd.DataFrame(
        {"slug": ["a", "b", "c", "d"], "rank": [1, 2, 3, 4], "points": [90, 80, 40, 30]}
    )
    # c and d went down; e and f came up.
    ranks = carry_forward_baseline(previous, ["a", "b", "e", "f"])

    assert ranks["a"] == 1
    assert ranks["b"] == 2
    assert {ranks["e"], ranks["f"]} == {3, 4}


def test_baseline_covers_every_club_exactly_once():
    previous = pd.DataFrame({"slug": CLUBS, "rank": range(1, 21), "points": range(20, 0, -1)})
    ranks = carry_forward_baseline(previous, CLUBS)

    assert sorted(ranks.values()) == list(range(1, 21))


# ----------------------------------------------------------------------------------
# Leakage - the test that matters
# ----------------------------------------------------------------------------------


def test_training_window_excludes_the_predicted_season(synthetic_matches, monkeypatch):
    """The model must never see a single match from the season it is scored on."""
    seen: list[pd.DataFrame] = []
    import src.models.backtest as backtest_module

    original = backtest_module.fit_dixon_coles

    def spy(matches, **kwargs):
        seen.append(matches)
        return original(matches, **kwargs)

    monkeypatch.setattr(backtest_module, "fit_dixon_coles", spy)
    backtest_season(synthetic_matches, "2023/24", n_simulations=50)

    assert seen, "the model was never fitted"
    for frame in seen:
        assert "2023/24" not in set(frame["season"]), "the predicted season leaked into training"


def test_decay_selection_never_sees_the_predicted_season(synthetic_matches, monkeypatch):
    """The hyperparameter is the subtle leak: it must be chosen inside training only."""
    scored: list[pd.DataFrame] = []
    import src.models.backtest as backtest_module

    def spy(train, validation, **kwargs):
        scored.append(validation)
        return 0.002, {}

    monkeypatch.setattr(backtest_module, "select_decay", spy)
    backtest_season(synthetic_matches, "2023/24", n_simulations=50)

    for frame in scored:
        assert "2023/24" not in set(frame["season"])


def test_decay_selection_holds_out_the_last_training_season(synthetic_matches):
    history = synthetic_matches[synthetic_matches["season"] != "2023/24"]
    decay = _select_decay_without_peeking(history)
    assert isinstance(decay, float)
    assert decay >= 0.0


def test_decay_selection_falls_back_with_one_season(synthetic_matches):
    single = synthetic_matches[synthetic_matches["season"] == "2021/22"]
    assert _select_decay_without_peeking(single) == 0.002


# ----------------------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------------------


def test_only_seasons_with_enough_history_are_eligible(synthetic_matches):
    eligible = backtestable_seasons(synthetic_matches)
    assert eligible == ["2023/24"]
    assert MIN_TRAINING_SEASONS == 2


def test_a_season_without_enough_history_is_refused(synthetic_matches):
    with pytest.raises(ValueError, match="prior season"):
        backtest_season(synthetic_matches, "2022/23", n_simulations=10)


def test_an_unknown_season_is_refused(synthetic_matches):
    with pytest.raises(ValueError, match="not in the data"):
        backtest_season(synthetic_matches, "1999/00", n_simulations=10)


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(synthetic_matches):
    return backtest_season(synthetic_matches, "2023/24", n_simulations=400)


def test_every_club_appears_once(result):
    assert len(result.per_club) == 20
    assert result.per_club["slug"].nunique() == 20


def test_predicted_ranks_are_a_permutation(result):
    assert sorted(result.per_club["predicted_rank"]) == list(range(1, 21))
    assert sorted(result.per_club["actual_rank"]) == list(range(1, 21))


def test_position_error_is_the_mean_absolute_difference(result):
    expected = (result.per_club["predicted_rank"] - result.per_club["actual_rank"]).abs().mean()
    assert result.mean_position_error == pytest.approx(expected)


def test_interval_coverage_counts_clubs_inside_their_own_band(result):
    inside = (
        (result.per_club["actual_rank"] >= result.per_club["interval_low"])
        & (result.per_club["actual_rank"] <= result.per_club["interval_high"])
    ).sum()
    assert result.interval_coverage == pytest.approx(inside / 20)
    assert 0.0 <= result.interval_coverage <= 1.0


def test_interval_bounds_are_ordered_and_in_range(result):
    assert (result.per_club["interval_low"] <= result.per_club["interval_high"]).all()
    assert result.per_club["interval_low"].between(1, 20).all()
    assert result.per_club["interval_high"].between(1, 20).all()


def test_champion_probability_is_a_probability(result):
    assert 0.0 <= result.champion_probability <= 1.0


def test_overlaps_cannot_exceed_the_band_size(result):
    assert 0 <= result.ucl_overlap <= 4
    assert 0 <= result.relegation_overlap <= 3


def test_a_learnable_league_is_predicted_well(result):
    """The synthetic league is deterministic, so the model should nearly nail it."""
    assert result.champion_correct
    assert result.mean_position_error < 2.0
    assert result.spearman > 0.9


def test_summarise_averages_across_seasons(synthetic_matches, result):
    summary = summarise([result])
    assert summary["seasons"] == 1
    assert summary["mean_position_error"] == pytest.approx(result.mean_position_error)
    assert summary["champion_hit_rate"] == pytest.approx(float(result.champion_correct))


def test_summarise_handles_no_results():
    assert summarise([]) == {}


def test_run_backtest_covers_every_eligible_season(synthetic_matches):
    results = run_backtest(synthetic_matches, n_simulations=50)
    assert [r.season for r in results] == backtestable_seasons(synthetic_matches)
