"""Tests for the award models and the assembled predictions payload."""

import numpy as np
import pandas as pd
import pytest

from src.features.player_features import QUALIFYING_MINUTES
from src.models.awards import (
    FEATURE_COLUMNS,
    POTS_WEIGHTS,
    TOP_N,
    award_probabilities,
    build_player_state,
    build_preseason_state,
    player_of_the_season,
)
from src.models.predictions import PREDICTIONS_PATH, load_predictions, player_slug

# ----------------------------------------------------------------------------------
# Award probabilities
# ----------------------------------------------------------------------------------


def test_probabilities_sum_to_one():
    totals = np.array([10.0, 8.0, 6.0, 3.0])
    remaining = np.array([5.0, 6.0, 7.0, 2.0])
    probabilities = award_probabilities(totals, remaining, n_simulations=5000, seed=1)

    assert probabilities.sum() == pytest.approx(1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_an_unassailable_lead_is_near_certain():
    """20 goals clear with almost nothing left to play for should be ~100%."""
    totals = np.array([30.0, 5.0, 4.0])
    remaining = np.array([0.1, 0.1, 0.1])
    probabilities = award_probabilities(totals, remaining, n_simulations=5000, seed=2)

    assert probabilities[0] > 0.99


def test_identical_players_split_the_probability():
    """Tie handling: three indistinguishable players must each get about a third."""
    totals = np.array([5.0, 5.0, 5.0])
    remaining = np.array([4.0, 4.0, 4.0])
    probabilities = award_probabilities(totals, remaining, n_simulations=20000, seed=3)

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities == pytest.approx(np.full(3, 1 / 3), abs=0.02)


def test_a_dead_heat_splits_rather_than_picking_one():
    """With zero remaining goals and equal totals, credit is shared, not awarded."""
    probabilities = award_probabilities(
        np.array([10.0, 10.0]), np.array([0.0, 0.0]), n_simulations=100, seed=4
    )
    assert probabilities == pytest.approx(np.array([0.5, 0.5]))


def test_empty_candidate_pool_is_handled():
    assert len(award_probabilities(np.zeros(0), np.zeros(0))) == 0


def test_trailing_player_can_still_win():
    """A player behind but with far more expected goals must have real probability."""
    totals = np.array([12.0, 6.0])
    remaining = np.array([1.0, 12.0])
    probabilities = award_probabilities(totals, remaining, n_simulations=5000, seed=5)
    assert probabilities[1] > probabilities[0]


# ----------------------------------------------------------------------------------
# Feature building and leakage
# ----------------------------------------------------------------------------------


def _fake_player_matches(n_matches: int = 30) -> pd.DataFrame:
    """Two clubs, four players, deterministic scoring."""
    rows = []
    date = pd.Timestamp("2024-08-17")
    for match in range(n_matches):
        for team, opponent in (("alpha", "bravo"), ("bravo", "alpha")):
            for player in range(2):
                pid = f"{team}-{player}"
                rows.append(
                    {
                        "player_id": pid,
                        "player_name": f"Player {pid}",
                        "position": "FW" if player == 0 else "MC",
                        "minutes": 90.0,
                        "goals": float((match + player) % 3 == 0),
                        "assists": float((match + player) % 4 == 0),
                        "xg": 0.4,
                        "xa": 0.3,
                        "match_id": f"m{match}-{team}",
                        "team_slug": team,
                        "opponent_slug": opponent,
                        "date": date,
                        "season": "2024/25",
                        "goals_per90_last5": 0.5,
                        "assists_per90_last5": 0.2,
                        "trend_goals": 0.0,
                        "trend_assists": 0.0,
                    }
                )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _fake_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_slug": ["alpha", "bravo"],
            "team_elo": [1600.0, 1400.0],
            "team_attack": [0.3, -0.2],
            "team_defence": [0.2, -0.1],
        }
    )


def test_player_state_has_every_feature_column():
    state = build_player_state(_fake_player_matches(), "2024/25", 10, _fake_ratings())
    missing = [column for column in FEATURE_COLUMNS if column not in state.columns]
    assert missing == [], f"missing feature columns: {missing}"
    assert not state[FEATURE_COLUMNS].isna().any().any()


def test_features_do_not_leak_matches_after_the_cutoff():
    """The row for cutoff 10 must be identical whether or not later matches exist.

    Same guarantee, and the same tampering trick, as the feature-layer leakage test.
    A model trained on leaked rows validates beautifully and predicts nothing.
    """
    full = _fake_player_matches(30)
    truncated = full[
        full["match_id"].isin([f"m{i}-{t}" for i in range(10) for t in ("alpha", "bravo")])
    ]

    from_full = build_player_state(full, "2024/25", 10, _fake_ratings())
    from_truncated = build_player_state(truncated, "2024/25", 10, _fake_ratings())

    columns = [*FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        from_full.sort_values("player_id")[columns].reset_index(drop=True),
        from_truncated.sort_values("player_id")[columns].reset_index(drop=True),
    )


def test_targets_only_count_matches_after_the_cutoff():
    matches = _fake_player_matches(30)
    state = build_player_state(matches, "2024/25", 10, _fake_ratings()).set_index("player_id")

    player = matches[matches["player_id"] == "alpha-0"]
    order = player.sort_values("date")
    expected = order.iloc[10:]["goals"].sum()

    assert state.loc["alpha-0", "goals_remaining"] == pytest.approx(expected)


def test_preseason_state_starts_everyone_at_zero():
    """Before a ball is kicked nobody has scored, and a full season remains."""
    state = build_preseason_state(_fake_player_matches(), "2024/25", _fake_ratings())

    assert (state["goals_to_date"] == 0).all()
    assert (state["assists_to_date"] == 0).all()
    assert (state["minutes_to_date"] == 0).all()
    assert (state["matches_remaining"] == 38).all()
    # The rates must survive, or there is no signal left at all.
    assert (state["goals_per90_to_date"] > 0).any()


def test_preseason_state_supplies_every_feature_column():
    state = build_preseason_state(_fake_player_matches(), "2024/25", _fake_ratings())
    missing = [column for column in FEATURE_COLUMNS if column not in state.columns]
    assert missing == [], f"missing feature columns: {missing}"
    assert not state[FEATURE_COLUMNS].isna().any().any()


def test_preseason_and_midseason_states_share_a_feature_schema():
    """Train and serve must agree, or the model extrapolates outside what it saw."""
    mid = build_player_state(_fake_player_matches(), "2024/25", 10, _fake_ratings())
    pre = build_preseason_state(_fake_player_matches(), "2024/25", _fake_ratings())
    assert set(FEATURE_COLUMNS) <= set(mid.columns)
    assert set(FEATURE_COLUMNS) <= set(pre.columns)


# ----------------------------------------------------------------------------------
# Player of the season
# ----------------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert sum(POTS_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(POTS_WEIGHTS) == {"attacking", "team", "minutes"}


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["Best", "Middle", "Cameo"],
            "team_slug": ["alpha", "bravo", "alpha"],
            "predicted_goals": [25.0, 10.0, 1.0],
            "predicted_assists": [10.0, 5.0, 0.0],
            "predicted_minutes": [3200.0, 2500.0, 100.0],
        }
    )


def test_leader_on_every_component_ranks_first():
    ranked = player_of_the_season(_candidates(), {"alpha": 88.0, "bravo": 50.0})
    assert ranked.iloc[0]["player_name"] == "Best"
    assert ranked.iloc[0]["score"] == pytest.approx(1.0)


def test_low_minute_players_are_excluded():
    ranked = player_of_the_season(_candidates(), {"alpha": 88.0, "bravo": 50.0})
    assert "Cameo" not in set(ranked["player_name"])
    assert QUALIFYING_MINUTES == 450


def test_score_is_the_weighted_sum_of_its_components():
    ranked = player_of_the_season(_candidates(), {"alpha": 88.0, "bravo": 50.0})
    row = ranked.iloc[1]
    expected = (
        POTS_WEIGHTS["attacking"] * row["score_attacking"]
        + POTS_WEIGHTS["team"] * row["score_team"]
        + POTS_WEIGHTS["minutes"] * row["score_minutes"]
    )
    assert row["score"] == pytest.approx(expected)


def test_team_performance_actually_moves_the_ranking():
    """Two equal players at different clubs must be separated by team strength."""
    candidates = pd.DataFrame(
        {
            "player_name": ["AtGoodClub", "AtBadClub"],
            "team_slug": ["alpha", "bravo"],
            "predicted_goals": [15.0, 15.0],
            "predicted_assists": [5.0, 5.0],
            "predicted_minutes": [3000.0, 3000.0],
        }
    )
    ranked = player_of_the_season(candidates, {"alpha": 90.0, "bravo": 30.0})
    assert ranked.iloc[0]["player_name"] == "AtGoodClub"


def test_empty_pool_returns_empty():
    empty = pd.DataFrame(
        columns=[
            "player_name",
            "team_slug",
            "predicted_goals",
            "predicted_assists",
            "predicted_minutes",
        ]
    )
    assert player_of_the_season(empty, {}).empty


# ----------------------------------------------------------------------------------
# Slugs
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Erling Haaland", "erling-haaland"),
        ("Bruno Fernandes", "bruno-fernandes"),
        ("Enzo Fernández", "enzo-fernandez"),
        ("N'Golo Kanté", "n-golo-kante"),
    ],
)
def test_player_slug_is_ascii_kebab_case(name, expected):
    assert player_slug(name) == expected


# ----------------------------------------------------------------------------------
# The written payload
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def payload():
    if not PREDICTIONS_PATH.exists():
        pytest.skip("predictions.json not built - run src.models.cli --awards")
    return load_predictions()


def test_payload_carries_the_full_table(payload):
    table = payload["table"]
    assert len(table) == 20
    assert {"team", "slug", "predicted_position", "title_probability"} <= set(table[0])


def test_payload_has_every_award(payload):
    for key in ("champion", "top_scorer", "top_assists", "player_of_the_season"):
        assert key in payload, key


@pytest.mark.parametrize("award", ["top_scorer", "top_assists", "player_of_the_season"])
def test_each_award_reports_five_candidates(payload, award):
    candidates = payload[award]["candidates"]
    assert len(candidates) == TOP_N
    assert len({row["player"] for row in candidates}) == TOP_N


@pytest.mark.parametrize("award", ["top_scorer", "top_assists"])
def test_candidate_probabilities_are_ordered_and_valid(payload, award):
    probabilities = [row["probability"] for row in payload[award]["candidates"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_predicted_counts_are_never_negative(payload):
    for row in payload["top_scorer"]["candidates"]:
        assert row["predicted_goals"] >= 0
    for row in payload["top_assists"]["candidates"]:
        assert row["predicted_assists"] >= 0


def test_pots_components_are_normalised(payload):
    for row in payload["player_of_the_season"]["candidates"]:
        for value in row["components"].values():
            assert 0.0 <= value <= 1.0


def test_assumptions_are_recorded_machine_readably(payload):
    """A caveat nobody can read is a caveat that gets forgotten."""
    assumptions = payload["assumptions"]
    assert isinstance(assumptions, list)
    if payload["season"] == "2026/27":
        assert assumptions, "a carried-forward-squad prediction must say so"
        assert any("carried forward" in note for note in assumptions)


DASHBOARD_TOP_LEVEL = (
    "generated_at",
    "season",
    "model_version",
    "n_simulations",
    "assumptions",
    "table",
    "champion",
    "top_scorer",
    "top_assists",
    "player_of_the_season",
)

DASHBOARD_TABLE_FIELDS = (
    "team",
    "slug",
    "predicted_rank",
    "title_probability",
    "expected_points",
    "position_distribution",
)


@pytest.mark.parametrize("key", DASHBOARD_TOP_LEVEL)
def test_payload_has_the_fields_the_dashboard_reads(payload, key):
    """The dashboard reads these by name; a rename should break a test, not the page."""
    assert key in payload


@pytest.mark.parametrize("field", DASHBOARD_TABLE_FIELDS)
def test_every_table_row_has_the_dashboard_fields(payload, field):
    for row in payload["table"]:
        assert field in row, f"{row.get('slug')} is missing {field}"


def test_position_distribution_supports_the_finish_range(payload):
    """The range mark walks this to the 10th/50th/90th percentile - it must be complete."""
    for row in payload["table"]:
        distribution = row["position_distribution"]
        assert set(distribution) == {str(p) for p in range(1, 21)}
        assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("award", ["top_scorer", "top_assists", "player_of_the_season"])
def test_award_candidates_carry_the_fields_the_cards_render(payload, award):
    for row in payload[award]["candidates"]:
        assert {"player", "slug", "team", "team_slug"} <= set(row)


def test_champion_matches_the_top_of_the_table(payload):
    """The hero card and the table must not disagree about who wins."""
    assert payload["champion"]["slug"] == payload["table"][0]["slug"]
    assert payload["champion"]["probability"] == payload["table"][0]["title_probability"]


def test_validation_scores_are_reported(payload):
    validation = payload["validation"]
    for stat in ("goals", "assists"):
        assert f"{stat}_mae" in validation
        assert f"{stat}_baseline_mae" in validation
