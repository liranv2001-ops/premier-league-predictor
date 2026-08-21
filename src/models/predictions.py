"""Assemble the full prediction payload: table, champion and the three awards.

Writes ``data/processed/predictions.json`` in the shape CLAUDE.md documents, so it is a
single copy step away from the dashboard.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data_collection.config import PROJECT_ROOT, RAW_DB_PATH
from src.data_collection.storage import read_table
from src.features.config import (
    MATCH_FEATURES_TABLE,
    PLAYER_MATCH_FEATURES_TABLE,
    PROCESSED_DB_PATH,
)
from src.models import (
    DEFAULT_DECAY,
    display_name,
    load_matches,
    run_season_simulation,
    split_season,
)
from src.models.awards import (
    TOP_N,
    AwardModels,
    award_probabilities,
    build_player_state,
    build_preseason_state,
    build_training_set,
    fit_award_models,
    player_of_the_season,
)
from src.models.config import N_SIMULATIONS, RANDOM_SEED
from src.models.dixon_coles import fit_dixon_coles

logger = logging.getLogger(__name__)

MODEL_VERSION = "dc-xgb-v1"
PREDICTIONS_PATH = PROCESSED_DB_PATH.parent / "predictions.json"

#: Where the dashboard reads from. Publishing is an explicit step so the page can never
#: quietly serve a payload someone forgot to copy across.
PUBLISHED_PATH = PROJECT_ROOT / "frontend" / "public" / "predictions.json"


def publish_assets() -> dict[str, int]:
    """Copy generated badges, photos and both mapping files into ``frontend/public``.

    Vite only serves what lives under ``public/``, and the dashboard needs the mappings
    as well as the images - the credits block is built from them, and attribution that
    is not displayed is not attribution.

    Returns:
        Counts of files copied per kind.
    """
    import shutil

    public = PROJECT_ROOT / "frontend" / "public"
    copied = {"logos": 0, "players": 0, "mappings": 0}

    for kind, patterns in (("logos", ("*.svg",)), ("players", ("*.jpg", "*.png", "*.svg"))):
        source_dir = PROJECT_ROOT / "assets" / kind
        target_dir = public / kind
        target_dir.mkdir(parents=True, exist_ok=True)

        # Clear first: the badges replaced earlier PNG crests, and a stale .png would
        # keep winning because the dashboard would still find it.
        for stale in target_dir.iterdir():
            if stale.is_file():
                stale.unlink()

        for pattern in patterns:
            for path in sorted(source_dir.glob(pattern)):
                shutil.copy2(path, target_dir / path.name)
                copied[kind] += 1

    for kind in ("logos", "players"):
        mapping = PROJECT_ROOT / "assets" / kind / "mapping.json"
        if mapping.exists():
            shutil.copy2(mapping, public / f"{kind}-mapping.json")
            copied["mappings"] += 1

    logger.info(
        "Published %d badges, %d player images and %d mapping files",
        copied["logos"],
        copied["players"],
        copied["mappings"],
    )
    return copied


def publish(source: Path = PREDICTIONS_PATH, destination: Path = PUBLISHED_PATH) -> Path:
    """Copy the payload to where the dashboard fetches it.

    Args:
        source: The generated payload.
        destination: The path Vite serves.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: If the payload has not been generated yet.
    """
    if not source.exists():
        raise FileNotFoundError(
            f"{source} does not exist - run the models with --awards before publishing."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Published %s -> %s", source.name, destination)
    return destination


def player_slug(name: str) -> str:
    """Turn a player name into an asset-friendly slug.

    Args:
        name: Player name as the source spells it.

    Returns:
        A kebab-case ASCII slug, matching ``assets/players/{slug}.jpg``.
    """
    slug: str = (
        pd.Series([name])
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
        .iloc[0]
    )
    return slug


def team_ratings_for(matches: pd.DataFrame, features: pd.DataFrame, season: str) -> pd.DataFrame:
    """Build the per-club quality table the award features need.

    Combines ELO at the club's most recent match with the Dixon-Coles attack and
    defence ratings, both fitted on data strictly before the season in question.

    Args:
        matches: Every match available.
        features: ``match_features`` rows.
        season: Season the ratings describe.

    Returns:
        One row per club with ``team_slug``, ``team_elo``, ``team_attack``,
        ``team_defence``.
    """
    history = matches[matches["season"] < season]
    if history.empty:
        history = matches

    model = fit_dixon_coles(history, decay=DEFAULT_DECAY)

    season_features = features[features["season"] == season]
    if season_features.empty:
        season_features = features

    elo_rows = pd.concat(
        [
            season_features[["home_slug", "home_elo_pre"]].rename(
                columns={"home_slug": "team_slug", "home_elo_pre": "team_elo"}
            ),
            season_features[["away_slug", "away_elo_pre"]].rename(
                columns={"away_slug": "team_slug", "away_elo_pre": "team_elo"}
            ),
        ]
    )
    elo = elo_rows.groupby("team_slug", as_index=False)["team_elo"].mean()

    elo["team_attack"] = elo["team_slug"].map(model.attack)
    elo["team_defence"] = elo["team_slug"].map(model.defence)
    promoted_attack, promoted_defence = model.promoted_ratings()
    elo["team_attack"] = elo["team_attack"].fillna(promoted_attack)
    elo["team_defence"] = elo["team_defence"].fillna(promoted_defence)
    return elo


def _award_block(
    candidates: pd.DataFrame,
    stat: str,
    probability_column: str,
) -> dict[str, object]:
    """Build one award entry with its top-N shortlist.

    Args:
        candidates: Scored candidates, best first.
        stat: ``"goals"`` or ``"assists"``.
        probability_column: Column holding the win probability.

    Returns:
        The award block.
    """
    shortlist = candidates.head(TOP_N)
    rows = [
        {
            "player": row["player_name"],
            "slug": player_slug(str(row["player_name"])),
            "team": display_name(str(row["team_slug"])),
            "team_slug": row["team_slug"],
            f"predicted_{stat}": round(float(row[f"predicted_{stat}"]), 2),
            f"{stat}_so_far": int(row[f"{stat}_to_date"]),
            "probability": round(float(row[probability_column]), 4),
        }
        for _, row in shortlist.iterrows()
    ]
    leader = dict(rows[0]) if rows else {}
    leader["candidates"] = rows
    return leader


def build_predictions(
    season: str,
    *,
    cutoff_matchweek: int = 0,
    teams: list[str] | None = None,
    squad_season: str | None = None,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
    write: bool = True,
) -> dict[str, object]:
    """Produce the full prediction payload for a season.

    Args:
        season: Season to predict, e.g. ``"2026/27"``.
        cutoff_matchweek: Matchweeks already played in that season.
        teams: Club list, required when the season has no matches yet.
        squad_season: Season whose squads stand in for ``season``. Required when
            ``season`` has no player data of its own.
        n_simulations: Monte Carlo runs.
        seed: Random seed.
        write: Whether to write ``predictions.json``.

    Returns:
        The payload.
    """
    matches = load_matches()
    features = read_table(MATCH_FEATURES_TABLE, PROCESSED_DB_PATH)
    player_matches = read_table(PLAYER_MATCH_FEATURES_TABLE, PROCESSED_DB_PATH)

    simulation = run_season_simulation(
        season,
        cutoff_matchweek=cutoff_matchweek if cutoff_matchweek else 38,
        teams=teams,
        n_simulations=n_simulations,
        seed=seed,
        db_path=RAW_DB_PATH,
        write_json=False,
    )
    table = simulation["teams"]
    assert isinstance(table, list)
    team_points = {str(row["slug"]): float(row["expected_points"]) for row in table}

    assumptions: list[str] = []

    # --- awards -------------------------------------------------------------------
    available_seasons = sorted(player_matches["season"].unique())
    have_player_data = season in available_seasons
    source_season = season if have_player_data else (squad_season or available_seasons[-1])

    if not have_player_data:
        assumptions.append(
            f"Player squads carried forward from {source_season}: {season} has no player "
            f"data yet. Historically only ~53% of players remain at the same club from "
            f"one season to the next, so club assignments are approximate."
        )

    ratings = team_ratings_for(matches, features, source_season)

    # The model must never train on the season it predicts.
    training_seasons = [s for s in available_seasons if s != source_season]
    ratings_by_season = {s: team_ratings_for(matches, features, s) for s in available_seasons}
    training = build_training_set(
        player_matches, ratings_by_season, seasons=[*training_seasons, source_season]
    )
    models = fit_award_models(training, validation_season=source_season, seed=seed)

    if have_player_data:
        # Mid-season: keep what has been scored so far and predict the remainder.
        state = build_player_state(player_matches, source_season, cutoff_matchweek, ratings)
        state["matches_remaining"] = 38 - cutoff_matchweek
        state["expected_remaining_minutes"] = (
            state["minutes_per_appearance"] * state["matches_remaining"]
        )
    else:
        # A fresh season. This uses the *same* builder the pre-season training rows come
        # from, so the model is asked exactly the question it was trained on. Building
        # a full-season state and then overriding matches_remaining would put every
        # feature outside the range the model ever saw.
        state = build_preseason_state(player_matches, source_season, ratings)

    # Only clubs actually in the season being predicted can supply candidates. Filter
    # BEFORE reading the totals out - taking them first and slicing afterwards would
    # pair every player with some other player's goals the moment a row is dropped.
    season_clubs = {str(row["slug"]) for row in table}
    dropped = state[~state["team_slug"].isin(season_clubs)]
    state = state[state["team_slug"].isin(season_clubs)].reset_index(drop=True)

    goals_so_far = state["goals_to_date"].to_numpy(dtype=float)
    assists_so_far = state["assists_to_date"].to_numpy(dtype=float)

    if not dropped.empty:
        assumptions.append(
            f"{len(dropped)} players excluded: their {source_season} club is not in {season}."
        )
    missing_clubs = sorted(season_clubs - set(state["team_slug"]))
    if missing_clubs:
        assumptions.append(
            "No player candidates for "
            + ", ".join(display_name(c) for c in missing_clubs)
            + " - no top-flight player data exists for them."
        )

    predicted_goals, predicted_assists = models.predict(state)
    state["predicted_goals"] = goals_so_far + predicted_goals
    state["predicted_assists"] = assists_so_far + predicted_assists
    state["predicted_minutes"] = (
        state["minutes_to_date"] + state["expected_remaining_minutes"]
    ).astype(float)

    state["goal_probability"] = award_probabilities(
        goals_so_far, predicted_goals, n_simulations=n_simulations, seed=seed
    )
    state["assist_probability"] = award_probabilities(
        assists_so_far, predicted_assists, n_simulations=n_simulations, seed=seed + 1
    )

    scorers = state.sort_values("goal_probability", ascending=False).reset_index(drop=True)
    assisters = state.sort_values("assist_probability", ascending=False).reset_index(drop=True)
    pots = player_of_the_season(state, team_points)

    champion = table[0]
    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "season": season,
        "model_version": MODEL_VERSION,
        "cutoff_matchweek": cutoff_matchweek,
        "n_simulations": n_simulations,
        "seed": seed,
        "assumptions": assumptions,
        "validation": {key: round(value, 4) for key, value in models.scores.items()},
        "table": table,
        "champion": {
            "team": champion["team"],
            "slug": champion["slug"],
            "probability": champion["title_probability"],
        },
        "top_scorer": _award_block(scorers, "goals", "goal_probability"),
        "top_assists": _award_block(assisters, "assists", "assist_probability"),
        "player_of_the_season": _pots_block(pots),
    }

    if write:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", PREDICTIONS_PATH)

    return payload


def _pots_block(pots: pd.DataFrame) -> dict[str, object]:
    """Build the player-of-the-season entry, exposing its component scores.

    Args:
        pots: Output of :func:`player_of_the_season`, best first.

    Returns:
        The award block, with the three weighted components kept visible so a reader
        can see *why* a player ranks where they do.
    """
    if pots.empty:
        return {"candidates": []}

    rows = [
        {
            "player": row["player_name"],
            "slug": player_slug(str(row["player_name"])),
            "team": display_name(str(row["team_slug"])),
            "team_slug": row["team_slug"],
            "score": round(float(row["score"]), 4),
            "components": {
                "attacking": round(float(row["score_attacking"]), 4),
                "team": round(float(row["score_team"]), 4),
                "minutes": round(float(row["score_minutes"]), 4),
            },
            "predicted_goals": round(float(row["predicted_goals"]), 2),
            "predicted_assists": round(float(row["predicted_assists"]), 2),
            "predicted_minutes": int(row["predicted_minutes"]),
        }
        for _, row in pots.head(TOP_N).iterrows()
    ]
    leader = dict(rows[0])
    leader["candidates"] = rows
    return leader


def load_predictions(path: Path = PREDICTIONS_PATH) -> dict[str, object]:
    """Read a previously written payload.

    Args:
        path: File to read.

    Returns:
        The payload.
    """
    return dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "PREDICTIONS_PATH",
    "PUBLISHED_PATH",
    "AwardModels",
    "build_predictions",
    "load_predictions",
    "publish",
    "split_season",
]
