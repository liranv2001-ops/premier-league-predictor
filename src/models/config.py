"""Constants and paths for the modelling layer."""

from __future__ import annotations

from pathlib import Path

from src.data_collection.config import PROJECT_ROOT
from src.features.config import PROCESSED_DIR

#: Single source of truth for every seeded random draw in the project.
RANDOM_SEED = 42

#: Monte Carlo runs per simulation.
N_SIMULATIONS = 10_000

#: Largest scoreline the joint score matrix represents. A 10-10 draw has probability
#: ~1e-20 under any realistic lambda, so truncating here loses nothing measurable.
MAX_GOALS = 10

#: Candidate time-decay rates, in units of 1/day, searched by held-out likelihood.
#: 0.0 means "no decay"; 0.005 halves a match's weight in about 140 days.
DECAY_GRID = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005)

#: Points awarded for a win and a draw.
WIN_POINTS = 3
DRAW_POINTS = 1

#: Clubs in a Premier League season.
LEAGUE_SIZE = 20

SIMULATION_DIR = PROCESSED_DIR
MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "models"


def simulation_json_path(season: str) -> Path:
    """Return the output path for a season's simulation JSON.

    Args:
        season: Season label such as ``"2025/26"``.

    Returns:
        Path to write the JSON to; the slash in the season label is replaced.
    """
    return SIMULATION_DIR / f"simulation_{season.replace('/', '-')}.json"
