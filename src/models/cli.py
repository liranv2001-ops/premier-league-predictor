"""Command line entry point for fitting and simulating.

Examples:
    python -m src.models.cli --season 2025/26 --cutoff-matchweek 20
    python -m src.models.cli --season 2025/26 --cutoff-matchweek 38
    python -m src.models.cli --season 2026/27 --teams-file teams.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data_collection.config import UnknownTeamError, normalise_team
from src.models import (
    DEFAULT_DECAY,
    MissingTeamsError,
    compare_variants,
    load_matches,
    run_season_simulation,
    score_backtest,
    split_season,
)
from src.models.config import N_SIMULATIONS, RANDOM_SEED, simulation_json_path
from src.models.evaluate import Scores
from src.models.predictions import (
    PREDICTIONS_PATH,
    build_predictions,
    publish,
    publish_assets,
)

logger = logging.getLogger(__name__)


def read_teams_file(path: Path) -> list[str]:
    """Read and normalise a club list.

    Args:
        path: File with one club name per line; blank lines and ``#`` comments ignored.

    Returns:
        Canonical club slugs.

    Raises:
        UnknownTeamError: If a name has no slug mapping.
    """
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [normalise_team(line) for line in lines]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.models.cli",
        description="Fit Dixon-Coles and simulate a season's remaining fixtures.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Train on everything before each eligible season and predict it from "
        "matchweek 0, reporting accuracy against a carry-forward baseline. Ignores "
        "--season.",
    )
    parser.add_argument("--season", help='Season label, e.g. "2025/26".')
    parser.add_argument(
        "--cutoff-matchweek",
        type=int,
        default=38,
        help="Matchweeks treated as already played (default: 38, the whole season).",
    )
    parser.add_argument(
        "--teams-file",
        type=Path,
        help="File listing the season's clubs, one per line. Required for a season "
        "with no matches in the database.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=N_SIMULATIONS,
        help=f"Monte Carlo runs (default: {N_SIMULATIONS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})."
    )
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Also fit Dixon-Coles with src/features covariates and report which "
        "variant scores better on the held-out matches.",
    )
    parser.add_argument(
        "--awards",
        action="store_true",
        help="Also fit the award models and write data/processed/predictions.json "
        "with the table, champion, top scorer, top assists and player of the season.",
    )
    parser.add_argument(
        "--squad-season",
        help="Season whose squads stand in for the predicted one. Needed when the "
        "target season has no player data yet (e.g. 2025/26 squads for 2026/27).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Copy predictions.json to frontend/public so the dashboard serves it, "
        "and fetch any club badges and player photos still missing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def print_awards(payload: dict[str, object]) -> None:
    """Print the three award shortlists.

    Args:
        payload: Output of ``build_predictions``.
    """
    validation = payload.get("validation", {})
    assert isinstance(validation, dict)
    print("\nAward model validation (lower MAE is better):")
    for stat in ("goals", "assists"):
        model_mae = validation.get(f"{stat}_mae")
        baseline = validation.get(f"{stat}_baseline_mae")
        if model_mae is None or baseline is None:
            continue
        verdict = "beats baseline" if model_mae < baseline else "LOSES to baseline"
        print(f"  {stat:<8} MAE {model_mae:.3f}  vs naive {baseline:.3f}   -> {verdict}")

    for key, label, stat in (
        ("top_scorer", "Top scorer", "goals"),
        ("top_assists", "Top assists", "assists"),
    ):
        block = payload.get(key, {})
        assert isinstance(block, dict)
        print(f"\n{label}:")
        for i, row in enumerate(block.get("candidates", []), start=1):
            print(
                f"  {i}. {row['player']:<24}{row['team']:<20}"
                f"{row[f'predicted_{stat}']:>6.1f} {stat}"
                f"{100 * row['probability']:>8.1f}%"
            )

    pots = payload.get("player_of_the_season", {})
    assert isinstance(pots, dict)
    print("\nPlayer of the season:")
    for i, row in enumerate(pots.get("candidates", []), start=1):
        parts = row["components"]
        print(
            f"  {i}. {row['player']:<24}{row['team']:<20}score {row['score']:.3f}"
            f"   (att {parts['attacking']:.2f} / team {parts['team']:.2f}"
            f" / mins {parts['minutes']:.2f})"
        )

    assumptions = payload.get("assumptions", [])
    assert isinstance(assumptions, list)
    if assumptions:
        print("\nAssumptions carried into this prediction:")
        for note in assumptions:
            print(f"  - {note}")


def print_variant_comparison(season: str, cutoff_matchweek: int) -> None:
    """Fit both model variants and print their held-out scores.

    Args:
        season: Season to hold out within.
        cutoff_matchweek: Matchweek to split at.
    """
    featured = load_matches(with_features=True)
    history, _, remaining = split_season(featured, season, cutoff_matchweek)
    if remaining.empty:
        print("\nNo held-out matches at this cutoff - skipping variant comparison.")
        return

    report = compare_variants(history, remaining, decay=DEFAULT_DECAY)
    print(f"\nHeld-out comparison ({len(remaining)} matches, higher log-likelihood wins):")
    print(f"  {'variant':<18}{'logLik/match':>14}{'log-loss':>12}{'Brier':>10}")
    for key in ("classic", "with_covariates"):
        scores = report[key]
        assert isinstance(scores, Scores)
        print(
            f"  {key:<18}{scores.log_likelihood:>14.4f}"
            f"{scores.log_loss:>12.4f}{scores.brier:>10.4f}"
        )
    print(f"  winner: {report['winner']}   betas: {report['betas']}")


def print_backtest(n_simulations: int, seed: int) -> int:
    """Run the backtest across every eligible season and print the metrics.

    Args:
        n_simulations: Monte Carlo runs per season.
        seed: Random seed.

    Returns:
        Process exit code.
    """
    from src.models.backtest import run_backtest, summarise

    try:
        matches = load_matches()
    except (ValueError, FileNotFoundError) as exc:
        logging.error("Could not load matches: %s", exc)
        return 1

    results = run_backtest(matches, n_simulations=n_simulations, seed=seed)
    if not results:
        logging.error("No season has enough prior history to backtest.")
        return 1

    print("\nTrained on every season before the one predicted; predicted from matchweek 0.")
    print("Baseline = last season's table with promoted clubs in the vacated places.\n")

    header = (
        f"{'season':<10}{'trained':>8}{'MAE':>7}{'base':>7}{'rho':>7}"
        f"{'champion':>26}{'P(champ)':>10}{'top4':>6}{'rel':>5}{'cover':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        verdict = "HIT" if result.champion_correct else f"miss ({result.predicted_champion})"
        print(
            f"{result.season:<10}{len(result.training_seasons):>8}"
            f"{result.mean_position_error:>7.2f}{result.baseline_position_error:>7.2f}"
            f"{result.spearman:>7.2f}"
            f"{result.actual_champion + ' ' + verdict:>26}"
            f"{100 * result.champion_probability:>9.1f}%"
            f"{result.ucl_overlap:>5}/4{result.relegation_overlap:>4}/3"
            f"{100 * result.interval_coverage:>6.0f}%"
        )

    summary = summarise(results)
    print("-" * len(header))
    print(
        f"{'average':<10}{'':>8}{summary['mean_position_error']:>7.2f}"
        f"{summary['baseline_position_error']:>7.2f}{summary['spearman']:>7.2f}"
        f"{'':>26}{100 * summary['mean_champion_probability']:>9.1f}%"
        f"{summary['ucl_overlap']:>5.1f}/4{summary['relegation_overlap']:>4.1f}/3"
        f"{100 * summary['interval_coverage']:>6.0f}%"
    )

    beat = sum(r.beats_baseline for r in results)
    hits = sum(r.champion_correct for r in results)
    print(f"\nChampion identified in {hits}/{len(results)} seasons.")
    print(f"Model ordered the table better than the baseline in {beat}/{len(results)} seasons.")
    print(f"Mean points error: {summary['mean_points_error']:.1f} points per club.")
    print(
        f"80% interval covered {100 * summary['interval_coverage']:.0f}% of clubs "
        "(80% would be perfectly calibrated)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a simulation and print a summary table.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code - 0 on success, 1 on a handled failure.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if args.backtest:
        return print_backtest(args.simulations, args.seed)

    if not args.season:
        logging.error("--season is required unless --backtest is given.")
        return 1

    teams = None
    if args.teams_file:
        try:
            teams = read_teams_file(args.teams_file)
        except (OSError, UnknownTeamError) as exc:
            logging.error("Could not read the club list: %s", exc)
            return 1

    try:
        payload = run_season_simulation(
            args.season,
            cutoff_matchweek=args.cutoff_matchweek,
            teams=teams,
            n_simulations=args.simulations,
            seed=args.seed,
        )
    except MissingTeamsError as exc:
        logging.error("%s", exc)
        return 1
    except ValueError as exc:
        logging.error("Could not run the simulation: %s", exc)
        return 1

    rows = payload["teams"]
    assert isinstance(rows, list)

    print(f"\n{args.season} - {payload['n_simulations']} simulations, seed {payload['seed']}")
    print(f"{payload['matches_played']} played, {payload['matches_remaining']} remaining\n")
    print(f"{'#':>3}  {'club':<22}{'mean pos':>9}{'title %':>9}{'exp pts':>9}")
    print("-" * 55)
    for row in rows:
        print(
            f"{row['predicted_rank']:>3}  {row['team']:<22}"
            f"{row['predicted_position']:>9.2f}"
            f"{100 * float(row['title_probability']):>9.1f}"
            f"{row['expected_points']:>9.1f}"
        )

    # When the season is already complete, we can say how good the prediction was.
    matches = load_matches()
    actual = matches[matches["season"] == args.season]
    if not actual.empty and len(actual) == 380:
        report = score_backtest(payload, actual)
        print("\nBack-test against the real season:")
        for key, value in report.items():
            print(f"  {key:<32} {value}")

    if args.awards:
        try:
            predictions = build_predictions(
                args.season,
                cutoff_matchweek=args.cutoff_matchweek if args.cutoff_matchweek < 38 else 0,
                teams=teams,
                squad_season=args.squad_season,
                n_simulations=args.simulations,
                seed=args.seed,
            )
        except (ValueError, KeyError) as exc:
            logging.error("Award models failed: %s", exc)
            return 1
        print_awards(predictions)
        print(f"\nWrote {PREDICTIONS_PATH}")

    if args.publish:
        try:
            destination = publish()
            copied = publish_assets()
        except (FileNotFoundError, OSError) as exc:
            logging.error("Publishing failed: %s", exc)
            return 1
        print(f"\nPublished to {destination}")
        print(
            f"  {copied['logos']} club badges, {copied['players']} player images, "
            f"{copied['mappings']} mapping files"
        )
        print("  anything missing renders as a monogram in the dashboard")

    if args.compare_variants:
        try:
            print_variant_comparison(args.season, args.cutoff_matchweek)
        except (ValueError, KeyError) as exc:
            logging.error("Variant comparison failed: %s", exc)

    print(f"\nWrote {simulation_json_path(args.season)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
