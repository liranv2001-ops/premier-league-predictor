"""Command line entry point for data collection.

Examples:
    python -m src.data_collection.cli
    python -m src.data_collection.cli --seasons 8 --source players
    python -m src.data_collection.cli --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.data_collection import CHEAP_SOURCES, DEFAULT_COMPLETED_SEASONS, collect_all
from src.data_collection.config import (
    DEFAULT_MIN_INTERVAL_SECONDS,
    RAW_DB_PATH,
    UnknownTeamError,
)
from src.data_collection.understat import UnderstatFormatError


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data_collection.cli",
        description="Collect Premier League match results and player statistics.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=DEFAULT_COMPLETED_SEASONS,
        help="Number of completed seasons to fetch, on top of the current one "
        f"(default: {DEFAULT_COMPLETED_SEASONS}).",
    )
    parser.add_argument(
        "--source",
        choices=["all", "matches", "players", "player_matches", "badges", "photos", "assets"],
        default="all",
        help="Which collector to run. 'all' means matches + players; "
        "'player_matches' is separate because it costs one request per match "
        "(~1,900 for five seasons, roughly an hour).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached responses and re-fetch everything.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_SECONDS,
        help="Minimum seconds between requests to the same host "
        f"(default: {DEFAULT_MIN_INTERVAL_SECONDS}). Lower it only for the long "
        "player_matches run, and stay polite - these are undocumented endpoints.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging, including rate-limit sleeps.",
    )
    return parser


def build_assets(source: str) -> int:
    """Generate club badges and/or fetch free-licensed player photos.

    Args:
        source: ``"badges"``, ``"photos"`` or ``"assets"`` for both.

    Returns:
        Process exit code.
    """
    import json

    from src.data_collection.club_badges import build_all
    from src.data_collection.wikimedia import collect_player_photos
    from src.models.predictions import PREDICTIONS_PATH

    if not PREDICTIONS_PATH.exists():
        logging.error(
            "%s does not exist - run the models with --awards first, so we know which "
            "clubs and players need assets.",
            PREDICTIONS_PATH,
        )
        return 1

    payload = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))

    if source in ("badges", "assets"):
        clubs = [(str(row["slug"]), str(row["team"])) for row in payload["table"]]
        entries = build_all(clubs)
        print(f"\nGenerated {len(entries)} club badges in assets/logos/")
        print("  These are monogram badges drawn by this project, NOT official crests.")

    if source in ("photos", "assets"):
        # The club comes along so the collector can verify the article is about this
        # player - a mononym like "Thiago" otherwise resolves to a famous namesake.
        candidates: dict[str, tuple[str, str]] = {}
        for award in ("top_scorer", "top_assists", "player_of_the_season"):
            for row in payload.get(award, {}).get("candidates", []):
                candidates[str(row["slug"])] = (str(row["player"]), str(row["team"]))

        mapping = collect_player_photos(candidates)
        players = mapping["players"]
        placeholders = mapping["placeholders"]
        assert isinstance(players, dict) and isinstance(placeholders, list)

        print(f"\nFree-licensed photos: {len(players)}/{len(candidates)} candidates")
        for entry in players.values():
            print(f"  {entry['player']:<24}{entry['licence']:<16}{entry['author'][:34]}")
        if placeholders:
            print(f"\nUsing the generic placeholder ({len(placeholders)}):")
            for entry in placeholders:
                print(f"  {entry['player']:<24}{entry['reason']}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the collection pipeline.

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

    # Asset generation reads predictions.json rather than the match database, so it runs
    # on its own path instead of through collect_all.
    if args.source in ("badges", "photos", "assets"):
        return build_assets(args.source)

    sources = CHEAP_SOURCES if args.source == "all" else (args.source,)

    try:
        written = collect_all(
            args.seasons,
            sources=sources,
            force_refresh=args.force_refresh,
            min_interval=args.min_interval,
        )
    except UnknownTeamError as exc:
        logging.error("%s", exc)
        return 1
    except UnderstatFormatError as exc:
        logging.error("Understat returned an unexpected response.\n%s", exc)
        return 1

    print(f"\nStored in {RAW_DB_PATH}")
    for table, rows in written.items():
        print(f"  {table:<22} {rows:>6} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
