"""Command line entry point for feature building.

Examples:
    python -m src.features.cli
    python -m src.features.cli --only team
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.features import build_all
from src.features.config import PROCESSED_DB_PATH


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.features.cli",
        description="Build model-ready feature tables from the raw database.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "team", "player"],
        default="all",
        help="Which feature families to build (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the feature tables.

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

    try:
        written = build_all(only=args.only)
    except ValueError as exc:
        logging.error(
            "Could not read the raw database - has collection run?\n  %s",
            exc,
        )
        return 1

    print(f"\nStored in {PROCESSED_DB_PATH}")
    for table, rows in written.items():
        print(f"  {table:<26} {rows:>6} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
