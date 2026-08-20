"""Smoke tests: the project layout imports and the ML stack is installed.

These exist so a broken environment fails loudly on `pytest` rather than
halfway through a training run.
"""

import importlib

import pytest

SRC_PACKAGES = [
    "src",
    "src.data_collection",
    "src.features",
    "src.models",
]

THIRD_PARTY = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "sqlalchemy",
    "requests",
]


@pytest.mark.parametrize("name", SRC_PACKAGES)
def test_src_packages_are_importable(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", THIRD_PARTY)
def test_third_party_dependencies_are_installed(name):
    assert importlib.import_module(name) is not None


def test_sqlite_is_available():
    """SQLite ships with the stdlib; the pipeline writes data/processed/pl.db."""
    import sqlite3

    with sqlite3.connect(":memory:") as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
