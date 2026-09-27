"""Shared pytest fixtures.

The repository root is put on the import path so the tests exercise the same
``src`` package the scripts import, rather than a copy of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.utils.env import configure_determinism  # noqa: E402

#: Synthetic case table shipped with the repository.
SYNTHETIC_CASES = REPOSITORY_ROOT / "data" / "synthetic" / "synthetic_cases.csv"

#: Pool used by the end-to-end test.
#:
#: The disagreement block compares named experts: it anchors on the convex
#: combination expert and measures signed and absolute differences against ``lgb``,
#: ``xgb``, ``median``, ``catboost`` and ``ridge``. A pool that omits any of those
#: cannot produce the documented block, so the reduced pool used here is exactly the
#: convex combination expert's base pool plus the convex expert itself. That keeps
#: the block widths, the check on the convex expert's base pool and the meta width
#: arithmetic identical to the full pool, while dropping the four experts that only
#: add columns, which is what makes the suite fast enough to run routinely.
TEST_POOL = ("median", "ridge", "rf", "xgb", "lgb", "catboost", "c3")

#: Demonstration-scale settings for the end-to-end test. Smaller than the settings
#: used for the published demonstration, so that the suite stays quick.
TEST_CASES = 220
TEST_FOLDS = 2
TEST_INNER_FOLDS = 2
TEST_SEED = 42


@pytest.fixture(scope="session", autouse=True)
def _deterministic_environment():
    """Pin the thread and hash environment once for the whole session."""
    configure_determinism()


@pytest.fixture(scope="session")
def synthetic_path() -> Path:
    """Location of the shipped synthetic case table."""
    if not SYNTHETIC_CASES.exists():
        pytest.skip(f"synthetic case table not found: {SYNTHETIC_CASES}")
    return SYNTHETIC_CASES


@pytest.fixture(scope="session")
def synthetic_frame(synthetic_path):
    """The shipped synthetic case table, loaded once per session."""
    return load_cases(synthetic_path)


@pytest.fixture(scope="session")
def test_pool():
    """Pool used by the end-to-end tests."""
    return TEST_POOL


@pytest.fixture(scope="session")
def test_pipeline_result(synthetic_frame):
    """One full end-to-end run shared by the tests that need a fitted pipeline.

    Session-scoped because a run trains the expert pool once per fold; repeating it
    per test would dominate the suite's runtime while adding no coverage.
    """
    config = PipelineConfig(
        n_outer_folds=TEST_FOLDS,
        n_inner_folds=TEST_INNER_FOLDS,
        seeds=(TEST_SEED,),
        pool=TEST_POOL,
        max_cases=TEST_CASES,
        label="test scale",
    )
    return run_pipeline(synthetic_frame, config)
