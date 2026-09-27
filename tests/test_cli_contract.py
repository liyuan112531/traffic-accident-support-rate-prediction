"""Contract tests for the command-line entry points.

The scope here is narrow and deliberate: every stage script must be importable, must
answer ``--help``, and must write where it says it writes. The end-to-end tests cover
the method; this module covers the seam between the shell and the package, which is
where a released script most often breaks -- an import that only resolves under one
layout, an argument renamed in one script and not in another, a run that writes to a
location the documentation does not mention.

Each script is run as a subprocess rather than by calling its ``main``. That is the
only way to exercise the import block at the top of the file, which is exactly the part
that differs between running from the repository root and running from elsewhere.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

#: The canonical item universe. Imported rather than repeated, so that this module
#: cannot assert against a stale copy of the list -- which is exactly the mistake it
#: would otherwise be guarding against.
from src.preprocessing.schema import ITEMS, N_ITEMS  # noqa: E402

#: Every executable script. Written out rather than discovered by globbing, so that
#: removing a script fails this test instead of quietly shrinking its coverage.
STAGE_SCRIPTS = (
    "make_synthetic_data.py",
    "make_data_dictionary.py",
    "train_experts.py",
    "train_meta.py",
    "evaluate.py",
    "bootstrap.py",
    "permutation_test.py",
    "shap_analysis.py",
    "run_synthetic_demo.py",
)

#: Scripts that produce one named file, so their ``--output`` is a file path. Every
#: other script produces a directory of tables.
GENERATOR_SCRIPTS = ("make_synthetic_data.py", "make_data_dictionary.py")

#: The generator refuses to emit a table too small to split into folds. Kept here as a
#: named constant so the test states the contract rather than an arbitrary number.
GENERATOR_MIN_CASES = 50
GENERATOR_TEST_CASES = 60


def run_script(name: str, *arguments: str, timeout: int = 600):
    """Run a released script in a fresh interpreter from the repository root."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_every_script_answers_help(name: str) -> None:
    """Each script imports cleanly and builds its argument parser."""
    result = run_script(name, "--help")
    assert result.returncode == 0, f"{name} --help failed:\n{result.stderr}"
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("name", GENERATOR_SCRIPTS)
def test_generator_output_convention_is_documented(name: str) -> None:
    """A generator says its ``--output`` is a path, so the help text is not silent."""
    result = run_script(name, "--help")
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert "path" in result.stdout.lower()


def test_stage_output_convention_is_documented() -> None:
    """A stage script says its ``--output`` is a directory."""
    result = run_script("train_experts.py", "--help")
    assert result.returncode == 0
    assert "directory" in result.stdout.lower()


@pytest.mark.parametrize("name", GENERATOR_SCRIPTS)
def test_generator_rejects_a_directory(name: str, tmp_path: Path) -> None:
    """Passing a directory is reported as an argument error, not as a write failure.

    This is the slip the two conventions invite. The failure mode being guarded
    against is a bare permission error from the write that follows, which says nothing
    about which argument was wrong.
    """
    result = run_script(name, "--output", str(tmp_path))
    assert result.returncode == 2, (
        f"{name} accepted a directory as --output:\n{result.stdout}\n{result.stderr}"
    )
    assert "is a directory" in result.stderr


def test_data_dictionary_writes_to_the_requested_file(tmp_path: Path) -> None:
    """The dictionary is generated, and every row describes a column rather than a case."""
    target = tmp_path / "dictionary.csv"
    result = run_script("make_data_dictionary.py", "--output", str(target))
    assert result.returncode == 0, result.stderr
    assert target.exists()

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "the dictionary has no rows"
    assert {"name", "dtype", "description", "role", "public_release_safe"} <= set(rows[0])
    # A dictionary row is a column description; nothing here may be flagged unsafe.
    assert {row["public_release_safe"] for row in rows} == {"yes"}
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names)), "the dictionary documents a column twice"

    # The case table's claim and award columns are documented under their schema names.
    for item in ITEMS:
        assert f"{item}_claim" in names
        assert f"{item}_award" in names

    # The claim side of the meta representation is addressed positionally. All
    # thirty-three positions must be documented, and they must be the real names: the
    # manuscript's meta matrix has no column called ``log1p_<item>_claim``.
    positional = [f"claim_{i}" for i in range(2 * N_ITEMS + 1)]
    for name in positional:
        assert name in names, f"{name} is missing from the data dictionary"


def test_data_dictionary_covers_the_whole_meta_representation(tmp_path: Path) -> None:
    """Every one of the seventy-three meta columns is documented, under its real name.

    The meta representation is the interface the method is described in terms of, so a
    reader working from the dictionary alone should be able to look up each of its
    columns. Asserting the names rather than only the count catches a block that keeps
    its width while losing a column, and it catches the specific failure this test was
    written after: a dictionary listing descriptive aliases -- ``log1p_medical_claim``,
    ``item_is_medical`` -- that exist nowhere in the code.
    """
    from src.experts.registry import EXPERT_NAMES
    from src.features.meta_features import (
        DEFAULT_DISAGREEMENT_SPEC,
        META_DIMENSION,
        meta_block_boundaries,
        meta_feature_names,
    )

    target = tmp_path / "dictionary.csv"
    result = run_script("make_data_dictionary.py", "--output", str(target))
    assert result.returncode == 0, result.stderr

    with target.open(newline="", encoding="utf-8") as handle:
        names = {row["name"] for row in csv.DictReader(handle)}

    expected = meta_feature_names(tuple(EXPERT_NAMES), DEFAULT_DISAGREEMENT_SPEC)
    assert len(expected) == META_DIMENSION == 73

    missing = [name for name in expected if name not in names]
    assert not missing, f"meta columns missing from the data dictionary: {missing}"

    # The documented claim side must be the full forty-nine, not the thirty-three
    # request features alone: the sixteen item indicators are part of the block.
    from src.features.claim_features import N_REQUEST_FEATURES
    from src.features.item_indicators import N_ITEM_INDICATORS

    claim_side = meta_block_boundaries(len(EXPERT_NAMES))["claim_side"]
    assert claim_side[1] - claim_side[0] == 49
    assert 49 == N_REQUEST_FEATURES + N_ITEM_INDICATORS


def test_shipped_dictionary_matches_the_generator(tmp_path: Path) -> None:
    """The committed dictionary has not drifted from the code that produces it."""
    shipped = REPOSITORY_ROOT / "data" / "schema" / "data_dictionary.csv"
    assert shipped.exists(), "the shipped data dictionary is missing"

    regenerated = tmp_path / "dictionary.csv"
    result = run_script("make_data_dictionary.py", "--output", str(regenerated))
    assert result.returncode == 0, result.stderr

    assert regenerated.read_bytes() == shipped.read_bytes(), (
        "data/schema/data_dictionary.csv is stale; regenerate it with "
        "scripts/make_data_dictionary.py"
    )


def test_generator_refuses_a_table_too_small_for_folds(tmp_path: Path) -> None:
    """The documented minimum is enforced rather than silently producing a bad table."""
    target = tmp_path / "cases.csv"
    result = run_script(
        "make_synthetic_data.py",
        "--n-cases", str(GENERATOR_MIN_CASES - 1),
        "--output", str(target),
    )
    assert result.returncode != 0
    assert "at least" in result.stderr
    assert not target.exists()


def test_synthetic_generator_writes_the_documented_shape(tmp_path: Path) -> None:
    """The generator emits the claim/award pairs the pipeline expects."""
    target = tmp_path / "cases.csv"
    result = run_script(
        "make_synthetic_data.py",
        "--n-cases", str(GENERATOR_TEST_CASES),
        "--output", str(target),
    )
    assert result.returncode == 0, result.stderr

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == GENERATOR_TEST_CASES
    for item in ITEMS:
        assert f"{item}_claim" in rows[0]
        assert f"{item}_award" in rows[0]
    # Every identifier announces itself as synthetic.
    assert all(row["case_id"].startswith("SYNTH-") for row in rows)


def test_synthetic_generator_is_reproducible(tmp_path: Path) -> None:
    """The same seed gives byte-identical output, a different seed does not.

    The shipped synthetic table is reproducible from its recorded seed, which is what
    makes the expected-output tables checkable.
    """
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    other = tmp_path / "other.csv"

    for target, seed in ((first, "7"), (second, "7"), (other, "8")):
        result = run_script(
            "make_synthetic_data.py",
            "--n-cases", str(GENERATOR_TEST_CASES),
            "--seed", seed,
            "--output", str(target),
        )
        assert result.returncode == 0, result.stderr

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != other.read_bytes()


def test_shipped_synthetic_table_matches_its_recorded_seed(tmp_path: Path) -> None:
    """The committed synthetic table is exactly what the documented seed produces."""
    shipped = REPOSITORY_ROOT / "data" / "synthetic" / "synthetic_cases.csv"
    assert shipped.exists(), "the shipped synthetic table is missing"

    with shipped.open(newline="", encoding="utf-8") as handle:
        n_cases = sum(1 for _ in csv.DictReader(handle))

    regenerated = tmp_path / "cases.csv"
    result = run_script(
        "make_synthetic_data.py",
        "--n-cases", str(n_cases),
        "--seed", "20260927",
        "--output", str(regenerated),
    )
    assert result.returncode == 0, result.stderr
    assert regenerated.read_bytes() == shipped.read_bytes(), (
        "data/synthetic/synthetic_cases.csv is not reproducible from its recorded seed"
    )
