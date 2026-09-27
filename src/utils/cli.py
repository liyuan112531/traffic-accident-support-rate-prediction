"""Shared command-line helpers for the stage scripts.

Kept in the package rather than duplicated in each script so that the default
scale, the default case table and the labelling of a demonstration run are defined
once. Every stage script is a thin wrapper over :mod:`src`; the reason this helper
exists is to stop the wrappers drifting apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .paths import PROJECT_ROOT, load_config
from .env import configure_determinism


def default_case_table() -> Path:
    """Path of the synthetic demonstration case table."""
    return PROJECT_ROOT / "data" / "synthetic" / "synthetic_cases.csv"


def default_output_dir() -> Path:
    """Default directory for run artefacts."""
    return PROJECT_ROOT / "examples" / "synthetic_demo" / "expected_output"


def demonstration_defaults() -> Dict[str, Any]:
    """The demonstration-scale settings from ``config/experiment_config.yaml``."""
    section = load_config().get("demonstration", {})
    if not isinstance(section, dict):
        section = {}
    return {
        "max_cases": int(section.get("max_cases", 200)),
        "n_outer_folds": int(section.get("n_outer_folds", 2)),
        "inner_folds": int(section.get("inner_folds", 2)),
        "seeds": [int(s) for s in section.get("seeds", [42])],
        "bootstrap_replicates": int(section.get("bootstrap_replicates", 400)),
        "permutation_replicates": int(section.get("permutation_replicates", 400)),
        "label": str(section.get("label", "demonstration scale")),
    }


def add_common_arguments(parser: argparse.ArgumentParser,
                         with_split: bool = True) -> None:
    """Add the arguments every stage script understands."""
    configure_determinism()
    defaults = demonstration_defaults()
    parser.add_argument("--data", type=Path, default=default_case_table(),
                        help="case table to run on; defaults to the synthetic file")
    parser.add_argument("--max-cases", type=int, default=defaults["max_cases"],
                        help="restrict to a deterministic subset of cases")
    if with_split:
        parser.add_argument("--folds", type=int, default=defaults["n_outer_folds"],
                            help="number of outer folds")
        parser.add_argument("--inner-folds", type=int, default=defaults["inner_folds"],
                            help="number of inner folds used for cross-fitting")
        parser.add_argument("--seeds", type=int, nargs="+", default=defaults["seeds"],
                            help="random seeds; one run per seed")
    parser.add_argument("--with-tabpfn", action="store_true",
                        help="include TabPFN; requires downloading model weights")
    parser.add_argument("--pool", nargs="+", default=None,
                        help="explicit expert subset; defaults to the frozen pool. The "
                             "convex combination expert is dropped unless its whole "
                             "base pool is present, and the disagreement block "
                             "requires the experts it compares, so a reduced pool must "
                             "still contain them.")
    parser.add_argument("--output", type=Path, default=default_output_dir(),
                        help="directory for run artefacts")


def resolve_output_file(output, default: Path) -> Path:
    """Resolve the single output file of a generator script.

    ``make_synthetic_data`` and ``make_data_dictionary`` each write exactly one named
    file, so for those two scripts ``--output`` is a file path, while for every stage
    script it is a directory. Passing a directory to a generator is an easy slip and
    would otherwise surface as a permission error from the write that follows, so it
    is reported as the argument error it is.

    Raises ``ValueError`` with an actionable message; callers are expected to print it
    and exit non-zero.
    """
    target = Path(output) if output is not None else Path(default)
    if target.is_dir():
        raise ValueError(
            f"--output takes a file path, but '{target}' is a directory. "
            f"Pass a file, for example --output '{target / Path(default).name}'."
        )
    return target


def banner(title: str) -> None:
    """Print a section heading."""
    line = "=" * 78
    print()
    print(line)
    print(title)
    print(line)


def display_path(path) -> str:
    """Render a path relative to the repository when it lies inside it.

    Scripts may be pointed at an output directory outside the repository, so the
    relative form is a convenience and never an assumption. Printing an absolute
    path for an outside directory is correct behaviour, not a failure.
    """
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)
