#!/usr/bin/env python
"""Run the synthetic demonstration from the examples directory.

A convenience wrapper so the demonstration can be started from here without
remembering the path into ``scripts/``. All arguments are forwarded unchanged; run
``python examples/synthetic_demo/run_demo.py --help`` for the full list.

This file contains no method code. ``scripts/run_synthetic_demo.py`` is the
demonstration; this wrapper exists only so the example is self-contained in its own
directory.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_synthetic_demo.py"


def main() -> int:
    if not DEMO_SCRIPT.exists():
        print(f"demonstration script not found: {DEMO_SCRIPT}")
        return 2
    # The demonstration derives the repository root from its own location, so it can
    # be executed from anywhere and needs no working-directory assumption.
    sys.argv = [str(DEMO_SCRIPT), *sys.argv[1:]]
    try:
        runpy.run_path(str(DEMO_SCRIPT), run_name="__main__")
    except SystemExit as outcome:
        # The demonstration signals its verdict through its exit status; forward it
        # rather than swallowing it, so a failing verification is visible to a caller.
        code = outcome.code
        return int(code) if isinstance(code, int) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
