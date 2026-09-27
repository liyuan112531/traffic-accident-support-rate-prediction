"""Runtime environment configuration for reproducible execution.

Thread counts are pinned so that floating-point reduction order does not vary
between machines or between runs. This is a deliberate trade of speed for
reproducibility: results that shift in the last few digits because a parallel
reduction happened to be scheduled differently are difficult to distinguish from
real differences, and they cost more to investigate than the runtime they save.

Every variable is set with ``setdefault``, so an explicit value in the
environment always wins over the default applied here.
"""

from __future__ import annotations

import os
from typing import Dict

#: Environment variables applied by :func:`configure_determinism`.
DETERMINISTIC_ENVIRONMENT: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
    "PYTHONHASHSEED": "0",
}


def configure_determinism() -> Dict[str, str]:
    """Apply the deterministic thread settings and return them.

    Call before importing libraries that read these variables at import time.
    """
    for name, value in DETERMINISTIC_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    return dict(DETERMINISTIC_ENVIRONMENT)
