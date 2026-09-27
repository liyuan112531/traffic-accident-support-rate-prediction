"""Multiplicity correction for families of simultaneous comparisons.

Comparing several model pairs on several metrics produces a family of p-values.
Reporting them at their nominal level would overstate the evidence, because the
expected number of spurious findings grows with the size of the family. The
Benjamini-Hochberg step-up procedure controls the false discovery rate, which is
the appropriate target when the goal is to screen a set of comparisons rather
than to declare a single confirmatory result.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, monotone in the input order.

    The returned array preserves the input ordering, so it can be aligned with
    the table of comparisons that produced it.
    """
    values = np.asarray([float(p) for p in p_values], dtype=float)
    if values.size == 0:
        return values
    if np.any(~np.isfinite(values)):
        raise ValueError("p-values must be finite")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    n = values.size
    order = np.argsort(values)
    ranked = values[order] * n / (np.arange(n) + 1.0)
    monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, controlling the family-wise error rate.

    Provided alongside the false-discovery-rate procedure because the two answer
    different questions: Holm bounds the probability of *any* false rejection,
    which is stricter and appropriate for a small confirmatory family.
    """
    values = np.asarray([float(p) for p in p_values], dtype=float)
    if values.size == 0:
        return values
    n = values.size
    order = np.argsort(values)
    ranked = np.empty(n, dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        candidate = (n - position) * values[index]
        running = max(running, candidate)
        ranked[index] = min(1.0, running)
    return ranked
