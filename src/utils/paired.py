"""Build the paired arrays the resampling routines consume.

Both resampling routines compare two prediction vectors on the same units, and
both need the same four quantities: the target, the two predictions and the
grouping labels. Extracting them is identical between the two scripts, so it is
defined once here rather than copied, which is also what keeps the two scripts
comparable by construction.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import numpy as np
import pandas as pd

#: Column names expected in a paired table.
CASE_COLUMN: str = "case_id"
ITEM_COLUMN: str = "item"
TARGET_COLUMN: str = "y_true"


class PairedUnits(NamedTuple):
    """The four arrays a paired comparison needs, plus a short provenance note."""

    y: np.ndarray
    prediction_a: np.ndarray
    prediction_b: np.ndarray
    items: np.ndarray
    cases: np.ndarray
    n_dropped: int
    dropped: Dict[str, int]


def paired_units(frame: pd.DataFrame, variant_a: str, variant_b: str,
                 case_column: str = CASE_COLUMN,
                 item_column: str = ITEM_COLUMN,
                 target_column: str = TARGET_COLUMN) -> PairedUnits:
    """Select the units where both variants produced a prediction.

    A unit is usable only when the target and both predictions are finite. The
    dropped counts are returned rather than silently discarded so that the caller
    can report them: a comparison that quietly discarded most of its rows would
    otherwise look like a complete comparison.
    """
    for column in (case_column, item_column, target_column, variant_a, variant_b):
        if column not in frame.columns:
            raise KeyError(
                f"column {column!r} is missing; available columns are "
                f"{sorted(frame.columns)}"
            )

    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(frame[variant_a], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(frame[variant_b], errors="coerce").to_numpy(dtype=float)

    dropped = {
        "target_not_finite": int((~np.isfinite(target)).sum()),
        f"{variant_a}_not_finite": int((~np.isfinite(a)).sum()),
        f"{variant_b}_not_finite": int((~np.isfinite(b)).sum()),
    }
    keep = np.isfinite(target) & np.isfinite(a) & np.isfinite(b)
    n_dropped = int((~keep).sum())

    return PairedUnits(
        y=target[keep],
        prediction_a=a[keep],
        prediction_b=b[keep],
        items=frame[item_column].to_numpy()[keep],
        cases=frame[case_column].to_numpy()[keep],
        n_dropped=n_dropped,
        dropped=dropped,
    )


def per_item_slices(units: PairedUnits) -> Dict[str, np.ndarray]:
    """Row indices of every item with at least two usable units."""
    labels = np.asarray([str(v) for v in units.items])
    return {
        item: np.flatnonzero(labels == item)
        for item in sorted(set(labels.tolist()))
        if int((labels == item).sum()) >= 2
    }


def describe_pair(units: PairedUnits, variant_a: str, variant_b: str) -> Dict[str, Any]:
    """Summary of a paired comparison, suitable for recording in a report."""
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_units": int(units.y.size),
        "n_cases": int(np.unique(units.cases).size),
        "n_items": int(np.unique([str(v) for v in units.items]).size),
        "n_dropped": int(units.n_dropped),
        "dropped": units.dropped,
    }
