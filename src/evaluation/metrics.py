"""Item-level, macro-averaged and cell-level regression metrics.

Two averaging conventions are used, and the difference between them is not
cosmetic.

``macro`` averages the metric over the 16 items, giving each item weight
``1/16`` regardless of how many cases claimed it. This is the reported primary
criterion: it answers the question "how well does the model predict each head of
claim", which is the legally meaningful question, because a rarely claimed item
is as important as a frequently claimed one.

``micro`` pools every case-item unit without regard to item, so frequently claimed
items dominate. It answers "how well does the model predict a randomly chosen
case-item unit". Both are reported; neither replaces the other.

Zero-variance items
-------------------
R-squared divides by the total sum of squares. An item evaluated on a subset that
happens to have a constant target has zero total sum of squares and an undefined
R-squared. Such items are skipped for R-squared only, and the number skipped is
reported, because silently substituting 0 or 1 would fabricate an agreement
between the models that the data does not support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

#: Keys returned by :func:`macro_micro_metrics`.
METRIC_KEYS: tuple = ("macro_MAE", "macro_RMSE", "macro_R2", "micro_MAE")

#: An item must contribute at least this many units to be scored.
MIN_UNITS_PER_ITEM: int = 2


def _item_statistics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Absolute error, squared error and R-squared for one group of units."""
    residual = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan"),
    }


def item_metrics(y: np.ndarray, prediction: np.ndarray,
                 items: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-item metrics, sorted by item identifier.

    Items contributing fewer than :data:`MIN_UNITS_PER_ITEM` units are omitted:
    a single observation has no defined variance and no meaningful error
    distribution.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    labels = np.asarray([str(i) for i in items])
    if not (y.size == prediction.size == labels.size):
        raise ValueError("target, prediction and item labels must have equal length")
    rows: List[Dict[str, Any]] = []
    for item in sorted(set(labels.tolist())):
        mask = labels == item
        if int(mask.sum()) < MIN_UNITS_PER_ITEM:
            continue
        rows.append({"item": item, **_item_statistics(y[mask], prediction[mask])})
    return rows


def macro_metrics(y: np.ndarray, prediction: np.ndarray,
                  items: Sequence[str]) -> Dict[str, float]:
    """Mean of the per-item metrics over items with at least two units."""
    rows = item_metrics(y, prediction, items)
    if not rows:
        raise ValueError("no item had enough units to compute a macro average")
    r2_values = np.asarray([r["R2"] for r in rows], dtype=float)
    finite_r2 = r2_values[np.isfinite(r2_values)]
    return {
        "macro_MAE": float(np.mean([r["MAE"] for r in rows])),
        "macro_RMSE": float(np.mean([r["RMSE"] for r in rows])),
        # Guarded rather than a bare nanmean: when every item has a constant target
        # no item has a defined R-squared, and nanmean of an all-NaN slice is NaN with
        # a runtime warning. Reporting NaN is correct here, the warning is not.
        "macro_R2": float(np.mean(finite_r2)) if finite_r2.size else float("nan"),
        "n_items": len(rows),
        "n_items_with_defined_r2": int(finite_r2.size),
    }


def micro_mae(y: np.ndarray, prediction: np.ndarray) -> float:
    """Cell-level mean absolute error pooled over all units."""
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if y.size != prediction.size:
        raise ValueError("target and prediction disagree on length")
    return float(np.mean(np.abs(y - prediction)))


def macro_micro_metrics(y: np.ndarray, prediction: np.ndarray,
                        items: Sequence[str]) -> Dict[str, float]:
    """All four reported metrics for one out-of-fold prediction vector."""
    macro = macro_metrics(y, prediction, items)
    return {
        "macro_MAE": macro["macro_MAE"],
        "macro_RMSE": macro["macro_RMSE"],
        "macro_R2": macro["macro_R2"],
        "micro_MAE": micro_mae(y, prediction),
        "n_items": macro["n_items"],
        "n_items_with_defined_r2": macro["n_items_with_defined_r2"],
        "n_units": int(np.asarray(y).size),
    }


def case_metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Case-level metrics, delegating to the aggregation package's definition."""
    from ..aggregation.claim_weighted import case_level_metrics

    return case_level_metrics(y, prediction)


def aggregate_over_seeds(seed_predictions: np.ndarray) -> np.ndarray:
    """Average predictions across seeds before scoring.

    The official convention is to average the per-seed predictions first and then
    compute the metrics once, so that seed-to-seed variance enters the reported
    score through its effect on the averaged prediction rather than as an extra
    averaging step on the metrics themselves.
    """
    values = np.asarray(seed_predictions, dtype=float)
    if values.ndim == 1:
        return values
    return np.nanmean(values, axis=0)
