"""Item-balanced training weights.

The problem this solves
-----------------------
The meta-learner is trained on pooled case-item units. Pooling means an item
contributes as many rows as it has cases, so a frequently claimed item such as
medical expenses contributes many more rows than a rarely claimed one such as
funeral expenses. Under an unweighted loss, the optimisation objective is
therefore weighted by item frequency.

The evaluation, by contrast, is a *macro* average: every item counts equally, and
each item's contribution to the reported metric is ``1/16`` regardless of how
often it is claimed. Training and evaluation therefore optimise different
objectives, and the mismatch is structural rather than accidental.

The correction
--------------
Give each unit of item ``j`` the weight::

    w_ij = 1 / (J * n_j)

where ``J = 16`` is the number of items and ``n_j`` the number of training units
belonging to item ``j``. Then the total weight of item ``j`` is::

    sum_i w_ij = n_j * 1 / (J * n_j) = 1 / J

independently of ``n_j``, so every item carries exactly the same total weight,
matching the macro evaluation. The vector is then rescaled so that the mean
sample weight is one (``sum(w) = N``), which leaves the objective unchanged but
keeps the effective learning rate comparable across folds.

What this is not
----------------
This is not ``class_weight="balanced"``, and it is not inverse-frequency
weighting applied to a classifier. It is a per-row weight constructed so that
each item's *total* contribution is exactly ``1/J``, computed from the item
frequency of the training fold only. Because the weights depend on ``n_j``, they
are recomputed inside every fold and are never transferred across folds.

Units with a structurally absent rate (``claim == 0``) are excluded from the
training universe before weighting; they are never imputed and never given a
weight.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Number of compensation modeling items.
N_ITEMS: int = 16


def item_balanced_weights(items: Sequence[str], n_items: int = N_ITEMS
                          ) -> np.ndarray:
    """Per-unit weights whose per-item totals are all ``1 / n_items``.

    The returned vector is rescaled so that ``sum(weights) == len(items)``.

    Parameters
    ----------
    items:
        Item identifier for every training unit, length ``N``.
    n_items:
        Number of items defining the macro average, ``J``. Defaults to 16.
    """
    values = np.asarray([str(x) for x in items])
    if values.size == 0:
        raise ValueError("no training units supplied")
    if int(n_items) <= 0:
        raise ValueError("the number of items must be positive")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / (float(n_items) * counts[inverse].astype(float))
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("degenerate item-balanced weights: sum is not positive")
    return weights * (float(values.size) / total)


def uniform_weights(items: Sequence[str]) -> np.ndarray:
    """All-ones weights: the unweighted pooled objective used as attribution control.

    Comparing a model trained with these weights against one trained with
    :func:`item_balanced_weights` isolates the effect of item balancing, because
    nothing else differs between the two runs.
    """
    return np.ones(len(items), dtype=float)


def weight_check(items: Sequence[str], weights: np.ndarray,
                 tol: float = 1e-8) -> Dict[str, Any]:
    """Verify the invariants the weighting scheme must satisfy.

    Returns a dictionary suitable for logging or asserting in a test:
    ``balanced`` is ``True`` when the largest and smallest per-item totals differ
    by less than ``tol`` in relative terms.
    """
    values = np.asarray([str(x) for x in items])
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != values.size:
        raise ValueError(
            f"weights and items disagree on length: {weights.size} vs {values.size}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("weights contain NaN or infinite values")
    unique = sorted(set(values.tolist()))
    totals = {u: float(np.sum(weights[values == u])) for u in unique}
    series = np.asarray(list(totals.values()), dtype=float)
    n_units = int(values.size)
    # The raw totals live on the rescaled axis, where the whole training set sums
    # to ``n_units`` rather than to one. Dividing by that sum puts every item back
    # on the probability-like axis on which the ideal is exactly ``1 / n_items``,
    # which is the axis the item-balanced objective is defined on.
    share = series / float(np.sum(weights))
    return {
        "n_units": n_units,
        "n_items": len(unique),
        "sum_w": float(np.sum(weights)),
        "sum_w_over_units": float(np.mean(weights)),
        "item_total_min": float(series.min()),
        "item_total_max": float(series.max()),
        "item_total_ideal": float(n_units) / float(N_ITEMS),
        "item_share_min": float(share.min()),
        "item_share_max": float(share.max()),
        "item_share_ideal": 1.0 / float(N_ITEMS),
        "max_deviation_from_share": float(np.max(np.abs(share - 1.0 / float(N_ITEMS)))),
        "balanced": bool(series.max() - series.min() < tol * max(1.0, series.max())),
        "all_positive": bool((weights > 0).all()),
    }


def objective_alignment_table(items: Sequence[str],
                              n_items: int = N_ITEMS) -> pd.DataFrame:
    """Compare the training objective with the evaluation objective, per item.

    For each item the table reports its share of training rows, the weight it
    receives under each weighting scheme, and the ratio of the effective training
    weight to the macro target weight (``1/J``). Under uniform weights the ratio
    equals ``J * n_j / N``, which is the magnitude of the misalignment; under
    item-balanced weights the ratio is 1 by construction.
    """
    values = np.asarray([str(x) for x in items])
    unique = sorted(set(values.tolist()))
    total = float(values.size)
    uniform = uniform_weights(values)
    balanced = item_balanced_weights(values, n_items)
    rows = []
    for item in unique:
        mask = values == item
        n_j = int(mask.sum())
        effective_uniform = float(np.sum(uniform[mask])) / total
        effective_balanced = float(np.sum(balanced[mask])) / total
        target = 1.0 / float(n_items)
        rows.append(
            {
                "item": item,
                "n_units": n_j,
                "share_of_training_rows": n_j / total,
                "macro_target_weight": target,
                "effective_weight_uniform": effective_uniform,
                "effective_weight_item_balanced": effective_balanced,
                # Computed rather than asserted, so that the table is evidence for
                # the balancing property instead of a restatement of it.
                "alignment_ratio_uniform": effective_uniform / target,
                "alignment_ratio_item_balanced": effective_balanced / target,
            }
        )
    return pd.DataFrame(rows).sort_values("n_units", ascending=False).reset_index(drop=True)


def describe_weighting(items: Sequence[str], n_items: int = N_ITEMS) -> str:
    """One-line human-readable summary of the weighting invariants.

    Reported on the share axis, where the ideal is exactly ``1 / n_items`` and is
    therefore comparable across folds of different size. The raw per-item totals are
    reported by :func:`weight_check` for callers that need them.
    """
    weights = item_balanced_weights(items, n_items)
    check = weight_check(items, weights)
    return (
        "item-balanced weights: {n_units} units over {n_items} items, "
        "per-item share in [{lo:.6f}, {hi:.6f}] against an ideal of 1/{n_items}, "
        "mean weight {mean:.6f}".format(
            n_units=check["n_units"],
            n_items=check["n_items"],
            lo=check["item_share_min"],
            hi=check["item_share_max"],
            mean=check["sum_w_over_units"],
        )
    )
