"""Paired resampling tests for comparing two prediction methods.

Both procedures share one design decision: the two methods are compared *paired*,
on exactly the same resampled units. Comparing independently resampled samples
would inflate the variance of the difference and destroy the correlation between
the two predictors on the same case, which is precisely the quantity of interest.

The resampling unit is the case, not the case-item unit. Units belonging to the
same case are not independent: they share a claim structure, a procedure and a
year, so resampling units would understate the standard error. Cases are drawn
with replacement and all of a drawn case's units move together.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Default number of bootstrap replicates.
DEFAULT_BOOTSTRAP: int = 2000

#: Default number of permutation replicates.
DEFAULT_PERMUTATIONS: int = 2000


def _metric_matrix(counts: np.ndarray, row_case: np.ndarray, y: np.ndarray,
                   prediction: np.ndarray, item_rows: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted per-replicate MAE, RMSE and R-squared for one item.

    ``counts`` has shape ``(n_replicates, n_cases)``; ``item_rows`` selects the
    rows of this item. Returns three arrays of length ``n_replicates``.
    """
    weights = counts[:, row_case[item_rows]]
    truth = y[item_rows]
    pred = prediction[item_rows]
    residual = truth[None, :] - pred[None, :]
    total_weight = weights.sum(axis=1)
    weighted_mean = (weights * truth[None, :]).sum(axis=1) / np.maximum(total_weight, 1e-300)
    mae = (weights * np.abs(residual)).sum(axis=1) / np.maximum(total_weight, 1e-300)
    mse = (weights * residual ** 2).sum(axis=1) / np.maximum(total_weight, 1e-300)
    total_ss = (weights * (truth[None, :] - weighted_mean[:, None]) ** 2).sum(axis=1)
    r2 = np.where(total_ss > 0, 1.0 - (weights * residual ** 2).sum(axis=1)
                  / np.where(total_ss > 0, total_ss, 1.0), np.nan)
    return mae, np.sqrt(mse), r2


def case_stratified_bootstrap(y: np.ndarray, prediction_a: np.ndarray,
                              prediction_b: np.ndarray,
                              items: Sequence[str],
                              cases: Sequence[Any],
                              n_replicates: int = DEFAULT_BOOTSTRAP,
                              seed: int = 0,
                              alpha: float = 0.05) -> Dict[str, Any]:
    """Paired case-stratified bootstrap percentile intervals for the macro metrics.

    Returns one row per metric with the observed difference ``a - b``, the
    bootstrap standard deviation, the percentile interval, and the fraction of
    replicates in which ``a`` is the better method. A negative difference means
    ``a`` has the lower error, which is the better outcome for MAE and RMSE; for
    R-squared a positive difference is better.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction_a = np.asarray(prediction_a, dtype=float).reshape(-1)
    prediction_b = np.asarray(prediction_b, dtype=float).reshape(-1)
    labels = np.asarray([str(i) for i in items])
    if not (y.size == prediction_a.size == prediction_b.size == labels.size):
        raise ValueError("target, predictions and item labels must have equal length")

    unique_cases, row_case = np.unique(np.asarray(cases), return_inverse=True)
    n_cases = int(unique_cases.size)
    if n_cases < 2:
        raise ValueError("bootstrap requires at least two distinct cases")

    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_cases, size=(int(n_replicates), n_cases))
    counts = np.empty((int(n_replicates), n_cases), dtype=float)
    for row in range(int(n_replicates)):
        counts[row] = np.bincount(draws[row], minlength=n_cases)

    mae_a, rmse_a, r2_a = [], [], []
    mae_b, rmse_b, r2_b = [], [], []
    usable_items = 0
    for item in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == item)
        if rows.size < 2:
            continue
        usable_items += 1
        for values, prediction, store_mae, store_rmse, store_r2 in (
            (y, prediction_a, mae_a, rmse_a, r2_a),
            (y, prediction_b, mae_b, rmse_b, r2_b),
        ):
            mae, rmse, r2 = _metric_matrix(counts, row_case, values, prediction, rows)
            store_mae.append(mae)
            store_rmse.append(rmse)
            store_r2.append(r2)
    if usable_items == 0:
        raise ValueError("no item had enough units for the bootstrap")

    def _stack(rows: List[np.ndarray]) -> np.ndarray:
        return np.nanmean(np.vstack(rows), axis=0)

    replicates = {
        "macro_MAE": (_stack(mae_a) - _stack(mae_b), "lower_is_better"),
        "macro_RMSE": (_stack(rmse_a) - _stack(rmse_b), "lower_is_better"),
        "macro_R2": (_stack(r2_a) - _stack(r2_b), "higher_is_better"),
    }

    from ..evaluation.metrics import macro_metrics

    observed = {
        key: macro_metrics(y, prediction_a, items)[key]
        - macro_metrics(y, prediction_b, items)[key]
        for key in ("macro_MAE", "macro_RMSE", "macro_R2")
    }

    rows: List[Dict[str, Any]] = []
    for metric, (distribution, direction) in replicates.items():
        finite = distribution[np.isfinite(distribution)]
        low, high = (np.percentile(finite, [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)])
                     if finite.size else (np.nan, np.nan))
        better = (finite < 0) if direction == "lower_is_better" else (finite > 0)
        rows.append(
            {
                "metric": metric,
                "delta_observed": float(observed[metric]),
                "bootstrap_mean": float(np.nanmean(finite)) if finite.size else np.nan,
                "bootstrap_sd": float(np.nanstd(finite)) if finite.size else np.nan,
                "ci_low": float(low),
                "ci_high": float(high),
                "direction_probability": float(np.mean(better)) if finite.size else np.nan,
                "ci_crosses_zero": bool(low <= 0.0 <= high),
                "n_replicates": int(n_replicates),
                "n_cases": n_cases,
                "n_items": usable_items,
                "seed": int(seed),
            }
        )
    return {"rows": rows, "n_cases": n_cases, "n_items": usable_items}


def paired_permutation_test(y: np.ndarray, prediction_a: np.ndarray,
                            prediction_b: np.ndarray,
                            items: Sequence[str],
                            n_replicates: int = DEFAULT_PERMUTATIONS,
                            seed: int = 0) -> Dict[str, Any]:
    """Paired permutation test on the macro metrics.

    Under the null hypothesis the two methods are exchangeable, so for each
    case-item unit the two predictions may be swapped without changing the joint
    distribution. Swapping randomly and recomputing the macro difference
    generates a null distribution centred at zero. The two-sided p-value is the
    proportion of replicates whose absolute difference is at least the observed
    one.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction_a = np.asarray(prediction_a, dtype=float).reshape(-1)
    prediction_b = np.asarray(prediction_b, dtype=float).reshape(-1)
    labels = np.asarray([str(i) for i in items])
    if not (y.size == prediction_a.size == prediction_b.size == labels.size):
        raise ValueError("target, predictions and item labels must have equal length")

    rng = np.random.default_rng(int(seed))
    n_replicates = int(n_replicates)
    abs_a = np.abs(y - prediction_a)
    abs_b = np.abs(y - prediction_b)
    sq_a = (y - prediction_a) ** 2
    sq_b = (y - prediction_b) ** 2

    per_item = {"mae_a": [], "mae_b": [], "rmse_a": [], "rmse_b": [],
                "r2_a": [], "r2_b": []}
    for item in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == item)
        if rows.size < 2:
            continue
        swap = rng.random((n_replicates, rows.size)) < 0.5
        per_item["mae_a"].append(np.where(swap, abs_b[rows], abs_a[rows]).mean(axis=1))
        per_item["mae_b"].append(np.where(swap, abs_a[rows], abs_b[rows]).mean(axis=1))
        rmse_a = np.sqrt(np.where(swap, sq_b[rows], sq_a[rows]).mean(axis=1))
        rmse_b = np.sqrt(np.where(swap, sq_a[rows], sq_b[rows]).mean(axis=1))
        per_item["rmse_a"].append(rmse_a)
        per_item["rmse_b"].append(rmse_b)
        total_ss = float(np.sum((y[rows] - y[rows].mean()) ** 2))
        if total_ss > 0:
            per_item["r2_a"].append(
                1.0 - np.where(swap, sq_b[rows], sq_a[rows]).sum(axis=1) / total_ss
            )
            per_item["r2_b"].append(
                1.0 - np.where(swap, sq_a[rows], sq_b[rows]).sum(axis=1) / total_ss
            )

    from ..evaluation.metrics import macro_metrics

    reference_a = macro_metrics(y, prediction_a, items)
    reference_b = macro_metrics(y, prediction_b, items)
    rows: List[Dict[str, Any]] = []
    for metric, key_a, key_b in (("macro_MAE", "mae_a", "mae_b"),
                                 ("macro_RMSE", "rmse_a", "rmse_b"),
                                 ("macro_R2", "r2_a", "r2_b")):
        if not per_item[key_a]:
            continue
        null = np.nanmean(np.vstack(per_item[key_a]), axis=0) - np.nanmean(
            np.vstack(per_item[key_b]), axis=0
        )
        observed = reference_a[metric] - reference_b[metric]
        p_two_sided = float(
            (np.sum(np.abs(null) >= abs(observed)) + 1.0) / (null.size + 1.0)
        )
        better = (null <= observed) if metric != "macro_R2" else (null >= observed)
        p_one_sided = float((np.sum(better) + 1.0) / (null.size + 1.0))
        rows.append(
            {
                "metric": metric,
                "delta_observed": float(observed),
                "permutation_mean": float(np.nanmean(null)),
                "permutation_sd": float(np.nanstd(null)),
                "p_two_sided": p_two_sided,
                "p_one_sided_a_better": p_one_sided,
                "n_replicates": n_replicates,
                "seed": int(seed),
            }
        )
    return {"rows": rows, "n_replicates": n_replicates}
