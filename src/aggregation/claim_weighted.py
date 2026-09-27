"""Hierarchical aggregation from item support rates to the case-level rate.

The construction
----------------
For case ``i`` and item ``j`` define the item-level support rate and the
case-specific aggregation weight::

    r_ij = Award_ij / Claim_ij
    w_ij = Claim_ij / sum_k Claim_ik

Both quantities are observable before judgment is imposed on the model: the
claimed amounts are known when the case is filed, and the awarded amounts are
the target. The case-level support rate is then reconstructed as::

    R_i = sum_j w_ij * r_ij

This identity is exact, not an approximation. Substituting the definitions::

    sum_j (Claim_ij / sum_k Claim_ik) * (Award_ij / Claim_ij)
      = sum_j Award_ij / sum_k Claim_ik
      = Award_total_i / Claim_total_i
      = R_i

so the aggregation layer introduces no additional learnable parameters and no
additional degrees of freedom. It is a deterministic, algebraic re-expression of
the case-level rate in terms of item-level rates.

Scope of the claim
------------------
What is parameter-free is the aggregation layer. The experts and the
meta-learner that produce ``r_hat_ij`` do contain learnable parameters. The
distinction matters and is stated here explicitly so the statement is not read as
a claim about the whole model.

Why it is worth having
----------------------
Because the weights come from the case's own request structure, the reconstruction
is automatically case-adaptive: a case that claimed almost entirely medical
expenses is aggregated with a weight vector concentrated on medical expenses,
without any learned routing. The weight vector is also a description of the case
in its own right, summarised here by the Herfindahl-Hirschman index.

Structural absence
------------------
Items with ``Claim_ij = 0`` cannot contribute: their weight is zero and their
rate is undefined. Summing over them is therefore harmless, and no imputation is
needed. This is why ``w_ij`` already sums to one over the requested items alone.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..preprocessing.schema import AMOUNT_ZERO_TOL

#: Weight of each item under a uniform aggregation, for reference comparisons.
UNIFORM_WEIGHT: float = 1.0 / 16.0


def request_weights(claim_matrix: np.ndarray,
                    tol: float = AMOUNT_ZERO_TOL) -> np.ndarray:
    """Row-normalised claim shares ``w_ij = Claim_ij / sum_k Claim_ik``.

    Rows whose claim total is not positive cannot be normalised; the case has no
    defined support rate at all and is reported rather than silently zeroed.
    """
    claim = np.asarray(claim_matrix, dtype=float)
    if claim.ndim != 2:
        raise ValueError("the claim matrix must be two-dimensional")
    if not np.isfinite(claim).all():
        raise ValueError("the claim matrix contains NaN or infinite values")
    if (claim < -tol).any():
        raise ValueError("the claim matrix contains negative amounts")
    total = claim.sum(axis=1, keepdims=True)
    undefined = total[:, 0] <= tol
    if undefined.any():
        raise ValueError(
            f"{int(undefined.sum())} case(s) have a non-positive claim total, so no "
            "support rate is defined for them; exclude these cases before "
            "aggregation rather than imputing a value"
        )
    return claim / total


def aggregate(weights: np.ndarray, item_predictions: np.ndarray,
              defined: Optional[np.ndarray] = None) -> np.ndarray:
    """``R_hat_i = sum_j w_ij * r_hat_ij``, restricted to defined units.

    ``weights`` has shape ``(n_cases, n_items)`` and ``item_predictions`` the
    same. ``defined`` marks which case-item units carry a genuine prediction;
    units outside it contribute nothing. Undefined units already carry zero
    weight under :func:`request_weights`, so the result is the same as summing
    over requested items only.
    """
    weights = np.asarray(weights, dtype=float)
    predictions = np.asarray(item_predictions, dtype=float)
    if weights.shape != predictions.shape:
        raise ValueError(
            f"weights {weights.shape} and item predictions {predictions.shape} "
            "must have the same shape"
        )
    if defined is None:
        defined = np.isfinite(predictions)
    defined = np.asarray(defined, dtype=bool)
    if defined.shape != weights.shape:
        raise ValueError("the definedness mask must match the weight matrix shape")

    filled = np.where(defined, np.nan_to_num(predictions, nan=0.0, posinf=0.0,
                                             neginf=0.0), 0.0)
    numerator = np.sum(weights * filled, axis=1)
    denominator = np.sum(np.where(defined, weights, 0.0), axis=1)

    out = np.full(weights.shape[0], np.nan, dtype=float)
    resolvable = denominator > 0
    out[resolvable] = numerator[resolvable] / denominator[resolvable]
    return np.clip(out, 0.0, 1.0)


def conservation_deviation(claim_matrix: np.ndarray, award_matrix: np.ndarray
                           ) -> Dict[str, Any]:
    """Measure how exactly ``sum_j w_ij r_ij`` reproduces ``Award_total / Claim_total``.

    With exact arithmetic the deviation is zero; in double precision it is bounded
    by accumulated rounding error, typically of the order of ``1e-16``. The
    function reports the magnitude actually observed so the identity is verified
    rather than asserted.
    """
    claim = np.asarray(claim_matrix, dtype=float)
    award = np.asarray(award_matrix, dtype=float)
    if claim.shape != award.shape:
        raise ValueError("claim and award matrices must have the same shape")
    claim_total = claim.sum(axis=1)
    defined = claim_total > AMOUNT_ZERO_TOL
    excluded = int((~defined).sum())

    weights = request_weights(claim[defined])
    rates = np.full(claim[defined].shape, 0.0, dtype=float)
    positive = claim[defined] > AMOUNT_ZERO_TOL
    rates[positive] = award[defined][positive] / claim[defined][positive]
    reconstructed = aggregate(weights, rates, defined=positive)
    direct = award[defined].sum(axis=1) / claim_total[defined]
    deviation = np.abs(reconstructed - direct)

    return {
        "n_cases_checked": int(defined.sum()),
        "n_cases_excluded": excluded,
        "max_abs_deviation": float(deviation.max()) if deviation.size else 0.0,
        "mean_abs_deviation": float(deviation.mean()) if deviation.size else 0.0,
        "exact_match_fraction": float(np.mean(deviation <= 0.0))
        if deviation.size else 1.0,
        "within_1e-12_fraction": float(np.mean(deviation <= 1e-12))
        if deviation.size else 1.0,
    }


def herfindahl(weights: np.ndarray) -> Dict[str, float]:
    """Concentration of the aggregation weights, per case.

    ``HHI_i = sum_j w_ij^2`` equals ``1/16`` when a case spreads its claim evenly
    across all sixteen items and approaches 1 when a case claims a single item.
    It summarises how much of the aggregation is driven by one head of claim.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("the weight matrix must be two-dimensional")
    index = np.sum(weights ** 2, axis=1)
    used = np.sum(weights > 0, axis=1)
    return {
        "mean": float(index.mean()),
        "median": float(np.median(index)),
        "min": float(index.min()),
        "max": float(index.max()),
        "n_cases": int(index.size),
        "uniform_reference": float(1.0 / weights.shape[1]),
        "n_cases_single_item": int(np.sum(used == 1)),
    }


def uniform_benchmark(n_items: int = 16) -> np.ndarray:
    """Uniform weight vector ``1 / n_items``, as a diagnostic benchmark only.

    Uniform weighting is *not* the proposed aggregation and is not a valid
    substitute: it discards the case's request structure. It is provided so that
    the contribution of structure-aware weighting can be quantified.
    """
    return np.full(int(n_items), 1.0 / float(n_items), dtype=float)


def case_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Mean absolute error, root mean squared error and R-squared at case level."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError("target and prediction disagree on length")
    residual = y_true - y_pred
    total_sum_squares = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": float(1.0 - np.sum(residual ** 2) / total_sum_squares)
        if total_sum_squares > 0 else float("nan"),
        "n_cases": int(y_true.size),
    }
