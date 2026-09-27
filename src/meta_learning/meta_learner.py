"""The Item-Balanced L1 meta-learner.

Definition
----------
The meta head is a gradient-boosted tree ensemble trained on the 73-dimensional
representation with the absolute-error objective, using the item-balanced sample
weights of :mod:`src.meta_learning.item_balanced`::

    L(theta) = sum_j (1 / J) * sum_{i in I_j} | y_ij - g_theta(z_ij) |

so each of the ``J = 16`` items contributes a total of ``1/J`` to the objective.
The mean absolute error is used rather than the mean squared error because the
support rate is a bounded ratio with point mass at 0 and 1; a squared loss
over-penalises the resulting heavy tails and pushes the fit towards the interior
of the range.

Three variants are exposed so that the contribution of item balancing and of the
loss shape can be attributed separately:

``l1``   absolute error with item-balanced weights. The proposed configuration.
``l2``   squared error with item-balanced weights. Same weights, different loss.
``mix``  per-unit convex combination ``(1 - lam) * |e| + lam * e^2``, smoothed
         with a constant inside the square root.

The mixture is implemented with analytic gradient and hessian so that it can be
passed to a second-order booster. The smoothing constant is fixed in advance; it
only affects the behaviour of the loss for very small residuals, where the
absolute and squared losses agree to first order.

Everything except ``sample_weight`` and the loss shape is identical to the
pre-registered hyperparameters used elsewhere in the study, so the comparison
between variants isolates a single factor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np

#: Label used in tables, figures and documentation.
LOSS_LABEL: str = "Item-Balanced L1"

#: Meta-head hyperparameters, identical to the pre-registered tree settings.
META_HYPERPARAMETERS: Dict[str, Any] = {
    "n_estimators": 300,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
    "n_jobs": 1,
    "verbose": -1,
}

#: Smoothing constant inside the mixture loss.
MIX_DELTA: float = 1e-4


def mix_gradient_hessian(lam: float, delta: float = MIX_DELTA
                         ) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Analytic gradient and hessian of the smoothed mixture loss.

    With ``e = prediction - truth``::

        L(e) = (1 - lam) * sqrt(e^2 + delta) + lam * e^2
        dL/de = (1 - lam) * e / sqrt(e^2 + delta) + 2 * lam * e
        d2L/de2 = (1 - lam) * delta / (e^2 + delta)^{3/2} + 2 * lam

    ``lam = 0`` recovers a smoothed absolute error, ``lam = 1`` the squared error.
    """
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1], received {lam}")

    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        root = np.sqrt(error ** 2 + float(delta))
        gradient = (1.0 - lam) * error / root + 2.0 * lam * error
        hessian = (1.0 - lam) * float(delta) / (root ** 3) + 2.0 * lam
        return gradient, hessian

    return objective


def _objective_keyword(kind: str) -> str:
    """Map a variant name onto LightGBM's objective keyword."""
    if kind == "l1":
        return "regression_l1"
    if kind == "l2":
        return "regression"
    raise ValueError(f"unknown meta-loss variant: {kind!r}; expected 'l1', 'l2' or 'mix'")


def fit_meta_head(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                  seed: int, kind: str = "l1", lam: Optional[float] = None,
                  hyperparameters: Optional[Dict[str, Any]] = None):
    """Fit the meta head and return the fitted LightGBM regressor.

    Parameters
    ----------
    x:
        Meta representation, shape ``(n_units, n_features)``.
    y:
        Observed support rates, length ``n_units``.
    sample_weight:
        Per-unit weights. Use :func:`src.meta_learning.item_balanced_weights` for
        the proposed configuration, or an all-ones vector for the unweighted
        attribution control.
    seed:
        Random seed.
    kind:
        ``"l1"``, ``"l2"`` or ``"mix"``.
    lam:
        Mixture parameter, required when ``kind == "mix"``.
    """
    from lightgbm import LGBMRegressor

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if x.shape[0] != y.size or sample_weight.size != y.size:
        raise ValueError(
            "meta head received inconsistent lengths: "
            f"x={x.shape[0]}, y={y.size}, weights={sample_weight.size}"
        )
    if not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
        raise ValueError("sample weights must be finite and strictly positive")

    params = dict(META_HYPERPARAMETERS if hyperparameters is None else hyperparameters)
    if kind == "mix":
        if lam is None:
            raise ValueError("kind='mix' requires an explicit lam")
        params.pop("objective", None)
        model = LGBMRegressor(
            objective=mix_gradient_hessian(float(lam)),
            random_state=int(seed),
            **params,
        )
    else:
        model = LGBMRegressor(
            objective=_objective_keyword(kind), random_state=int(seed), **params
        )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def predict_meta_head(model, x: np.ndarray) -> np.ndarray:
    """Predict and clip to the admissible support-rate interval ``[0, 1]``.

    Clipping is a property of the target's range, not a learned correction: a
    support rate cannot fall outside ``[0, 1]``, so predictions are projected onto
    the interval. No calibration layer is applied.
    """
    x = np.asarray(x, dtype=float)
    return np.clip(np.asarray(model.predict(x), dtype=float), 0.0, 1.0)
