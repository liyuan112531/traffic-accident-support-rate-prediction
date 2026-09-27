"""Convex combination expert (the seventh expert of the frozen pool).

Rather than a single learner, this expert solves for the best non-negative
convex combination of the base experts *per item*, using only out-of-fold
predictions:

    minimise_w   MAE(P w, y) + l2 * ||w||^2
    subject to   w >= 0,  sum(w) = 1

where the columns of ``P`` are the base experts' out-of-fold predictions. The
combination weights are therefore constrained to the probability simplex, so the
combined prediction stays inside the range of its inputs and the combination
layer itself has no free scale.

Two design points deserve attention.

The weights are fitted on out-of-fold predictions rather than in-sample fits.
Fitting on in-sample predictions would let the solver reward the base model that
overfits hardest, because in-sample errors are optimistic in proportion to a
model's capacity.

The tiny L2 term is not regularisation in the usual sense. It is there to make
the solution unique when several weight vectors attain the same minimum MAE; the
coefficient is fixed a priori and is not tuned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

#: Default L2 coefficient, pre-registered to break ties between equivalent optima.
DEFAULT_L2: float = 1e-3

#: Default number of inner folds used to build the out-of-fold prediction matrix.
DEFAULT_INNER_SPLITS: int = 3


def effective_number(weights: np.ndarray) -> float:
    """``1 / sum(w^2)``: the effective number of contributing experts.

    Equals ``k`` when ``k`` experts share the weight equally, and approaches 1
    when a single expert dominates.
    """
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights ** 2))
    return float(1.0 / denominator) if denominator > 0 else float("nan")


def convex_weights(p_oof: np.ndarray, y: np.ndarray, l2: float = DEFAULT_L2,
                   maxiter: int = 400, tol: float = 1e-10) -> Dict[str, Any]:
    """Solve the simplex-constrained minimum-MAE combination.

    Parameters
    ----------
    p_oof:
        Out-of-fold predictions, shape ``(n_observations, n_base_experts)``.
    y:
        Observed support rates, length ``n_observations``.
    l2:
        Tie-breaking ridge coefficient. Must be small and fixed in advance.

    Returns
    -------
    dict
        ``weights``, ``oof_mae``, ``n_effective`` and solver diagnostics. When
        the solver fails, equal weights are returned and ``solver_success`` is
        ``False``; failures are surfaced rather than hidden because an equal
        weight fallback changes the expert's behaviour.
    """
    p_oof = np.asarray(p_oof, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if p_oof.ndim != 2:
        raise ValueError("out-of-fold prediction matrix must be two-dimensional")
    if p_oof.shape[0] != y.size:
        raise ValueError(
            f"prediction matrix has {p_oof.shape[0]} rows but y has {y.size}"
        )
    if not (np.isfinite(p_oof).all() and np.isfinite(y).all()):
        raise ValueError("convex combination inputs contain NaN or infinite values")
    n_base = p_oof.shape[1]
    if n_base == 0:
        raise ValueError("no base experts supplied")

    def objective(w: np.ndarray) -> float:
        return float(np.mean(np.abs(p_oof @ w - y))) + float(l2) * float(w @ w)

    start = np.full(n_base, 1.0 / n_base, dtype=float)
    constraint = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
        "jac": lambda w: np.ones(n_base, dtype=float),
    }
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_base,
        constraints=[constraint],
        options={"maxiter": int(maxiter), "ftol": float(tol)},
    )
    weights = np.asarray(result.x, dtype=float)
    converged = bool(result.success) and np.isfinite(weights).all() and weights.sum() > 0
    if not converged:
        weights = start.copy()
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / weights.sum()
    return {
        "weights": weights,
        "oof_mae": float(np.mean(np.abs(p_oof @ weights - y))),
        "n_effective": effective_number(weights),
        "solver_success": converged,
        "solver_message": str(result.message)[:160],
        "n_iterations": int(getattr(result, "nit", 0)),
    }


class ConvexCombinationExpert:
    """Simplex-weighted combination of base experts, fitted per item.

    The class exposes the same ``fit`` / ``predict`` interface as the other experts
    so the pipeline can treat all eleven uniformly.

    Sharing mechanism
    -----------------
    Fitting the combination by refitting all base models internally would duplicate
    work the pipeline already performs: the base experts are members of the same
    pool and are fitted on the same data for the same items. When the pipeline
    already has out-of-fold base predictions and base models fitted on the full
    data, it passes them in through :meth:`fit_fitted` and the internal refitting is
    skipped. The weights are then identical to what an internal fit would produce,
    because they depend only on the out-of-fold prediction matrix and the target.
    """

    def __init__(self, base_names: Sequence[str],
                 base_builder=None,
                 n_inner_splits: int = DEFAULT_INNER_SPLITS,
                 l2: float = DEFAULT_L2) -> None:
        self.base_names = list(base_names)
        self._base_builder = base_builder
        self.n_inner_splits = int(n_inner_splits)
        self.l2 = float(l2)
        self.weights_: Optional[np.ndarray] = None
        self.fitted_: List[Any] = []
        self.diagnostics_: Dict[str, Any] = {}
        self.shared_: bool = False

    # ------------------------------------------------------------------ internal
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ConvexCombinationExpert":
        """Fit by cross-fitting the base experts internally.

        Used when the caller has no reusable base fits. Every base model is fitted
        once per inner fold to build the out-of-fold matrix, then once on the full
        data to serve predictions.
        """
        if self._base_builder is None:
            raise ValueError(
                "this expert has no base builder, so it can only be fitted through "
                "fit_fitted with out-of-fold predictions and fitted base models"
            )
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape[0] != y.size:
            raise ValueError("feature matrix and target disagree on length")
        if x.shape[0] < self.n_inner_splits:
            raise ValueError(
                f"need at least {self.n_inner_splits} rows to cross-fit the convex "
                f"combination, received {x.shape[0]}"
            )
        out_of_fold = np.full((x.shape[0], len(self.base_names)), np.nan, dtype=float)
        splitter = KFold(n_splits=self.n_inner_splits, shuffle=True, random_state=0)
        for train_index, valid_index in splitter.split(x):
            for column, name in enumerate(self.base_names):
                model = self._base_builder(name, 0)
                model.fit(x[train_index], y[train_index])
                out_of_fold[valid_index, column] = np.clip(
                    model.predict(x[valid_index]), 0.0, 1.0
                )
        if not np.isfinite(out_of_fold).all():
            raise RuntimeError("cross-fitting left undefined out-of-fold predictions")
        fitted = []
        for name in self.base_names:
            model = self._base_builder(name, 0)
            model.fit(x, y)
            fitted.append(model)
        return self.fit_fitted(out_of_fold, y, fitted, shared=False)

    def fit_fitted(self, out_of_fold: np.ndarray, y: np.ndarray,
                   fitted_base_models: Sequence[Any],
                   shared: bool = True) -> "ConvexCombinationExpert":
        """Fit the weights from an out-of-fold matrix and pre-fitted base models.

        ``out_of_fold`` has shape ``(n_observations, n_base_experts)`` and must
        contain genuine out-of-fold predictions: fitting the weights on in-sample
        predictions would reward whichever base model overfits most.
        """
        out_of_fold = np.asarray(out_of_fold, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if out_of_fold.ndim != 2:
            raise ValueError("the out-of-fold matrix must be two-dimensional")
        if out_of_fold.shape[1] != len(self.base_names):
            raise ValueError(
                f"the out-of-fold matrix has {out_of_fold.shape[1]} columns but the "
                f"base pool has {len(self.base_names)} members"
            )
        if len(fitted_base_models) != len(self.base_names):
            raise ValueError("the number of fitted base models does not match the pool")
        solution = convex_weights(out_of_fold, y, l2=self.l2)
        self.weights_ = solution["weights"]
        self.diagnostics_ = solution
        self.fitted_ = list(fitted_base_models)
        self.shared_ = bool(shared)
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("the convex combination expert has not been fitted")
        x = np.asarray(x, dtype=float)
        stacked = np.column_stack(
            [np.clip(model.predict(x), 0.0, 1.0) for model in self.fitted_]
        )
        return np.clip(stacked @ self.weights_, 0.0, 1.0)

    def weight_table(self) -> Dict[str, float]:
        """Mapping from base expert name to fitted weight."""
        if self.weights_ is None:
            return {}
        return {name: float(w) for name, w in zip(self.base_names, self.weights_)}


def weights_from_out_of_fold(out_of_fold: np.ndarray, y: np.ndarray,
                             l2: float = DEFAULT_L2) -> np.ndarray:
    """Simplex weights minimising the out-of-fold mean absolute error.

    Standalone entry point used when the weights are needed without constructing an
    expert object, for instance to compute a nested combination inside the pipeline.
    """
    return convex_weights(np.asarray(out_of_fold, dtype=float),
                          np.asarray(y, dtype=float), l2=l2)["weights"]
