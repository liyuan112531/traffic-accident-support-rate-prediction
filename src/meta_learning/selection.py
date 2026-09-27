"""Pre-registered model-selection rules for the meta layer.

All selection happens on inner cross-validation predictions inside the training
fold. Outer-test data never participates in any choice made here, because doing
so would turn an evaluation into a selection and inflate the reported
performance.

Two rules are provided.

The primary rule is lexicographic rather than scalarised: among candidate
configurations whose mean absolute error satisfies the constraint on the root
mean squared error, take the one with the lowest mean absolute error. A scalar
penalty (``MAE + C * max(0, RMSE - cap)^2``) is also provided, but it is used
only for ranking candidates inside a fold, never as a training gradient. Treating
a macro-level metric as if it were a per-sample differentiable loss would be a
category error: it is not decomposable over units.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Pre-registered penalty coefficient for the scalarised ranking rule.
PENALTY_C: float = 100.0


def blend(prediction_a: np.ndarray, prediction_b: np.ndarray,
          alpha: float) -> np.ndarray:
    """Weighted blend ``alpha * a + (1 - alpha) * b``, clipped to ``[0, 1]``."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], received {alpha}")
    a = np.asarray(prediction_a, dtype=float)
    b = np.asarray(prediction_b, dtype=float)
    return np.clip(alpha * a + (1.0 - alpha) * b, 0.0, 1.0)


def penalty_score(mae: float, rmse: float, rmse_cap: float,
                  c: float = PENALTY_C) -> float:
    """Scalarised ranking score ``MAE + C * max(0, RMSE - cap)^2``."""
    return float(mae) + float(c) * max(0.0, float(rmse) - float(rmse_cap)) ** 2


def pick_min_mae_under_rmse_cap(rows: Sequence[Dict[str, Any]],
                                mae_key: str = "macro_MAE",
                                rmse_key: str = "macro_RMSE",
                                rmse_cap: Optional[float] = None
                                ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return the feasible candidate with the lowest MAE, and how it was chosen.

    ``rows`` is a sequence of dictionaries holding at least the two metric keys.
    When ``rmse_cap`` is ``None`` the rule degrades to plain minimum MAE, which is
    the honest behaviour when no constraint has been declared.
    """
    if not rows:
        raise ValueError("no candidate rows supplied")
    if rmse_cap is None:
        best = min(rows, key=lambda r: float(r[mae_key]))
        return best, "unconstrained_min_MAE"
    feasible = [r for r in rows if float(r[rmse_key]) <= float(rmse_cap)]
    if feasible:
        best = min(feasible, key=lambda r: float(r[mae_key]))
        return best, "feasible_min_MAE"
    nearest = min(
        rows,
        key=lambda r: (float(r[mae_key]) ** 2 + float(r[rmse_key]) ** 2),
    )
    return nearest, "diagnostic_nearest_when_infeasible"


def select_lambda(rows: Sequence[Dict[str, Any]], rmse_cap: float,
                  c: float = PENALTY_C) -> Tuple[Dict[str, Any], str]:
    """Pick the mixture parameter by the scalarised ranking rule."""
    if not rows:
        raise ValueError("no candidate rows supplied")
    scored: List[Tuple[float, Dict[str, Any]]] = [
        (penalty_score(r["macro_MAE"], r["macro_RMSE"], rmse_cap, c), r) for r in rows
    ]
    score, best = min(scored, key=lambda item: (item[0], float(item[1]["macro_MAE"])))
    return best, f"min_penalty_score({score:.6f})"


def select_alpha(inner_predictions: Dict[float, np.ndarray], y: np.ndarray,
                 items: Sequence[str], metric_fn,
                 alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                 rmse_cap: Optional[float] = None
                 ) -> Tuple[float, List[Dict[str, Any]], str]:
    """Choose the blend weight on inner-CV predictions.

    ``metric_fn(y, prediction, items)`` must return a mapping containing
    ``macro_MAE`` and ``macro_RMSE``.
    """
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        key = float(alpha)
        if key not in inner_predictions:
            raise KeyError(f"inner predictions missing for alpha={key}")
        metrics = metric_fn(y, inner_predictions[key], items)
        rows.append({"alpha": key, **metrics})
    best, how = pick_min_mae_under_rmse_cap(rows, rmse_cap=rmse_cap)
    if best is None:  # pragma: no cover - guarded by the non-empty check above
        raise RuntimeError("candidate selection returned no result")
    return float(best["alpha"]), rows, how
