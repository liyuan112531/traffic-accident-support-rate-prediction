"""Declarative registry of the 11 experts with pre-registered hyperparameters.

Every value in :data:`SPECS` was fixed before any experimental result was
inspected and is reproduced here unchanged. There is no hyperparameter search in
this repository: the same parameters are applied to all 16 items, to every
feature set, and to every fold and seed, so no part of the model configuration
depends on the data.

Pool composition
----------------
The frozen 11-expert pool used in the manuscript is::

    median, ridge, rf, xgb, lgb, catboost, c3, extratrees, histgb, svr_rbf, tabpfn

``c3`` is the item-wise convex combination expert, which itself combines the six
base experts listed in :data:`C3_BASE_POOL`. It is counted once in the pool.

Availability
------------
:func:`is_available` reports whether an expert can be constructed in the current
environment. Only TabPFN can be unavailable in practice, because it needs model
weights that are downloaded on first use; those weights are not distributed with
this repository. Unavailable experts are skipped and reported, never silently
substituted.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Base experts combined by the convex combination expert.
C3_BASE_POOL: tuple = ("median", "ridge", "rf", "xgb", "lgb", "catboost")

#: The frozen 11-expert pool, in the order used for the meta representation.
FROZEN_POOL: tuple = (
    "median",
    "ridge",
    "rf",
    "xgb",
    "lgb",
    "catboost",
    "c3",
    "extratrees",
    "histgb",
    "svr_rbf",
    "tabpfn",
)


@dataclass(frozen=True)
class ExpertSpec:
    """One expert: how to build it, what it needs, and what it is."""

    name: str
    family: str
    description: str
    requires: tuple = ()
    optimises: str = "mae"
    #: ``True`` when the expert needs an external artefact at first use.
    needs_external_weights: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


#: Pre-registered specification of every expert.
SPECS: Dict[str, ExpertSpec] = {
    "median": ExpertSpec(
        name="median",
        family="baseline",
        description=(
            "Predicts the training-fold median support rate of the item and "
            "ignores the features. Anchors what is achievable without any "
            "request-side information."
        ),
        optimises="mae",
        params={"how": "median"},
    ),
    "ridge": ExpertSpec(
        name="ridge",
        family="linear",
        description="L2-regularised linear regression on the request-side features.",
        requires=("sklearn",),
        optimises="l2",
        params={"alpha": 1.0},
    ),
    "rf": ExpertSpec(
        name="rf",
        family="bagged trees",
        description="Random forest regressor; averaged deep trees on bootstrapped samples.",
        requires=("sklearn",),
        optimises="l2",
        params={
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": 1,
        },
    ),
    "xgb": ExpertSpec(
        name="xgb",
        family="boosted trees",
        description="Gradient-boosted trees with second-order split gains.",
        requires=("xgboost",),
        optimises="l2",
        params={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": 1,
        },
    ),
    "lgb": ExpertSpec(
        name="lgb",
        family="boosted trees",
        description="Leaf-wise gradient boosting with histogram splits.",
        requires=("lightgbm",),
        optimises="l2",
        params={
            "n_estimators": 300,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "min_child_samples": 20,
            "n_jobs": 1,
            "verbose": -1,
        },
    ),
    "catboost": ExpertSpec(
        name="catboost",
        family="boosted trees",
        description="Ordered-boosting gradient trees with symmetric splits.",
        requires=("catboost",),
        optimises="l2",
        params={
            "iterations": 300,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": 1,
        },
    ),
    "c3": ExpertSpec(
        name="c3",
        family="combination",
        description=(
            "Item-wise convex combination of the six base experts, with the "
            "combination weights fitted on out-of-fold predictions over the "
            "probability simplex."
        ),
        requires=("sklearn", "scipy"),
        optimises="mae",
        params={"n_inner_splits": 3, "l2": 1e-3},
    ),
    "extratrees": ExpertSpec(
        name="extratrees",
        family="bagged trees",
        description="Extremely randomised trees with random split thresholds.",
        requires=("sklearn",),
        optimises="l2",
        params={
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_jobs": 1,
        },
    ),
    "histgb": ExpertSpec(
        name="histgb",
        family="boosted trees",
        description="Histogram-based gradient boosting with leaf-wise growth.",
        requires=("sklearn",),
        optimises="l2",
        params={
            "max_iter": 300,
            "learning_rate": 0.06,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "early_stopping": False,
        },
    ),
    "svr_rbf": ExpertSpec(
        name="svr_rbf",
        family="kernel",
        description="Radial-basis-function support vector regression, standardised inputs.",
        requires=("sklearn",),
        optimises="epsilon-insensitive",
        params={"C": 10.0, "gamma": "scale", "epsilon": 0.1},
    ),
    "tabpfn": ExpertSpec(
        name="tabpfn",
        family="prior-data fitted network",
        description=(
            "Prior-data fitted tabular transformer performing in-context "
            "regression. Needs pretrained weights downloaded at first use; those "
            "weights are not distributed with this repository."
        ),
        requires=("tabpfn",),
        optimises="l2",
        needs_external_weights=True,
        params={
            "n_estimators": 1,
            "device": "cpu",
            "ignore_pretraining_limits": True,
        },
    ),
}

#: Names in registry order.
EXPERT_NAMES: tuple = tuple(SPECS.keys())


def _module_missing(module: str) -> bool:
    """Whether ``module`` cannot be imported in this environment."""
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return True
    return False


def _unmet_requirement(name: str,
                       allow_external_weights: bool = False) -> Optional[str]:
    """Reason the named expert cannot be built here, or ``None`` if it can.

    The convex combination expert additionally requires every member of its base
    pool. It is reported as unavailable rather than silently redefined over a
    reduced pool, because a combination of six models and a combination of four are
    different experts.
    """
    if name not in SPECS:
        return f"unknown expert: {name!r}"
    spec = SPECS[name]
    missing = [m for m in spec.requires if _module_missing(m)]
    if missing:
        return f"missing dependency: {', '.join(missing)}"
    if spec.needs_external_weights and not allow_external_weights:
        return (
            "requires pretrained weights that are not distributed with this "
            "repository; pass allow_external_weights=True to enable"
        )
    if name == "c3":
        missing_base = [
            base for base in C3_BASE_POOL
            if _unmet_requirement(base, allow_external_weights) is not None
        ]
        if missing_base:
            return (
                "the convex combination expert requires its full base pool; "
                f"unavailable: {', '.join(missing_base)}"
            )
    return None


def is_available(name: str, allow_external_weights: bool = False) -> bool:
    """Whether the named expert can be built here.

    ``allow_external_weights`` controls whether experts needing downloaded weights
    (TabPFN) are considered available. It defaults to ``False`` so the default code
    path never triggers a network download.
    """
    return _unmet_requirement(name, allow_external_weights) is None


def availability(allow_external_weights: bool = False) -> Dict[str, bool]:
    """Availability of every registered expert."""
    return {name: is_available(name, allow_external_weights) for name in EXPERT_NAMES}


def enforce_convex_base(pool: Sequence[str],
                        skipped: Optional[Dict[str, str]] = None
                        ) -> tuple:
    """Drop ``c3`` unless its whole base pool is present in the same pool.

    ``c3`` is defined as a combination over exactly six named models. Combining
    whichever subset happens to be present would produce a different expert under
    the same name, so the pool is corrected instead and the reason is recorded. A
    run that asks for a reduced pool therefore loses ``c3`` rather than silently
    redefining it.

    Returns ``(pool, skipped)``.
    """
    names = [str(name) for name in pool]
    if "c3" not in names:
        return tuple(names), dict(skipped or {})
    missing = [base for base in C3_BASE_POOL if base not in names]
    if not missing:
        return tuple(names), dict(skipped or {})
    reason = (
        "the convex combination expert is defined over its full base pool; "
        f"missing from the requested pool: {', '.join(missing)}"
    )
    reduced = [name for name in names if name != "c3"]
    updated = dict(skipped or {})
    updated["c3"] = reason
    return tuple(reduced), updated


def resolve_pool(requested: Optional[Sequence[str]] = None,
                 allow_external_weights: bool = False
                 ) -> tuple:
    """Return the requested pool restricted to what is available, plus reasons.

    Returns ``(pool, skipped)`` where ``skipped`` maps each dropped expert to the
    reason it was dropped, so a run can report exactly what it did not use. The
    convex combination expert is additionally removed when its base pool is not
    fully present; see :func:`enforce_convex_base`.
    """
    requested = list(FROZEN_POOL if requested is None else requested)
    unknown = [name for name in requested if name not in SPECS]
    if unknown:
        raise KeyError(f"unknown expert(s) requested: {unknown}")
    pool: List[str] = []
    skipped: Dict[str, str] = {}
    for name in requested:
        reason = _unmet_requirement(name, allow_external_weights)
        if reason is not None:
            skipped[name] = reason
            continue
        pool.append(name)
    return enforce_convex_base(pool, skipped)


def _build_base_median(how: str = "median"):
    """Constant predictor returning the training-fold mean or median."""

    class _ConstantPredictor:
        def __init__(self, how_: str) -> None:
            self.how_ = how_
            self.value_ = float("nan")

        def fit(self, x, y) -> "_ConstantPredictor":
            y = list(y)
            if not y:
                raise ValueError("constant predictor received an empty training set")
            import numpy as np

            self.value_ = float(np.median(y)) if self.how_ == "median" else float(np.mean(y))
            return self

        def predict(self, x):
            import numpy as np

            return np.full(len(x), self.value_, dtype=float)

    return _ConstantPredictor(how)


def build_expert(name: str, seed: int = 0):
    """Construct an unfitted expert by name.

    ``seed`` is injected into the estimator's own random-state parameter so that
    a single seed controls every stochastic component of the run.
    """
    if name not in SPECS:
        raise KeyError(f"unknown expert: {name!r}; registered: {list(SPECS)}")
    spec = SPECS[name]
    params = dict(spec.params)

    if name == "median":
        return _build_base_median(params.get("how", "median"))

    if name == "c3":
        from .convex import ConvexCombinationExpert

        return ConvexCombinationExpert(
            base_names=C3_BASE_POOL,
            base_builder=build_expert,
            n_inner_splits=params.get("n_inner_splits", 3),
            l2=params.get("l2", 1e-3),
        )

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=params["alpha"])

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=int(seed), **params)

    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(random_state=int(seed), **params)

    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=int(seed), **params)

    if name == "svr_rbf":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        return make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=params["C"], gamma=params["gamma"],
                epsilon=params["epsilon"]),
        )

    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=int(seed), **params)

    if name == "lgb":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=int(seed), **params)

    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=int(seed), **params)

    if name == "tabpfn":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(random_state=int(seed), **params)

    raise KeyError(f"no builder registered for expert {name!r}")


def describe_experts() -> List[Dict[str, Any]]:
    """Tabular description of the pool, for documentation and the audit trail."""
    rows: List[Dict[str, Any]] = []
    for name in EXPERT_NAMES:
        spec = SPECS[name]
        rows.append(
            {
                "expert": name,
                "family": spec.family,
                "description": spec.description,
                "requires": ", ".join(spec.requires) or "-",
                "needs_external_weights": spec.needs_external_weights,
                "optimises": spec.optimises,
                "hyperparameters": params_to_string(spec.params),
            }
        )
    return rows


def params_to_string(params: Dict[str, Any]) -> str:
    """Render a hyperparameter mapping as ``key=value`` pairs."""
    if not params:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in params.items())
