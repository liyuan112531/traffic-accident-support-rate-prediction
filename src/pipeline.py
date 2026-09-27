"""End-to-end pipeline: experts, meta-learning, aggregation and evaluation.

This module is the single place where the whole method is assembled, so the
executable scripts stay thin and every run is reproducible from a configuration
object.

Protocol
--------
Cross-validation is at the *case* level, and the same fold assignment is shared by
every item and every expert. Two consequences follow. A case never appears in both
the training and the evaluation part of a fold, so no case-item unit of the same
case can leak across the boundary. And because the folds are shared, differences
between experts reflect the models rather than different data splits.

The meta-learner is trained on nested out-of-fold predictions. Inside each outer
training fold the experts are cross-fitted, so every meta training unit is
described by expert predictions produced by models that never saw that unit.
Training the meta-learner on in-sample expert predictions would let it learn to
exploit the experts' overfitting, which does not generalise.

No configuration choice is ever made on the outer evaluation part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .aggregation.claim_weighted import (
    aggregate,
    case_level_metrics,
    conservation_deviation,
    herfindahl,
    request_weights,
)
from .evaluation.metrics import item_metrics as compute_item_metrics
from .evaluation.metrics import macro_micro_metrics
from .experts.convex import weights_from_out_of_fold
from .experts.registry import C3_BASE_POOL, SPECS, build_expert, resolve_pool
from .features.claim_features import build_request_features
from .features.item_indicators import item_indicator_matrix
from .features.meta_features import (
    DEFAULT_DISAGREEMENT_SPEC,
    N_CLAIM_SIDE,
    N_DISAGREEMENT,
    build_disagreement_block,
    meta_feature_names,
)
from .meta_learning.item_balanced import (
    item_balanced_weights,
    objective_alignment_table,
    uniform_weights,
    weight_check,
)
from .meta_learning.meta_learner import fit_meta_head, predict_meta_head
from .preprocessing.pipeline import (
    case_matrices,
    case_totals,
    item_support_rate,
    to_item_long,
)
from .preprocessing.schema import ITEMS, TARGET_NAME

#: Minimum number of training units for an item to justify its own meta model.
#: Items below the threshold are served by the pooled model instead, matching the
#: pre-registered protocol. The threshold is a property of the protocol rather
#: than of the data, so it is not tuned.
MIN_META_ROWS: int = 100

#: Minimum number of training units for an expert to be fitted on an item.
MIN_EXPERT_ROWS: int = 8

#: Variant labels used in every result table.
VARIANT_PROPOSED: str = "Item-Balanced L1"
VARIANT_SQUARED: str = "Item-Balanced L2"
VARIANT_UNWEIGHTED: str = "Pooled L1, uniform weights"
VARIANT_AGG_PROPOSED: str = "Aggregated, Item-Balanced L1"
VARIANT_AGG_SQUARED: str = "Aggregated, Item-Balanced L2"
VARIANT_AGG_UNWEIGHTED: str = "Aggregated, uniform-weight L1"
VARIANT_DIRECT: str = "Direct case-level L1"

#: Unit-level variants scored with the item-averaged metrics.
UNIT_VARIANTS: Tuple[str, ...] = (
    VARIANT_PROPOSED,
    VARIANT_SQUARED,
    VARIANT_UNWEIGHTED,
)

#: Case-level variants scored after aggregation.
CASE_AGGREGATION_VARIANTS: Tuple[Tuple[str, str], ...] = (
    (VARIANT_AGG_PROPOSED, VARIANT_PROPOSED),
    (VARIANT_AGG_SQUARED, VARIANT_SQUARED),
    (VARIANT_AGG_UNWEIGHTED, VARIANT_UNWEIGHTED),
)


@dataclass
class PipelineConfig:
    """Everything that determines a run.

    The defaults reproduce the pre-registered protocol. The synthetic
    demonstration overrides only the number of folds and seeds, and reports those
    overrides in the run settings, so a demonstration run is never mistaken for a
    reproduction of the study.
    """

    n_outer_folds: int = 5
    n_inner_folds: int = 3
    seeds: Sequence[int] = (42, 2024, 3407)
    pool: Optional[Sequence[str]] = None
    allow_external_weights: bool = False
    min_meta_rows: int = MIN_META_ROWS
    min_expert_rows: int = MIN_EXPERT_ROWS
    max_cases: Optional[int] = None
    label: str = "pre-registered protocol"
    progress: bool = False

    def resolved_seeds(self) -> Tuple[int, ...]:
        """Seeds as a tuple of ints."""
        return tuple(int(s) for s in self.seeds)

    def describe(self) -> Dict[str, Any]:
        """Run description, recorded alongside the results."""
        return {
            "label": self.label,
            "outer_folds": int(self.n_outer_folds),
            "inner_folds": int(self.n_inner_folds),
            "seeds": list(self.resolved_seeds()),
            "min_meta_rows": int(self.min_meta_rows),
            "max_cases": self.max_cases,
        }


@dataclass
class PipelineResult:
    """Container for everything a run produces."""

    settings: Dict[str, Any] = field(default_factory=dict)
    unit_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    case_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    item_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    case_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weights_check: Dict[str, Any] = field(default_factory=dict)
    alignment: pd.DataFrame = field(default_factory=pd.DataFrame)
    conservation: Dict[str, Any] = field(default_factory=dict)
    concentration: Dict[str, float] = field(default_factory=dict)
    fold_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The last fitted Item-Balanced L1 head, together with the cross-fitted meta
    #: matrix it was trained on. Retained so the interpretation module can explain a
    #: real fitted model. It is the final fold of the final seed, which is a
    #: representative member of the ensemble of folds rather than a selected one:
    #: nothing is chosen on the basis of these values.
    meta_model: Any = None
    meta_training_matrix: Optional[np.ndarray] = None
    meta_training_truth: Optional[np.ndarray] = None
    meta_names: List[str] = field(default_factory=list)

    def unit_summary(self) -> pd.DataFrame:
        """Item-averaged metrics per unit-level variant."""
        return pd.DataFrame(
            [{"variant": name, **values} for name, values in self.metrics.items()]
        )

    def case_summary(self) -> pd.DataFrame:
        """Case-level metrics per variant."""
        return pd.DataFrame(
            [{"variant": name, **values} for name, values in self.case_metrics.items()]
        )


def assert_meta_rows_sufficient(n_rows: int, minimum: int, where: str) -> None:
    """Reject a fold whose cross-fitted meta training matrix is too small.

    The threshold exists because a meta head fitted on a handful of rows is not a
    fitted model, and reporting its metrics as though it were would be misleading.
    Raising is preferred to skipping silently: a run that is too small is a
    configuration error, and it should be visible in the traceback rather than
    inferred from a gap in the fold log.
    """
    if int(n_rows) < int(minimum):
        raise RuntimeError(
            f"{where}: the cross-fitted meta training matrix has {int(n_rows)} "
            f"complete rows, below the configured minimum of {int(minimum)}. "
            "Increase the number of cases, or lower min_meta_rows explicitly if a "
            "smaller fit is intended."
        )


def _select_cases(frame: pd.DataFrame, max_cases: Optional[int],
                  seed: int = 0) -> pd.DataFrame:
    """Optionally restrict the run to a deterministic subset of cases.

    Used only to bound the cost of a demonstration run. The subset is drawn with a
    fixed seed and the subset size is recorded in the run settings.
    """
    if max_cases is None or max_cases >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(len(frame), size=int(max_cases), replace=False))
    return frame.iloc[chosen].reset_index(drop=True)


def _fit_expert_pool(expert_names: Sequence[str],
                     base_pool: Sequence[str],
                     items: Sequence[str],
                     request_units: np.ndarray,
                     y_units: np.ndarray,
                     item_codes: np.ndarray,
                     request_eval_cases: np.ndarray,
                     unit_partition: np.ndarray,
                     n_partitions: int,
                     c3_l2: float,
                     min_expert_rows: int,
                     seed: int
                     ) -> Dict[str, Any]:
    """Fit the whole expert pool per item, for both the evaluation and meta parts.

    Two prediction sets are produced in one pass.

    ``evaluation predictions``
        Each expert fitted on the full training fold, predicting the evaluation
        cases. Shape ``(n_eval_cases, n_items)`` per expert.

    ``cross-fitted predictions``
        Each expert fitted on the training cases outside an inner partition,
        predicting the units inside it. Shape ``(n_train_units, n_items)`` per
        expert, undefined outside a unit's own item. These are the inputs the
        meta-learner is trained on, and they are genuine out-of-fold predictions:
        no unit is ever described by a model that saw its own case.

    Both sets are produced together because the convex combination expert composes
    the very same base experts. Producing them jointly lets that expert reuse the
    base fits instead of refitting six models per call, which is where almost all
    of the runtime would otherwise go. The weights it derives are unchanged, since
    they depend only on the out-of-fold prediction matrix and the target.

    An item with fewer than ``min_expert_rows`` training units is not fitted at all.
    Its entries are filled with the training-fold median of that item's support
    rate, which is the constant median expert, and the item is reported as a
    fallback. Nothing is imputed at the label level.
    """
    n_units = y_units.shape[0]
    n_eval = request_eval_cases.shape[0]
    evaluation: Dict[str, np.ndarray] = {
        name: np.full((n_eval, len(items)), np.nan, dtype=float)
        for name in expert_names
    }
    cross_fitted: Dict[str, np.ndarray] = {
        name: np.full((n_units, len(items)), np.nan, dtype=float)
        for name in expert_names
    }
    fallback_items: List[str] = []
    combination_weights: Dict[str, Dict[str, float]] = {}

    base_names = [name for name in base_pool if name in expert_names]
    other_names = [name for name in expert_names
                   if name != "c3" and name not in base_names]
    has_c3 = "c3" in expert_names
    global_median = float(np.median(y_units)) if y_units.size else 0.0

    def cross_fit_single(name: str, position: int, rows: np.ndarray,
                         target: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Out-of-fold predictions of one expert on one item's units."""
        column = np.full(rows.size, np.nan, dtype=float)
        for partition in range(n_partitions):
            inside = unit_partition[rows] == partition
            if not inside.any():
                continue
            outside = ~inside
            if int(outside.sum()) < min_expert_rows:
                column[inside] = (float(np.median(target[outside]))
                                  if outside.any() else float(np.median(target)))
                continue
            model = build_expert(name, seed=seed)
            model.fit(request_units[rows][outside], target[outside])
            column[inside] = np.clip(
                model.predict(request_units[rows][inside]), 0.0, 1.0
            )
        out[rows, position] = column
        return column

    for position, item in enumerate(items):
        rows = np.flatnonzero(item_codes == position)
        if rows.size < min_expert_rows:
            fallback = (float(np.median(y_units[rows])) if rows.size
                        else global_median)
            for name in expert_names:
                evaluation[name][:, position] = fallback
                cross_fitted[name][rows, position] = fallback
            fallback_items.append(str(item))
            continue
        target = y_units[rows]

        base_oof_columns: List[np.ndarray] = []
        base_eval_columns: List[np.ndarray] = []
        for name in base_names:
            model = build_expert(name, seed=seed)
            model.fit(request_units[rows], target)
            evaluation[name][:, position] = np.clip(
                model.predict(request_eval_cases), 0.0, 1.0
            )
            base_eval_columns.append(evaluation[name][:, position])
            base_oof_columns.append(
                cross_fit_single(name, position, rows, target, cross_fitted[name])
            )
        for name in other_names:
            model = build_expert(name, seed=seed)
            model.fit(request_units[rows], target)
            evaluation[name][:, position] = np.clip(
                model.predict(request_eval_cases), 0.0, 1.0
            )
            cross_fit_single(name, position, rows, target, cross_fitted[name])

        if not has_c3:
            continue
        oof_matrix = (np.column_stack(base_oof_columns) if base_oof_columns
                      else np.empty((rows.size, 0)))
        if oof_matrix.shape[1] < 2 or not np.isfinite(oof_matrix).all():
            fallback = float(np.median(target))
            evaluation["c3"][:, position] = fallback
            cross_fitted["c3"][rows, position] = fallback
            fallback_items.append(str(item))
            continue
        eval_matrix = np.column_stack(base_eval_columns)
        weights_full = weights_from_out_of_fold(oof_matrix, target, l2=c3_l2)
        evaluation["c3"][:, position] = np.clip(eval_matrix @ weights_full, 0.0, 1.0)
        combination_weights[str(item)] = {
            name: float(w) for name, w in zip(base_names, weights_full)
        }
        column = np.full(rows.size, np.nan, dtype=float)
        for partition in range(n_partitions):
            inside = unit_partition[rows] == partition
            if not inside.any():
                continue
            outside = ~inside
            if int(outside.sum()) < 2:
                column[inside] = float(np.median(target))
                continue
            weights_nested = weights_from_out_of_fold(
                oof_matrix[outside], target[outside], l2=c3_l2
            )
            column[inside] = np.clip(oof_matrix[inside] @ weights_nested, 0.0, 1.0)
        cross_fitted["c3"][rows, position] = column

    return {
        "evaluation": evaluation,
        "cross_fitted": cross_fitted,
        "fallback_items": fallback_items,
        "combination_weights": combination_weights,
        "base_pool": base_names,
    }


def _meta_matrix_for_units(expert_matrices: Dict[str, np.ndarray],
                           expert_names: Sequence[str],
                           item_indices: np.ndarray,
                           request_rows: np.ndarray) -> np.ndarray:
    """Assemble the meta representation for a set of case-item units.

    Expert predictions are read from the ``(unit, item)`` prediction matrices; the
    request-side and indicator blocks come from the unit's own case. Units whose
    expert predictions are incomplete are not filled here; the caller decides how
    to treat them.
    """
    n_units = item_indices.size
    block_a = np.empty((n_units, len(expert_names)), dtype=float)
    row_index = np.arange(n_units)
    for column, name in enumerate(expert_names):
        block_a[:, column] = expert_matrices[name][row_index, item_indices]
    prediction_map = {
        name: block_a[:, column] for column, name in enumerate(expert_names)
    }
    block_b = build_disagreement_block(prediction_map, expert_names,
                                       DEFAULT_DISAGREEMENT_SPEC)
    indicators = item_indicator_matrix([ITEMS[j] for j in item_indices], ITEMS)
    block_c = np.hstack([request_rows, indicators])
    if block_c.shape[1] != N_CLAIM_SIDE:
        raise AssertionError(
            f"claim-side block has {block_c.shape[1]} columns, expected {N_CLAIM_SIDE}"
        )
    return np.hstack([block_a, block_b, block_c])


def _run_single_fold(fold_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one outer fold: experts, meta heads, aggregation, evaluation."""
    (seed, fold, train_cases, test_cases) = (
        fold_context["seed"],
        fold_context["fold"],
        fold_context["train_cases"],
        fold_context["test_cases"],
    )
    expert_names = fold_context["expert_names"]
    items = fold_context["items"]
    config: PipelineConfig = fold_context["config"]
    request_features = fold_context["request_features"]
    long_frame = fold_context["long_frame"]
    unit_case_position = fold_context["unit_case_position"]
    matrices = fold_context["matrices"]
    case_ids = fold_context["case_ids"]
    totals = fold_context["totals"]

    n_cases = request_features.shape[0]
    train_mask = np.zeros(n_cases, dtype=bool)
    train_mask[train_cases] = True
    test_mask = np.zeros(n_cases, dtype=bool)
    test_mask[test_cases] = True

    train_rows = np.flatnonzero(train_mask[unit_case_position])
    test_rows = np.flatnonzero(test_mask[unit_case_position])
    train_position = unit_case_position[train_rows]
    test_position = unit_case_position[test_rows]
    train_position_local = np.searchsorted(train_cases, train_position)
    test_position_local = np.searchsorted(test_cases, test_position)

    train_request = request_features[train_cases]
    test_request = request_features[test_cases]
    # Feature rows are per case; expand them to per unit so the expert-fitting
    # routines can index a case-item unit directly by its own row.
    train_request_units = train_request[train_position_local]
    train_truth = long_frame["y"].to_numpy(dtype=float)[train_rows]
    train_item_codes = long_frame["item_index"].to_numpy()[train_rows]
    test_item_codes = long_frame["item_index"].to_numpy()[test_rows]

    # ---- inner partitions over *cases*, shared by every item and every expert.
    # They serve two purposes at once: they define the cross-fitting that produces
    # the meta training inputs, and they supply the out-of-fold base predictions
    # from which the convex combination expert derives its weights.
    inner_splitter = KFold(n_splits=int(config.n_inner_folds), shuffle=True,
                           random_state=int(seed))
    partition_of_local_case = np.empty(len(train_cases), dtype=int)
    for partition, (_, valid) in enumerate(inner_splitter.split(np.arange(len(train_cases)))):
        partition_of_local_case[valid] = partition
    unit_partition = partition_of_local_case[train_position_local]

    pool_fit = _fit_expert_pool(
        expert_names=expert_names,
        base_pool=C3_BASE_POOL,
        items=items,
        request_units=train_request_units,
        y_units=train_truth,
        item_codes=train_item_codes,
        request_eval_cases=test_request,
        unit_partition=unit_partition,
        n_partitions=int(config.n_inner_folds),
        c3_l2=float(SPECS["c3"].params.get("l2", 1e-3)),
        min_expert_rows=int(config.min_expert_rows),
        seed=int(seed),
    )
    expert_eval = pool_fit["evaluation"]
    expert_train = pool_fit["cross_fitted"]
    expert_eval_units = {
        name: expert_eval[name][test_position_local] for name in expert_names
    }

    meta_train = _meta_matrix_for_units(
        expert_train, expert_names, train_item_codes,
        train_request[train_position_local],
    )
    complete = np.isfinite(meta_train).all(axis=1)
    meta_train = meta_train[complete]
    train_truth_complete = train_truth[complete]
    train_items_complete = train_item_codes[complete]
    assert_meta_rows_sufficient(
        meta_train.shape[0], int(config.min_meta_rows),
        f"fold {fold} of seed {seed}",
    )

    meta_eval = _meta_matrix_for_units(
        expert_eval_units, expert_names, test_item_codes,
        test_request[test_position_local],
    )
    if not np.isfinite(meta_eval).all():
        raise RuntimeError(
            "the meta representation of the evaluation part contains undefined "
            "expert predictions"
        )

    balanced = item_balanced_weights(train_items_complete, n_items=len(items))
    uniform = uniform_weights(train_items_complete)
    checks = {
        "seed": int(seed),
        "fold": int(fold),
        **weight_check(train_items_complete, balanced, tol=1e-8),
    }

    heads = {
        VARIANT_PROPOSED: fit_meta_head(meta_train, train_truth_complete, balanced,
                                        int(seed), kind="l1"),
        VARIANT_SQUARED: fit_meta_head(meta_train, train_truth_complete, balanced,
                                       int(seed), kind="l2"),
        VARIANT_UNWEIGHTED: fit_meta_head(meta_train, train_truth_complete, uniform,
                                          int(seed), kind="l1"),
    }
    unit_predictions = {
        variant: predict_meta_head(model, meta_eval) for variant, model in heads.items()
    }

    # ---- direct case-level baseline: predict the total rate from the case's own
    # request-side features, without decomposing the case into items
    total_truth = totals.loc[train_mask, TARGET_NAME].to_numpy(dtype=float)
    usable = np.isfinite(total_truth)
    if int(usable.sum()) >= int(config.min_expert_rows):
        direct_model = fit_meta_head(
            train_request[usable], total_truth[usable],
            np.ones(int(usable.sum())), int(seed), kind="l1",
        )
        direct_case = predict_meta_head(direct_model, test_request)
    else:
        direct_case = np.full(len(test_cases), np.nan, dtype=float)

    # ---- case-level aggregation of the item predictions
    case_claim = matrices["claim"][test_cases]
    case_award = matrices["award"][test_cases]
    claim_total = case_claim.sum(axis=1)
    defined_units = case_claim > 0
    case_weights = request_weights(case_claim)
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_case = np.where(claim_total > 0, case_award.sum(axis=1) / claim_total,
                              np.nan)

    aggregated: Dict[str, np.ndarray] = {}
    for label, variant in CASE_AGGREGATION_VARIANTS:
        rate_matrix = np.full(case_claim.shape, np.nan, dtype=float)
        rate_matrix[test_position_local, test_item_codes] = unit_predictions[variant]
        aggregated[label] = aggregate(case_weights, rate_matrix, defined=defined_units)

    return {
        "seed": int(seed),
        "fold": int(fold),
        "weight_check": checks,
        "fallback_items": pool_fit["fallback_items"],
        "n_fallback_items": len(pool_fit["fallback_items"]),
        "combination_weights": pool_fit["combination_weights"],
        "n_train_cases": int(train_mask.sum()),
        "n_test_cases": int(test_mask.sum()),
        "n_meta_train_rows": int(meta_train.shape[0]),
        "n_units_dropped_incomplete": int((~complete).sum()),
        "meta_width": int(meta_train.shape[1]),
        "meta_model": heads[VARIANT_PROPOSED],
        "meta_training_matrix": meta_train,
        "meta_training_truth": train_truth_complete,
        "unit_rows": {
            "seed": int(seed),
            "fold": int(fold),
            "case_position": test_position,
            "item_index": test_item_codes,
            "y_true": long_frame["y"].to_numpy(dtype=float)[test_rows],
            **{variant: unit_predictions[variant] for variant in UNIT_VARIANTS},
        },
        "case_rows": {
            "seed": int(seed),
            "fold": int(fold),
            "case_ids": case_ids[test_cases],
            "y_true": truth_case,
            **aggregated,
            VARIANT_DIRECT: direct_case,
        },
    }


def run_pipeline(frame: pd.DataFrame,
                 config: Optional[PipelineConfig] = None,
                 progress_callback: Optional[Callable[[str], None]] = None
                 ) -> PipelineResult:
    """Execute the full protocol and return every intermediate result."""
    config = config or PipelineConfig()

    def announce(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    items = list(ITEMS)
    seed_for_subset = config.resolved_seeds()[0] if config.seeds else 0
    frame = _select_cases(frame, config.max_cases, seed=seed_for_subset)

    pool, skipped = resolve_pool(config.pool, config.allow_external_weights)
    if not pool:
        raise RuntimeError("no expert is available in this environment")
    expert_names = list(pool)

    names = meta_feature_names(expert_names, DEFAULT_DISAGREEMENT_SPEC)
    matrices = case_matrices(frame, items)
    request_features = build_request_features(frame, items).to_numpy(dtype=float)
    totals = case_totals(frame, items)
    case_ids = matrices["case_id"]
    n_cases = len(frame)

    long_frame = to_item_long(frame, items)
    lookup = {cid: i for i, cid in enumerate(case_ids)}
    unit_case_position = np.array([lookup[c] for c in long_frame["case_id"]])

    result = PipelineResult()
    result.settings = {
        "n_cases": int(n_cases),
        "n_units": int(len(long_frame)),
        "items": items,
        "experts": expert_names,
        "pool_size": len(expert_names),
        "experts_skipped": skipped,
        "meta_width": len(names),
        "meta_blocks": {
            "expert_predictions": len(expert_names),
            "disagreement": N_DISAGREEMENT,
            "claim_side": N_CLAIM_SIDE,
        },
        "frozen_pool_meta_width": 11 + N_DISAGREEMENT + N_CLAIM_SIDE,
        "run": config.describe(),
        "uses_external_weights": bool(config.allow_external_weights),
    }
    announce(
        f"cases={n_cases} units={len(long_frame)} experts={len(expert_names)} "
        f"meta_width={len(names)}"
    )

    fold_context_base = {
        "expert_names": expert_names,
        "items": items,
        "config": config,
        "request_features": request_features,
        "long_frame": long_frame,
        "unit_case_position": unit_case_position,
        "matrices": matrices,
        "case_ids": case_ids,
        "totals": totals,
    }

    unit_blocks: List[Dict[str, Any]] = []
    case_blocks: List[Dict[str, Any]] = []
    fold_records: List[Dict[str, Any]] = []
    weight_checks: List[Dict[str, Any]] = []

    for seed in config.resolved_seeds():
        splitter = KFold(n_splits=int(config.n_outer_folds), shuffle=True,
                         random_state=int(seed))
        for fold, (train_cases, test_cases) in enumerate(
            splitter.split(np.arange(n_cases))
        ):
            announce(f"seed {seed}, fold {fold + 1}/{config.n_outer_folds}")
            outcome = _run_single_fold(
                {**fold_context_base, "seed": int(seed), "fold": fold,
                 "train_cases": np.sort(train_cases), "test_cases": np.sort(test_cases)}
            )
            weight_checks.append(outcome["weight_check"])
            result.meta_model = outcome["meta_model"]
            result.meta_training_matrix = outcome["meta_training_matrix"]
            result.meta_training_truth = outcome["meta_training_truth"]
            result.meta_names = meta_feature_names(expert_names,
                                                   DEFAULT_DISAGREEMENT_SPEC)
            unit_blocks.append(outcome["unit_rows"])
            case_blocks.append(outcome["case_rows"])
            fold_records.append(
                {
                    "seed": outcome["seed"],
                    "fold": outcome["fold"],
                    "n_train_cases": outcome["n_train_cases"],
                    "n_test_cases": outcome["n_test_cases"],
                    "n_meta_train_rows": outcome["n_meta_train_rows"],
                    "meta_width": outcome["meta_width"],
                    "n_units_dropped_incomplete": outcome["n_units_dropped_incomplete"],
                    "n_items_constant_fallback": outcome["n_fallback_items"],
                    "fallback_items": ", ".join(outcome["fallback_items"]),
                }
            )

    result.fold_log = pd.DataFrame(fold_records)
    result.weights_check = _summarise_weight_checks(weight_checks, len(items))
    result.alignment = objective_alignment_table(long_frame["item"].to_numpy(),
                                                  n_items=len(items))
    result.conservation = conservation_deviation(matrices["claim"], matrices["award"])
    result.concentration = herfindahl(request_weights(matrices["claim"]))

    _score_unit_level(result, unit_blocks, case_ids, items)
    _score_case_level(result, case_blocks)
    return result


def _summarise_weight_checks(checks: Sequence[Dict[str, Any]],
                             n_items: int) -> Dict[str, Any]:
    """Aggregate the per-fold weight invariants into one record."""
    if not checks:
        return {}
    frame = pd.DataFrame(checks)
    return {
        "folds_checked": int(len(frame)),
        "all_balanced": bool(frame["balanced"].all()),
        "all_positive": bool(frame["all_positive"].all()),
        "min_item_total": float(frame["item_total_min"].min()),
        "max_item_total": float(frame["item_total_max"].max()),
        "min_item_share": float(frame["item_share_min"].min()),
        "max_item_share": float(frame["item_share_max"].max()),
        "ideal_item_share": 1.0 / n_items,
        "min_mean_weight": float(frame["sum_w_over_units"].min()),
        "max_mean_weight": float(frame["sum_w_over_units"].max()),
        # Reported on the share axis, where the ideal is exactly 1 / n_items. On the
        # rescaled axis the per-item total depends on the fold size and would not be
        # comparable with a fixed ideal.
        "max_deviation_from_ideal": float(
            np.max(np.abs(frame["item_share_min"] - 1.0 / n_items).max())
        ),
        "max_deviation_from_ideal_share": float(
            frame["max_deviation_from_share"].max()
        ),
    }


def _score_unit_level(result: PipelineResult, blocks: Sequence[Dict[str, Any]],
                      case_ids: np.ndarray, items: Sequence[str]) -> None:
    """Average predictions across folds and seeds, then compute item-level metrics."""
    if not blocks:
        return
    records: List[pd.DataFrame] = []
    for block in blocks:
        frame = pd.DataFrame(
            {
                "seed": block["seed"],
                "fold": block["fold"],
                "case_id": case_ids[block["case_position"]],
                "item": [items[j] for j in block["item_index"]],
                "y_true": block["y_true"],
                **{variant: block[variant] for variant in UNIT_VARIANTS},
            }
        )
        records.append(frame)
    stacked = pd.concat(records, ignore_index=True)
    averaged = (
        stacked.groupby(["case_id", "item"], as_index=False)
        .agg({"y_true": "first", **{v: "mean" for v in UNIT_VARIANTS}})
    )
    result.unit_frame = averaged
    item_rows: List[Dict[str, Any]] = []
    for variant in UNIT_VARIANTS:
        subset = averaged.dropna(subset=[variant])
        if subset.empty:
            continue
        result.metrics[variant] = macro_micro_metrics(
            subset["y_true"].to_numpy(float),
            subset[variant].to_numpy(float),
            subset["item"].to_numpy(),
        )
        for row in compute_item_metrics(
            subset["y_true"].to_numpy(float),
            subset[variant].to_numpy(float),
            subset["item"].to_numpy(),
        ):
            item_rows.append({"variant": variant, **row})
    result.item_frame = pd.DataFrame(item_rows)


def _score_case_level(result: PipelineResult,
                      blocks: Sequence[Dict[str, Any]]) -> None:
    """Average case-level predictions across folds and seeds, then score them."""
    if not blocks:
        return
    labels = [c for c in blocks[0] if c not in ("seed", "fold", "case_ids", "y_true")]
    records: List[pd.DataFrame] = []
    for block in blocks:
        frame = pd.DataFrame(
            {
                "seed": block["seed"],
                "fold": block["fold"],
                "case_id": block["case_ids"],
                "y_true": block["y_true"],
                **{label: block[label] for label in labels},
            }
        )
        records.append(frame)
    stacked = pd.concat(records, ignore_index=True)
    averaged = (
        stacked.groupby("case_id", as_index=False)
        .agg({"y_true": "first", **{label: "mean" for label in labels}})
    )
    result.case_frame = averaged
    for label in labels:
        subset = averaged.dropna(subset=[label])
        if subset.empty:
            continue
        result.case_metrics[label] = case_level_metrics(
            subset["y_true"].to_numpy(float), subset[label].to_numpy(float)
        )
