"""The 73-dimensional meta representation.

Composition
-----------
The meta-learner operates on case-item units. Each unit is described by three
concatenated blocks::

    block A : expert predictions        11 columns
    block B : expert disagreement       13 columns
    block C : claim / request side      49 columns
                                        -----------
                                        73 columns

with block C splitting further into::

    33 request features  (claim_0 ... claim_32)
  + 16 item indicators   (item_<item>, one-hot item membership)
  = 49 claim-side features

Block A has one column per expert, so its width equals the size of the expert pool.
The manuscript's frozen configuration uses an 11-expert pool, which is what makes
the total 73. The builder takes the pool size as an argument rather than
hard-coding 11, so a demonstration run with a reduced pool produces a correspondingly
narrower matrix while every block identity is preserved.

Block B: the 13 disagreement features
-------------------------------------
Disagreement between experts is informative in its own right: when the pool agrees,
the unit is easy; when the pool splits, the unit is hard. Thirteen statistics
summarise the spread, in this fixed order::

    3  signed differences     anchor - comparison, three comparisons
    5  dispersion             max, min, range, std, agg_median
    3  absolute differences   |comparison_a - comparison_b|, three pairs
    1  mean of the other experts, excluding the anchor
    1  pool size
    --
    13

All thirteen are functions of the expert predictions alone. They are not calibrated
uncertainty estimates: they measure how much the pool disagrees, not how likely the
prediction is to be wrong, and the manuscript states this explicitly.

Naming
------
Identifiers reproduce the study's frozen meta feature matrix, including the
positional request-side names, so that the feature labels in the manuscript's
figures correspond one to one with the columns produced here. The meaning of a
positional request-side identifier is given by
:func:`src.features.claim_features.request_feature_semantics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .claim_features import N_REQUEST_FEATURES
from .item_indicators import N_ITEM_INDICATORS

#: Width of the expert-prediction block under the manuscript's frozen pool.
N_EXPERT_PREDICTIONS: int = 11

#: Width of the disagreement block. Fixed by construction.
N_DISAGREEMENT: int = 13

#: Width of the claim-side block: request features plus item indicators.
N_CLAIM_SIDE: int = N_REQUEST_FEATURES + N_ITEM_INDICATORS  # 49

#: Total width of the meta representation under the frozen configuration.
META_DIMENSION: int = N_EXPERT_PREDICTIONS + N_DISAGREEMENT + N_CLAIM_SIDE  # 73

#: Alias kept for readability in documentation and tests.
CLAIM_SIDE_DIMENSION: int = N_CLAIM_SIDE

#: Identifier used for the median of the expert predictions. It is not ``pred_median``
#: because that identifier already denotes the ``median`` expert's own prediction.
MEDIAN_OF_PREDICTIONS: str = "agg_median"


@dataclass(frozen=True)
class DisagreementSpec:
    """Which comparisons and which dispersion statistics to compute, and in what order.

    The defaults reproduce the frozen configuration of the manuscript: the item-wise
    convex combination expert is the anchor, three experts are compared against it by
    signed difference, five dispersion statistics are added, and three pairs are
    compared by absolute difference.
    """

    anchor: str = "c3"
    signed: Tuple[str, ...] = ("lgb", "xgb", "median")
    dispersion: Tuple[str, ...] = ("max", "min", "range", "std", "median")
    absolute: Tuple[Tuple[str, str], ...] = (
        ("xgb", "lgb"),
        ("c3", "catboost"),
        ("c3", "ridge"),
    )
    mean_others: bool = True
    pool_size: bool = True

    def width(self) -> int:
        """Number of disagreement columns this specification produces."""
        return (
            len(self.signed)
            + len(self.dispersion)
            + len(self.absolute)
            + int(self.mean_others)
            + int(self.pool_size)
        )

    def referenced_experts(self) -> Tuple[str, ...]:
        """Every expert identifier the specification refers to."""
        names = {self.anchor}
        names.update(self.signed)
        for left, right in self.absolute:
            names.update((left, right))
        return tuple(sorted(names))


#: Default specification; the name list below is derived from it.
DEFAULT_DISAGREEMENT_SPEC = DisagreementSpec()


def _dispersion_name(statistic: str) -> str:
    """Column name of one dispersion statistic."""
    if statistic == "median":
        return MEDIAN_OF_PREDICTIONS
    return f"pred_{statistic}"


def disagreement_names(pool: Sequence[str],
                       spec: DisagreementSpec = DEFAULT_DISAGREEMENT_SPEC
                       ) -> List[str]:
    """Ordered names of the disagreement block for a given expert pool.

    Raises :class:`KeyError` when the specification refers to an expert that the
    pool does not contain, so a mis-specified pool fails at construction time
    rather than producing a silently truncated block.
    """
    present = {str(p) for p in pool}
    unknown = sorted(set(spec.referenced_experts()) - present)
    if unknown:
        raise KeyError(
            "disagreement specification references experts absent from the pool: "
            f"{unknown}; pool = {sorted(present)}"
        )
    names = [f"diff_{spec.anchor}_{c}" for c in spec.signed]
    names += [_dispersion_name(stat) for stat in spec.dispersion]
    names += [f"disagree_{a}_{b}" for a, b in spec.absolute]
    if spec.mean_others:
        names.append(f"pred_mean_others_{spec.anchor}")
    if spec.pool_size:
        names.append("n_experts")
    return names


#: Canonical disagreement names under the frozen 11-expert configuration.
DISAGREEMENT_NAMES: List[str] = disagreement_names(
    ("median", "ridge", "rf", "xgb", "lgb", "catboost", "c3",
     "extratrees", "histgb", "svr_rbf", "tabpfn")
)


def build_expert_block(predictions: Mapping[str, np.ndarray],
                       pool: Sequence[str]) -> np.ndarray:
    """Stack expert predictions into block A, shape ``(n_units, len(pool))``."""
    if not pool:
        raise ValueError("the expert pool is empty")
    columns = []
    for name in pool:
        if name not in predictions:
            raise KeyError(f"missing predictions for expert {name!r}")
        column = np.asarray(predictions[name], dtype=float).reshape(-1)
        columns.append(column)
    widths = {c.shape[0] for c in columns}
    if len(widths) != 1:
        raise ValueError(
            f"expert prediction vectors disagree in length: {sorted(widths)}"
        )
    return np.column_stack(columns)


def build_disagreement_block(predictions: Mapping[str, np.ndarray],
                             pool: Sequence[str],
                             spec: DisagreementSpec = DEFAULT_DISAGREEMENT_SPEC
                             ) -> np.ndarray:
    """Compute block B, shape ``(n_units, spec.width())``."""
    names = disagreement_names(pool, spec)
    stack = build_expert_block(predictions, pool)
    anchor = np.asarray(predictions[spec.anchor], dtype=float).reshape(-1)
    others = [p for p in pool if p != spec.anchor]
    columns: List[np.ndarray] = []

    for comparison in spec.signed:
        columns.append(
            anchor - np.asarray(predictions[comparison], dtype=float).reshape(-1)
        )
    for statistic in spec.dispersion:
        if statistic == "max":
            columns.append(stack.max(axis=1))
        elif statistic == "min":
            columns.append(stack.min(axis=1))
        elif statistic == "range":
            columns.append(stack.max(axis=1) - stack.min(axis=1))
        elif statistic == "std":
            columns.append(
                stack.std(axis=1, ddof=1) if stack.shape[1] > 1
                else np.zeros(stack.shape[0])
            )
        elif statistic == "median":
            columns.append(np.median(stack, axis=1))
        else:
            raise ValueError(f"unknown dispersion statistic: {statistic!r}")
    for left, right in spec.absolute:
        columns.append(
            np.abs(
                np.asarray(predictions[left], dtype=float).reshape(-1)
                - np.asarray(predictions[right], dtype=float).reshape(-1)
            )
        )
    if spec.mean_others:
        if not others:
            raise ValueError("mean-of-others requires at least two experts in the pool")
        columns.append(
            np.mean(
                [np.asarray(predictions[k], dtype=float).reshape(-1) for k in others],
                axis=0,
            )
        )
    if spec.pool_size:
        columns.append(np.full(stack.shape[0], float(len(pool))))

    block = np.column_stack(columns)
    if block.shape[1] != len(names):
        raise AssertionError(
            f"disagreement block width {block.shape[1]} contradicts its name list "
            f"({len(names)})"
        )
    return block


def build_claim_side_block(request_features: np.ndarray,
                           item_indicators: np.ndarray) -> np.ndarray:
    """Build block C, shape ``(n_units, 49)``.

    ``request_features`` contributes 33 columns and ``item_indicators`` 16, so the
    concatenation is exactly the 49-dimensional claim-side block.
    """
    request_features = np.asarray(request_features, dtype=float)
    item_indicators = np.asarray(item_indicators, dtype=float)
    if request_features.ndim != 2 or item_indicators.ndim != 2:
        raise ValueError("both claim-side inputs must be two-dimensional")
    if request_features.shape[0] != item_indicators.shape[0]:
        raise ValueError("request features and item indicators disagree on row count")
    if request_features.shape[1] != N_REQUEST_FEATURES:
        raise ValueError(
            f"request block must have {N_REQUEST_FEATURES} columns, "
            f"found {request_features.shape[1]}"
        )
    if item_indicators.shape[1] != N_ITEM_INDICATORS:
        raise ValueError(
            f"indicator block must have {N_ITEM_INDICATORS} columns, "
            f"found {item_indicators.shape[1]}"
        )
    block = np.hstack([request_features, item_indicators])
    if block.shape[1] != N_CLAIM_SIDE:
        raise AssertionError(f"claim-side block width {block.shape[1]} != {N_CLAIM_SIDE}")
    return block


def build_meta_matrix(predictions: Mapping[str, np.ndarray],
                      request_features: np.ndarray,
                      item_indicators: np.ndarray,
                      pool: Sequence[str],
                      spec: DisagreementSpec = DEFAULT_DISAGREEMENT_SPEC
                      ) -> np.ndarray:
    """Assemble the full meta matrix ``[A | B | C]``.

    Returns shape ``(n_units, len(pool) + 13 + 49)``. With the frozen 11-expert pool
    this is ``(n_units, 73)``.
    """
    block_a = build_expert_block(predictions, pool)
    block_b = build_disagreement_block(predictions, pool, spec)
    block_c = build_claim_side_block(request_features, item_indicators)
    if not (block_a.shape[0] == block_b.shape[0] == block_c.shape[0]):
        raise ValueError("meta blocks disagree on row count")
    return np.hstack([block_a, block_b, block_c])


def meta_feature_names(pool: Sequence[str],
                       spec: DisagreementSpec = DEFAULT_DISAGREEMENT_SPEC
                       ) -> List[str]:
    """Ordered names of the full meta representation for a given pool."""
    from .claim_features import request_feature_names
    from .item_indicators import item_indicator_names

    names = [f"pred_{p}" for p in pool]
    names += disagreement_names(pool, spec)
    names += request_feature_names()
    names += item_indicator_names()
    return names


def meta_dimension(pool_size: int) -> int:
    """Width of the meta representation for an arbitrary pool size."""
    return int(pool_size) + N_DISAGREEMENT + N_CLAIM_SIDE


def meta_block_boundaries(pool_size: int) -> Dict[str, Tuple[int, int]]:
    """Half-open index ranges of the three blocks, for attribution and auditing."""
    start_b = int(pool_size)
    start_c = start_b + N_DISAGREEMENT
    end = start_c + N_CLAIM_SIDE
    return {
        "expert_predictions": (0, start_b),
        "disagreement": (start_b, start_c),
        "claim_side": (start_c, end),
    }
