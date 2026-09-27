"""Request-side feature block.

This module builds the 33-dimensional claim-side feature set used by every expert
model and by the claim-side block of the meta representation::

    claim_0 ... claim_15    raw claimed amount, one per item, canonical item order
    claim_16                claim_total, the sum of the sixteen claimed amounts
    claim_17 ... claim_32   log1p of claim_0 ... claim_15
    --------------------------------------------------------------------------
    = 33 request features

Only pre-judgment information enters this block. The claimed amounts are known
when the lawsuit is filed, so no award-side or outcome-derived quantity can reach a
model through these columns. The guard :func:`assert_no_leakage` enforces that
boundary mechanically rather than by convention.

Identifier convention
---------------------
The columns are named positionally, ``claim_0`` through ``claim_32``, matching the
identifiers used by the study's meta feature matrix and therefore by the feature
labels in the manuscript's figures. Positional names are terse, so
:func:`request_feature_semantics` returns the meaning of each index and is the
authority used to build the published data dictionary.

The log transform is applied because compensation amounts are strongly
right-skewed. It is monotone, so it adds no information beyond the raw amount; it
changes the geometry the tree splits operate on.

Claim totals are not stored anywhere in this package. They are derived from the
sixteen amounts, so a total can never disagree with its components.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..preprocessing.schema import (
    ITEMS,
    N_ITEMS,
    claim_columns,
    item_claim_column,
)

#: Column-name fragments that mark a post-outcome or label-derived quantity.
LEAKAGE_TOKENS: Sequence[str] = (
    "award",
    "support_rate",
    "n_nonzero_items",
    "liability_ratio",
    "disability_level",
    "text_len",
    "p_zero",
    "p_partial",
    "p_full",
    "state_entropy",
    "cluster",
    "mds",
)

#: Position of the claim total inside the request-side block.
CLAIM_TOTAL_INDEX: int = N_ITEMS  # 16

#: Total width of the request-side block: 16 raw + 1 total + 16 log1p = 33.
N_REQUEST_FEATURES: int = 2 * N_ITEMS + 1  # 33


def assert_no_leakage(columns: Sequence[str], where: str) -> None:
    """Raise when any column name marks a post-outcome or label-derived field."""
    hits: List[str] = []
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in LEAKAGE_TOKENS):
            hits.append(str(column))
    if hits:
        raise ValueError(f"{where}: forbidden post-outcome field(s): {hits}")


def request_feature_names(items: Sequence[str] = ITEMS) -> List[str]:
    """Ordered names of the 33 request-side features."""
    if len(items) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} items, received {len(items)}")
    return [f"claim_{i}" for i in range(N_REQUEST_FEATURES)]


def request_feature_semantics(
        items: Sequence[str] = ITEMS) -> List[Tuple[str, str, str]]:
    """``(name, kind, meaning)`` for every request-side feature.

    ``kind`` is one of ``raw``, ``total`` or ``log1p``. This is the single source
    of truth for the meaning of a positional identifier, used by the data
    dictionary generator and by the documentation.
    """
    if len(items) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} items, received {len(items)}")
    rows: List[Tuple[str, str, str]] = []
    for j, item in enumerate(items):
        rows.append((f"claim_{j}", "raw", f"claimed amount, {item}"))
    rows.append((
        f"claim_{CLAIM_TOTAL_INDEX}", "total",
        "claim_total, the sum of the sixteen claimed amounts",
    ))
    for j, item in enumerate(items):
        rows.append((
            f"claim_{CLAIM_TOTAL_INDEX + 1 + j}", "log1p",
            f"log1p-transformed claimed amount, {item}",
        ))
    return rows


#: Canonical names of the request-side block.
REQUEST_FEATURE_NAMES: List[str] = request_feature_names(ITEMS)


def request_feature_dictionary() -> Dict[str, str]:
    """Mapping from request-side feature name to its human-readable meaning."""
    return {name: meaning for name, _, meaning in request_feature_semantics(ITEMS)}


def build_request_features(frame: pd.DataFrame,
                           items: Sequence[str] = ITEMS) -> pd.DataFrame:
    """Build the request-side feature matrix, indexed like ``frame``.

    Raises if a claim column is missing or carries a non-finite or negative value,
    because a malformed request side would silently corrupt every downstream
    expert.
    """
    if len(items) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} items, received {len(items)}")
    needed = list(claim_columns(items))
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f"request-side columns missing from the frame: {missing}")
    raw = frame[needed].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("claim amounts contain NaN or infinite values")
    if (raw < 0).any():
        raise ValueError("claim amounts contain negative values")
    total = raw.sum(axis=1, keepdims=True)
    values = np.column_stack([raw, total, np.log1p(raw)])
    out = pd.DataFrame(values, columns=request_feature_names(items),
                       index=frame.index)
    assert_no_leakage(list(out.columns), "request-side feature block")
    return out


def request_feature_block_dimension() -> Dict[str, int]:
    """Decomposition of the request-side block, for documentation and tests."""
    return {
        "raw_claims": N_ITEMS,
        "claim_total": 1,
        "log1p_claims": N_ITEMS,
        "total": N_REQUEST_FEATURES,
    }


def descriptive_alias(name: str, items: Sequence[str] = ITEMS) -> str:
    """Descriptive column name for a positional request-side identifier.

    Provided so that tables and documentation can label a feature readably without
    renaming the underlying column, which would break alignment with the study's
    frozen feature matrix.

    The alias is derived from the column's position rather than from parsing its
    meaning, so it cannot drift if a description is reworded::

        claim_0  ... claim_15   ->  <item>_claim
        claim_16                ->  claim_total
        claim_17 ... claim_32   ->  log1p_<item>_claim
    """
    names = request_feature_names(items)
    if name not in names:
        raise KeyError(f"unknown request-side feature: {name!r}")
    position = names.index(name)
    if position < N_ITEMS:
        return item_claim_column(items[position])
    if position == CLAIM_TOTAL_INDEX:
        return "claim_total"
    offset = position - CLAIM_TOTAL_INDEX - 1
    if not 0 <= offset < len(items):  # pragma: no cover - guarded by the name lookup
        raise ValueError(f"request-side position {position} is outside the block")
    return f"log1p_{item_claim_column(items[offset])}"
