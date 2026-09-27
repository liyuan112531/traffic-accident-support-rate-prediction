"""Construction of case-level and item-level frames from a claim/award table.

The public release operates on two kinds of table:

``case frame``
    One row per case, 16 claim columns and 16 award columns, plus non-outcome
    case attributes. Totals and the case-level support rate are derived here and
    never stored, so the target cannot drift from the amounts.

``item long frame``
    One row per (case, item) unit, restricted to units with a defined support
    rate (``claim > 0``). This is the training and evaluation universe.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .schema import (
    AMOUNT_ZERO_TOL,
    CASE_KEY,
    FORBIDDEN_FEATURE_TOKENS,
    ITEMS,
    N_ITEMS,
    TARGET_NAME,
    YEAR_COLUMN,
    assert_contract,
    award_columns,
    claim_columns,
)


def load_cases(path) -> pd.DataFrame:
    """Read a case-level CSV.

    ``path`` may be a ``str`` or :class:`pathlib.Path`. The file is expected to
    hold one row per case with paired ``<item>_claim`` / ``<item>_award``
    columns. The synthetic demonstration file in ``data/synthetic/`` follows
    this layout.
    """
    frame = pd.read_csv(path)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Raise a descriptive error when the frame breaks the data contract."""
    assert_contract(ITEMS)
    missing = [
        column
        for column in (*claim_columns(ITEMS), *award_columns(ITEMS))
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            "case table is missing %d required column(s): %s"
            % (len(missing), ", ".join(missing))
        )
    if CASE_KEY in frame.columns and frame[CASE_KEY].duplicated().any():
        duplicates = int(frame[CASE_KEY].duplicated().sum())
        raise ValueError(f"case identifier is not unique ({duplicates} duplicates)")
    amounts = list(claim_columns(ITEMS)) + list(award_columns(ITEMS))
    values = frame[amounts].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("claim/award columns contain NaN or infinite values")
    if (values < -AMOUNT_ZERO_TOL).any():
        raise ValueError("claim/award columns contain negative amounts")


def _case_ids(frame: pd.DataFrame) -> np.ndarray:
    """Case identifiers, synthesised from the row position when absent."""
    if CASE_KEY in frame.columns:
        return frame[CASE_KEY].to_numpy()
    return np.arange(len(frame))


def case_matrices(frame: pd.DataFrame,
                  items: Sequence[str] = ITEMS) -> Dict[str, np.ndarray]:
    """Return the claim and award matrices plus their derived row sums.

    Shapes are ``(n_cases, n_items)`` with columns in canonical item order.
    """
    claim = frame[list(claim_columns(items))].to_numpy(dtype=float)
    award = frame[list(award_columns(items))].to_numpy(dtype=float)
    return {
        "claim": claim,
        "award": award,
        "claim_total": claim.sum(axis=1),
        "award_total": award.sum(axis=1),
        "case_id": _case_ids(frame),
    }


def item_support_rate(claim: np.ndarray, award: np.ndarray) -> np.ndarray:
    """Element-wise support rate with structural absence encoded as ``NaN``.

    ``claim == 0``  -> ``NaN`` (the item was not claimed; no rate exists)
    ``award == 0``  -> ``0.0`` when ``claim > 0`` (a claim that was rejected)
    """
    claim = np.asarray(claim, dtype=float)
    award = np.asarray(award, dtype=float)
    out = np.full(claim.shape, np.nan, dtype=float)
    defined = claim > AMOUNT_ZERO_TOL
    out[defined] = award[defined] / claim[defined]
    return out


def case_totals(frame: pd.DataFrame, items: Sequence[str] = ITEMS) -> pd.DataFrame:
    """Case-level claim total, award total and total support rate.

    The total support rate is the claim-amount-weighted mean of the item-level
    rates. It is ``NaN`` for a case with no claim at all, which is the
    case-level analogue of structural absence.
    """
    matrices = case_matrices(frame, items)
    claim_total = matrices["claim_total"]
    award_total = matrices["award_total"]
    total_rate = np.full(claim_total.shape, np.nan, dtype=float)
    defined = claim_total > AMOUNT_ZERO_TOL
    total_rate[defined] = award_total[defined] / claim_total[defined]
    return pd.DataFrame(
        {
            CASE_KEY: matrices["case_id"],
            "claim_total": claim_total,
            "award_total": award_total,
            TARGET_NAME: total_rate,
        }
    )


def to_item_long(frame: pd.DataFrame,
                 items: Sequence[str] = ITEMS) -> pd.DataFrame:
    """Long (case, item) frame restricted to units with a defined rate.

    Columns: ``case_id``, ``item``, ``item_index``, ``claim``, ``award``, ``y``.
    Units with ``claim == 0`` are dropped without imputation and are reported by
    :func:`structural_absence_report`.
    """
    if len(items) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} items, received {len(items)}")
    matrices = case_matrices(frame, items)
    case_ids = matrices["case_id"]
    rate = item_support_rate(matrices["claim"], matrices["award"])
    rows: List[Dict[str, object]] = []
    for j, item in enumerate(items):
        defined = np.isfinite(rate[:, j])
        idx = np.flatnonzero(defined)
        for i in idx:
            rows.append(
                {
                    CASE_KEY: case_ids[i],
                    "item": item,
                    "item_index": j,
                    "claim": float(matrices["claim"][i, j]),
                    "award": float(matrices["award"][i, j]),
                    "y": float(rate[i, j]),
                }
            )
    return pd.DataFrame(rows)


def structural_absence_report(frame: pd.DataFrame,
                              items: Sequence[str] = ITEMS) -> pd.DataFrame:
    """Per-item count of defined and structurally absent case-item units.

    Exposed so the retention of the item universe can be inspected rather than
    assumed. No imputation is performed anywhere in this package.
    """
    matrices = case_matrices(frame, items)
    claim = matrices["claim"]
    defined = claim > AMOUNT_ZERO_TOL
    behavioural_zero = defined & (matrices["award"] <= AMOUNT_ZERO_TOL)
    n_cases = claim.shape[0]
    return pd.DataFrame(
        {
            "item": list(items),
            "item_index": np.arange(len(items)),
            "n_cases": n_cases,
            "n_defined": defined.sum(axis=0),
            "n_structurally_absent": (~defined).sum(axis=0),
            "n_behavioural_zero": behavioural_zero.sum(axis=0),
            "defined_fraction": defined.mean(axis=0),
        }
    )


def assert_no_forbidden_columns(columns: Sequence[str], where: str) -> None:
    """Reject any column whose name marks it as a post-outcome attribute."""
    hits = [
        str(c)
        for c in columns
        if any(token in str(c).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if hits:
        raise ValueError(f"{where}: forbidden post-outcome field(s) present: {hits}")


def year_vector(frame: pd.DataFrame) -> np.ndarray:
    """Document year as a float array, ``NaN`` where unavailable.

    Used only to define the temporal extrapolation split; it is never a feature.
    """
    if YEAR_COLUMN not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[YEAR_COLUMN], errors="coerce").to_numpy(dtype=float)


def temporal_split(frame: pd.DataFrame, train_year: int = 2022):
    """Boolean masks ``(train_mask, test_mask)`` for the extrapolation protocol.

    Training uses cases from ``train_year``; evaluation uses strictly later
    cases. The split is positional and never consults the label, and an
    overlapping index is treated as a hard error.
    """
    year = year_vector(frame)
    known = np.isfinite(year)
    train = known & (year == float(train_year))
    test = known & (year > float(train_year))
    if int((train & test).sum()) != 0:
        raise ValueError("temporal split produced overlapping train/test cases")
    return train, test
