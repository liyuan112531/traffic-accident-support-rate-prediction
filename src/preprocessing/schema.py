"""Declarative data contract for the 16 compensation modeling items.

The contract is deliberately declarative: every downstream module imports the
item list and the column-naming rules from here instead of re-declaring them,
so the number of items and their identifiers cannot drift between modules.

Target definition
-----------------
For case ``i`` and item ``j``::

    r_ij = award_ij / claim_ij        defined only when claim_ij > 0

Two distinct cases are kept apart, exactly as in the manuscript:

``structural absence``
    ``claim_ij == 0``. The item was not claimed, so the support rate does not
    exist. It is encoded as ``NaN`` and never as ``0.0``; the case-item unit is
    excluded from training and evaluation rather than imputed.

``behavioural zero``
    ``claim_ij > 0`` and ``award_ij == 0``. The claim was asserted and fully
    rejected. This is an informative observation with value ``0.0``.

Collapsing these two situations into a single zero would silently inject
16 x n_cases fabricated labels, so the distinction is enforced in code.
"""

from __future__ import annotations

from typing import Sequence, Tuple

#: The 16 compensation modeling items, in the canonical order used throughout
#: the manuscript. Order matters: matrices and indicator blocks rely on it.
ITEMS: Tuple[str, ...] = (
    "medical",
    "future_medical",
    "hospital_meal",
    "nutrition",
    "nursing",
    "lost_income",
    "transport",
    "accommodation",
    "disability_comp",
    "assistive_device",
    "death_comp",
    "funeral",
    "mental_solatium",
    "property_loss",
    "appraisal",
    "other",
)

#: Number of compensation modeling items. Asserted rather than assumed.
N_ITEMS: int = 16

#: Column-name suffixes for the paired claim and award amounts.
CLAIM_SUFFIX: str = "_claim"
AWARD_SUFFIX: str = "_award"

#: Name of the case-level total support-rate target.
TARGET_NAME: str = "total_support_rate"

#: Columns that identify a case and describe its procedural context. They carry
#: no outcome information and are therefore usable as grouping keys.
CASE_KEY: str = "case_id"
YEAR_COLUMN: str = "doc_year"
REGION_COLUMN: str = "doc_region"
PROCEDURE_COLUMN: str = "doc_proc_level"

#: Columns that must never reach a feature matrix. They are post-outcome
#: attributes (decided by, or observed only after, the judgment) and using them
#: would leak the label. The list is enforced by
#: :func:`src.features.claim_features.assert_no_leakage`.
FORBIDDEN_FEATURE_TOKENS: Tuple[str, ...] = (
    "award",
    "support_rate",
    "liability_ratio",
    "disability_level",
    "n_nonzero_items",
    "text_len",
)

#: Tolerance below which a monetary amount is treated as "not claimed".
AMOUNT_ZERO_TOL: float = 1e-9

#: Tolerance used when comparing support rates for equality.
RATE_TOL: float = 1e-9


def item_claim_column(item: str) -> str:
    """Claim-amount column name for ``item``."""
    return f"{item}{CLAIM_SUFFIX}"


def item_award_column(item: str) -> str:
    """Award-amount column name for ``item``."""
    return f"{item}{AWARD_SUFFIX}"


def claim_columns(items: Sequence[str] = ITEMS) -> Tuple[str, ...]:
    """All 16 claim-amount column names, in canonical order."""
    return tuple(item_claim_column(i) for i in items)


def award_columns(items: Sequence[str] = ITEMS) -> Tuple[str, ...]:
    """All 16 award-amount column names, in canonical order."""
    return tuple(item_award_column(i) for i in items)


def assert_contract(items: Sequence[str] = ITEMS) -> None:
    """Fail loudly if the item contract has been edited inconsistently."""
    if len(items) != N_ITEMS:
        raise AssertionError(
            f"the item contract declares {len(items)} items, expected {N_ITEMS}"
        )
    if len(set(items)) != len(items):
        raise AssertionError(f"duplicate item identifiers in the contract: {items}")
