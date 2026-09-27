"""Data preprocessing: schema contract and item-level frame construction."""

from .schema import (
    CLAIM_SUFFIX,
    AWARD_SUFFIX,
    ITEMS,
    N_ITEMS,
    TARGET_NAME,
    item_award_column,
    item_claim_column,
)
from .pipeline import (
    case_matrices,
    case_totals,
    load_cases,
    to_item_long,
    validate_schema,
)

__all__ = [
    "CLAIM_SUFFIX",
    "AWARD_SUFFIX",
    "ITEMS",
    "N_ITEMS",
    "TARGET_NAME",
    "item_award_column",
    "item_claim_column",
    "case_matrices",
    "case_totals",
    "load_cases",
    "to_item_long",
    "validate_schema",
]
