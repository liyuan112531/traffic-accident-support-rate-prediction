"""Resampling-based inference: paired bootstrap and paired permutation."""

from .resampling import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_PERMUTATIONS,
    case_stratified_bootstrap,
    paired_permutation_test,
)
from .multiplicity import benjamini_hochberg, holm_adjust

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_PERMUTATIONS",
    "case_stratified_bootstrap",
    "paired_permutation_test",
    "benjamini_hochberg",
    "holm_adjust",
]
