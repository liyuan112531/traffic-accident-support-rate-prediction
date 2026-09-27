"""Public reference implementation of the judicial support-rate prediction framework.

The package implements the methodology described in the accompanying manuscript:
a 16-item decomposition of civil compensation claims, an 11-expert ensemble, a
73-dimensional meta representation, Item-Balanced L1 meta-learning, and
claim-structure-driven hierarchical aggregation.

Nothing in this package reads, downloads, or reconstructs the restricted
case-level dataset. All paths are resolved relative to the repository root.
"""

__all__ = [
    "preprocessing",
    "features",
    "experts",
    "meta_learning",
    "aggregation",
    "evaluation",
    "statistics",
    "interpretation",
    "utils",
]

__version__ = "1.0.0"
