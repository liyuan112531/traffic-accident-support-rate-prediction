"""Post-hoc interpretability: TreeSHAP attribution over the meta representation."""

from .shap_analysis import (
    BLOCK_A_EXPERT,
    BLOCK_B_DISAGREEMENT,
    BLOCK_C_CLAIM,
    block_contribution,
    disagreement_quartile_table,
    expert_contribution,
    feature_inventory,
    tree_shap,
)

__all__ = [
    "BLOCK_A_EXPERT",
    "BLOCK_B_DISAGREEMENT",
    "BLOCK_C_CLAIM",
    "block_contribution",
    "disagreement_quartile_table",
    "expert_contribution",
    "feature_inventory",
    "tree_shap",
]
