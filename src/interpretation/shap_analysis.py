"""Attribution of meta-learner behaviour over the 73-dimensional input.

Scope
-----
Everything produced here describes the *behaviour of the fitted model*. It says
which inputs the model relies on, not which quantities cause the outcome. The
observed support rate is a legal decision recorded in a judgment, not a
randomised treatment, so a large attribution is not a causal effect and no
counterfactual is available against which one could be estimated.

In particular the disagreement block measures how much the expert pool
disagrees. It is a prediction-difficulty signal. It is not a calibrated
uncertainty estimate: nothing here has been calibrated against realised coverage,
so a large value means the pool split, not that a confidence interval is valid.

Method
------
TreeSHAP is used, which for a tree ensemble yields additive attributions that
sum to the difference between the prediction and the model's expected value. The
feature blocks follow the construction in :mod:`src.features.meta_features`, so
attributions can be aggregated to block level without any assumption about which
columns belong where.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Block identifiers used throughout the attribution tables.
BLOCK_A_EXPERT: str = "A_expert_prediction"
BLOCK_B_DISAGREEMENT: str = "B_expert_disagreement"
BLOCK_C_CLAIM: str = "C_claim_request"

#: Sub-blocks inside block C.
SUB_CLAIM_FEATURES: str = "C1_request_features"
SUB_ITEM_INDICATOR: str = "C2_item_indicator"


def feature_inventory(names: Sequence[str], n_pool: int,
                      n_disagreement: int = 13,
                      n_claim_side: int = 49,
                      n_indicator: int = 16) -> pd.DataFrame:
    """Assign every meta feature to its block and sub-block.

    The boundaries are supplied explicitly rather than inferred from the names, so
    a mismatch between the name list and the declared block widths raises instead
    of producing a plausible but wrong assignment.
    """
    names = list(names)
    expected = n_pool + n_disagreement + n_claim_side
    if len(names) != expected:
        raise ValueError(
            f"feature name list has {len(names)} entries but the block widths "
            f"declare {n_pool} + {n_disagreement} + {n_claim_side} = {expected}"
        )
    claim_feature_end = n_pool + n_disagreement + (n_claim_side - n_indicator)
    indicator_end = n_pool + n_disagreement + n_claim_side
    rows: List[Dict[str, Any]] = []
    for position, name in enumerate(names):
        if position < n_pool:
            block, sub = BLOCK_A_EXPERT, ""
        elif position < n_pool + n_disagreement:
            block, sub = BLOCK_B_DISAGREEMENT, ""
        elif position < claim_feature_end:
            block, sub = BLOCK_C_CLAIM, SUB_CLAIM_FEATURES
        elif position < indicator_end:
            block, sub = BLOCK_C_CLAIM, SUB_ITEM_INDICATOR
        else:  # pragma: no cover - guarded by the length check above
            raise AssertionError("feature position outside the declared blocks")
        rows.append(
            {"index": position, "feature": name, "block": block, "sub_block": sub}
        )
    inventory = pd.DataFrame(rows)
    inventory["block_label"] = inventory["block"].map(
        {
            BLOCK_A_EXPERT: "Expert predictions",
            BLOCK_B_DISAGREEMENT: "Expert disagreement",
            BLOCK_C_CLAIM: "Claim / request side",
        }
    )
    return inventory


def tree_shap(model, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """TreeSHAP values of a fitted tree ensemble, shape ``(n_units, n_features)``.

    Returns the attribution matrix and the model's expected value.
    """
    import shap

    x = np.asarray(x, dtype=float)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    expected = np.asarray(explainer.expected_value).ravel()
    if expected.size != 1:
        raise ValueError(
            "tree_shap expects a scalar expected value; received "
            f"{expected.size} values, which suggests a multi-output model"
        )
    return values, float(expected[0])


def global_attribution(shap_values: np.ndarray,
                       inventory: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute attribution per feature, ranked, with shares."""
    values = np.asarray(shap_values, dtype=float)
    if values.shape[1] != len(inventory):
        raise ValueError(
            f"attribution matrix has {values.shape[1]} columns but the inventory "
            f"lists {len(inventory)} features"
        )
    table = inventory.copy()
    table["mean_abs_shap"] = np.mean(np.abs(values), axis=0)
    table["mean_shap"] = np.mean(values, axis=0)
    table["median_abs_shap"] = np.median(np.abs(values), axis=0)
    table["n_units"] = int(values.shape[0])
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    total = float(table["mean_abs_shap"].sum())
    table["share"] = table["mean_abs_shap"] / total if total > 0 else np.nan
    return table


def block_contribution(global_table: pd.DataFrame) -> pd.DataFrame:
    """Share of total attribution carried by each block.

    ``contribution`` is the sum of the block's mean absolute attributions divided
    by the total, so the block contributions sum to one.
    """
    grouped = (
        global_table.groupby(["block", "block_label"], as_index=False)
        .agg(
            n_features=("feature", "size"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap_per_feature=("mean_abs_shap", "mean"),
        )
    )
    total = float(grouped["total_mean_abs_shap"].sum())
    grouped["contribution"] = grouped["total_mean_abs_shap"] / total if total > 0 else np.nan
    grouped["contribution_pct"] = 100.0 * grouped["contribution"]
    return grouped.sort_values("contribution", ascending=False).reset_index(drop=True)


def expert_contribution(global_table: pd.DataFrame,
                        prefix: str = "pred_") -> pd.DataFrame:
    """Per-expert shares within the expert-prediction block.

    Also reports the effective number of contributing experts,
    ``1 / sum(share^2)``, which is directly comparable with the pool size: a value
    close to the pool size means the load is spread, a value close to one means a
    single expert dominates.
    """
    block = global_table[global_table["block"] == BLOCK_A_EXPERT].copy()
    if block.empty:
        return pd.DataFrame(
            columns=["expert", "mean_abs_shap", "share", "expert_rank"]
        )
    block["expert"] = block["feature"].str.replace(prefix, "", regex=False)
    total = float(block["mean_abs_shap"].sum())
    block["share"] = block["mean_abs_shap"] / total if total > 0 else np.nan
    block = block.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    block["expert_rank"] = np.arange(1, len(block) + 1)
    block.attrs["n_experts"] = len(block)
    block.attrs["dominance_ratio"] = (
        float(block["mean_abs_shap"].iloc[0] / total) if total > 0 else np.nan
    )
    block.attrs["effective_n_experts"] = (
        float(1.0 / np.sum(block["share"] ** 2)) if total > 0 else np.nan
    )
    return block


def disagreement_quartile_table(disagreement: np.ndarray, truth: np.ndarray,
                                prediction: np.ndarray,
                                n_quantiles: int = 4) -> pd.DataFrame:
    """Error by quartile of expert disagreement.

    If disagreement carries information about difficulty, error should increase
    across the quartiles. The table reports the quartile boundaries so the pattern
    can be checked against the actual split rather than asserted.
    """
    disagreement = np.asarray(disagreement, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if not (disagreement.size == truth.size == prediction.size):
        raise ValueError("disagreement, truth and prediction must have equal length")
    edges = np.quantile(disagreement, np.linspace(0.0, 1.0, n_quantiles + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bucket = np.digitize(disagreement, edges[1:-1], right=False)
    residual = truth - prediction
    rows: List[Dict[str, Any]] = []
    for level in range(n_quantiles):
        mask = bucket == level
        if not mask.any():
            continue
        rows.append(
            {
                "quartile": level + 1,
                "lower_bound": float(edges[level]),
                "upper_bound": float(edges[level + 1]),
                "n_units": int(mask.sum()),
                "mean_disagreement": float(disagreement[mask].mean()),
                "MAE": float(np.mean(np.abs(residual[mask]))),
                "RMSE": float(np.sqrt(np.mean(residual[mask] ** 2))),
            }
        )
    return pd.DataFrame(rows)
