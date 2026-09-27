"""TreeSHAP attribution of the fitted meta-learner over the 73-dimensional input.

What this measures
------------------
The attribution splits the meta-learner's output among the inputs it actually used.
Attributions are aggregated to the three blocks the representation is built from,
and within the expert block to individual experts, so the question answered is
which part of the representation the fit relies on.

What this does not measure
--------------------------
A large attribution is not a causal effect. The observed support rate is a decision
recorded in a judgment, not assigned at random, so no intervention is available to
compare against and nothing here supports a counterfactual reading.

The disagreement block is a prediction-difficulty signal. It is not a calibrated
uncertainty estimate: nothing here has been checked against realised coverage, so a
large value means the expert pool split, not that an interval is valid.

Usage
-----
::

    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --max-cases 400 --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.interpretation.shap_analysis import (  # noqa: E402
    block_contribution as block_table,
    disagreement_quartile_table,
    expert_contribution,
    feature_inventory,
    global_attribution,
    tree_shap,
)
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    display_path,
)
from src.utils.env import configure_determinism  # noqa: E402

#: Column used as the scalar summary of expert disagreement.
DISAGREEMENT_FEATURE: str = "pred_std"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--top", type=int, default=15,
                        help="how many features to print in the ranked table")
    parser.add_argument("--disagreement-feature", default=DISAGREEMENT_FEATURE,
                        help="column used as the scalar disagreement summary")
    arguments = parser.parse_args(argv)

    if not arguments.data.exists():
        print(f"case table not found: {arguments.data}")
        return 2

    configure_determinism()
    try:
        import shap  # noqa: F401
    except ImportError:
        print("this stage requires the 'shap' package; install it and retry")
        return 2

    frame = load_cases(arguments.data)
    config = PipelineConfig(
        n_outer_folds=int(arguments.folds),
        n_inner_folds=int(arguments.inner_folds),
        seeds=(int(arguments.seeds[0]),),
        allow_external_weights=bool(arguments.with_tabpfn),
        max_cases=int(arguments.max_cases) if arguments.max_cases else None,
        pool=tuple(arguments.pool) if arguments.pool else None,
        label="synthetic demonstration (interpretation stage)",
    )

    banner("Interpretation of the fitted meta-learner")
    result = run_pipeline(frame, config)
    matrix = result.meta_training_matrix
    truth = result.meta_training_truth
    names = list(result.meta_names)
    if matrix is None or truth is None or result.meta_model is None:
        print("the run produced no fitted meta-learner; increase --max-cases")
        return 1

    n_pool = len(result.settings["experts"])
    inventory = feature_inventory(names, n_pool)
    print(f"explained rows   : {matrix.shape[0]}")
    print(f"meta width       : {matrix.shape[1]}")
    print(f"blocks           : {inventory['block_label'].value_counts().to_dict()}")

    values, expected = tree_shap(result.meta_model, matrix)
    print(f"attribution shape: {values.shape}")
    print(f"model expectation: {expected:.6f}")

    additive_error = float(
        np.max(np.abs(values.sum(axis=1) + expected - result.meta_model.predict(matrix)))
    )
    banner("Additivity check")
    print(f"  max |sum(shap) + expectation - raw prediction| : {additive_error:.3e}")
    print("  TreeSHAP is additive, so this residual should be at numerical noise level.")

    banner(f"Top {arguments.top} features by mean absolute attribution")
    global_table = global_attribution(values, inventory)
    print(f"  {'rank':>4}  {'feature':<26}{'block':<24}{'mean|shap|':>12}{'share':>9}")
    for _, row in global_table.head(int(arguments.top)).iterrows():
        print(f"  {int(row['rank']):>4}  {str(row['feature']):<26}"
              f"{str(row['block_label']):<24}{row['mean_abs_shap']:>12.6f}"
              f"{row['share']:>9.4f}")

    banner("Block contribution")
    blocks = block_table(global_table)
    print(f"  {'block':<24}{'features':>10}{'mean|shap|':>13}{'contribution':>14}")
    for _, row in blocks.iterrows():
        print(f"  {str(row['block_label']):<24}{int(row['n_features']):>10}"
              f"{row['mean_abs_shap_per_feature']:>13.6f}"
              f"{row['contribution_pct']:>13.2f}%")

    banner("Contribution within the expert block")
    experts = expert_contribution(global_table)
    if experts.empty:
        print("  the run used no expert columns, so this table is empty")
    else:
        print(f"  {'expert':<16}{'mean|shap|':>13}{'share':>9}")
        for _, row in experts.iterrows():
            print(f"  {str(row['expert']):<16}{row['mean_abs_shap']:>13.6f}"
                  f"{row['share']:>9.4f}")
        print()
        print(f"  effective number of contributing experts : "
              f"{experts.attrs['effective_n_experts']:.3f} of {experts.attrs['n_experts']}")
        print(f"  dominance ratio of the leading expert    : "
              f"{experts.attrs['dominance_ratio']:.4f}")

    banner("Error by quartile of expert disagreement")
    quartiles = None
    if arguments.disagreement_feature not in names:
        print(f"  {arguments.disagreement_feature!r} is not part of this "
              f"representation; available disagreement columns are "
              f"{[n for n in names[n_pool:n_pool + 13]]}")
    else:
        column = names.index(arguments.disagreement_feature)
        prediction = np.clip(result.meta_model.predict(matrix), 0.0, 1.0)
        quartiles = disagreement_quartile_table(matrix[:, column], truth, prediction)
        print(f"  disagreement summary : {arguments.disagreement_feature}")
        print(f"  {'quartile':>9}{'n units':>9}{'mean disagreement':>19}"
              f"{'MAE':>10}{'RMSE':>10}")
        for _, row in quartiles.iterrows():
            print(f"  {int(row['quartile']):>9}{int(row['n_units']):>9}"
                  f"{row['mean_disagreement']:>19.6f}{row['MAE']:>10.6f}"
                  f"{row['RMSE']:>10.6f}")
        print()
        print("  A rising error across the quartiles means the pool's disagreement")
        print("  marks the units the model finds hard. It does not mean the")
        print("  disagreement value is a calibrated standard error.")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    global_path = output / "shap_global_attribution.csv"
    blocks_path = output / "shap_block_contribution.csv"
    experts_path = output / "shap_expert_contribution.csv"
    global_table.to_csv(global_path, index=False)
    blocks.to_csv(blocks_path, index=False)
    experts.to_csv(experts_path, index=False)
    written = [global_path, blocks_path, experts_path]
    if quartiles is not None:
        quartile_path = output / "shap_disagreement_quartiles.csv"
        quartiles.to_csv(quartile_path, index=False)
        written.append(quartile_path)

    record = {
        "n_rows": int(matrix.shape[0]),
        "meta_width": int(matrix.shape[1]),
        "n_pool": int(n_pool),
        "expected_value": float(expected),
        "additivity_max_abs_error": additive_error,
        "block_contribution_pct": {
            str(row["block_label"]): float(row["contribution_pct"])
            for _, row in blocks.iterrows()
        },
        "run": result.settings.get("run", {}),
        "scope": {
            "is_causal": False,
            "is_calibrated_uncertainty": False,
            "disagreement_reading": "prediction-difficulty signal",
        },
        "note": ("attribution of a model fitted on the synthetic demonstration "
                 "dataset; no causal or uncertainty interpretation is supported"),
    }
    record_path = output / "shap_diagnostics.json"
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    written.append(record_path)

    print()
    for path in written:
        print(f"wrote {display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
