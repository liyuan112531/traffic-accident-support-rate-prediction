"""Inspect the meta-learner stage in isolation.

The meta-learner is the only supervised component that sits on top of the expert
pool. It consumes the cross-fitted meta representation and is trained with the
Item-Balanced L1 objective. This script runs the protocol on the synthetic
demonstration dataset, then reports:

* the block structure of the meta representation,
* the item-balancing property of the training weights,
* the pre-registered hyperparameters of the meta head,
* the fitted head's behaviour on the cross-fitted matrix it was trained on.

Nothing here is a study result. The numbers describe the demonstration dataset and
are reported only so that the training stage can be inspected.

Usage
-----
::

    python scripts/train_meta.py
    python scripts/train_meta.py --max-cases 300 --folds 2 --inner-folds 2
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

from src.meta_learning.item_balanced import (  # noqa: E402
    describe_weighting,
    item_balanced_weights,
    objective_alignment_table,
    uniform_weights,
    weight_check,
)
from src.meta_learning.meta_learner import (  # noqa: E402
    LOSS_LABEL,
    META_HYPERPARAMETERS,
    predict_meta_head,
)
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.preprocessing.pipeline import load_cases, to_item_long  # noqa: E402
from src.preprocessing.schema import ITEMS  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    display_path,
)
from src.utils.env import configure_determinism  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--verbose", action="store_true",
                        help="report progress for every seed and fold")
    arguments = parser.parse_args(argv)

    if not arguments.data.exists():
        print(f"case table not found: {arguments.data}")
        return 2

    frame = load_cases(arguments.data)
    inner_folds = int(arguments.inner_folds)

    banner(f"Meta-learner stage - objective: {LOSS_LABEL}")
    print(f"cases            : {min(len(frame), arguments.max_cases or len(frame))}"
          f" of {len(frame)} available")
    print(f"inner partitions : {inner_folds}")

    long_frame = to_item_long(frame, list(ITEMS))
    items = long_frame["item"].to_numpy()

    banner("Item-balanced training weights")
    print(describe_weighting(items, n_items=len(ITEMS)))
    balanced = item_balanced_weights(items, n_items=len(ITEMS))
    uniform = uniform_weights(items)
    check = weight_check(items, balanced, tol=1e-8)
    print()
    print(f"  uniform weight per unit           : {uniform[0]:.8f}")
    print(f"  item-balanced weight per unit     : {balanced[0]:.8f} (item 0)")
    print(f"  sum of weights                    : {check['sum_w']:.6f}"
          f" (= number of units {check['n_units']})")
    print(f"  every item total weight           : {check['item_total_min']:.6f}"
          f" .. {check['item_total_max']:.6f}")
    print(f"  every item share of total weight  : {check['item_share_min']:.8f}"
          f" .. {check['item_share_max']:.8f}")
    print(f"  ideal share                       : {check['item_share_ideal']:.8f}"
          f" (= 1/{len(ITEMS)})")
    print(f"  balanced within tolerance         : {check['balanced']}")
    print(f"  every unit strictly positive      : {check['all_positive']}")

    banner("Objective alignment by item")
    alignment = objective_alignment_table(items, n_items=len(ITEMS))
    print(f"  {'item':<20}{'units':>7}{'row share':>11}{'uniform ratio':>16}"
          f"{'balanced ratio':>16}")
    for _, row in alignment.iterrows():
        print(f"  {str(row['item']):<20}{int(row['n_units']):>7}"
              f"{row['share_of_training_rows']:>11.4f}"
              f"{row['alignment_ratio_uniform']:>16.4f}"
              f"{row['alignment_ratio_item_balanced']:>16.4f}")
    print()
    print("  ratio 1.0 means the item's total training contribution matches its "
          "intended share; 1/16 for the item-balanced objective.")

    configure_determinism()
    config = PipelineConfig(
        n_outer_folds=int(arguments.folds),
        n_inner_folds=inner_folds,
        seeds=(int(arguments.seeds[0]),),
        max_cases=int(arguments.max_cases) if arguments.max_cases else None,
        pool=tuple(arguments.pool) if arguments.pool else None,
        label="synthetic demonstration (meta stage)",
        progress=bool(arguments.verbose),
    )
    banner("Running the protocol to obtain a cross-fitted meta matrix")
    result = run_pipeline(frame, config,
                          progress_callback=(print if arguments.verbose else None))

    matrix = result.meta_training_matrix
    truth = result.meta_training_truth
    names = list(result.meta_names)
    if matrix is None or truth is None:
        print("the run produced no meta training matrix; increase --max-cases")
        return 1

    banner("Cross-fitted meta representation")
    print(f"  rows                     : {matrix.shape[0]}")
    print(f"  width                    : {matrix.shape[1]}")
    print(f"  documented width         : 11 expert + 13 disagreement + 49 claim-side")
    print(f"  blocks                   : A=[0:11]  B=[11:24]  C=[24:73]")
    print(f"  distinct feature names   : {len(set(names))} of {len(names)}")
    print(f"  finite entries           : {np.isfinite(matrix).mean() * 100:.2f}%")

    banner("Meta head hyperparameters")
    for key, value in META_HYPERPARAMETERS.items():
        print(f"  {key:<26}: {value}")

    if result.meta_model is not None:
        prediction = predict_meta_head(result.meta_model, matrix)
        residual = np.abs(truth - prediction)
        banner("Fitted head on its own cross-fitted training matrix")
        print(f"  MAE                      : {float(residual.mean()):.6f}")
        print(f"  RMSE                     : {float(np.sqrt(np.mean((truth - prediction) ** 2))):.6f}")
        print(f"  predictions within [0,1] : {bool((prediction >= -1e-12).all() and (prediction <= 1 + 1e-12).all())}")
        print(f"  truth within [0,1]       : {bool((truth >= -1e-12).all() and (truth <= 1 + 1e-12).all())}")
        print()
        print("  in-sample fit only; the honest performance figure is the item-level")
        print("  and case-level table produced by scripts/evaluate.py")

    banner("Item-level metrics, all variants")
    print(result.unit_summary().to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    banner("Case-level metrics, all variants")
    print(result.case_summary().to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output / "objective_alignment.csv", index=False)
    result.unit_summary().to_csv(output / "unit_level_metrics.csv", index=False)
    result.case_summary().to_csv(output / "aggregated_metrics.csv", index=False)
    diagnostics = {
        "objective": LOSS_LABEL,
        "meta_width": int(matrix.shape[1]),
        "meta_rows": int(matrix.shape[0]),
        "feature_names": names,
        "weight_check": check,
        "meta_hyperparameters": {k: str(v) for k, v in META_HYPERPARAMETERS.items()},
        "run": result.settings.get("run", {}),
        "note": ("synthetic demonstration dataset; figures are not manuscript results"),
    }
    with (output / "meta_stage_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
    print()
    print(f"wrote {display_path(output / 'meta_stage_diagnostics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
