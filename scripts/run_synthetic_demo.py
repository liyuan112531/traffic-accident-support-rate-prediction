"""Execute the full pipeline on the synthetic demonstration dataset.

What this demonstrates
----------------------
That the published code runs from end to end on a case table, in the order the
method prescribes::

    synthetic cases
      -> request-side features
      -> eleven expert models fitted per item
      -> expert predictions and the disagreement statistics
      -> the 73-dimensional meta representation
      -> Item-Balanced L1 meta-learning
      -> claim-amount-driven aggregation
      -> case-level support rate
      -> item-level, case-level and resampling-based evaluation

What this does not demonstrate
------------------------------
The numerical results of the manuscript. Those depend on the restricted
case-level corpus, which is not distributed. The scale here is deliberately
reduced so that a checkout can be exercised quickly, and every output is labelled
accordingly.

Usage
-----
::

    python scripts/run_synthetic_demo.py
    python scripts/run_synthetic_demo.py --max-cases 600 --folds 3
    python scripts/run_synthetic_demo.py --with-tabpfn
    python scripts/run_synthetic_demo.py --write-predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.env import configure_determinism  # noqa: E402

configure_determinism()

from src.evaluation.metrics import macro_micro_metrics  # noqa: E402
from src.features.claim_features import N_REQUEST_FEATURES  # noqa: E402
from src.features.item_indicators import N_ITEM_INDICATORS  # noqa: E402
from src.features.meta_features import (  # noqa: E402
    N_CLAIM_SIDE,
    N_DISAGREEMENT,
    meta_block_boundaries,
    meta_dimension,
)
from src.pipeline import (  # noqa: E402
    PipelineConfig,
    VARIANT_DIRECT,
    VARIANT_PROPOSED,
    run_pipeline,
)
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.preprocessing.schema import ITEMS  # noqa: E402
from src.statistics.resampling import (  # noqa: E402
    case_stratified_bootstrap,
    paired_permutation_test,
)
from src.utils.cli import display_path  # noqa: E402
from src.utils.paths import PROJECT_ROOT, load_config  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "synthetic" / "synthetic_cases.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "synthetic_demo" / "expected_output"

BANNER = "=" * 78


def _print_header(title: str) -> None:
    print()
    print(BANNER)
    print(title)
    print(BANNER)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def verify_structure(pool_size: int) -> Dict[str, Any]:
    """Confirm the dimensional identities of the meta representation.

    Checked against the code's own declarations rather than against literals, so
    the check fails if a block is ever widened or narrowed by mistake.
    """
    return {
        "request_features": N_REQUEST_FEATURES,
        "request_features_is_33": N_REQUEST_FEATURES == 33,
        "item_indicators": N_ITEM_INDICATORS,
        "item_indicators_is_16": N_ITEM_INDICATORS == 16,
        "claim_side": N_CLAIM_SIDE,
        "claim_side_is_49": N_CLAIM_SIDE == 33 + 16,
        "disagreement": N_DISAGREEMENT,
        "disagreement_is_13": N_DISAGREEMENT == 13,
        "meta_width_for_pool": meta_dimension(pool_size),
        "block_boundaries_for_pool": meta_block_boundaries(pool_size),
        "frozen_pool_width": 11 + N_DISAGREEMENT + N_CLAIM_SIDE,
        "frozen_pool_width_is_73": (11 + N_DISAGREEMENT + N_CLAIM_SIDE) == 73,
        "frozen_pool_boundaries": meta_block_boundaries(11),
        "frozen_pool_boundaries_are_11_24_73":
            meta_block_boundaries(11)["claim_side"] == (24, 73),
    }


def main(argv=None) -> int:
    config_yaml = load_config().get("demonstration", {})
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int,
                        default=config_yaml.get("max_cases", 320))
    parser.add_argument("--folds", type=int,
                        default=config_yaml.get("n_outer_folds", 2))
    parser.add_argument("--inner-folds", type=int,
                        default=config_yaml.get("inner_folds", 2))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=config_yaml.get("seeds", [42]))
    parser.add_argument("--bootstrap", type=int,
                        default=config_yaml.get("bootstrap_replicates", 400))
    parser.add_argument("--permutations", type=int,
                        default=config_yaml.get("permutation_replicates", 400))
    parser.add_argument("--with-tabpfn", action="store_true",
                        help="include TabPFN; requires downloading model weights")
    parser.add_argument("--pool", nargs="+", default=None,
                        help="explicit expert subset; defaults to the frozen pool")
    parser.add_argument("--write-predictions", action="store_true",
                        help="also write per-unit and per-case predictions")
    arguments = parser.parse_args(argv)

    started = time.time()
    _print_header("Synthetic demonstration of the support-rate pipeline")

    if not arguments.data.exists():
        print(f"the synthetic dataset is missing: {arguments.data}")
        print("generate it with:  python scripts/make_synthetic_data.py")
        return 2

    frame = load_cases(arguments.data)
    print(f"dataset          : {display_path(arguments.data)}")
    print(f"cases            : {len(frame)}")
    print(f"items            : {len(ITEMS)}")
    print("synthetic        : yes, by construction; not derived from real judgments")

    config = PipelineConfig(
        n_outer_folds=int(arguments.folds),
        n_inner_folds=int(arguments.inner_folds),
        seeds=tuple(int(s) for s in arguments.seeds),
        pool=arguments.pool,
        allow_external_weights=bool(arguments.with_tabpfn),
        max_cases=int(arguments.max_cases),
        label=str(config_yaml.get("label", "demonstration scale")),
    )
    print(f"run scale        : {json.dumps(config.describe())}")

    result = run_pipeline(frame, config, progress_callback=lambda m: print(f"  .. {m}"))
    elapsed = time.time() - started

    settings = result.settings
    pool = settings["experts"]
    width = settings["meta_width"]

    # ------------------------------------------------------------------ structure
    _print_header("1. Method structure")
    structure = verify_structure(len(pool))
    print(f"expert pool      : {len(pool)} experts -> {', '.join(pool)}")
    if settings["experts_skipped"]:
        for name, reason in settings["experts_skipped"].items():
            print(f"  skipped        : {name} - {reason}")
    print(f"meta dimension   : {width} = "
          f"{len(pool)} expert predictions + {N_DISAGREEMENT} disagreement + "
          f"{N_CLAIM_SIDE} claim-side")
    print(f"claim-side split : {N_REQUEST_FEATURES} request features + "
          f"{N_ITEM_INDICATORS} item indicators = {N_CLAIM_SIDE}")
    print(f"frozen-pool width: {structure['frozen_pool_width']} "
          f"(11 + 13 + 49), which the manuscript reports")
    if len(pool) != 11:
        print(f"  note           : this run does not include the full frozen pool, so "
              f"its meta width is {width}, not {structure['frozen_pool_width']}; the "
              "block identities are unchanged")

    # ------------------------------------------------------------------ weighting
    _print_header("2. Item-balanced training weights")
    checks = result.weights_check
    print(f"folds checked    : {checks.get('folds_checked')}")
    print(f"per-item share   : [{checks.get('min_item_share'):.8f}, "
          f"{checks.get('max_item_share'):.8f}] against an ideal of "
          f"{checks.get('ideal_item_share'):.8f} (1/16)")
    print(f"per-item total   : [{checks.get('min_item_total'):.4f}, "
          f"{checks.get('max_item_total'):.4f}] on the rescaled axis, where the "
          "whole training set sums to its number of units")
    print(f"mean weight      : [{checks.get('min_mean_weight'):.8f}, "
          f"{checks.get('max_mean_weight'):.8f}] (all-ones target)")
    print(f"balanced         : {_verdict(bool(checks.get('all_balanced')))}")
    print(f"strictly positive: {_verdict(bool(checks.get('all_positive')))}")

    print()
    print("Objective alignment by item (largest and smallest items by unit count):")
    alignment = result.alignment
    if len(alignment):
        head = alignment.head(3)
        tail = alignment.tail(3)
        print(f"  {'item':<20}{'units':>7}{'row share':>11}"
              f"{'uniform ratio':>15}{'balanced ratio':>16}")
        for _, row in list(head.iterrows()) + list(tail.iterrows()):
            print(f"  {row['item']:<20}{int(row['n_units']):>7}"
                  f"{row['share_of_training_rows']:>11.4f}"
                  f"{row['alignment_ratio_uniform']:>15.4f}"
                  f"{row['alignment_ratio_item_balanced']:>16.4f}")
        print("  ratio 1.0 means the item's total training weight matches its "
              "weight in the macro average")

    # ------------------------------------------------------------------ aggregation
    _print_header("3. Claim-structure-driven aggregation")
    conservation = result.conservation
    print(f"conservation identity  sum_j w_ij r_ij == award_total_i / claim_total_i")
    print(f"  cases checked        : {conservation['n_cases_checked']}")
    print(f"  cases excluded       : {conservation['n_cases_excluded']} "
          f"(no claim at all, so no rate is defined)")
    print(f"  max absolute deviation: {conservation['max_abs_deviation']:.3e}")
    print(f"  mean absolute deviation: {conservation['mean_abs_deviation']:.3e}")
    print(f"  identity holds       : "
          f"{_verdict(conservation['max_abs_deviation'] <= 1e-9)}")
    concentration = result.concentration
    print(f"  weight concentration : HHI mean {concentration['mean']:.4f}, "
          f"uniform reference {concentration['uniform_reference']:.4f}, "
          f"single-item cases {concentration['n_cases_single_item']}")

    # ------------------------------------------------------------------ metrics
    _print_header("4. Item-level performance (macro over the 16 items)")
    unit_summary = result.unit_summary()
    if len(unit_summary):
        print(f"  {'variant':<34}{'macro MAE':>11}{'macro RMSE':>12}"
              f"{'macro R2':>10}{'micro MAE':>11}")
        for _, row in unit_summary.iterrows():
            print(f"  {str(row['variant']):<34}{row['macro_MAE']:>11.4f}"
                  f"{row['macro_RMSE']:>12.4f}{row['macro_R2']:>10.4f}"
                  f"{row['micro_MAE']:>11.4f}")
        print()
        print("  note: in the synthetic generator the support rate depends on a latent")
        print("        case-level severity, while the observable request features are only")
        print("        weakly related to it. R-squared close to zero or negative is a")
        print("        property of that construction, not a property of the method.")

    proposed = result.metrics.get(VARIANT_PROPOSED, {})
    control = result.metrics.get("Pooled L1, uniform weights", {})
    if proposed and control:
        improvement = (
            control["macro_MAE"] - proposed["macro_MAE"]
        ) / control["macro_MAE"] * 100.0
        print()
        print(f"  item balancing changes macro MAE by {improvement:+.2f}% relative to "
              "the uniform-weight control")
        print("  this is a demonstration-scale figure and is not comparable with the "
              "manuscript, which uses the restricted corpus")

    _print_header("5. Case-level aggregation versus direct prediction")
    case_summary = result.case_summary()
    if len(case_summary):
        print(f"  {'variant':<40}{'MAE':>9}{'RMSE':>10}{'R2':>9}{'cases':>8}")
        for _, row in case_summary.iterrows():
            print(f"  {str(row['variant']):<40}{row['MAE']:>9.4f}"
                  f"{row['RMSE']:>10.4f}{row['R2']:>9.4f}{int(row['n_cases']):>8}")

    # ------------------------------------------------------------------ inference
    _print_header("6. Paired resampling inference")
    inference: Dict[str, Any] = {}
    units = result.unit_frame
    if len(units):
        comparison = "Pooled L1, uniform weights"
        usable = units.dropna(subset=[VARIANT_PROPOSED, comparison])
        if len(usable):
            items_vector = usable["item"].to_numpy()
            cases_vector = usable["case_id"].to_numpy()
            truth = usable["y_true"].to_numpy(float)
            prediction_a = usable[VARIANT_PROPOSED].to_numpy(float)
            prediction_b = usable[comparison].to_numpy(float)
            bootstrap = case_stratified_bootstrap(
                truth, prediction_a, prediction_b, items_vector, cases_vector,
                n_replicates=int(arguments.bootstrap), seed=11,
            )
            print(f"  paired case-stratified bootstrap of "
                  f"{VARIANT_PROPOSED!r} minus {comparison!r}")
            print(f"  {'metric':<12}{'delta':>10}{'95% interval':>26}{'P(better)':>11}")
            for row in bootstrap["rows"]:
                interval = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
                print(f"  {row['metric']:<12}{row['delta_observed']:>10.4f}"
                      f"{interval:>26}{row['direction_probability']:>11.3f}")
            inference["bootstrap"] = bootstrap["rows"]

            permutation = paired_permutation_test(
                truth, prediction_a, prediction_b, items_vector,
                n_replicates=int(arguments.permutations), seed=12,
            )
            print()
            print("  paired permutation test on the same comparison")
            print(f"  {'metric':<12}{'delta':>10}{'p (two-sided)':>15}")
            for row in permutation["rows"]:
                print(f"  {row['metric']:<12}{row['delta_observed']:>10.4f}"
                      f"{row['p_two_sided']:>15.4f}")
            inference["permutation"] = permutation["rows"]

    # ------------------------------------------------------------------ artefacts
    _print_header("7. Artefacts")
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if len(unit_summary):
        path = output_dir / "unit_level_metrics.csv"
        unit_summary.to_csv(path, index=False)
        written.append(path)
    if len(case_summary):
        path = output_dir / "aggregated_metrics.csv"
        case_summary.to_csv(path, index=False)
        written.append(path)
    if len(result.item_frame):
        path = output_dir / "item_level_metrics.csv"
        result.item_frame.to_csv(path, index=False)
        written.append(path)
    if len(alignment):
        path = output_dir / "objective_alignment.csv"
        alignment.to_csv(path, index=False)
        written.append(path)

    diagnostics = {
        "run_settings": settings,
        "structure_check": structure,
        "weight_invariants": result.weights_check,
        "conservation_identity": result.conservation,
        "weight_concentration": result.concentration,
        "inference": inference,
        "elapsed_seconds": round(elapsed, 2),
        "disclaimer": (
            "Produced from the synthetic demonstration dataset. These figures are "
            "not the results reported in the manuscript and are not comparable "
            "with them."
        ),
    }
    path = output_dir / "run_diagnostics.json"
    path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    written.append(path)

    if arguments.write_predictions:
        if len(result.unit_frame):
            path = output_dir / "unit_predictions.csv"
            result.unit_frame.to_csv(path, index=False)
            written.append(path)
        if len(result.case_frame):
            path = output_dir / "aggregated_predictions.csv"
            result.case_frame.to_csv(path, index=False)
            written.append(path)

    for path in written:
        print(f"  wrote {display_path(path)}")

    # ------------------------------------------------------------------ checks
    _print_header("8. Verification")
    checks = {
        "meta representation has the documented block widths":
            structure["claim_side_is_49"] and structure["disagreement_is_13"]
            and structure["frozen_pool_width_is_73"],
        "item-balanced weights are exactly balanced within every fold":
            bool(result.weights_check.get("all_balanced", False)),
        "every unit receives a strictly positive weight":
            bool(result.weights_check.get("all_positive", False)),
        "aggregation reproduces the case-level rate to machine precision":
            conservation["max_abs_deviation"] <= 1e-9,
        "at least one item-level metric was computed":
            bool(result.metrics),
        "at least one case-level metric was computed":
            bool(result.case_metrics),
        "all predictions lie inside [0, 1]":
            _predictions_in_range(result),
    }
    for name, ok in checks.items():
        print(f"  [{_verdict(ok)}] {name}")
    print()
    print(f"elapsed          : {elapsed:.1f} s")
    print(f"overall          : "
          f"{_verdict(all(checks.values()))}")
    return 0 if all(checks.values()) else 1


def _predictions_in_range(result) -> bool:
    """Whether every stored prediction lies inside the admissible interval."""
    columns = [
        c for c in result.unit_frame.columns
        if c not in ("case_id", "item", "y_true")
    ]
    for column in columns:
        values = result.unit_frame[column].dropna().to_numpy(dtype=float)
        if values.size and (values.min() < -1e-12 or values.max() > 1 + 1e-12):
            return False
    case_columns = [
        c for c in result.case_frame.columns if c not in ("case_id", "y_true")
    ]
    for column in case_columns:
        values = result.case_frame[column].dropna().to_numpy(dtype=float)
        if values.size and (values.min() < -1e-12 or values.max() > 1 + 1e-12):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
