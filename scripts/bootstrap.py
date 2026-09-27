"""Case-stratified paired bootstrap over the macro metrics.

The bootstrap resamples cases with replacement and recomputes the per-item metrics
for both methods on the same resampled cases, so the two methods see the same
draw. Resampling at the case level rather than the unit level is what keeps the
dependence between the items of one case: the sixteen units of a case share a
request side and a judge, and treating them as independent draws would understate
the interval width.

The comparison is by default the item-balanced objective against the unweighted
pooled objective. Nothing else differs between those two variants, so the interval
describes the effect of item balancing alone.

Usage
-----
::

    python scripts/bootstrap.py
    python scripts/bootstrap.py --variant-a "Item-Balanced L1" \\
        --variant-b "Item-Balanced L2" --replicates 2000

To bootstrap a table that was produced elsewhere, pass ``--predictions`` pointing
at a CSV with columns ``case_id``, ``item``, ``y_true`` and the two variant columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pipeline import (  # noqa: E402
    VARIANT_PROPOSED,
    VARIANT_UNWEIGHTED,
    PipelineConfig,
    run_pipeline,
)
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.statistics.resampling import case_stratified_bootstrap  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    demonstration_defaults,
    display_path,
)
from src.utils.env import configure_determinism  # noqa: E402
from src.utils.paired import describe_pair, paired_units, per_item_slices  # noqa: E402


def _load_units(arguments) -> tuple[pd.DataFrame, dict]:
    """Either run the protocol or read a paired table supplied by the caller."""
    if arguments.predictions is not None:
        frame = pd.read_csv(arguments.predictions)
        return frame, {
            "source": str(arguments.predictions.name),
            "label": "supplied prediction table",
        }

    defaults = demonstration_defaults()
    config = PipelineConfig(
        n_outer_folds=int(arguments.folds),
        n_inner_folds=int(arguments.inner_folds),
        seeds=tuple(int(s) for s in arguments.seeds),
        allow_external_weights=bool(arguments.with_tabpfn),
        max_cases=int(arguments.max_cases) if arguments.max_cases else None,
        pool=tuple(arguments.pool) if arguments.pool else None,
        label="synthetic demonstration (bootstrap stage)",
    )
    frame = load_cases(arguments.data)
    result = run_pipeline(frame, config)
    return result.unit_frame, {
        "source": arguments.data.name,
        "label": config.label,
        "run": result.settings.get("run", {}),
        "default_replicates": defaults["bootstrap_replicates"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="paired table to use instead of running the protocol")
    parser.add_argument("--variant-a", default=VARIANT_PROPOSED,
                        help="first variant; the difference is reported as a - b")
    parser.add_argument("--variant-b", default=VARIANT_UNWEIGHTED,
                        help="second variant")
    parser.add_argument("--replicates", type=int, default=None,
                        help="bootstrap replicates; defaults to the demonstration scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="bootstrap seed")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="two-sided interval level")
    parser.add_argument("--per-item", action="store_true",
                        help="also bootstrap each item separately")
    arguments = parser.parse_args(argv)

    if arguments.predictions is None and not arguments.data.exists():
        print(f"case table not found: {arguments.data}")
        return 2

    configure_determinism()
    frame, provenance = _load_units(arguments)

    replicates = (int(arguments.replicates) if arguments.replicates
                  else int(provenance.get("default_replicates", 2000)))
    seed = int(arguments.seed) if arguments.seed is not None else int(
        (provenance.get("run") or {}).get("seeds", [42])[0]
    )

    units = paired_units(frame, arguments.variant_a, arguments.variant_b)
    summary = describe_pair(units, arguments.variant_a, arguments.variant_b)

    banner("Case-stratified paired bootstrap")
    print(f"comparison       : {arguments.variant_a}  minus  {arguments.variant_b}")
    print(f"source           : {provenance.get('source')}")
    print(f"usable units     : {summary['n_units']} over {summary['n_cases']} cases "
          f"and {summary['n_items']} items")
    print(f"dropped units    : {summary['n_dropped']}")
    print(f"replicates       : {replicates}")
    print(f"seed             : {seed}")
    print(f"interval level   : {100.0 * (1 - arguments.alpha):.1f}%")
    if summary["n_units"] == 0:
        print("no usable units; nothing to bootstrap")
        return 1

    outcome = case_stratified_bootstrap(
        units.y, units.prediction_a, units.prediction_b, units.items, units.cases,
        n_replicates=replicates, seed=seed, alpha=arguments.alpha,
    )
    rows = pd.DataFrame(outcome["rows"])

    banner("Macro metrics")
    print(f"  {'metric':<12}{'observed delta':>16}{'bootstrap sd':>15}"
          f"{'CI low':>11}{'CI high':>11}{'P(a better)':>13}{'crosses 0':>11}")
    for _, row in rows.iterrows():
        print(f"  {row['metric']:<12}{row['delta_observed']:>16.6f}"
              f"{row['bootstrap_sd']:>15.6f}{row['ci_low']:>11.6f}"
              f"{row['ci_high']:>11.6f}{row['direction_probability']:>13.4f}"
              f"{str(bool(row['ci_crosses_zero'])):>11}")
    print()
    print("  A negative delta means the first variant has the lower error for MAE and")
    print("  RMSE, which is the better outcome there. For R-squared the better outcome")
    print("  is the higher value, so a positive delta favours the first variant.")
    print("  'P(a better)' is the fraction of replicates in which the first variant is")
    print("  the better method on that metric; it is not a p-value.")

    tables = {"macro": rows}
    if arguments.per_item:
        slices = per_item_slices(units)
        item_rows = []
        for item, index in slices.items():
            item_outcome = case_stratified_bootstrap(
                units.y[index], units.prediction_a[index], units.prediction_b[index],
                ["item"] * index.size, units.cases[index],
                n_replicates=replicates, seed=seed, alpha=arguments.alpha,
            )
            for row in item_outcome["rows"]:
                item_rows.append({"item": item, **row})
        item_table = pd.DataFrame(item_rows)
        tables["per_item"] = item_table

        banner("Per item, macro_MAE")
        subset = item_table[item_table["metric"] == "macro_MAE"]
        print(f"  {'item':<20}{'n cases':>9}{'delta':>10}{'CI low':>11}"
              f"{'CI high':>11}{'P(a better)':>13}")
        for _, row in subset.iterrows():
            print(f"  {str(row['item']):<20}{int(row['n_cases']):>9}"
                  f"{row['delta_observed']:>10.4f}{row['ci_low']:>11.4f}"
                  f"{row['ci_high']:>11.4f}{row['direction_probability']:>13.4f}")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = output / (f"bootstrap_{name}.csv" if name != "macro"
                         else "bootstrap_macro.csv")
        table.to_csv(path, index=False)
        written.append(path)

    record = {
        "comparison": summary,
        "replicates": replicates,
        "seed": seed,
        "alpha": arguments.alpha,
        "provenance": provenance,
        "rows": tables["macro"].to_dict(orient="records"),
        "note": ("computed on the synthetic demonstration dataset; these intervals "
                 "describe the demonstration, not the study"),
    }
    record_path = output / "bootstrap_diagnostics.json"
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    written.append(record_path)

    print()
    for path in written:
        print(f"wrote {display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
