"""Paired permutation test with a multiple-comparison correction.

Under the null hypothesis the two methods are exchangeable, so for every case-item
unit the two predictions may be swapped. Swapping at random and recomputing the
macro difference traces out a null distribution centred at zero, and the observed
difference is located in it. Because the swap happens within a unit, the test
conditions on the case and the item and needs no assumption about how units relate
to one another.

Two corrections are reported. Benjamini-Hochberg controls the expected share of
false discoveries and is the primary figure. Holm controls the family-wise error
rate and is reported alongside it because it is the more conservative of the two;
when both agree the conclusion does not depend on that choice.

Usage
-----
::

    python scripts/permutation_test.py
    python scripts/permutation_test.py --per-item --replicates 5000

To test a table produced elsewhere, pass ``--predictions`` pointing at a CSV with
columns ``case_id``, ``item``, ``y_true`` and the two variant columns.
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

from src.pipeline import (  # noqa: E402
    VARIANT_PROPOSED,
    VARIANT_UNWEIGHTED,
    PipelineConfig,
    run_pipeline,
)
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.statistics.multiplicity import benjamini_hochberg, holm_adjust  # noqa: E402
from src.statistics.resampling import paired_permutation_test  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    demonstration_defaults,
    display_path,
)
from src.utils.env import configure_determinism  # noqa: E402
from src.utils.paired import describe_pair, paired_units, per_item_slices  # noqa: E402


def _load_units(arguments) -> tuple[pd.DataFrame, dict]:
    if arguments.predictions is not None:
        return pd.read_csv(arguments.predictions), {
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
        label="synthetic demonstration (permutation stage)",
    )
    frame = load_cases(arguments.data)
    result = run_pipeline(frame, config)
    return result.unit_frame, {
        "source": arguments.data.name,
        "label": config.label,
        "run": result.settings.get("run", {}),
        "default_replicates": defaults["permutation_replicates"],
    }


def _with_corrections(table: pd.DataFrame) -> pd.DataFrame:
    """Attach Benjamini-Hochberg and Holm adjusted p-values.

    The correction is applied across whichever family the caller has assembled; the
    caller decides that family, which is the only honest way to do it because the
    family size is a property of the reported set and not of the test.
    """
    if table.empty:
        return table
    p_values = table["p_two_sided"].to_numpy(dtype=float)
    table = table.copy()
    table["q_value_bh"] = benjamini_hochberg(p_values)
    table["p_value_holm"] = holm_adjust(p_values)
    table["significant_bh_0.05"] = table["q_value_bh"] < 0.05
    table["significant_holm_0.05"] = table["p_value_holm"] < 0.05
    return table


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
                        help="permutation replicates; defaults to the demonstration scale")
    parser.add_argument("--seed", type=int, default=None, help="permutation seed")
    parser.add_argument("--per-item", action="store_true",
                        help="also test each item separately, corrected as one family")
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

    banner("Paired permutation test")
    print(f"comparison       : {arguments.variant_a}  minus  {arguments.variant_b}")
    print(f"source           : {provenance.get('source')}")
    print(f"usable units     : {summary['n_units']} over {summary['n_cases']} cases "
          f"and {summary['n_items']} items")
    print(f"dropped units    : {summary['n_dropped']}")
    print(f"replicates       : {replicates}")
    print(f"seed             : {seed}")
    if summary["n_units"] == 0:
        print("no usable units; nothing to test")
        return 1

    outcome = paired_permutation_test(
        units.y, units.prediction_a, units.prediction_b, units.items,
        n_replicates=replicates, seed=seed,
    )
    macro = _with_corrections(pd.DataFrame(outcome["rows"]))

    banner("Macro metrics, corrected across three metrics")
    print(f"  {'metric':<12}{'delta':>12}{'null sd':>11}{'p two-sided':>13}"
          f"{'q (BH)':>11}{'p (Holm)':>11}")
    for _, row in macro.iterrows():
        print(f"  {row['metric']:<12}{row['delta_observed']:>12.6f}"
              f"{row['permutation_sd']:>11.6f}{row['p_two_sided']:>13.4f}"
              f"{row['q_value_bh']:>11.4f}{row['p_value_holm']:>11.4f}")
    print()
    print("  A small p-value means the observed difference is unlikely under the")
    print("  hypothesis that the two methods are exchangeable. It is not a statement")
    print("  about the size or the practical importance of the difference.")

    tables = {"macro": macro}
    if arguments.per_item:
        slices = per_item_slices(units)
        item_rows = []
        for item, index in slices.items():
            item_outcome = paired_permutation_test(
                units.y[index], units.prediction_a[index], units.prediction_b[index],
                ["item"] * index.size, n_replicates=replicates, seed=seed,
            )
            for row in item_outcome["rows"]:
                item_rows.append({"item": item, **row})
        # One family: every (item, metric) pair reported below.
        item_table = _with_corrections(pd.DataFrame(item_rows))
        item_table = item_table.sort_values(
            ["metric", "p_two_sided", "item"]
        ).reset_index(drop=True)
        tables["per_item"] = item_table

        banner(f"Per item, corrected across {len(item_table)} "
               f"item-metric comparisons")
        print(f"  {'metric':<12}{'item':<20}{'delta':>10}{'p':>10}{'q (BH)':>10}"
              f"{'p (Holm)':>11}{'BH<0.05':>9}")
        for _, row in item_table.iterrows():
            print(f"  {row['metric']:<12}{str(row['item']):<20}"
                  f"{row['delta_observed']:>10.4f}{row['p_two_sided']:>10.4f}"
                  f"{row['q_value_bh']:>10.4f}{row['p_value_holm']:>11.4f}"
                  f"{str(bool(row['significant_bh_0.05'])):>9}")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = output / (f"permutation_{name}.csv" if name != "macro"
                         else "permutation_macro.csv")
        table.to_csv(path, index=False)
        written.append(path)

    record = {
        "comparison": summary,
        "replicates": replicates,
        "seed": seed,
        "provenance": provenance,
        "macro_rows": macro.to_dict(orient="records"),
        "correction_note": ("Benjamini-Hochberg controls the false discovery rate and "
                            "is the primary correction; Holm controls the family-wise "
                            "error rate and is reported alongside it"),
        "note": ("computed on the synthetic demonstration dataset; these p-values "
                 "describe the demonstration, not the study"),
    }
    record_path = output / "permutation_diagnostics.json"
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    written.append(record_path)

    print()
    for path in written:
        print(f"wrote {display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
