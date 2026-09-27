"""Evaluate a run and write the item-level, case-level and fold-level tables.

This is the reporting stage. It runs the protocol and writes the metric tables
without any further modelling, so the numbers it reports are exactly the numbers
:func:`src.pipeline.run_pipeline` produces.

Every table it writes is produced from the synthetic demonstration dataset unless
``--data`` is pointed at another case table that follows
``data/schema/data_dictionary.csv``.

Usage
-----
::

    python scripts/evaluate.py
    python scripts/evaluate.py --folds 3 --seeds 42 2024 --write-predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pipeline import (  # noqa: E402
    CASE_AGGREGATION_VARIANTS,
    UNIT_VARIANTS,
    PipelineConfig,
    run_pipeline,
)
from src.preprocessing.pipeline import load_cases  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    display_path,
)
from src.utils.env import configure_determinism  # noqa: E402


def _table(frame, columns=None) -> str:
    view = frame if columns is None else frame[columns]
    return view.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--write-predictions", action="store_true",
                        help="also write the per-unit and per-case prediction tables")
    arguments = parser.parse_args(argv)

    if not arguments.data.exists():
        print(f"case table not found: {arguments.data}")
        return 2

    configure_determinism()
    frame = load_cases(arguments.data)
    config = PipelineConfig(
        n_outer_folds=int(arguments.folds),
        n_inner_folds=int(arguments.inner_folds),
        seeds=tuple(int(s) for s in arguments.seeds),
        allow_external_weights=bool(arguments.with_tabpfn),
        max_cases=int(arguments.max_cases) if arguments.max_cases else None,
        pool=tuple(arguments.pool) if arguments.pool else None,
        label="synthetic demonstration (evaluation stage)",
    )

    banner("Evaluation")
    print(f"case table       : {arguments.data.name}")
    print(f"cases            : {min(len(frame), arguments.max_cases or len(frame))}")
    print(f"outer folds      : {config.n_outer_folds}")
    print(f"inner folds      : {config.n_inner_folds}")
    print(f"seeds            : {list(config.resolved_seeds())}")

    result = run_pipeline(frame, config)
    settings = result.settings

    banner("Run settings")
    print(f"  units                    : {settings['n_units']}")
    print(f"  experts used             : {settings['pool_size']} "
          f"({', '.join(settings['experts'])})")
    for name, reason in (settings.get("experts_skipped") or {}).items():
        print(f"  skipped                  : {name} - {reason}")
    print(f"  meta width               : {settings['meta_width']} "
          f"(blocks {settings['meta_blocks']})")
    print("  " + json.dumps(settings["run"], ensure_ascii=False))

    banner("Item-level metrics by variant")
    print(_table(result.unit_summary()))

    banner("Case-level metrics by variant")
    print(_table(result.case_summary()))

    banner("Weight invariants over every fold")
    weights = result.weights_check
    print(f"  folds checked            : {weights.get('folds_checked')}")
    print(f"  per-item share range     : [{weights.get('min_item_share'):.8f}, "
          f"{weights.get('max_item_share'):.8f}]")
    print(f"  ideal share              : {weights.get('ideal_item_share'):.8f}")
    print(f"  all items balanced       : {weights.get('all_balanced')}")
    print(f"  all units positive       : {weights.get('all_positive')}")

    banner("Aggregation invariants")
    conservation = result.conservation
    print(f"  conservation identity    : max |deviation| "
          f"{conservation['max_abs_deviation']:.3e} over "
          f"{conservation['n_cases_checked']} cases")
    print(f"  weight concentration HHI : mean {result.concentration['mean']:.4f}, "
          f"uniform reference {result.concentration['uniform_reference']:.4f}")

    banner("Per-item detail, proposed variant")
    if not result.item_frame.empty:
        detail = result.item_frame[result.item_frame["variant"] == UNIT_VARIANTS[0]]
        print(_table(detail, ["item", "n", "MAE", "RMSE", "R2"]))
    else:
        print("  no item-level rows were produced")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    unit_path = output / "unit_level_metrics.csv"
    case_path = output / "aggregated_metrics.csv"
    item_path = output / "item_level_metrics.csv"
    result.unit_summary().to_csv(unit_path, index=False)
    result.case_summary().to_csv(case_path, index=False)
    result.item_frame.to_csv(item_path, index=False)
    result.fold_log.to_csv(output / "fold_log.csv", index=False)

    written = [unit_path, case_path, item_path]
    if arguments.write_predictions:
        unit_predictions = output / "unit_predictions.csv"
        case_predictions = output / "aggregated_predictions.csv"
        result.unit_frame.to_csv(unit_predictions, index=False)
        result.case_frame.to_csv(case_predictions, index=False)
        written += [unit_predictions, case_predictions]

    diagnostics = {
        "run_settings": {k: v for k, v in settings.items() if k != "items"}
        | {"items": settings["items"]},
        "unit_variants": list(UNIT_VARIANTS),
        "case_variants": list(CASE_AGGREGATION_VARIANTS),
        "weight_invariants": weights,
        "conservation_identity": conservation,
        "weight_concentration": result.concentration,
        "note": ("synthetic demonstration dataset; these figures are not the "
                 "manuscript results and are not comparable with them"),
    }
    diagnostics_path = output / "evaluation_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
    written.append(diagnostics_path)

    print()
    for path in written:
        print(f"wrote {display_path(path)}")
    if arguments.write_predictions:
        print("note: the prediction tables contain synthetic cases only; no case "
              "identifier from the study dataset can appear in them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
