"""Fit and evaluate the expert pool item by item.

This stage is isolated so that expert behaviour can be inspected before the
meta-learner is involved. It fits every expert on every item using the request-side
features only, and reports the per-item mean absolute error on held-out cases.

The folds are shared across items and experts, and they are assigned to cases, so
no case contributes to both the fitting and the evaluation part.

Usage
-----
::

    python scripts/train_experts.py
    python scripts/train_experts.py --max-cases 400 --folds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.aggregation.claim_weighted import case_level_metrics  # noqa: E402
from src.experts.registry import build_expert, resolve_pool  # noqa: E402
from src.features.claim_features import build_request_features  # noqa: E402
from src.preprocessing.pipeline import load_cases, to_item_long  # noqa: E402
from src.preprocessing.schema import ITEMS  # noqa: E402
from src.utils.cli import (  # noqa: E402
    add_common_arguments,
    banner,
    display_path,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--min-train-units", type=int, default=8,
                        help="minimum training units for an item to be fitted")
    arguments = parser.parse_args(argv)

    if not arguments.data.exists():
        print(f"case table not found: {arguments.data}")
        return 2

    frame = load_cases(arguments.data)
    if arguments.max_cases and arguments.max_cases < len(frame):
        rng = np.random.default_rng(int(arguments.seeds[0]))
        keep = np.sort(rng.choice(len(frame), size=int(arguments.max_cases),
                                  replace=False))
        frame = frame.iloc[keep].reset_index(drop=True)

    pool, skipped = resolve_pool(
        tuple(arguments.pool) if arguments.pool else None,
        bool(arguments.with_tabpfn),
    )
    banner("Expert pool, fitted per item on request-side features")
    print(f"cases            : {len(frame)}")
    print(f"experts          : {len(pool)} -> {', '.join(pool)}")
    for name, reason in skipped.items():
        print(f"  unavailable    : {name} - {reason}")

    request = build_request_features(frame).to_numpy(dtype=float)
    long_frame = to_item_long(frame, list(ITEMS))
    item_codes = long_frame["item_index"].to_numpy()
    truth = long_frame["y"].to_numpy(dtype=float)
    lookup = {cid: i for i, cid in enumerate(frame["case_id"])}
    unit_case = np.array([lookup[c] for c in long_frame["case_id"]])
    # The request-side matrix has one row per case while the feature rows below are
    # indexed per case-item unit, so the features are expanded to the unit level
    # before anything is indexed by a unit position.
    request_units = request[unit_case]
    n_cases = len(frame)

    records: List[Dict[str, object]] = []
    for seed in (int(s) for s in arguments.seeds):
        folds = KFold(n_splits=int(arguments.folds), shuffle=True,
                      random_state=seed)
        for fold, (train_cases, test_cases) in enumerate(
            folds.split(np.arange(n_cases))
        ):
            train_case_set = set(int(c) for c in train_cases)
            is_train = np.array([c in train_case_set for c in unit_case])
            for position, item in enumerate(ITEMS):
                train_rows = np.flatnonzero(is_train & (item_codes == position))
                test_rows = np.flatnonzero(~is_train & (item_codes == position))
                if train_rows.size < arguments.min_train_units or test_rows.size == 0:
                    continue
                for name in pool:
                    model = build_expert(name, seed=seed)
                    model.fit(request_units[train_rows], truth[train_rows])
                    prediction = np.clip(model.predict(request_units[test_rows]),
                                        0.0, 1.0)
                    records.append({
                        "seed": seed,
                        "fold": fold,
                        "item": str(item),
                        "expert": name,
                        "n_train": int(train_rows.size),
                        "n_test": int(test_rows.size),
                        "MAE": float(np.mean(np.abs(truth[test_rows] - prediction))),
                        "RMSE": float(np.sqrt(np.mean((truth[test_rows] - prediction) ** 2))),
                    })

    table = pd.DataFrame(records)
    if table.empty:
        print("no item had enough training units to be fitted; increase --max-cases")
        return 1

    summary = (
        table.groupby("expert", as_index=False)
        .agg(mean_MAE=("MAE", "mean"), mean_RMSE=("RMSE", "mean"),
             n_item_folds=("MAE", "size"))
        .sort_values("mean_MAE")
        .reset_index(drop=True)
    )
    banner("Averaged over items")
    print(f"  {'expert':<12}{'mean item MAE':>15}{'mean item RMSE':>16}{'cells':>8}")
    for _, row in summary.iterrows():
        print(f"  {row['expert']:<12}{row['mean_MAE']:>15.4f}"
              f"{row['mean_RMSE']:>16.4f}{int(row['n_item_folds']):>8}")

    banner("Mean absolute error by item")
    by_item = (
        table.pivot_table(index="item", columns="expert", values="MAE", aggfunc="mean")
    )
    by_item = by_item.reindex([i for i in ITEMS if i in by_item.index])
    header = "  " + f"{'item':<20}" + "".join(f"{c[:8]:>9}" for c in by_item.columns)
    print(header)
    for item, row in by_item.iterrows():
        cells = "".join(
            f"{row[c]:>9.4f}" if np.isfinite(row[c]) else f"{'-':>9}"
            for c in by_item.columns
        )
        print(f"  {str(item):<20}{cells}")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "expert_item_errors.csv"
    table.to_csv(path, index=False)
    print()
    print(f"wrote {display_path(path)}")
    print("note: these figures come from the synthetic demonstration dataset and are "
          "not comparable with any value reported in the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
