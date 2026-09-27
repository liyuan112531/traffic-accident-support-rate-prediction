"""Build the published data dictionary.

The dictionary is generated from the code rather than typed by hand, so the
documented column names cannot drift from the names the code actually produces.

The file documents schema, types, roles and disclosure status only. It contains no
data values, no case identifiers, no personal information and no empirical
results; a dictionary row describes a column, never a record.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.experts.registry import EXPERT_NAMES, SPECS  # noqa: E402
from src.features.claim_features import (  # noqa: E402
    request_feature_names,
    request_feature_semantics,
)
from src.features.item_indicators import item_indicator_names  # noqa: E402
from src.features.meta_features import (  # noqa: E402
    DEFAULT_DISAGREEMENT_SPEC,
    disagreement_names,
)
from src.preprocessing.schema import ITEMS, N_ITEMS  # noqa: E402
from src.utils.cli import resolve_output_file  # noqa: E402

FROZEN_POOL = (
    "median", "ridge", "rf", "xgb", "lgb", "catboost", "c3",
    "extratrees", "histgb", "svr_rbf", "tabpfn",
)

COLUMNS: List[str] = [
    "name",
    "dtype",
    "description",
    "role",
    "allowed_values_or_encoding",
    "pre_judgment_information",
    "used_in_expert_models",
    "used_in_meta_learner",
    "public_release_safe",
]

PRESENT = "yes"
ABSENT = "no"


def _row(name: str, dtype: str, description: str, role: str, allowed: str,
         pre_judgment: str, in_experts: str, in_meta: str,
         safe: str = PRESENT) -> Dict[str, str]:
    return {
        "name": name,
        "dtype": dtype,
        "description": description,
        "role": role,
        "allowed_values_or_encoding": allowed,
        "pre_judgment_information": pre_judgment,
        "used_in_expert_models": in_experts,
        "used_in_meta_learner": in_meta,
        "public_release_safe": safe,
    }


def build_rows() -> List[Dict[str, str]]:
    """Every documented column of the release, in a deterministic order."""
    rows: List[Dict[str, str]] = []

    rows.append(_row(
        "case_id", "string",
        "Synthetic case identifier of the form SYNTH-0001. Not derived from any "
        "real case or judgment number.",
        "identifier", "pattern SYNTH-nnnn", PRESENT, ABSENT, ABSENT,
    ))
    rows.append(_row(
        "doc_year", "integer", "Year of the judgment, used only to define the "
        "temporal extrapolation split.",
        "procedural context", "2022 or 2023", PRESENT, ABSENT, ABSENT,
    ))
    rows.append(_row(
        "doc_region", "string", "Court region. Constant in the synthetic file.",
        "procedural context", "SYNTHETIC-REGION", PRESENT, ABSENT, ABSENT,
    ))
    rows.append(_row(
        "doc_proc_level", "string", "Instance of the proceeding.",
        "procedural context", "first instance", PRESENT, ABSENT, ABSENT,
    ))
    rows.append(_row(
        "injury_severity_synthetic", "categorical",
        "Illustrative ordinal severity label. Not used by any model in this "
        "repository: the manuscript excludes post-outcome case attributes of this "
        "class from every feature set. Included only to show the schema.",
        "illustrative only", "none | minor | moderate | severe | fatal",
        "ambiguous", ABSENT, ABSENT,
    ))

    for item in ITEMS:
        rows.append(_row(
            f"{item}_claim", "float", f"Claimed amount for the {item} head of claim, in yuan.",
            "request side", ">= 0; 0 means the item was not claimed",
            PRESENT, PRESENT, "indirectly",
        ))
    for item in ITEMS:
        rows.append(_row(
            f"{item}_award", "float", f"Awarded amount for the {item} head of claim, in yuan.",
            "outcome (target side)", ">= 0; <= <item>_claim",
            ABSENT, ABSENT, ABSENT,
        ))
    rows.append(_row(
        "claim_total", "float", "Sum of the sixteen claimed amounts.",
        "request side", "> 0", PRESENT, PRESENT, "indirectly",
    ))
    rows.append(_row(
        "award_total", "float", "Sum of the sixteen awarded amounts.",
        "outcome (target side)", ">= 0; <= claim_total", ABSENT, ABSENT, ABSENT,
    ))

    for item in ITEMS:
        rows.append(_row(
            f"r_{item}", "float",
            f"Item-level support rate {item}_award / {item}_claim. Undefined, "
            "encoded as NULL, when the item was not claimed.",
            "target", "[0, 1] or NULL", ABSENT, ABSENT, ABSENT,
        ))
    rows.append(_row(
        "total_support_rate", "float",
        "Case-level support rate award_total / claim_total.",
        "target", "[0, 1]", ABSENT, ABSENT, ABSENT,
    ))
    rows.append(_row(
        "w_<item>", "float",
        "Case-specific aggregation weight claim_<item> / claim_total. Zero for "
        "items that were not claimed.",
        "aggregation weight", "[0, 1]; the sixteen weights sum to 1",
        PRESENT, ABSENT, ABSENT,
    ))

    # The meta representation addresses the request side positionally: its thirty-three
    # claim-side columns are named claim_0 ... claim_32. Sixteen of those are the same
    # quantities as the case-table columns documented above, under different names, so
    # the two namings are cross-referenced here rather than left looking disjoint. The
    # block is listed in full -- all thirty-three positions -- because a reader who
    # counts the claim-side columns of the meta matrix has to arrive at thirty-three.
    for position, (name, kind, _meaning) in enumerate(request_feature_semantics()):
        if position < N_ITEMS:
            description = (
                f"Request-side feature: the {ITEMS[position]} claimed amount. This is "
                f"the meta-matrix name for the column documented above as "
                f"{ITEMS[position]}_claim."
            )
        elif position == N_ITEMS:
            description = (
                "Request-side feature: claim_total, the sum of the sixteen claimed "
                "amounts."
            )
        else:
            item = ITEMS[position - N_ITEMS - 1]
            description = (
                f"Request-side feature: the log1p transform of the {item} claimed "
                "amount. The transform is monotone, so it adds no information beyond "
                "the raw amount; it changes the geometry the tree splits operate on."
            )
        rows.append(_row(
            name, "float", description,
            "meta input, claim-side block", ">= 0", PRESENT, PRESENT, PRESENT,
        ))
    for name in item_indicator_names():
        rows.append(_row(
            name, "float", "One-hot indicator of the item this case-item unit "
            "belongs to.",
            "meta input, claim-side block", "0 or 1", PRESENT, ABSENT, PRESENT,
        ))
    names = disagreement_names(FROZEN_POOL, DEFAULT_DISAGREEMENT_SPEC)
    rows.append(_row(
        names[0], "float",
        "Signed difference between the anchor expert and a comparison expert. "
        "Measures pool disagreement, not calibrated uncertainty.",
        "meta input, disagreement block", "[-1, 1]", PRESENT, ABSENT, PRESENT,
    ))
    for name in names[1:]:
        rows.append(_row(
            name, "float",
            "Expert-disagreement summary statistic. Measures pool disagreement, "
            "not calibrated uncertainty.",
            "meta input, disagreement block",
            "[-1, 1]" if name.startswith("diff_") else "[0, inf)", PRESENT, ABSENT,
            PRESENT,
        ))
    for name in EXPERT_NAMES:
        rows.append(_row(
            f"pred_{name}", "float",
            f"Support-rate prediction of the {name} expert "
            f"({SPECS[name].family}).",
            "meta input, expert-prediction block", "[0, 1]", PRESENT, PRESENT,
            PRESENT,
        ))
    rows.append(_row(
        "total_support_rate_prediction", "float",
        "Case-level support-rate prediction after claim-weighted aggregation of the "
        "item-level predictions.",
        "output", "[0, 1]", ABSENT, ABSENT, PRESENT,
    ))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV path; defaults to "
                             "data/schema/data_dictionary.csv")
    arguments = parser.parse_args(argv)

    try:
        output = resolve_output_file(
            arguments.output,
            REPOSITORY_ROOT / "data" / "schema" / "data_dictionary.csv",
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    print(f"  documented columns: {len(rows)}")
    roles = {}
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    for role, count in sorted(roles.items(), key=lambda kv: -kv[1]):
        print(f"  {role:<34} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
