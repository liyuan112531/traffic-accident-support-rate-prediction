"""Generate the synthetic demonstration dataset.

Why this file exists
--------------------
The empirical study rests on court judgment documents that contain personally
identifiable information and that are not publicly accessible from the original
source. Those documents, and every case-level derivative of them, are therefore
not distributed with this repository.

To let the published code be executed from end to end, this script generates a
dataset from a parametric model instead. Nothing here is sampled, copied,
perturbed or summarised from the real corpus: every value is drawn from
distributions declared in this file, using a fixed random seed.

What the synthetic dataset is for
---------------------------------
It exercises the code path. It reproduces the *structure* of the problem: sixteen
paired claim and award amounts per case, a support rate that is bounded in
``[0, 1]`` with point mass at both ends, case-level heterogeneity, and a year
effect so that the temporal split is non-degenerate.

It does not reproduce the *results*. The reported performance of the manuscript
depends on the real data, and no number produced from this file should be
compared with any figure in the manuscript.

Usage
-----
::

    python scripts/make_synthetic_data.py
    python scripts/make_synthetic_data.py --n-cases 2000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The script is run as a file, so the repository root is put on the import path
# explicitly. No absolute path is used; the root is derived from this file.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.preprocessing.schema import ITEMS, N_ITEMS  # noqa: E402
from src.utils.cli import resolve_output_file  # noqa: E402

#: Probability that a case claims each item. The pattern reflects the qualitative
#: observation that a traffic accident case nearly always claims treatment costs
#: whereas burial and appraisal costs arise in a minority of cases. The rare items
#: are set higher than their real-world frequency on purpose: the demonstration
#: runs at a few hundred cases, and an item claimed in under ten per cent of cases
#: would have too few training units to fit at all. These numbers are not estimates
#: of the real frequencies and should not be read as such.
CLAIM_PROBABILITY: dict = {
    "medical": 0.93,
    "future_medical": 0.30,
    "hospital_meal": 0.42,
    "nutrition": 0.32,
    "nursing": 0.44,
    "lost_income": 0.55,
    "transport": 0.72,
    "accommodation": 0.20,
    "disability_comp": 0.46,
    "assistive_device": 0.22,
    "death_comp": 0.16,
    "funeral": 0.14,
    "mental_solatium": 0.58,
    "property_loss": 0.62,
    "appraisal": 0.18,
    "other": 0.16,
}

#: Median claimed amount per item, in yuan, and the lognormal shape parameter.
CLAIM_MEDIAN: dict = {
    "medical": 18000.0,
    "future_medical": 12000.0,
    "hospital_meal": 1500.0,
    "nutrition": 3000.0,
    "nursing": 9000.0,
    "lost_income": 16000.0,
    "transport": 2000.0,
    "accommodation": 2500.0,
    "disability_comp": 120000.0,
    "assistive_device": 8000.0,
    "death_comp": 900000.0,
    "funeral": 45000.0,
    "mental_solatium": 30000.0,
    "property_loss": 6000.0,
    "appraisal": 2400.0,
    "other": 5000.0,
}

CLAIM_SIGMA: float = 0.85

#: Support-rate generator per item: probability of a complete rejection, of full
#: support, the mean of the interior mass, and its concentration. A high
#: concentration gives a narrow distribution around the mean, a low one gives a
#: spread-out one. The parameters are chosen so the difficulty ranking resembles
#: the qualitative pattern in the literature: standardised, receipt-backed items
#: are decided more consistently than discretionary ones.
SUPPORT_SPEC: dict = {
    "medical": (0.04, 0.14, 0.88, 7.0),
    "future_medical": (0.18, 0.05, 0.60, 3.5),
    "hospital_meal": (0.08, 0.12, 0.80, 6.0),
    "nutrition": (0.12, 0.08, 0.70, 4.5),
    "nursing": (0.08, 0.12, 0.78, 5.5),
    "lost_income": (0.10, 0.10, 0.74, 5.0),
    "transport": (0.06, 0.20, 0.84, 6.5),
    "accommodation": (0.18, 0.06, 0.62, 4.0),
    "disability_comp": (0.08, 0.14, 0.82, 6.0),
    "assistive_device": (0.15, 0.10, 0.68, 4.5),
    "death_comp": (0.04, 0.32, 0.90, 8.0),
    "funeral": (0.05, 0.28, 0.88, 7.5),
    "mental_solatium": (0.22, 0.04, 0.58, 3.0),
    "property_loss": (0.12, 0.10, 0.72, 5.0),
    "appraisal": (0.06, 0.24, 0.86, 6.5),
    "other": (0.16, 0.10, 0.66, 4.0),
}

#: Severity categories used only to label the illustrative severity column.
SEVERITY_LABELS: tuple = ("none", "minor", "moderate", "severe", "fatal")

#: Cumulative cut points on the latent severity scale.
SEVERITY_CUTS: tuple = (0.0, 0.35, 0.75, 1.35, 2.1)

#: Procedural levels the synthetic cases are drawn from, and their probabilities.
#: Included so that a procedural column is present and non-degenerate: a column that
#: never varies cannot be grouped by, and grouping by the procedure is one of the ways
#: a reader might want to slice the output.
PROCEDURE_LEVELS: tuple = ("first instance", "second instance")
PROCEDURE_PROBABILITY: tuple = (0.78, 0.22)

#: Probability that a synthetic case is dated to the earlier of the two years. The
#: two years are the ones the study corpus spans.
YEAR_2022_PROBABILITY: float = 0.62


def _latent_severity(rng: np.random.Generator, n_cases: int) -> np.ndarray:
    """Case-level latent severity driving both claim size and support rate."""
    return rng.normal(loc=0.0, scale=1.0, size=n_cases)


def _severity_category(severity: np.ndarray) -> np.ndarray:
    """Map latent severity onto an ordinal label."""
    index = np.digitize(severity, SEVERITY_CUTS[1:], right=False)
    index = np.clip(index, 0, len(SEVERITY_LABELS) - 1)
    return np.asarray([SEVERITY_LABELS[i] for i in index], dtype=object)


def _draw_claims(rng: np.random.Generator, n_cases: int,
                 severity: np.ndarray) -> np.ndarray:
    """Claimed amounts, shape ``(n_cases, n_items)``; zero means not claimed."""
    claims = np.zeros((n_cases, N_ITEMS), dtype=float)
    for j, item in enumerate(ITEMS):
        claimed = rng.random(n_cases) < CLAIM_PROBABILITY[item]
        median = CLAIM_MEDIAN[item]
        # Severity inflates the claim, with a mild exponent so the effect stays
        # plausible for the largest heads of claim.
        inflation = np.exp(0.18 * severity)
        draws = rng.lognormal(mean=np.log(median), sigma=CLAIM_SIGMA, size=n_cases)
        claims[:, j] = np.where(claimed, np.maximum(draws * inflation, 1.0), 0.0)
    return np.round(claims, 2)


def _draw_support_rates(rng: np.random.Generator, n_cases: int,
                        severity: np.ndarray) -> np.ndarray:
    """Support rates on the claimed items, shape ``(n_cases, n_items)``."""
    rates = np.full((n_cases, N_ITEMS), np.nan, dtype=float)
    for j, item in enumerate(ITEMS):
        p_zero, p_full, interior_mean, concentration = SUPPORT_SPEC[item]
        draw = rng.random(n_cases)
        zero = draw < p_zero
        full = (draw >= p_zero) & (draw < p_zero + p_full)

        # Interior mass: a Beta whose mean is pulled down for severe cases, which
        # creates a genuine case-level effect rather than item-level noise alone.
        shift = 0.05 * severity
        mean = np.clip(interior_mean - shift, 0.05, 0.95)
        concentration_vec = np.full(n_cases, concentration, dtype=float)
        alpha = mean * concentration_vec
        beta = (1.0 - mean) * concentration_vec
        interior = rng.beta(alpha, beta)

        value = np.where(zero, 0.0, np.where(full, 1.0, interior))
        rates[:, j] = value
    return np.round(rates, 6)


def generate(n_cases: int = 600, seed: int = 20260927) -> pd.DataFrame:
    """Return the synthetic case table.

    Columns
    -------
    ``case_id``
        Synthetic identifier of the form ``SYNTH-0001``. It has no relation to any
        real case.
    ``doc_year``, ``doc_region``, ``doc_proc_level``
        Procedural context used to define the temporal split and to give the output
        something to group by. Region is constant by construction and carries no
        personal information; the procedure varies across the two procedural levels.
    ``injury_severity_synthetic``
        Illustrative ordinal label. It is *not* used by any model in this
        repository; the manuscript excludes this class of post-outcome attribute
        from every feature set.
    ``<item>_claim``, ``<item>_award``
        The sixteen paired amounts. Awards follow from the claimed amount and the
        drawn support rate, so the target is internally consistent.
    """
    if n_cases < 50:
        raise ValueError("request at least 50 cases so the folds remain meaningful")
    rng = np.random.default_rng(int(seed))

    severity = _latent_severity(rng, n_cases)
    claims = _draw_claims(rng, n_cases, severity)
    rates = _draw_support_rates(rng, n_cases, severity)
    rates = np.where(claims > 0, rates, 0.0)
    awards = np.round(claims * rates, 2)

    # A mild year effect: later cases are supported slightly more often, so the
    # temporal split is informative rather than degenerate.
    year = np.where(rng.random(n_cases) < YEAR_2022_PROBABILITY, 2022, 2023)
    year_lift = np.where(year == 2023, 0.02, 0.0)
    awards = np.round(np.minimum(claims, awards * (1.0 + year_lift[:, None])), 2)

    procedure = np.asarray(
        rng.choice(PROCEDURE_LEVELS, size=n_cases, p=PROCEDURE_PROBABILITY),
        dtype=object,
    )

    data = {
        "case_id": [f"SYNTH-{i + 1:04d}" for i in range(n_cases)],
        "doc_year": year.astype(int),
        "doc_region": ["SYNTHETIC-REGION"] * n_cases,
        "doc_proc_level": procedure,
        "injury_severity_synthetic": _severity_category(severity),
    }
    for j, item in enumerate(ITEMS):
        data[f"{item}_claim"] = claims[:, j]
        data[f"{item}_award"] = awards[:, j]
    frame = pd.DataFrame(data)
    frame["claim_total"] = frame[[f"{i}_claim" for i in ITEMS]].sum(axis=1).round(2)
    frame["award_total"] = frame[[f"{i}_award" for i in ITEMS]].sum(axis=1).round(2)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-cases", type=int, default=600,
                        help="number of synthetic cases to generate")
    parser.add_argument("--seed", type=int, default=20260927,
                        help="random seed; the default reproduces the published file")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV path; defaults to "
                             "data/synthetic/synthetic_cases.csv")
    arguments = parser.parse_args(argv)

    try:
        output = resolve_output_file(
            arguments.output,
            REPOSITORY_ROOT / "data" / "synthetic" / "synthetic_cases.csv",
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(arguments.n_cases, arguments.seed)
    frame.to_csv(output, index=False)

    claim_columns = [f"{i}_claim" for i in ITEMS]
    defined = (frame[claim_columns] > 0).sum().sum()
    print(f"wrote {output}")
    print(f"  cases            : {len(frame)}")
    print(f"  columns          : {frame.shape[1]}")
    print(f"  defined units    : {int(defined)} of {len(frame) * N_ITEMS}")
    print(f"  claim total (CNY): median {frame['claim_total'].median():,.0f}, "
          f"max {frame['claim_total'].max():,.0f}")
    print(f"  years            : {sorted(frame['doc_year'].unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
