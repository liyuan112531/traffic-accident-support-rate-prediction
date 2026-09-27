# Synthetic demonstration dataset

`synthetic_cases.csv` is a synthetic dataset of 600 cases with all sixteen claim and
award columns. It exists so that the published code can be executed from end to end by
anyone, and so that the tests have data to run on.

> The synthetic dataset is provided solely to demonstrate the execution of the public
> code. It was not used to generate the empirical results reported in the manuscript.

## How it was made

By `../scripts/make_synthetic_data.py`, from parametric distributions, with the fixed
seed `20260927`. Every value is drawn inside that script. Nothing is sampled, copied,
perturbed or summarised from the real corpus, and no real record contributed to it.

The generator works in four steps.

**Latent severity.** Each case draws a standard normal severity. It drives both the
size of the claims and the level of support, so the case-level variation is genuine
rather than item-level noise alone.

**Claims.** For each item, the case claims with a fixed probability; the claimed amount
is lognormal with a per-item median, inflated mildly by latent severity, and rounded to
two decimals. A zero means the item was not claimed.

**Support rates.** For each item, a fixed probability of complete rejection, a fixed
probability of full support, and otherwise a Beta draw whose mean is pulled down for
severe cases. This produces the shape the real target has: bounded in `[0, 1]` with
point mass at both ends and a dispersed interior.

**Awards and context.** The award is the claimed amount times the drawn rate, rounded,
so the target is internally consistent and no award exceeds its claim. A mild year
effect lifts the later year slightly so that the temporal split is not degenerate, and
a procedural level is drawn for each case.

## Columns

| Column | Meaning |
|---|---|
| `case_id` | `SYNTH-0001` … `SYNTH-0600`. Has no relation to any real case. |
| `doc_year` | 2022 or 2023 |
| `doc_region` | Constant `SYNTHETIC-REGION` |
| `doc_proc_level` | `first instance` or `second instance` |
| `injury_severity_synthetic` | One of `none`, `minor`, `moderate`, `severe`, `fatal` |
| `<item>_claim` | Claimed amount, CNY; zero means not claimed |
| `<item>_award` | Awarded amount, CNY; zero with a positive claim means refused |
| `claim_total`, `award_total` | Row sums, for convenience |

`injury_severity_synthetic` is an illustrative ordinal label. No model in this
repository uses it, and the manuscript excludes this class of post-judgment attribute
from every feature set. It is present because the real data has a comparable graded
attribute and the schema should show where it sits.

## Shape

At the default seed the file has 3,813 defined case-item units out of 9,600 possible,
and a median case-level claim total of about 155,000 CNY. The item-level coverage is
deliberately uneven, because real compensation claims are unevenly distributed across
heads of loss: `medical` is claimed in almost every case, while `funeral`,
`appraisal`, `death_comp` and `other` are claimed in a small minority.

**The item frequencies are not real.** The rare items were given higher claim
probabilities than they have in the study corpus on purpose: the demonstration runs at
a few hundred cases, and an item claimed in under ten per cent of them would have too
few training units to be fitted at all. Every probability in `CLAIM_PROBABILITY` is a
construction parameter. None of them is an estimate of anything and none should be read
as a frequency observed in real judgments.

## What may be concluded from it

The dataset supports one kind of claim: that the code executes, produces the meta
representation with the documented block widths — 73 columns for the frozen
eleven-expert pool, 72 for this demonstration's ten-expert pool, since TabPFN is
disabled by default — keeps every prediction inside `[0, 1]`, balances the item weights
exactly, and satisfies the aggregation identity to machine precision. The tests assert
those properties and the demonstration reports them.

It does not support any claim about accuracy. Because the item frequencies are
construction parameters rather than observations, and because the generator is not the
data-generating process of any real judgment, no error figure computed on this dataset
is interpretable as a level of model performance.

A near-zero or negative R-squared on this dataset is a property of the construction and
not of the method. The latent severity drives the support rate, while the observable
request-side features are related to severity only weakly and noisily; the generator
also injects rejection and full-support mass that the features do not predict. Those
two facts put a low ceiling on how much of the item-level variance any function of the
request side can explain. The consequence is that a model fitted here can look worse
than a per-item constant, and that is expected.

## Using a different seed

```bash
python scripts/make_synthetic_data.py --n-cases 1200 --seed 7 \
    --output /tmp/synthetic_seed7.csv
python scripts/run_synthetic_demo.py --data /tmp/synthetic_seed7.csv
```

A different seed gives a different dataset with the same structure. The structural
properties the demonstration reports should be unchanged; if they are not, that is a
finding about the code rather than about the data.
