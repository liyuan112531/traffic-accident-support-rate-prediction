# Synthetic demonstration

A runnable demonstration of the whole pipeline on the synthetic dataset. Its purpose
is narrow and worth stating plainly: it shows that the code executes, that the
dimensional structure is what the documentation says it is, and that the algebraic
properties the method depends on hold numerically.

It does not reproduce any result reported in the manuscript. The dataset it runs on is
synthetic, the scale is a fraction of the study protocol, and no figure it prints
should be compared with anything in the paper.

## Running it

```bash
python run_demo.py
```

or, from the repository root:

```bash
python scripts/run_synthetic_demo.py
python examples/synthetic_demo/run_demo.py --max-cases 120 --folds 2
```

The first invocation requires `data/synthetic/synthetic_cases.csv`, which is shipped.
It takes a few minutes at the default demonstration scale, because the expert pool is
genuinely trained on every item in every fold. Use `--max-cases` to shorten it.

```bash
python run_demo.py --help
```

lists every option. The ones worth knowing:

| Option | Effect |
|---|---|
| `--max-cases` | restrict to a deterministic subset of cases |
| `--folds`, `--inner-folds` | change the number of outer and inner partitions |
| `--seeds` | one run per seed |
| `--pool` | run an explicit expert subset |
| `--with-tabpfn` | include TabPFN, which downloads model weights on first use |
| `--bootstrap`, `--permutations` | number of resampling replicates |
| `--write-predictions` | also write the per-unit and per-case prediction tables |
| `--output` | where the artefact tables are written |

## What it does

1. Loads the synthetic case table and validates it against the schema contract.
2. Builds the request-side features and the long (case, item) frame.
3. Fits the expert pool item by item, on case-level folds shared by every expert.
4. Cross-fits the experts within each training fold to obtain genuine out-of-fold
   predictions.
5. Assembles the 73-dimensional meta representation and verifies its block widths.
6. Computes the item-balanced training weights and verifies that every item's total
   weight is `1/16`.
7. Trains the four meta heads, then aggregates the item predictions to the case level
   with the claim-structure weights.
8. Evaluates item-level and case-level metrics, and reports the conservation identity.
9. Runs the paired bootstrap and the paired permutation test between variants.
10. Reports a verification block and exits non-zero if any check fails.

## Reports

The verification block at the end is the part to read. On the default configuration it
reports:

| Check | Meaning |
|---|---|
| meta representation has the documented block widths | the frozen eleven-expert pool gives `73 = 11 + 13 + 49`, with the claim side itself `33 + 16`; this run's own width is `10 + 13 + 49 = 72`, because TabPFN is not in the default pool |
| item-balanced weights are exactly balanced within every fold | every item's share of the training weight is `1/16` |
| every unit receives a strictly positive weight | no unit is silently dropped by the weighting |
| aggregation reproduces the case-level rate to machine precision | the conservation identity holds |
| at least one item-level metric was computed | the item-level evaluation produced output |
| at least one case-level metric was computed | the case-level evaluation produced output |
| all predictions lie inside `[0, 1]` | the range of the target is respected |

The conservation identity is reported with the deviation it actually observed rather
than with a tolerance chosen to make it pass. On this dataset the maximum absolute
deviation is of order `1e-16`, which is the accumulated rounding error of
double-precision summation.

Two widths are reported, and the distinction matters. The method is defined by the
**73-column** meta representation built from the eleven-expert frozen pool: that is the
**full paper configuration**, and it is what the empirical experiments reported in the
manuscript use. This demonstration is the **lightweight demonstration configuration**.
It reaches ten of those eleven experts, because TabPFN needs pretrained weights that are
not distributed with this repository, so its own meta matrix is **72 columns** wide. The
run prints both numbers and states which one its own matrix uses; the block identities —
eleven or ten expert predictions, thirteen disagreement features, forty-nine claim-side
features — are unchanged either way, because only the expert-prediction block follows
the pool size. Pass `--with-tabpfn` to run the full eleven-expert configuration; expect
the first such run to fetch the weights and to be slower than the ones after it.

## Expected output

The tables the run writes are committed under `expected_output/`:

| File | Contents |
|---|---|
| `unit_level_metrics.csv` | macro and cell-level metrics for each of the three meta-head variants |
| `aggregated_metrics.csv` | case-level metrics for each aggregation variant |
| `item_level_metrics.csv` | per-item MAE, RMSE and R-squared, per variant |
| `objective_alignment.csv` | per item: share of training rows, and the ratio of effective training weight to macro weight under each weighting scheme |
| `run_diagnostics.json` | run settings, structure check, weight invariants, conservation identity, weight concentration, inference results and the verification block |

Compare your run against these to confirm the installation is sound. Two caveats.

The numbers will not match to the last digit. Floating-point addition is not
associative and BLAS implementations differ in how they sum, so a tree ensemble's
final decimal can differ between machines. Check the verification block and the shape
of the tables, not the last digit of every value.

On a fixed machine, environment and configuration the tables reproduce byte for byte.
The one field that cannot is `elapsed_seconds` in `run_diagnostics.json`, which records
the wall-clock time of the run: expect that single number to move and everything else to
match. `tests/test_cli_contract.py` asserts the byte-level reproducibility of the two
generated inputs, and the shipped tables here were confirmed against a fresh default run.

They will not match at all if you change the configuration. The committed tables
correspond to the demonstration settings in `config/experiment_config.yaml`; passing
`--max-cases` or `--seeds` will change every number, which is expected and is not a
failure.

## Reading the metric tables

Three points about the numbers, so that they are not misread.

**Item-level error is high and R-squared can be negative.** That is a property of the
synthetic construction, not a defect of the method. The generator's latent severity
drives the support rate, while the observable request-side features relate to severity
only weakly; it also injects rejection and full-support mass that the features do not
predict. Both facts put a low ceiling on how much of the item-level variance any
function of the request side can explain, so a model can be worse than a per-item
constant on this data. See `data/synthetic/README.md`.

**Uniform weighting may score better than item-balanced weighting on this data.** The
relative performance of the two depends on the misalignment between the unit
distribution and the macro metric, which is a property of the dataset. On the study
data, where the item frequencies are strongly uneven, the two differ substantially —
see the `objective_alignment.csv` table, which reports the misalignment directly. On
the synthetic data, where the construction parameters were chosen to keep every item
fittable, the frequencies are less uneven and the difference is smaller. Comparing two
variants on this dataset is therefore a check that both code paths run, not a finding
about which is better.

**The intervals and p-values are demonstration-scale.** The default replicate counts
are far below the study protocol's, so the Monte-Carlo error of an interval is
noticeably wider. They are computed by the same routines and are reported so those
routines can be inspected.
