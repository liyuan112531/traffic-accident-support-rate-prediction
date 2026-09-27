# Reproducibility

The code is at
<https://github.com/liyuan112531/traffic-accident-support-rate-prediction>.

## The claim this repository makes

This repository supports **methodological reproducibility**. Every stage of the
method is implemented, inspectable, executable and covered by tests, and the
dimensional and algebraic properties the method depends on are asserted rather than
asserted-in-prose.

It does **not** claim empirical reproducibility. The case-level dataset is not
distributed, so the numerical results reported in the manuscript cannot be
regenerated from this repository by anyone without access to the restricted
underlying data.

> The repository does not redistribute the empirical dataset. Therefore, users
> cannot reproduce the exact numerical results reported in the manuscript without
> access to the restricted underlying data. The repository is intended to support
> methodological inspection, implementation reuse, and execution of the provided
> synthetic demonstration.

The synthetic demonstration proves that the code executes end to end. It does not
reproduce any reported result. No script in this repository regenerates a figure or
a table from the manuscript, and no file in this repository contains a manuscript
figure hard-coded.

## What is reproducible, and how to check it

Run the tests:

```bash
python -m pytest tests/ -q
```

Then run the demonstration and compare its verification block:

```bash
python scripts/run_synthetic_demo.py
```

Each property below is reproduced by a script and asserted by a test.

| Property | Value | Where it is checked |
|---|---|---|
| The item universe | 16 items, in canonical order | `tests/test_preprocessing.py`, `tests/test_aggregation.py` |
| The request-side block | 33 columns = 16 raw + 1 total + 16 `log1p` | `tests/test_preprocessing.py` |
| The claim side | 49 columns = 33 request + 16 indicators | `tests/test_preprocessing.py` |
| The meta representation | 73 columns = 11 experts + 13 disagreement + 49 claim side | `tests/test_preprocessing.py`, `tests/test_synthetic_pipeline.py` |
| Block boundaries | `(0, 11)`, `(11, 24)`, `(24, 73)` | `tests/test_preprocessing.py` |
| Item balancing | each item's total training weight is exactly `1/16`, in every fold | `tests/test_item_balanced_l1.py`, `tests/test_synthetic_pipeline.py` |
| Weight normalisation | per-case aggregation weights sum to one | `tests/test_aggregation.py` |
| Aggregation identity | `sum_j w_ij r_ij == A_total_i / C_total_i` | `tests/test_aggregation.py` |
| Admissible range | every predicted rate lies in `[0, 1]` | `tests/test_synthetic_pipeline.py` |
| Structural absence | `claim == 0` gives `NaN`; `claim > 0, award == 0` gives `0.0` | `tests/test_preprocessing.py`, `tests/test_aggregation.py` |
| Determinism | the same configuration gives identical predictions | `tests/test_synthetic_pipeline.py` |

The conservation identity is reported by the demonstration rather than asserted on a
tolerance chosen to make it pass. On the synthetic dataset it holds to a maximum
absolute deviation of order `1e-16`, which is the accumulated rounding error of
double-precision summation, and the run reports the value it actually observed.

## Two configurations, one method

The meta representation is 73 columns wide for the frozen eleven-expert pool. The
repository ships two configurations of that pool:

| Configuration | Pool | Meta width | Selected by |
|---|---|---|---|
| **Lightweight demonstration** | 10 experts | 72 | the default; TabPFN disabled |
| **Full paper configuration** | 11 experts | 73 | `--with-tabpfn` |

TabPFN is the only difference, because it is the only expert that needs weights
downloaded on first use, and those weights are not distributed with this repository.
The empirical experiments reported in the manuscript use the full configuration.

The width the method is defined by is the frozen pool's 73, and that identity is
asserted against the frozen pool rather than against whatever pool a given run
happened to instantiate. A demonstration run that reaches only ten experts therefore
still fails the suite if the structure breaks — it does not quietly redefine the
method to be what it ran. Every run reports both widths so the two can never be
confused. When you enable the full configuration, expect the first run to fetch the
TabPFN weights and to be slower than the runs after it.

## Reproducing a run

A run is determined by:

1. the case table, which must follow the schema in `data/schema/data_dictionary.csv`;
2. `config/experiment_config.yaml`, which mirrors the protocol, the pre-registered
   hyperparameters, the meta-representation decomposition and the aggregation
   definition. The header of that file lists the module each section mirrors; the code
   is authoritative and the file is read only for the `demonstration` section;
3. the command-line arguments, of which `--seeds`, `--folds`, `--inner-folds` and
   `--max-cases` are the ones that change the result.

Two runs with the same three inputs produce identical predictions. This is
guaranteed by `src.utils.env.configure_determinism`, which pins the environment
variables that let thread scheduling and hash randomisation affect floating-point
summation order, and which every script calls before importing a modelling library.

Reproducibility across machines is a weaker property. The final decimal of a tree
ensemble's prediction depends on the BLAS and threading libraries present, because
floating-point addition is not associative and different libraries sum in different
orders. Expect agreement to well within any reported precision, and check the
verification block and the shape of the tables rather than the last digit of every
value.

## Environment

The versions the code was developed and executed against are pinned in
`requirements.txt` and `environment.yml`. They are not incidental: LightGBM's split
search, XGBoost's histogram builder, CatBoost's ordered boosting and scikit-learn's
forest implementation have all changed behaviour across releases, and `shap`'s
attribution output has changed shape across releases as well.

`numpy` is held at 1.26.x deliberately. Several of the pinned libraries were built
against the 1.x ABI, and 2.x additionally changes scalar-promotion rules that the
metric code depends on.

`.github/workflows/tests.yml` applies the same pins on Python 3.12 for every push and
pull request: the fast tests, the end-to-end tests and the synthetic demonstration, in
three parallel jobs. It upgrades nothing, and it does not install the optional TabPFN
extra, whose weights are not distributed.

## What is not reproducible

- **The manuscript's reported numbers.** The underlying data is restricted; see
  `docs/data_access.md`.
- **The study's real item frequencies.** The synthetic dataset uses construction
  parameters chosen so every item has enough training units, not the frequencies
  observed in the study data.
- **The study's empirical splits.** The synthetic demonstration uses a bounded
  number of cases, folds and seeds so that it finishes in minutes. Those settings are
  recorded in `config/experiment_config.yaml` under `demonstration` and are reported
  in every run's settings, so a demonstration run can never be mistaken for a run of
  the study protocol.
- **The fitted models.** Model weights and caches are not included in the
  repository, and no fitted artefact is committed.
- **The eleventh expert out of the box.** TabPFN needs pretrained weights that are
  not distributed, so the default pool reaches ten experts. Pass `--with-tabpfn` to
  instantiate the full eleven-expert configuration.

## Reporting a discrepancy

If a documented property does not hold on your machine, the useful information is:
the Python and library versions, the exact command, the run settings block printed at
the start of the run, and the failing assertion. Please include the settings block,
because it records the pool that was actually available and any expert that was
skipped.
