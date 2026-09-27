# Predicting Judicial Support Rates in Traffic Accident Compensation

Reference implementation of a claim-structure-driven hierarchical model for
predicting judicial support rates in traffic accident compensation.

**Repository:** <https://github.com/liyuan112531/traffic-accident-support-rate-prediction>

A traffic accident compensation judgment decides, item by item, how much of each
claimed head of loss is granted. This repository implements a model that follows
that structure rather than ignoring it. A claim is decomposed into sixteen statutory
compensation items. Each item is modelled separately by a pool of heterogeneous
experts. The item-level support rates are then aggregated back to a case-level rate
using the case's own claimed amounts as weights, an aggregation that introduces no
learnable parameters. A meta-learner sits on top of the expert pool, consumes the
expert predictions together with their disagreement and the request side, and is
trained with an item-balanced absolute-error objective.

---

## Table of contents

- [What this repository is](#what-this-repository-is)
- [What this repository is not](#what-this-repository-is-not)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running the individual stages](#running-the-individual-stages)
- [Tests](#tests)
- [Data](#data)
- [Reproducibility](#reproducibility)
- [Results reported in the manuscript](#results-reported-in-the-manuscript)
- [Privacy and data protection](#privacy-and-data-protection)
- [Authors and roles](#authors-and-roles)
- [Funding](#funding)
- [Citation](#citation)
- [License](#license)

---

## What this repository is

- A complete, executable implementation of the method: preprocessing, feature
  construction, the expert pool, the meta-learner, the aggregation layer, the
  evaluation metrics, the resampling routines and the attribution analysis.
- A specification of the data contract: the sixteen-item schema, the 73-dimensional
  meta representation and the data dictionary.
- A runnable demonstration on a **synthetic** dataset that exercises the whole
  chain end to end, together with the expected output tables it produces.
- A tested codebase. The unit tests cover the aggregation invariants, the
  item-balanced weighting scheme, the metric conventions, the schema and the
  feature-block dimensions; one end-to-end test runs the full pipeline on the
  synthetic dataset.

## What this repository is not

- It does **not** contain the empirical dataset. No judgment text, no case-level
  record, no real feature matrix, no real out-of-fold prediction, and no
  case identifier from the study dataset appears anywhere in this repository.
- It does **not** contain fitted models. Model weights and caches are not included
  in the repository.
- It does **not** reproduce the numbers reported in the manuscript. The synthetic
  demonstration proves that the code runs; it does not reproduce the empirical
  results. See [Reproducibility](#reproducibility).

---

## Method

### Sixteen compensation items

Every claim is decomposed into the sixteen statutory items below, in canonical
order. They are the unit at which a judgment is actually decided, and therefore the
unit at which the model predicts.

| # | Item | # | Item |
|---|---|---|---|
| 1 | `medical` | 9 | `disability_comp` |
| 2 | `future_medical` | 10 | `assistive_device` |
| 3 | `hospital_meal` | 11 | `death_comp` |
| 4 | `nutrition` | 12 | `funeral` |
| 5 | `nursing` | 13 | `mental_solatium` |
| 6 | `lost_income` | 14 | `property_loss` |
| 7 | `transport` | 15 | `appraisal` |
| 8 | `accommodation` | 16 | `other` |

The item-level target is the support rate

```
r_ij = Award_ij / Claim_ij
```

with one asymmetry that matters throughout: an item with `Claim_ij = 0` was never
requested, so no rate exists for it and the value is `NaN` — **structural
absence**. An item with `Claim_ij > 0` and `Award_ij = 0` was requested and
refused, which is a real observation with value `0.0` — a **behavioural zero**.
Nothing is imputed anywhere. A structurally absent unit is dropped, not filled.

### Eleven heterogeneous experts

Each item is fitted independently by a pool that spans different inductive biases
rather than varying one model family. A pool of near-identical models would carry
almost no information in the disagreement block the meta-learner consumes.

| Expert | Family | Uniform internal name |
|---|---|---|
| Median baseline | constant predictor, no features | `median` |
| Ridge regression | linear, L2 | `ridge` |
| Random forest | bagged deep trees | `rf` |
| XGBoost | gradient-boosted trees, second-order gains | `xgb` |
| LightGBM | gradient-boosted trees, histogram splits | `lgb` |
| CatBoost | gradient-boosted trees, ordered boosting | `catboost` |
| Convex combination | item-wise simplex-constrained combination of the six base models | `c3` |
| Extremely randomised trees | randomised split thresholds | `extratrees` |
| Histogram gradient boosting | scikit-learn's histogram booster | `histgb` |
| RBF support vector regression | kernel machine | `svr_rbf` |
| TabPFN | prior-data fitted network | `tabpfn` |

`c3` is counted once in the pool even though it composes six base models. It is
defined over its full base pool; if any base model is missing from a run, the expert
is dropped and the reason is reported rather than the expert being silently
redefined over a subset.

Every hyperparameter is the pre-registered fixed value recorded in
`src/experts/registry.py`, and mirrored for readability in
`config/experiment_config.yaml`. There is no
hyperparameter search in this repository. The same parameters are applied to all
sixteen items, to every fold and to every seed, which structurally removes tuning
leakage.

### Two run configurations

The method is defined by the frozen **eleven-expert pool** and the
**73-dimensional** meta representation. The repository instantiates it in two
configurations, and they differ in exactly one place: whether the TabPFN expert is
available.

| | Lightweight demonstration | Full paper configuration |
|---|---|---|
| Expert pool | **10 experts**, TabPFN disabled | **11 experts**, TabPFN enabled |
| Meta representation | **72** columns (`10 + 13 + 49`) | **73** columns (`11 + 13 + 49`) |
| TabPFN weights | not required, nothing is downloaded | downloaded on first use |
| Selected when | `--with-tabpfn` is absent, which is the default | `--with-tabpfn` is passed |
| Used by | `scripts/run_synthetic_demo.py` and the test suite | the empirical experiments reported in the manuscript |

**The empirical experiments reported in the manuscript were run with the full
11-expert / 73-dimensional configuration.** The lightweight demonstration runs the
same pipeline at a reduced scale with that one expert omitted, so that the code can be
executed by anyone without a network download and without redistributing third-party
weights.

The three-block decomposition `[A | B | C]`, the thirteen-column disagreement block and
the forty-nine-column claim side are identical in both configurations; only the width
of block A follows the pool size. Every run therefore reports the width it actually
used, alongside the width of the frozen pool, and never substitutes one for the other.

**TabPFN's pretrained weights are not distributed with this repository.** They are
large binary artefacts fetched at run time, and redistributing them is neither
necessary nor appropriate here. Model weights and caches of any kind are excluded from
the repository; nothing is downloaded unless you explicitly pass `--with-tabpfn`.

### The 73-dimensional meta representation

The meta-learner does not see the raw features. It sees a 73-dimensional
representation assembled as three contiguous blocks `[A | B | C]`:

| Block | Width | Content |
|---|---|---|
| **A** — expert predictions | **11** | one column per expert: `pred_<expert>` |
| **B** — expert disagreement | **13** | signed differences against the anchor expert, dispersion statistics, absolute pairwise differences, the mean of the other experts, and the pool size |
| **C** — claim side | **49** | **33** request-side features plus **16** item indicators |

The 33 request-side features decompose as sixteen raw claimed amounts, the claim
total, and sixteen `log1p`-transformed claimed amounts. The claim side therefore
satisfies `33 + 16 = 49`, and the whole representation satisfies
`11 + 13 + 49 = 73`.

Block A and block B are constructed so the ordering is fixed: the boundaries are
exposed by `src.features.meta_features.meta_block_boundaries` and are used directly
by the attribution analysis, so a block can never be mis-attributed by an
off-by-one.

### Item-Balanced L1

The meta head is a gradient-boosted tree ensemble trained with the absolute-error
objective and item-balanced sample weights:

```
L(theta) = sum_j (1 / J) * sum_{i in I_j} | y_ij - g_theta(z_ij) |        J = 16
```

Each item's total contribution to the loss is exactly `1 / J = 1/16`, regardless of
how many units it contributes. This is what the weighting scheme achieves:

```
w_ij = 1 / (J * n_j),        rescaled so that sum_ij w_ij = N
```

so `sum_{i in I_j} w_ij` is the same for every `j`. Without it, the loss would be
dominated by the items that are claimed most often — on the study dataset the
largest item contributes roughly an order of magnitude more units than the smallest
— and the macro-averaged metric being reported would not correspond to the objective
being optimised.

This is deliberately **not** the same as a class-balancing scheme: balancing classes
would equalise the number of *units* per item, whereas this equalises each item's
total *weight* in the loss. `src/meta_learning/item_balanced.py` provides
`objective_alignment_table`, which reports, per item, the ratio of its effective
training weight to its weight in the macro average; under the item-balanced scheme
that ratio is 1 for every item, and under uniform weighting it is `J * n_j / N`.

### Claim-structure-driven aggregation

The case-level support rate is reconstructed from the item-level predictions using
the case's own claimed amounts as weights:

```
w_ij = C_ij / sum_k C_ik                       (aggregation weight)
R_hat_i = sum_j w_ij * r_hat_ij                (case-level prediction)
```

This identity is exact rather than approximate. Substituting the definitions gives

```
sum_j (C_ij / sum_k C_ik) * (A_ij / C_ij) = sum_j A_ij / sum_k C_ik = A_total_i / C_total_i
```

so the aggregation layer introduces **zero learnable parameters** and no additional
degrees of freedom. The scope of that statement is the aggregation layer only: the
experts and the meta-learner that produce `r_hat_ij` do contain learnable
parameters.

Because the weights come from the case's own request structure, the reconstruction
is automatically case-adaptive — a case that claimed almost entirely medical
expenses is aggregated with a weight vector concentrated on medical expenses, with
no learned routing. Items with `C_ij = 0` already carry zero weight, so structural
absence needs no imputation and the weights already sum to one over the requested
items alone.

`tests/test_aggregation.py` asserts the three properties this relies on: the weights
sum to one for every case, the reconstruction reproduces `A_total / C_total`, and a
convex combination of rates in `[0, 1]` stays in `[0, 1]`.

### Protocol

- **Case-level cross-validation.** Folds are assigned to cases and shared by every
  item and every expert, so a case never straddles the boundary and differences
  between experts reflect the models rather than different splits.
- **Nested cross-fitting.** Inside each outer training fold the experts are
  cross-fitted, so every meta training unit is described by expert predictions
  produced by models that never saw that unit. Training the meta-learner on
  in-sample expert predictions would let it learn to exploit the experts' overfitting.
- **No selection on the outer part.** Nothing is chosen using the outer evaluation
  part, at any stage.
- **Frozen hyperparameters.** No tuning, no early stopping on the evaluation data.
- **Repetition.** Three seeds, five outer folds and three inner folds in the study
  protocol. Seeded and deterministic given the configuration; the thread-affecting
  environment variables are pinned by `src.utils.env.configure_determinism`.

See [`docs/experimental_protocol.md`](docs/experimental_protocol.md) for the full
statement, and [`docs/model_description.md`](docs/model_description.md) for the model
specification.

---

## Repository layout

```
.
├── .github/
│   └── workflows/
│       └── tests.yml               CI: fast tests, end-to-end tests, demo
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── config/
│   ├── experiment_config.yaml      protocol, hyperparameters, block decomposition
│   └── paths.example.yaml           documented template for local paths
├── data/
│   ├── README.md
│   ├── schema/
│   │   └── data_dictionary.csv     one row per documented column
│   └── synthetic/
│       ├── README.md
│       └── synthetic_cases.csv     600 synthetic cases, 16 claim/award pairs
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 the orchestration module
│   ├── utils/                      paths, determinism, CLI helpers, paired comparison
│   ├── preprocessing/              item contract, matrices, structural absence
│   ├── features/                   request block, indicators, meta representation
│   ├── experts/                    registry and the convex combination expert
│   ├── meta_learning/              item-balanced weights and the meta head
│   ├── aggregation/                claim-weighted aggregation
│   ├── evaluation/                 item, macro, cell and case metrics
│   ├── statistics/                 bootstrap, permutation, multiplicity
│   └── interpretation/             TreeSHAP attribution
├── scripts/
│   ├── make_synthetic_data.py
│   ├── make_data_dictionary.py
│   ├── train_experts.py
│   ├── train_meta.py
│   ├── evaluate.py
│   ├── bootstrap.py
│   ├── permutation_test.py
│   ├── shap_analysis.py
│   └── run_synthetic_demo.py
├── tests/
│   ├── conftest.py
│   ├── test_aggregation.py
│   ├── test_item_balanced_l1.py
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   ├── test_synthetic_pipeline.py
│   └── test_cli_contract.py
├── examples/
│   └── synthetic_demo/
│       ├── README.md
│       ├── run_demo.py
│       └── expected_output/
├── docs/
│   ├── reproducibility.md
│   ├── data_access.md
│   ├── model_description.md
│   ├── experimental_protocol.md
│   └── privacy_and_data_policy.md
└── audit/
    ├── public_release_audit.md
    ├── data_leakage_audit.md
    ├── path_and_secret_audit.md
    ├── source_mapping.md
    └── public_release_summary.md
```

---

## Installation

Python 3.10 to 3.12 is supported. The pinned environment used for development is
Python 3.12.7.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

or, with conda:

```bash
conda env create -f environment.yml
conda activate jrsp
```

`numpy` is held at 1.26.x on purpose. Several of the pinned libraries were built
against the 1.x ABI, and 2.x additionally changes scalar-promotion rules that the
metric code depends on. Do not upgrade it casually.

If you use the versioned configuration file rather than `pyproject.toml`:

```bash
pip install -e ".[analysis,test]"
```

Nothing else is required for the lightweight demonstration. In particular, no model
weights are downloaded, because TabPFN is disabled by default and model weights and
caches are not included in the repository. The full paper configuration is the only
path that fetches anything, and it does so only when `--with-tabpfn` is passed. See
[Two run configurations](#two-run-configurations).

## Quick start

Run the synthetic demonstration:

```bash
python scripts/run_synthetic_demo.py
```

This loads `data/synthetic/synthetic_cases.csv`, builds the request-side features,
fits the expert pool on case-level folds, cross-fits the meta representation, trains
the meta head with the item-balanced objective, aggregates the item predictions to
the case level, evaluates the result, and prints its own verification checks. It
takes a few minutes at the demonstration scale because the expert pool is genuinely
trained; use `--max-cases` to shorten it.

The same command is available through the wrapper:

```bash
python examples/synthetic_demo/run_demo.py
```

The tables it produces are committed under
`examples/synthetic_demo/expected_output/`. Compare your run against them to confirm
the installation is sound. Small numerical differences can arise from the BLAS and
threading libraries present on your machine, so compare the verification checks and
the shape of the tables rather than the last digit of every value.

## Running the individual stages

Each stage is an executable script with `--help`. All of them default to the
synthetic dataset and to a bounded demonstration scale.

```bash
# Regenerate the synthetic dataset and the data dictionary.
python scripts/make_synthetic_data.py --n-cases 600 --seed 20260927
python scripts/make_data_dictionary.py

# Stage 1: fit the expert pool item by item and inspect per-item error.
python scripts/train_experts.py --max-cases 260

# Stage 2: inspect the item-balanced weighting and the meta head.
python scripts/train_meta.py --max-cases 260

# Stage 3: evaluate every variant and write the metric tables.
python scripts/evaluate.py --write-predictions

# Stage 4: statistical robustness.
python scripts/bootstrap.py --replicates 2000 --per-item
python scripts/permutation_test.py --replicates 2000 --per-item

# Stage 5: interpretation.
python scripts/shap_analysis.py --top 20
```

To run any stage on your own data, point `--data` at a case table that follows
`data/schema/data_dictionary.csv`. Expected columns are the sixteen
`<item>_claim` / `<item>_award` pairs plus a `case_id`; `claim_total`, `award_total`,
`doc_year` and the procedural columns are read when present.

`--output` means two different things, because the scripts produce two different
shapes of result. `make_synthetic_data.py` and `make_data_dictionary.py` each write one
named file, so for them `--output` is a file path. Every stage script writes a directory
of tables, so for those `--output` is a directory. Passing a directory to a generator is
reported as an argument error rather than left to fail on the write.

## Tests

```bash
python -m pytest tests/ -q
```

The suite has two parts. The unit tests are fast. The end-to-end test trains the
expert pool once per fold and takes a couple of minutes; it is tagged `slow` so you
can run the fast part alone:

```bash
python -m pytest tests/ -q -m "not slow"
```

What the tests establish:

| Area | Asserted |
|---|---|
| Aggregation | per-case weights sum to one; `sum_j w_ij r_ij == A_total / C_total`; a convex combination of rates in `[0, 1]` stays in `[0, 1]`; a requested-and-refused item is `0.0`, not missing |
| Item-Balanced L1 | every one of the sixteen items receives a total training weight of `1/16`; per-unit weight is inversely proportional to item frequency; balancing differs from uniform weighting; item balancing is not class balancing |
| Metrics | macro averages over items, not units; cell-level averages over units; items with fewer than two units are excluded; an item with a constant target has an undefined R-squared and is excluded from the mean |
| Preprocessing | the sixteen-item contract; structural absence encoded as `NaN` and behavioural zero as `0.0`; the request block is 33 columns with the `16 + 1 + 16` decomposition; indicators are one-hot; the claim side is `33 + 16 = 49` |
| Meta representation | width is 73 for the frozen eleven-expert pool; blocks are `11 + 13 + 49`; boundaries are `(0, 11)`, `(11, 24)`, `(24, 73)`; the width arithmetic holds for any pool size |
| End to end | the pipeline runs; the fold count matches the configuration; every expert predicted every held-out unit; weights are balanced in every fold; every prediction lies in `[0, 1]`; the conservation identity holds to machine precision; the run is reproducible |
| Command line | every script imports and answers `--help`; a directory passed to a generator's file-valued `--output` is reported as an argument error; the committed data dictionary and synthetic table are byte-for-byte reproducible from their generators and recorded seeds; the dictionary documents all seventy-three meta columns under the names the code uses |

### Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on every push and on
every pull request. It uses Python 3.12 and the exact pins in `requirements.txt`, and it
loosens or upgrades none of them. The suite is split so that a fast failure is reported
quickly rather than after the slow tests have finished.

| Job | Command | Scope |
|---|---|---|
| `unit-tests` | `pytest tests/ -q -m "not slow"` | the 115 fast tests |
| `end-to-end` | `pytest tests/ -q -m "slow"` | the 20 end-to-end tests |
| `synthetic-demo` | `python scripts/run_synthetic_demo.py` | the demonstration, whose verification block must report `overall: PASS` |

The two marker selections partition the suite exactly — 115 plus 20 is the 135 of the
full run — so between them every test is executed.

TabPFN is not installed in CI. It is the one optional dependency, its weights are not
distributed, and neither the default pool nor the test fixtures include it, so
installing it would pull the PyTorch stack into every run for no coverage gain.

Each of the three command lines above was executed against this repository before the
workflow was written, so the commands themselves are known to work. The workflow's first
run on GitHub Actions is still to happen, because the repository has not been pushed
yet. Expect the `end-to-end` and `synthetic-demo` jobs to take several minutes; both
train the expert pool on every item in every fold.

## Data

### Data availability

> The original case-level dataset is not publicly distributed because the
> underlying court judgment documents contain personally identifiable information
> and are not currently publicly accessible from the original source.

This repository provides the code, the configuration, the data schema and synthetic
examples that support methodological reproducibility. It contains no judgment texts
and no real case-level data.

What is distributed:

- `data/schema/data_dictionary.csv` — one row per documented column, giving the
  feature name, its data type, its description, its role in the pipeline, its
  allowed values or encoding, whether it is used by the experts, whether it is used
  by the meta-learner, whether it is observable before the judgment, and whether it
  is safe for public release. It describes structure; it contains no data values.
- `data/synthetic/synthetic_cases.csv` — a synthetic dataset generated from
  parametric distributions by `scripts/make_synthetic_data.py`. It is not derived
  from any real judgment, it contains no identifier from the study dataset, and it
  carries no textual material. Every case identifier begins with `SYNTH-` and the
  region field is literally `SYNTHETIC-REGION`, so a synthetic row cannot be
  mistaken for a real one.

What is not distributed: the case-level dataset, the judgment documents, the
real feature matrices, the real out-of-fold predictions, the real attribution
values, any case identifier, and any personal data.

See [`docs/data_access.md`](docs/data_access.md) and
[`docs/privacy_and_data_policy.md`](docs/privacy_and_data_policy.md).

### The synthetic dataset

The synthetic dataset exists so that the code can be executed by anyone, and so
that the tests have data to run on. It reproduces the structure of the study data —
sixteen claim and award columns, a graded injury-severity attribute, a procedural
level, two academic years — while being generated entirely from a random seed.

It deliberately does **not** reproduce the real item frequencies. Rare items were
given higher claim probabilities than their real counterparts so that every item
has enough training units for the demonstration to be meaningful. The probabilities
in `scripts/make_synthetic_data.py` are documented as construction parameters, not
as estimates. For the same reason, no accuracy figure computed on this dataset is
interpretable as a level of performance, and a near-zero or negative R-squared is a
property of the construction — the latent severity drives the rate while the
observable request-side features relate to it only weakly — and not a property of
the method.

## Reproducibility

This repository supports **methodological reproducibility**: every step of the
method is implemented, inspectable, executable and tested. Given a case table that
follows the documented schema, the pipeline will run on it and produce the full set
of tables.

It does **not** claim empirical reproducibility. The empirical dataset is not
distributed, so the numbers reported in the manuscript cannot be regenerated from
this repository by anyone who does not have access to the restricted underlying
data. The synthetic demonstration shows that the implementation executes; it does
not reproduce any reported result, and none of the scripts here regenerate a
manuscript figure or table.

> The repository does not redistribute the empirical dataset. Therefore, users
> cannot reproduce the exact numerical results reported in the manuscript without
> access to the restricted underlying data. The repository is intended to support
> methodological inspection, implementation reuse, and execution of the provided
> synthetic demonstration.

What *is* reproducible here, and is reproduced by the scripts and the tests:

- the dimensional structure of the model — sixteen items, an eleven-expert frozen pool,
  a 73-dimensional meta representation with blocks `11 + 13 + 49` and a claim side of
  `33 + 16`. The lightweight demonstration reaches ten of those experts, so its own meta
  matrix is 72 wide; the 73-column identity is asserted against the frozen pool, and
  both widths are reported by every run. See
  [Two run configurations](#two-run-configurations);
- the item-balanced weighting property — every item's total training weight is
  `1/16`, in every fold;
- the aggregation identity — `sum_j w_ij r_ij` reproduces `A_total_i / C_total_i` to
  machine precision, with zero learnable parameters;
- the full execution of the pipeline on a case table that follows the schema.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Results reported in the manuscript

The manuscript reports item-level and case-level error figures, the effect of
replacing the item-balanced objective with the alternatives, the result of the
paired resampling tests with multiplicity correction, and the block-level
attribution of the fitted meta-learner.

Those values are **not** quoted in this repository and are **not** computed by any
script here, because the data that produced them is not distributed. No file in this
repository hard-codes a manuscript result, and no script prints a figure it did not
compute. If you need the numbers, take them from the manuscript itself; they cannot
be regenerated from this repository alone.

The variant labels used in the code and in the output tables are the formal names of
the analyses: core model performance, statistical robustness, case-level aggregation
validation, temporal extrapolation and interpretability analysis. The internal
experiment codes used during development are not part of the public interface and do
not appear in the code, the documentation or the configuration.

## Privacy and data protection

The judgment documents from which the study data was extracted contain personal
information, and the compensation amounts they record are linked to identifiable
individuals. Accordingly:

- No judgment text is distributed, in any form, including excerpts and paraphrases.
- No case-level record is distributed: not the claim and award amounts, not the
  derived rates, not the feature matrix and not the predictions.
- No identifier is distributed: no name, no identity number, no address, no
  telephone number, no judgment number and no case identifier.
- No file is distributed from which any of the above could be recovered or
  reconstructed, including aggregates small enough to identify a case.
- The synthetic dataset is generated from parametric distributions rather than by
  perturbing real records, so it is not an anonymised version of the study data.

[`audit/data_leakage_audit.md`](audit/data_leakage_audit.md) records the scan that
was performed over every data-bearing file type, and
[`docs/privacy_and_data_policy.md`](docs/privacy_and_data_policy.md) states the
policy in full.

## Authors and roles

| Role | Name |
|---|---|
| First author | Liyuan Gu |
| Corresponding author | Mingyue Miao |
| Co-author | Lihua Huang |
| Co-author | Fangxia Zhao |

The author order and the role assignments above match the manuscript. Please
correspond about this repository through the corresponding author, using the contact
details given in the manuscript.

## Funding

This work was supported by the General Program of Science and Technology of Beijing
Municipal Education Commission, Grant No. KM202410038001.

## Citation

If you use this code, please cite the accompanying manuscript. Repository metadata is
provided in [`CITATION.cff`](CITATION.cff).

The repository is at
<https://github.com/liyuan112531/traffic-accident-support-rate-prediction>.
No DOI is asserted here, because none has been assigned to the manuscript or to this
software, and no ORCID identifier is listed for any author, because none was supplied.
When a DOI is minted it belongs both in `CITATION.cff` and here.

## License

**License: MIT**

This project is released under the [MIT License](LICENSE).
Copyright (c) 2026 Liyuan Gu, Mingyue Miao, Lihua Huang, Fangxia Zhao.

The full text is in [`LICENSE`](LICENSE). The licence covers the code and the
documentation in this repository. It does **not** cover the empirical dataset, which is
not distributed here at all — see [Data](#data). It also does not cover third-party
model weights, which are likewise not distributed: TabPFN is the only component that
would fetch any, and it is disabled unless `--with-tabpfn` is passed, in which case the
weights remain under their own upstream licence.
