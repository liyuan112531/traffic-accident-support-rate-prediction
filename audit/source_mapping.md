# Source mapping

How the released code relates to the code the study was run on.

The release is a **rewrite organised by stage**, not a copy of the study codebase.
The study codebase grew as a sequence of numbered experiments: each experiment added a
module, and several modules ended up containing partial versions of the same logic
written against a slightly different interface. Reproducing that layout in a public
release would have meant publishing superseded code paths alongside the live ones,
which invites a reader to run the wrong one.

So each released module was written from the corresponding study module as its
specification, with the interface changed to a single self-consistent one and the
superseded paths dropped. The correspondence below is by **logical origin**: it records
which study module a released module was derived from, and what was taken from it.

Nothing is copied verbatim from a data file. No study data file is referenced at
runtime, and none is distributed.

## Module correspondence

### `src/preprocessing/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `schema.py` | the item contract in the study's data-contract configuration | The sixteen item identifiers in canonical order, the `_claim` / `_award` naming convention, the target name, the case key, the year column, the list of post-outcome fields banned as features |
| `pipeline.py` | the study's data-audit and shared loading helpers | Schema validation, the claim/award matrices in canonical column order, the item support rate with structural absence as `NaN`, the long (case, item) frame, the temporal split |

The study's data-audit experiment and its shared helpers were consolidated into one
module. The consolidation removed a duplicated support-rate implementation that
differed only in whether it used a strict or a permissive zero test; the released
version keeps the strict one and documents the tolerance.

### `src/features/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `claim_features.py` | the request-side feature builder | The 33-column construction: sixteen raw claims, the claim total, sixteen `log1p` claims; the leakage token list and the guard that enforces it |
| `item_indicators.py` | the same module's indicator helper | The one-hot indicator over the sixteen items |
| `meta_features.py` | the study's super-learner module | The three-block decomposition `[A | B | C]`, the block widths 11 / 13 / 49, the order of the thirteen disagreement columns, the anchor-and-members structure of the disagreement specification |

The column **order** of the meta representation was recovered from the frozen study
feature table's column names (`pred_*`, then the disagreement block in its published
order, then `claim_0 … claim_32`, then `item_*`). Only the **names and their order**
were taken. The table's values are real and neither the table nor any extract from it
is distributed.

The study's module additionally contained text-encoder and verified-fact feature
blocks used in other experiments. Those are not part of the method described in the
manuscript's main specification, so they are not in the release.

### `src/experts/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `registry.py` | the study's single-item baseline configuration and its model-class expansion module | The eleven experts, their families, their pre-registered hyperparameters exactly as recorded in the experiment configuration, and the composition of the convex combination expert's base pool |
| `convex.py` | the study's ensemble module | The simplex-constrained minimum-absolute-error fit with the L2 tie-break, the SLSQP formulation and bounds, the fallback behaviour, and the effective-number diagnostic |

The study's model-class expansion module also carried code for candidate experts that
were considered and not adopted. Those are not in the release, because publishing a
configuration that was not used invites a reader to think it was.

### `src/meta_learning/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `item_balanced.py` | the study's item-balanced optimisation module | The item-balanced weight formula and its rescaling to a mean weight of one, the balancing check, the objective-alignment table |
| `meta_learner.py` | the same module plus the super-learner module | The absolute-error objective, the meta-head hyperparameters, the smoothed mixture loss with its analytic gradient and hessian, the clipping to `[0, 1]` |
| `selection.py` | the same module's selection routines | The blend, the penalty score with the RMSE cap, and the selection of the mixture parameter and the blend weight |

The study module embedded its headline target values as constants used to score
candidate configurations. Those constants are **not** carried over: no manuscript
figure appears anywhere in the release, and the selection routines take their
thresholds as arguments instead.

### `src/aggregation/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `claim_weighted.py` | the study's aggregation-validation module | The case-specific aggregation weights `C_ij / sum_k C_ik`, the reconstruction `R_hat_i = sum_j w_ij r_hat_ij`, the conservation-identity check, the Herfindahl concentration, the uniform benchmark, the case-level metrics |

The study module reported its results into a fixed output location as a side effect of
computing them. The released version returns values and lets the caller decide where
they go, which is what allows the scripts to write only where they are told to.

### `src/evaluation/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `metrics.py` | the study's metric module | The four reported metrics, the macro convention (average over items, minimum two units per item), the cell-level convention, the undefined-R-squared handling, the seed-averaging order |

The study module also contained the plotting helpers for its experiment figures. Those
belong to the manuscript's presentation rather than to the method and are not in the
release.

### `src/statistics/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `resampling.py` | the study's final-statistics module | The case-stratified paired bootstrap, the paired permutation test with the finite-sample p-value correction, the per-item grouping conventions |
| `multiplicity.py` | the same module | The Benjamini-Hochberg and Holm corrections |

The study module additionally contained leave-one-item-out and seed-stability analyses
written against its own result layout. They are not in the release; the release
provides the resampling primitives and the stage script that applies them to the
comparisons the method defines.

### `src/interpretation/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `shap_analysis.py` | the study's interpretability module | The feature inventory that assigns every column to its block from declared widths, the TreeSHAP wrapper, the block and expert aggregation, the effective-number diagnostic, the disagreement quartile table, and the explicit statements that the attributions are neither causal nor calibrated uncertainty |

### `src/pipeline.py`

| Logical origin | What was taken |
|---|---|
| The study's proposed-model experiment and its cross-fitting module | The case-level fold construction shared across items and experts, the nested cross-fitting of the experts, the meta training matrix assembly, the four reported variants, the case-level aggregation, the averaging across folds and seeds before scoring |

Two behaviours were changed deliberately, and both are recorded here because they are
substantive rather than cosmetic.

The study assembled its evaluation and cross-fitted expert predictions in two separate
passes, each refitting the base models. The released version produces both in one pass
and shares the base fits with the convex combination expert. The weights are unchanged,
because they depend only on the out-of-fold matrix and the target; the runtime is
roughly a third of what it was.

The study filled unfittable items from a global default. The released version fills
them with the training fold's median rate for that item, which is exactly the constant
median expert, and reports every such item in the fold log. The change was made so that
the fill is a model output rather than a substituted observation, and so that a run
that relied on it heavily is visible.

### `src/utils/`

| Released module | Logical origin | What was taken |
|---|---|---|
| `paths.py` | new | No study equivalent. The study hard-coded its output locations; the release resolves everything from the repository root and an optional ignored configuration file. |
| `env.py` | the determinism setup scattered across the study scripts | The thread-count and hash-seed variables, applied with `setdefault` so a user's own settings are not overwritten |
| `cli.py` | new | No study equivalent. Shared argument defaults and the display-path helper. |
| `paired.py` | new | No study equivalent. Assembles the paired arrays the two resampling routines need, so the two scripts cannot diverge in how they select comparable units. |

## Scripts

| Released script | Logical origin |
|---|---|
| `make_synthetic_data.py` | new; no study equivalent |
| `make_data_dictionary.py` | new; derived from the schema declarations |
| `train_experts.py` | the study's single-item baseline experiment |
| `train_meta.py` | the study's proposed-model experiment, meta stage |
| `evaluate.py` | the study's proposed-model experiment, evaluation stage |
| `bootstrap.py` | the study's final-statistics experiment, bootstrap part |
| `permutation_test.py` | the study's final-statistics experiment, permutation part |
| `shap_analysis.py` | the study's interpretability experiment |
| `run_synthetic_demo.py` | new; assembles the stages into one runnable demonstration |

## Tests and documentation

The tests, the documentation, the configuration files, the packaging metadata
(`pyproject.toml`, `requirements.txt`, `environment.yml`), the licence, the citation
metadata, the CI workflow (`.github/workflows/tests.yml`) and the audit records are all
new. The study codebase had no public-facing test suite, no continuous integration, and
its configuration was expressed as per-experiment JSON records rather than as a single
documented file.

## What was read from the study codebase

Read for reference, and no study file was modified:

- the evaluation modules listed above;
- the model baseline module;
- the request-side feature module;
- the data-contract and feature-policy configuration files;
- the experiment configuration records that hold the pre-registered expert
  hyperparameters;
- the column names of the frozen meta feature table.

## What was neither read nor used

- No judgment document.
- No case-level data file.
- No fitted model artefact.
- No manuscript file.

The release contains no statement about, and no value derived from, any of these.

## Summary of the release specification

| Aspect | Resolution |
|---|---|
| Item universe | 16, unchanged |
| Expert pool | 11, unchanged |
| Meta representation | 73 columns, blocks 11 / 13 / 49, claim side 33 + 16, unchanged |
| Meta objective | Item-Balanced L1 with the item-balanced weights, unchanged |
| Aggregation | Claim-structure-driven, zero learnable parameters, unchanged |
| Expert hyperparameters | Pre-registered values, unchanged |
| Fold and seed structure | Case-level, shared folds, nested cross-fitting, unchanged |
| Study data | Not distributed |
| Study results | Not quoted, not reproducible from the release |
