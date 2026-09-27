# Experimental protocol

This document states the protocol the code implements. It is a description of
`src/pipeline.py` and `config/experiment_config.yaml`, not a restatement of the
manuscript.

## Summary

| Setting | Study protocol | Demonstration default |
|---|---|---|
| Expert pool | 11 experts, TabPFN enabled | 10 experts, TabPFN disabled |
| Meta representation width | 73 = `11 + 13 + 49` | 72 = `10 + 13 + 49` |
| Cross-validation | case-level, shared across items and experts | same |
| Outer folds | 5 | 2 |
| Inner folds (cross-fitting) | 3 | 2 |
| Seeds | 42, 2024, 3407 | 42 |
| Minimum meta training rows | 100 | 100 |
| Minimum expert training units per item | 8 | 8 |
| Cases | the full study corpus | 200 |
| Bootstrap replicates | 10,000 | 400 |
| Permutation replicates | 10,000 | 400 |
| Hyperparameter tuning | none | none |
| Selection on the outer evaluation part | none | none |

The study column is what the method is defined as. The demonstration column is what
the scripts run by default so that the code can be executed in minutes. The
demonstration settings are written in `config/experiment_config.yaml` under
`demonstration`, they are printed in every run's settings block, and they are
recorded in the output artefacts, so a demonstration run cannot be mistaken for a
run of the study protocol.

The expert pool is the one setting whose difference is not a matter of scale. TabPFN
needs pretrained weights that are not distributed with this repository, so it is
disabled unless `--with-tabpfn` is passed; the pool therefore reaches ten of the
eleven experts by default, and the meta representation narrows by exactly one column
because only block A follows the pool size. The method is defined by the full
configuration, and the repository names the two configurations consistently — see the
README for the table. Everything else in the protocol is identical between them.

## Fold construction

Folds are drawn with `KFold(shuffle=True, random_state=seed)` over **case indices**.
Two consequences follow, and both matter.

First, a case never appears in both the training and the evaluation part of a fold.
Because every item and every expert uses the same fold assignment, no case-item unit
of a case can straddle the boundary either. Splitting at the unit level instead would
put fifteen of a case's sixteen items in training and ask the model to predict the
sixteenth, which is a much easier problem and would make every metric optimistic.

Second, because the folds are shared, a difference between two experts reflects the
models rather than the split. Comparing experts across different splits would confound
the two and make the disagreement block partly a measure of data luck.

## Nested cross-fitting

Inside each outer training fold the experts are cross-fitted over the inner
partitions. For every outer fold the pipeline produces two prediction sets in one
pass:

- **evaluation predictions** — each expert fitted on the whole outer training fold
  and predicting the held-out cases;
- **cross-fitted predictions** — each expert fitted on the training cases outside an
  inner partition and predicting the units inside it.

The meta-learner is trained on the cross-fitted set. Its inputs are therefore genuine
out-of-fold predictions: no meta training unit is ever described by an expert that saw
that unit's own case. Training on in-sample expert predictions would let the
meta-learner learn to exploit whichever expert overfits most, because in-sample error
is optimistic in proportion to a model's capacity.

The two sets are produced together because the convex combination expert composes the
same base models. Sharing the fits avoids refitting six models per item per fold,
which is where almost all of the runtime would otherwise go. The weights are
unchanged by the sharing, because they depend only on the out-of-fold prediction
matrix and the target.

The convex combination expert additionally derives a nested weight vector for each
inner partition from the out-of-fold columns restricted to that partition's outside
rows, so its cross-fitted prediction is not fitted on data the meta-learner would
otherwise see.

## Insufficient units

Two thresholds guard against fitting on too little data. Both are checked per item,
per fold.

`MIN_EXPERT_ROWS = 8`. An item with fewer than eight training units in a fold is not
fitted by any expert. Every expert's entry for that item is filled with the training
fold's median rate for that item — which is exactly the constant median expert — and
the item is reported in `fallback_items` and counted in
`n_items_constant_fallback` in the fold log. Nothing is imputed at the level of the
label: the fill is a model output, not a substituted observation.

`MIN_META_ROWS = 100`. A fold whose cross-fitted meta training matrix has fewer than
100 complete rows is rejected with a descriptive error naming the fold and the seed,
rather than producing a meta head fitted on a handful of rows. The threshold is
enforced by `assert_meta_rows_sufficient`. A run that is too small is a configuration
error rather than a degraded result, so it fails loudly instead of silently reporting
a metric from an underfitted head.

Units with an undefined expert prediction are dropped from the meta training matrix
rather than filled, and the number dropped is recorded in the fold log as
`n_units_dropped_incomplete`. Every run reports both counts, so a run that relied
heavily on the constant fallback is visible rather than silent.

## Hyperparameters

Every hyperparameter is fixed in advance and recorded in
`src/experts/registry.py` (experts) and
`src/meta_learning/meta_learner.META_HYPERPARAMETERS` (meta head). There is no search,
no grid, no early stopping on the evaluation data and no model selection of any kind.
The same parameters apply to all sixteen items, to every fold and to every seed.

This is a deliberate design constraint, not a convenience. It removes tuning leakage
structurally: there is no path by which information from an evaluation fold could
influence the configuration, because the configuration does not depend on any data.

If you change a hyperparameter, the run is no longer comparable with the manuscript.

## Reported variants

Four meta heads are trained per fold so that the contribution of each design decision
can be attributed. They differ in exactly one respect each.

| Label | Loss | Weights |
|---|---|---|
| `Item-Balanced L1` | absolute | item-balanced |
| `Item-Balanced L2` | squared | item-balanced |
| `Pooled L1, uniform weights` | absolute | all ones |
| `Direct case-level L1` | absolute | all ones |

The first is the proposed configuration. The second isolates the shape of the loss.
The third isolates item balancing, because nothing else differs from the first. The
fourth predicts the case-level rate directly from the request side with no item
decomposition at all, which is the comparison that justifies the hierarchical
structure.

For the case-level results, the first three meta heads' item predictions are
aggregated with the claim-structure weights, and the direct variant is reported
alongside them.

## Aggregation

Case-level predictions are formed once, per fold, by aggregating the item predictions
with the case's own claim shares:

```
w_ij = C_ij / sum_k C_ik
R_hat_i = sum_j w_ij * r_hat_ij
```

No parameters are fitted at this stage, and no variant of the aggregation is selected
on the basis of its evaluation performance. Structural absence (`C_ij = 0`) gives
`w_ij = 0` and needs no imputation.

## Averaging across folds and seeds

Predictions are averaged across folds and seeds **before** the metrics are computed,
which is the official convention. Averaging the metrics instead would report a
different quantity: the mean of the fold-level errors rather than the error of the
mean prediction, and the two differ whenever the folds disagree.

The per-item metrics are then computed on the averaged predictions, and the macro
metrics are the mean of the per-item metrics. An item with fewer than two units is
excluded from the per-item metrics, and an item whose target is constant has an
undefined R-squared and is excluded from the R-squared average but retained for MAE
and RMSE.

## Statistical tests

Both resampling procedures compare two variants on the same units.

**Bootstrap.** Cases are drawn with replacement, `B` times, and both variants are
scored on the same draw. Resampling at the case level rather than the unit level
preserves the dependence between the sixteen units of a case: they share a request
side and a decision, so treating them as independent draws would understate the
interval width. Percentile intervals are reported at the requested level, together
with the fraction of replicates in which each variant is the better one. That fraction
is a direction probability, not a p-value.

**Permutation.** Under the null the two variants are exchangeable, so for each
case-item unit the two predictions may be swapped. Swapping at random generates a null
distribution for the macro difference, and the observed difference is located in it as
a two-sided p-value using the standard `(count + 1) / (replicates + 1)` correction,
which is finite-sample valid and never returns exactly zero. The swap is within a unit,
so the test conditions on the case and on the item.

**Multiplicity.** Benjamini-Hochberg is the primary correction because the reported
set is a family of exploratory comparisons and controlling the expected share of false
discoveries is the appropriate criterion. Holm controls the family-wise error rate and
is reported alongside it because it is the more conservative of the two; when both
agree, the conclusion does not depend on which is chosen. Correction is applied across
whichever family the caller assembles rather than a family hard-coded in the library,
because family size is a property of what is being reported.

Both procedures default to 2,000 replicates in the demonstration and to 10,000 in the
study protocol. Fewer replicates widen the Monte-Carlo error of the interval, so a
demonstration interval should not be read as a study interval.

## Attribution

TreeSHAP is applied to the fitted meta head of the final fold of the final seed.
Attributions are aggregated to the three blocks and, within the expert block, to
individual experts.

Scope: the explained model is a representative member of the ensemble of folds, not a
selected one; nothing is chosen on the basis of these values. The attributions are not
causal, because the observed support rate is a decision rather than a randomised
treatment, and the disagreement block is a prediction-difficulty signal rather than a
calibrated uncertainty estimate. See `docs/model_description.md`.

## Determinism

`src.utils.env.configure_determinism` sets the environment variables that let thread
scheduling and hash randomisation affect floating-point summation order — the OpenMP,
MKL, OpenBLAS, NumExpr and VecLib thread counts, `PYTHONHASHSEED`, and joblib's CPU
count — and every script calls it before importing a modelling library. Estimators are
pinned to a single thread so that summation order does not depend on how many cores are
free.

Two runs with the same configuration produce identical predictions, and
`tests/test_synthetic_pipeline.py` asserts this. Agreement across machines is weaker,
because BLAS implementations differ in how they sum, and floating-point addition is not
associative.

## What is never done

- No hyperparameter search, no tuning, no early stopping on evaluation data.
- No imputation of a structurally absent unit, and no default amount for a missing
  column.
- No post-outcome field is ever used as a feature; the leakage guard is executable.
- No selection of a variant, a seed, a fold or a feature set on the basis of
  evaluation performance.
- No case-level information crosses a fold boundary, and no unit is described by a
  model that saw its own case.
