# Model description

This document specifies the model as implemented in `src/`. It is a description of
the code, not of the manuscript; where the two could differ, the code is authoritative
for this repository.

## Structure

The model is hierarchical in the sense that it predicts at the level a judgment is
actually decided and then reconstructs the case-level quantity algebraically.

```
                       claim side (33 features)
                                |
                                v
        +-----------------------------------------------+
        |  item-wise expert pool, fitted independently   |
        |  for each of the 16 compensation items         |
        |  11 experts, pre-registered hyperparameters    |
        +-----------------------------------------------+
             |                        |
             | expert predictions     | disagreement
             v                        v
        +-----------------------------------------------+
        |  meta representation  [A | B | C]  -> 73 dim   |
        |  A: 11 expert predictions                      |
        |  B: 13 disagreement features                   |
        |  C: 49 claim-side features (33 + 16)           |
        +-----------------------------------------------+
                                |
                                v
        +-----------------------------------------------+
        |  meta head: gradient-boosted trees             |
        |  objective: Item-Balanced L1                   |
        +-----------------------------------------------+
                                |
                                v
                      item-level rate  r_hat_ij
                                |
                                v
        +-----------------------------------------------+
        |  claim-structure-driven aggregation            |
        |  w_ij = C_ij / sum_k C_ik                      |
        |  R_hat_i = sum_j w_ij * r_hat_ij               |
        |  zero learnable parameters                     |
        +-----------------------------------------------+
                                |
                                v
                      case-level rate  R_hat_i
```

## Notation

| Symbol | Meaning |
|---|---|
| `i` | case index |
| `j` | item index, `j = 1 ... J` |
| `J` | number of modelled items, `16` |
| `C_ij` | amount claimed for item `j` in case `i` |
| `A_ij` | amount awarded for item `j` in case `i` |
| `r_ij` | item-level support rate, `A_ij / C_ij` |
| `R_i` | case-level support rate, `A_total_i / C_total_i` |
| `z_ij` | the 73-dimensional meta representation of unit `(i, j)` |
| `r_hat_ij` | meta-learner prediction of `r_ij` |
| `w_ij` | aggregation weight, `C_ij / sum_k C_ik` |
| `R_hat_i` | case-level prediction, `sum_j w_ij r_hat_ij` |

## The target, and the asymmetry that matters

```
r_ij = A_ij / C_ij     defined only when C_ij > 0
```

Two situations look similar in a table and are not similar at all.

**Structural absence.** `C_ij = 0`. The item was never requested, so a rate does not
exist and none can be constructed. The value is `NaN`, the unit is excluded from
training and from evaluation, and it is reported in
`src.preprocessing.pipeline.structural_absence_report`. Nothing is imputed.

**Behavioural zero.** `C_ij > 0` and `A_ij = 0`. The item was requested and refused.
That is a real observation of the judicial decision and its rate is `0.0`. It stays in
the data with full weight.

Collapsing the two — for example by filling absent units with zero, or by dropping
refused items — would change what the model is estimating. The distinction is
enforced at the point of construction and asserted in `tests/test_preprocessing.py`
and `tests/test_aggregation.py`.

At the case level the same asymmetry appears once: `R_i` is `NaN` for a case with no
claim at all, and such a case is excluded rather than zeroed.

## Stage 1 — expert pool

Eleven experts are defined in `src/experts/registry.py` with their pre-registered
hyperparameters. The pool is fitted **per item**: each expert sees only the units of
one item, trained on the request-side features of those units' cases. This is a
deliberate choice. Items have different claim structures, different scales and
different decision patterns; pooling them would force one function to serve sixteen
different relationships, and the item indicator alone would not recover that.

| Expert | What it is | Loss it optimises |
|---|---|---|
| `median` | constant: the training-fold median rate of that item; ignores the features | absolute |
| `ridge` | L2-regularised linear regression, `alpha = 1.0` | squared |
| `rf` | random forest, 300 trees, `min_samples_leaf = 2`, `max_features = sqrt` | squared |
| `xgb` | XGBoost, 300 rounds, `max_depth = 3`, `lr = 0.05`, subsample 0.9, colsample 0.9, `reg_lambda = 1.0`, `hist` | squared |
| `lgb` | LightGBM, 300 rounds, `num_leaves = 15`, `lr = 0.05`, subsample 0.9 with `subsample_freq = 1`, colsample 0.9, `min_child_samples = 20` | squared |
| `catboost` | CatBoost, 300 iterations, `depth = 4`, `lr = 0.05`, `l2_leaf_reg = 3.0` | squared |
| `c3` | item-wise simplex-constrained convex combination of the six models above | absolute |
| `extratrees` | extremely randomised trees, 500 trees, `max_features = sqrt`, `min_samples_leaf = 1` | squared |
| `histgb` | histogram gradient boosting, 300 iterations, `lr = 0.06`, `max_leaf_nodes = 15`, `min_samples_leaf = 20` | squared |
| `svr_rbf` | standardised features, RBF kernel SVR, `C = 10.0`, `gamma = scale`, `epsilon = 0.1` | epsilon-insensitive |
| `tabpfn` | prior-data fitted network, `n_estimators = 1`, CPU | squared |

`median` exists to anchor what is achievable with no request-side information at all.
A model that does not beat it has learned nothing from the features.

### The convex combination expert

`c3` fits, for each item, non-negative weights over its base pool that sum to one:

```
minimise   mean( | P w - y | )  +  l2 * ||w||^2
subject to sum(w) = 1,  w >= 0
```

where the columns of `P` are the base experts' out-of-fold predictions. It is solved
by SLSQP on the simplex. The absolute error is the right objective here because it is
the quantity the expert is judged on, and because the target has point mass at 0 and 1
where a squared objective would be dominated by its heavy tails. The small L2 term
(`l2 = 1e-3`) is not regularisation in the usual sense: it breaks ties when several
weight vectors attain the same minimum, and its coefficient is fixed in advance rather
than tuned.

The fit is deliberately constrained rather than unconstrained. A convex combination
cannot extrapolate beyond the range of its members, so the combination stays inside
`[0, 1]` whenever its members do, and the combination layer carries no free scale of
its own. When the solver fails to converge, equal weights are returned and
`solver_success` is reported as `False`, because an equal-weight fallback is a
different expert and silently hiding it would misrepresent the run.

The weights are fitted on out-of-fold base predictions, never on in-sample ones. If
they were fitted in-sample the combination would chase whichever member overfits
most, which is precisely the failure the constraint is meant to prevent.

`c3` is defined over its full base pool of six models. If a run's pool omits any of
them, `enforce_convex_base` removes `c3` and records the reason, rather than
redefining it over a subset and keeping the name. A combination of six models and a
combination of four are different experts.

### Availability

Only TabPFN can be unavailable in practice, because it needs weights that are
downloaded on first use. Those weights are not distributed with this repository, so
TabPFN is disabled by default. When it is unavailable the run reports it under
`experts_skipped`, the meta width becomes `10 + 13 + 49 = 72`, and every downstream
table records the width it actually used. There is no silent substitution.

That produces two named configurations, and they are referred to by these names
throughout the repository:

| Configuration | Pool | Meta width | TabPFN weights |
|---|---|---|---|
| **Lightweight demonstration** (the default) | 10 experts | `72 = 10 + 13 + 49` | not needed, nothing is downloaded |
| **Full paper configuration** (`--with-tabpfn`) | 11 experts | `73 = 11 + 13 + 49` | downloaded on first use |

The empirical experiments reported in the manuscript use the **full paper
configuration**. The lightweight demonstration runs the same pipeline at a reduced
scale with that one expert omitted, so that the code can be executed without a network
download and without redistributing third-party weights. Block B and block C are
unaffected: the disagreement specification and the claim side do not reference TabPFN,
so only the width of block A follows the pool size.

## Stage 2 — the 73-dimensional meta representation

The meta-learner sees `z_ij = [A | B | C]`, three contiguous blocks.

### Block A — expert predictions, 11 columns

`pred_<expert>`, one column per expert in pool order.

### Block B — expert disagreement, 13 columns

Disagreement is encoded rather than summarised into a single number, because
different kinds of disagreement carry different information. The block is, in order:

| # | Column | Kind |
|---|---|---|
| 1–3 | `diff_c3_lgb`, `diff_c3_xgb`, `diff_c3_median` | signed: anchor minus member |
| 4–8 | `pred_max`, `pred_min`, `pred_range`, `pred_std`, `agg_median` | dispersion over the pool |
| 9–11 | `disagree_xgb_lgb`, `disagree_c3_catboost`, `disagree_c3_ridge` | absolute pairwise differences |
| 12 | `pred_mean_others_c3` | mean of the pool excluding the anchor |
| 13 | `n_experts` | pool size |

`c3` is the anchor because it is itself a combination of the base models, so the
signed differences against it say how far each base model sits from the combination.
The dispersion statistics are the uncalibrated spread of the pool; `pred_std` uses
`ddof = 1` and falls back to zero for a single-member pool.

The row positions of these columns are fixed and are exposed by
`meta_block_boundaries`, so a block cannot be mis-attributed by an off-by-one.

### Block C — claim side, 49 columns

Thirty-three request-side features followed by sixteen item indicators.

The request-side block decomposes as: sixteen raw claimed amounts, then
`claim_total`, then the sixteen `log1p`-transformed claimed amounts. The raw amounts
carry the case's scale; the logs carry its shape on a scale that does not let the
largest case dominate; the total carries the case's overall magnitude. The
indicator tells the meta-learner which item it is being asked about, which is
necessary because the same representation is used for all sixteen items.

Every request-side column is observable before the judgment is made. The block is
built from the claim columns alone, and a leakage guard rejects any column whose name
marks it as a post-outcome attribute.

## Stage 3 — the meta head

A LightGBM gradient-boosted tree ensemble trained with the absolute-error objective
and item-balanced sample weights. The hyperparameters are the pre-registered values
also used for the `lgb` expert: 300 rounds, `num_leaves = 15`, `lr = 0.05`, subsample
0.9 with `subsample_freq = 1`, colsample 0.9, `min_child_samples = 20`.

### Why absolute error

The support rate is a bounded ratio with point mass at 0 and 1. Squared error
over-penalises the resulting heavy tails and pulls the fit toward the interior of the
range, which is visible in the systematic under-prediction of full awards. Absolute
error is the loss that matches the quantity being reported, which is a mean absolute
error, so that the objective being optimised and the metric being reported agree.

### Why item-balanced weights

The metric is a **macro** average over items: each of the sixteen items contributes
one sixteenth of the reported error. A loss that simply averaged over units would
weight the frequently claimed items far more heavily than the macro metric does, so
the model would be optimised for something other than what is reported. The
item-balanced weights make the training objective's per-item contribution equal the
macro metric's:

```
L(theta) = sum_j (1 / J) * sum_{i in I_j} | y_ij - g_theta(z_ij) |
```

with per-unit weights

```
w_ij = 1 / (J * n_j)      rescaled so that sum_ij w_ij = N
```

so `sum_{i in I_j} w_ij` is `N / J` for every `j`. The rescaling makes the mean weight
exactly 1, which keeps the effective learning rate comparable across folds of
different sizes.

This is not class balancing. Balancing classes would equalise the number of units per
item; here the *number* of units is untouched and only the total *weight* per item is
equalised. The two coincide only when every item already has the same number of units.
`objective_alignment_table` reports, per item, the ratio of the effective training
weight to the weight in the macro average — 1 for every item under this scheme, and
`J * n_j / N` under uniform weighting.

### Clipping

Predictions are projected onto `[0, 1]`. This is a range property of the target, not a
learned correction, and no calibration layer is applied. It is the only post-hoc
operation in the model.

### Reported variants

The code trains four heads so that the contribution of each design decision can be
attributed:

| Variant label | Loss | Weights | What it isolates |
|---|---|---|---|
| Item-Balanced L1 | absolute | item-balanced | the proposed configuration |
| Item-Balanced L2 | squared | item-balanced | the shape of the loss |
| Pooled L1, uniform weights | absolute | all ones | the effect of item balancing |
| Direct case-level L1 | absolute | all ones | whether decomposing into items helps at all |

The last one predicts the case-level rate directly from the request side, without any
item decomposition, which is the comparison that justifies the hierarchical
structure.

## Stage 4 — aggregation

```
w_ij = C_ij / sum_k C_ik
R_hat_i = sum_j w_ij * r_hat_ij
```

The weights are the case's own claimed amounts normalised to sum to one, so they
carry no fitted parameters. The identity

```
sum_j w_ij r_ij = A_total_i / C_total_i
```

is exact: substituting `r_ij = A_ij / C_ij` cancels `C_ij` and leaves the ratio of the
row sums. The aggregation therefore adds no degrees of freedom; the learnable
structure lives entirely in the experts and the meta head.

Because `C_ij = 0` gives `w_ij = 0`, structurally absent items contribute nothing and
need no imputation, and the weights already sum to one over the requested items alone.
In practice only the requested items are scored at all; the case's weight vector is
reported over the full sixteen items for inspection, with `herfindahl` summarising its
concentration.

Two properties follow directly and are worth stating because they are easy to get
wrong:

- The weights are **case-specific**. Two cases with the same total claim but different
  compositions get different weight vectors.
- The aggregation is **not** a uniform average. A uniform benchmark is provided as a
  diagnostic only; it discards the request structure and is not a substitute.

## Stage 5 — evaluation

Four metrics are reported, and the difference between them matters:

| Metric | Averaged over | What it answers |
|---|---|---|
| `macro_MAE` | items | the typical error on an item, weighting rare and common items equally |
| `macro_RMSE` | items | the same, penalising large errors more |
| `macro_R2` | items | how much item-level variance is explained, relative to a per-item constant |
| `micro_MAE` | units | the typical error on a case-item unit, weighting common items more |

`macro_R2` is averaged over only those items with a defined R-squared; an item whose
target is constant has none, and including it as a zero would be wrong.

Predictions are averaged across seeds before the metrics are computed, so seed
variance enters through its effect on the averaged prediction rather than as an extra
averaging step on the metrics.

Case-level metrics are computed separately, on the aggregated prediction, because the
case-level error is not a function of the item-level errors alone — the aggregation
weights differ per case, so an error on a heavily weighted item costs more.

## Stage 6 — statistical robustness

- **Paired case-stratified bootstrap.** Cases are drawn with replacement and both
  methods are scored on the same draw. Resampling at the case level rather than the
  unit level preserves the dependence between the sixteen units of a case, which share
  a request side and a decision. Percentile intervals at the requested level are
  reported together with the fraction of replicates in which each method is the better
  one.
- **Paired permutation test.** Under the null the two methods are exchangeable, so for
  each case-item unit the two predictions may be swapped. Swapping at random generates
  a null distribution for the macro difference. The test conditions on the case and
  the item and needs no assumption about how units relate to one another.
- **Multiplicity.** Benjamini-Hochberg controls the expected share of false
  discoveries and is the primary correction; Holm controls the family-wise error rate
  and is reported alongside it. Correction is applied across whichever family the
  caller assembles, because the family size is a property of the reported set rather
  than of the test.

## Stage 7 — interpretation

TreeSHAP is used, which for a tree ensemble yields additive attributions that sum to
the difference between the prediction and the model's expected value. The additivity
is checked numerically and reported.

Attributions are aggregated to the three blocks, and within block A to individual
experts. `expert_contribution` also reports the effective number of contributing
experts, `1 / sum(share^2)`, which is directly comparable with the pool size: a value
near the pool size means the load is spread, a value near one means a single expert
dominates.

### Scope of interpretation — read this before quoting an attribution

**Not causal.** The observed support rate is a decision recorded in a judgment, not a
quantity assigned at random. No intervention is available to compare against, so a
large attribution is not a causal effect and nothing here supports a counterfactual
reading. The block structure describes what the fitted model relies on.

**Not calibrated uncertainty.** The disagreement block measures how much the expert
pool disagrees. It is a prediction-difficulty signal. Nothing has been calibrated
against realised coverage, so a large value means the pool split, not that a
confidence interval is valid.

**Attribution is of one fitted member.** The explained model is the meta head from the
final fold of the final seed. It is a representative member of the ensemble of folds,
not a selected one: nothing is chosen on the basis of these values.
