"""Tests for the Item-Balanced L1 objective.

The objective's defining property is that each of the sixteen items contributes a
total of ``1/16`` to the training loss. The tests below establish that property
directly, show that it differs from a plain unweighted fit on the same data, and
show that it is not the same thing as balancing classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.meta_learning.item_balanced import (
    N_ITEMS,
    describe_weighting,
    item_balanced_weights,
    objective_alignment_table,
    uniform_weights,
    weight_check,
)
from src.meta_learning.meta_learner import (
    LOSS_LABEL,
    MIX_DELTA,
    META_HYPERPARAMETERS,
    fit_meta_head,
    mix_gradient_hessian,
    predict_meta_head,
)
from src.preprocessing.schema import ITEMS

#: An unbalanced design: items differ by an order of magnitude in unit count, which
#: is the situation the objective exists to correct, only more so.
UNBALANCED_ITEMS = np.repeat(np.asarray(ITEMS, dtype=object), [400, 300, 200, 150,
                                                               120, 100, 90, 80,
                                                               60, 50, 40, 30,
                                                               20, 15, 10, 5])


def test_the_schema_defines_sixteen_items():
    assert len(ITEMS) == N_ITEMS == 16


def test_every_item_receives_the_same_total_weight():
    weights = item_balanced_weights(UNBALANCED_ITEMS)
    check = weight_check(UNBALANCED_ITEMS, weights)
    assert check["balanced"] is True
    assert check["item_share_min"] == pytest.approx(1.0 / N_ITEMS, abs=1e-12)
    assert check["item_share_max"] == pytest.approx(1.0 / N_ITEMS, abs=1e-12)
    assert check["max_deviation_from_share"] < 1e-12


def test_total_weight_equals_the_number_of_units():
    weights = item_balanced_weights(UNBALANCED_ITEMS)
    assert weights.sum() == pytest.approx(float(UNBALANCED_ITEMS.size))
    assert np.all(weights > 0.0)


def test_per_unit_weight_is_inversely_proportional_to_item_frequency():
    """Two items differing by a factor of 100 differ by 100 in per-unit weight."""
    items = np.array(["rare"] * 3 + ["common"] * 300)
    weights = item_balanced_weights(items)
    rare = weights[items == "rare"][0]
    common = weights[items == "common"][0]
    assert rare / common == pytest.approx(100.0, rel=1e-12)


def test_balancing_is_not_the_same_as_uniform_weighting():
    balanced = item_balanced_weights(UNBALANCED_ITEMS)
    uniform = uniform_weights(UNBALANCED_ITEMS)
    assert np.allclose(uniform, 1.0)
    assert not np.allclose(balanced, uniform)
    balanced_share = balanced[UNBALANCED_ITEMS == ITEMS[0]].sum() / balanced.sum()
    uniform_share = uniform[UNBALANCED_ITEMS == ITEMS[0]].sum() / uniform.sum()
    assert balanced_share == pytest.approx(1.0 / N_ITEMS, abs=1e-12)
    # medical is the largest item, so uniform weighting over-weights it
    assert uniform_share > 1.0 / N_ITEMS


def test_alignment_table_reports_one_for_the_balanced_objective():
    table = objective_alignment_table(UNBALANCED_ITEMS)
    assert len(table) == N_ITEMS
    assert np.allclose(table["alignment_ratio_item_balanced"], 1.0, atol=1e-12)
    assert (table["alignment_ratio_uniform"] > 1.0).any()
    assert (table["alignment_ratio_uniform"] < 1.0).any()
    assert np.allclose(table["macro_target_weight"], 1.0 / N_ITEMS)


def test_alignment_ratio_of_uniform_weighting_is_the_frequency_ratio():
    table = objective_alignment_table(UNBALANCED_ITEMS)
    expected = table["share_of_training_rows"] * N_ITEMS
    assert np.allclose(table["alignment_ratio_uniform"], expected, atol=1e-12)


def test_weights_reject_an_empty_training_set():
    with pytest.raises(ValueError, match="no training units"):
        item_balanced_weights([])


def test_weight_check_rejects_length_mismatch():
    with pytest.raises(ValueError, match="disagree on length"):
        weight_check(["a", "b"], np.ones(3))


def test_describe_weighting_mentions_the_ideal():
    text = describe_weighting(UNBALANCED_ITEMS)
    assert "1/16" in text or "0.062500" in text
    assert "units over 16 items" in text


# ------------------------------------------------------------------- the meta head


def _toy_problem(n_items: int = 4, per_item: int = 30, seed: int = 0):
    """A separable toy regression with unequal item sizes."""
    rng = np.random.default_rng(seed)
    items = np.repeat(np.asarray(ITEMS[:n_items], dtype=object), per_item)
    x = rng.normal(size=(items.size, 5))
    y = np.clip(x[:, 0] * 0.3 + 0.5 + rng.normal(scale=0.05, size=items.size),
                0.0, 1.0)
    return x, y, items


def test_fit_meta_head_with_item_balanced_weights_runs_and_predicts_in_range():
    x, y, items = _toy_problem()
    weights = item_balanced_weights(items)
    model = fit_meta_head(x, y, weights, seed=0, kind="l1")
    prediction = predict_meta_head(model, x)
    assert prediction.shape == y.shape
    assert np.all(prediction >= 0.0)
    assert np.all(prediction <= 1.0)


def test_fit_meta_head_rejects_weights_of_the_wrong_length():
    x, y, items = _toy_problem()
    with pytest.raises(ValueError, match="inconsistent lengths"):
        fit_meta_head(x, y, np.ones(y.size + 1), seed=0, kind="l1")


def test_fit_meta_head_rejects_non_positive_weights():
    x, y, _ = _toy_problem()
    weights = np.ones(y.size)
    weights[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_meta_head(x, y, weights, seed=0, kind="l1")


def test_unknown_loss_variant_is_rejected():
    x, y, items = _toy_problem()
    with pytest.raises(ValueError, match="unknown meta-loss variant"):
        fit_meta_head(x, y, item_balanced_weights(items), seed=0, kind="huber")


def test_predict_meta_head_clips_to_the_admissible_interval():
    """Clipping is a range property, so an out-of-range model is projected."""

    class _OutOfRange:
        def predict(self, x):
            return np.array([-0.5] * x.shape[0])

    clipped = predict_meta_head(_OutOfRange(), np.zeros((3, 2)))
    assert np.all(clipped == 0.0)


def test_meta_hyperparameters_are_the_pre_registered_values():
    assert META_HYPERPARAMETERS["n_estimators"] == 300
    assert META_HYPERPARAMETERS["num_leaves"] == 15
    assert META_HYPERPARAMETERS["learning_rate"] == 0.05
    assert META_HYPERPARAMETERS["min_child_samples"] == 20
    assert META_HYPERPARAMETERS["n_jobs"] == 1
    assert LOSS_LABEL == "Item-Balanced L1"


# ------------------------------------------------------------- the mixture variant


def test_mixture_objective_at_zero_recovers_the_smoothed_absolute_error():
    objective = mix_gradient_hessian(0.0)
    error = np.array([-0.2, 0.0, 0.3])
    gradient, hessian = objective(np.zeros(3), error)
    root = np.sqrt(error ** 2 + MIX_DELTA)
    assert np.allclose(gradient, error / root)
    assert np.allclose(hessian, MIX_DELTA / root ** 3)
    assert np.all(gradient <= 1.0) and np.all(gradient >= -1.0)


def test_mixture_objective_at_one_recovers_the_squared_error():
    objective = mix_gradient_hessian(1.0)
    error = np.array([-0.2, 0.0, 0.3])
    gradient, hessian = objective(np.zeros(3), error)
    assert np.allclose(gradient, 2.0 * error)
    assert np.allclose(hessian, 2.0)


def test_mixture_objective_rejects_a_lam_outside_the_unit_interval():
    with pytest.raises(ValueError, match="lam must lie in"):
        mix_gradient_hessian(1.5)
