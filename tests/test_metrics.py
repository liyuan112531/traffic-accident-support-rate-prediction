"""Tests for the item-level, macro-averaged and cell-level metrics.

The distinction the method depends on is the one between averaging over items and
averaging over units. The macro figure weights the rare items as heavily as the
common ones; the cell-level figure does not. Both are asserted here so that a
change in either convention is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    METRIC_KEYS,
    MIN_UNITS_PER_ITEM,
    aggregate_over_seeds,
    case_metrics,
    item_metrics,
    macro_metrics,
    macro_micro_metrics,
    micro_mae,
)


def test_item_metrics_omits_items_with_a_single_unit():
    y = np.array([0.2, 0.4, 0.6])
    prediction = np.array([0.2, 0.4, 0.6])
    items = ["a", "b", "b"]
    rows = item_metrics(y, prediction, items)
    assert [row["item"] for row in rows] == ["b"]
    assert MIN_UNITS_PER_ITEM == 2


def test_item_metrics_on_a_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    items = ["a", "a", "b", "b"]
    rows = {row["item"]: row for row in item_metrics(y, y.copy(), items)}
    assert rows["a"]["MAE"] == pytest.approx(0.0)
    assert rows["a"]["RMSE"] == pytest.approx(0.0)
    assert rows["a"]["R2"] == pytest.approx(1.0)
    assert rows["a"]["n"] == 2


def test_item_metrics_reject_a_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        item_metrics(np.zeros(3), np.zeros(2), ["a", "a", "a"])


def test_macro_metrics_is_the_mean_over_items_not_units():
    """Rare and common items must carry the same weight."""
    y = np.concatenate([np.full(100, 0.5), np.array([0.5, 0.5])])
    items = ["common"] * 100 + ["rare"] * 2
    prediction = np.concatenate([np.full(100, 0.5), np.array([1.0, 1.0])])
    macro = macro_metrics(y, prediction, items)
    # common MAE 0.0, rare MAE 0.5, so the macro average is 0.25
    assert macro["macro_MAE"] == pytest.approx(0.25)
    assert macro["n_items"] == 2
    # the cell-level figure is dominated by the 100 common units and sits far below
    assert micro_mae(y, prediction) == pytest.approx(1.0 / 102.0)
    assert micro_mae(y, prediction) < macro["macro_MAE"]


def test_macro_metrics_ignores_r2_of_a_constant_item():
    y = np.array([0.5, 0.5, 0.1, 0.9])
    items = ["flat", "flat", "varying", "varying"]
    prediction = np.array([0.4, 0.6, 0.1, 0.9])
    macro = macro_metrics(y, prediction, items)
    assert macro["n_items"] == 2
    assert macro["n_items_with_defined_r2"] == 1
    assert macro["macro_R2"] == pytest.approx(1.0)


def test_macro_metrics_rejects_an_empty_problem():
    with pytest.raises(ValueError, match="no item had enough units"):
        macro_metrics(np.array([0.5]), np.array([0.5]), ["only"])


def test_macro_micro_metrics_exposes_every_reported_metric():
    rng = np.random.default_rng(0)
    y = rng.uniform(size=40)
    prediction = np.clip(y + rng.normal(scale=0.1, size=40), 0.0, 1.0)
    items = np.repeat([f"i{k}" for k in range(4)], 10)
    report = macro_micro_metrics(y, prediction, items)
    for key in METRIC_KEYS:
        assert key in report
        assert np.isfinite(report[key])
    assert report["n_units"] == 40
    assert report["n_items"] == 4


def test_micro_mae_is_the_cell_level_average():
    y = np.array([0.0, 1.0])
    prediction = np.array([1.0, 0.0])
    assert micro_mae(y, prediction) == pytest.approx(1.0)


def test_aggregate_over_seeds_takes_the_mean_over_replicates():
    replicates = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert np.allclose(aggregate_over_seeds(replicates), [0.3, 0.4])


def test_case_metrics_on_a_perfect_prediction():
    truth = np.array([0.0, 0.25, 1.0])
    report = case_metrics(truth, truth.copy())
    assert report["MAE"] == pytest.approx(0.0)
    assert report["R2"] == pytest.approx(1.0)
    assert report["n_cases"] == 3


def test_case_metrics_reject_a_length_mismatch():
    with pytest.raises(ValueError, match="disagree on length"):
        case_metrics(np.zeros(3), np.zeros(4))
