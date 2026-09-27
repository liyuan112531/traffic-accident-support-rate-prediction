"""Tests for the claim-structure-driven aggregation layer.

The properties asserted here are the ones the method relies on: the weights form a
distribution over a case's requested items, the reconstruction reproduces the
case-level rate exactly, and the reconstructed rate stays inside the admissible
interval when the item-level inputs do.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.aggregation.claim_weighted import (
    UNIFORM_WEIGHT,
    aggregate,
    case_level_metrics,
    conservation_deviation,
    herfindahl,
    request_weights,
    uniform_benchmark,
)
from src.preprocessing.schema import ITEMS

N_ITEMS = len(ITEMS)


def _claim_matrix(seed: int = 0, n_cases: int = 40) -> np.ndarray:
    """A claim matrix with structural zeros, so absent items are present."""
    rng = np.random.default_rng(seed)
    claim = np.where(
        rng.random((n_cases, N_ITEMS)) < 0.45,
        0.0,
        np.exp(rng.normal(9.0, 1.1, size=(n_cases, N_ITEMS))),
    )
    # guarantee every case has at least one request
    empty = claim.sum(axis=1) <= 0
    if empty.any():
        claim[empty, 0] = 1.0
    return claim


# --------------------------------------------------------------------------- weights


def test_request_weights_sum_to_one_per_case():
    weights = request_weights(_claim_matrix(seed=1))
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12)


def test_request_weights_are_zero_where_nothing_was_claimed():
    claim = _claim_matrix(seed=2)
    weights = request_weights(claim)
    assert np.all(weights[claim <= 0] == 0.0)
    assert np.all(weights[claim > 0] > 0.0)


def test_request_weights_match_the_definition():
    claim = _claim_matrix(seed=3, n_cases=5)
    weights = request_weights(claim)
    expected = claim / claim.sum(axis=1, keepdims=True)
    assert np.allclose(weights, expected, atol=1e-15)


def test_request_weights_reject_a_case_with_no_claim():
    claim = _claim_matrix(seed=4, n_cases=3)
    claim[1, :] = 0.0
    with pytest.raises(ValueError, match="non-positive claim total"):
        request_weights(claim)


def test_request_weights_reject_negative_amounts():
    claim = _claim_matrix(seed=5, n_cases=2)
    claim[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative amounts"):
        request_weights(claim)


# ------------------------------------------------------------------------ aggregate


def test_aggregate_matches_the_closed_form():
    """sum_j w_ij r_ij must equal sum_j Award_ij / sum_j Claim_ij."""
    claim = _claim_matrix(seed=6)
    rng = np.random.default_rng(6)
    award = claim * rng.uniform(0.0, 1.0, size=claim.shape)
    weights = request_weights(claim)
    defined = claim > 0
    rates = np.zeros_like(claim)
    rates[defined] = award[defined] / claim[defined]

    reconstructed = aggregate(weights, rates, defined=defined)
    direct = award.sum(axis=1) / claim.sum(axis=1)
    assert np.allclose(reconstructed, direct, atol=1e-12)


def test_aggregate_prediction_stays_inside_the_unit_interval():
    """A convex combination of rates in [0, 1] cannot leave [0, 1]."""
    claim = _claim_matrix(seed=7)
    rng = np.random.default_rng(7)
    item_predictions = rng.uniform(0.0, 1.0, size=claim.shape)
    weights = request_weights(claim)
    reconstructed = aggregate(weights, item_predictions, defined=claim > 0)
    assert np.all(reconstructed >= 0.0)
    assert np.all(reconstructed <= 1.0)


def test_aggregate_of_constant_rates_is_that_constant():
    claim = _claim_matrix(seed=8)
    weights = request_weights(claim)
    rates = np.full_like(claim, 0.37)
    reconstructed = aggregate(weights, rates, defined=claim > 0)
    assert np.allclose(reconstructed, 0.37, atol=1e-12)


def test_aggregate_ignores_weight_of_undefined_units():
    """Undefined units already carry zero weight, so masking changes nothing."""
    claim = _claim_matrix(seed=9)
    rng = np.random.default_rng(9)
    award = claim * rng.uniform(0.0, 1.0, size=claim.shape)
    weights = request_weights(claim)
    defined = claim > 0
    rates = np.zeros_like(claim)
    rates[defined] = award[defined] / claim[defined]
    rates_with_nan = np.where(defined, rates, np.nan)

    with_mask = aggregate(weights, rates, defined=defined)
    with_nan = aggregate(weights, rates_with_nan, defined=defined)
    assert np.allclose(with_mask, with_nan, atol=1e-15)


def test_aggregate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        aggregate(np.ones((3, N_ITEMS)), np.ones((3, N_ITEMS - 1)))


# --------------------------------------------------------------- conservation identity


def test_conservation_identity_holds_to_machine_precision():
    claim = _claim_matrix(seed=10)
    rng = np.random.default_rng(10)
    award = claim * rng.uniform(0.0, 1.0, size=claim.shape)
    report = conservation_deviation(claim, award)
    assert report["n_cases_checked"] == claim.shape[0]
    assert report["n_cases_excluded"] == 0
    assert report["max_abs_deviation"] < 1e-12
    assert report["within_1e-12_fraction"] == 1.0


def test_conservation_identity_excludes_cases_without_a_claim():
    claim = _claim_matrix(seed=11, n_cases=6)
    claim[2, :] = 0.0
    award = claim * 0.5
    report = conservation_deviation(claim, award)
    assert report["n_cases_excluded"] == 1
    assert report["n_cases_checked"] == 5


def test_behavioural_zero_is_not_structural_absence():
    """A requested item with nothing awarded is a rate of zero, not a missing value."""
    claim = np.zeros((1, N_ITEMS))
    award = np.zeros((1, N_ITEMS))
    claim[0, 0] = 1000.0
    claim[0, 1] = 3000.0
    award[0, 0] = 0.0       # requested, refused
    award[0, 1] = 1500.0    # requested, half granted
    report = conservation_deviation(claim, award)
    assert report["max_abs_deviation"] < 1e-15
    weights = request_weights(claim)
    assert weights[0, 0] == pytest.approx(0.25)
    # 0.25 * 0.0 + 0.75 * 0.5 == 0.375 == 1500 / 4000
    assert 0.25 * 0.0 + 0.75 * 0.5 == pytest.approx(0.375)


# -------------------------------------------------------------------------- herfindahl


def test_herfindahl_matches_the_uniform_reference_when_the_claim_is_flat():
    claim = np.ones((4, N_ITEMS))
    index = herfindahl(request_weights(claim))
    assert index["mean"] == pytest.approx(1.0 / N_ITEMS)
    assert index["uniform_reference"] == pytest.approx(1.0 / N_ITEMS)
    assert index["n_cases_single_item"] == 0


def test_herfindahl_reaches_one_for_a_single_item_case():
    claim = np.zeros((1, N_ITEMS))
    claim[0, 3] = 500.0
    index = herfindahl(request_weights(claim))
    assert index["max"] == pytest.approx(1.0)
    assert index["n_cases_single_item"] == 1


def test_uniform_benchmark_is_a_distribution():
    benchmark = uniform_benchmark(N_ITEMS)
    assert benchmark.sum() == pytest.approx(1.0)
    assert np.all(benchmark == UNIFORM_WEIGHT)


# --------------------------------------------------------------------- case metrics


def test_case_level_metrics_on_a_perfect_prediction():
    truth = np.array([0.1, 0.4, 0.9])
    report = case_level_metrics(truth, truth.copy())
    assert report["MAE"] == pytest.approx(0.0)
    assert report["RMSE"] == pytest.approx(0.0)
    assert report["R2"] == pytest.approx(1.0)
    assert report["n_cases"] == 3


def test_case_level_metrics_r2_is_undefined_for_a_constant_target():
    truth = np.full(4, 0.5)
    report = case_level_metrics(truth, truth + 0.1)
    assert np.isnan(report["R2"])
    assert report["MAE"] == pytest.approx(0.1)
