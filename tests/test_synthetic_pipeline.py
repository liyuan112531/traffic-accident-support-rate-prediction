"""End-to-end execution of the pipeline on the synthetic demonstration dataset.

This is the test that establishes the code actually runs as a whole: the synthetic
case table is loaded, the expert pool is fitted item by item on case-level folds,
the cross-fitted meta representation is assembled and given its item-balanced
weights, the meta head is trained, and the item predictions are aggregated back to
the case level.

Nothing here asserts a level of accuracy. The synthetic dataset is a construction,
not a sample from the study population, so a target figure would be meaningless.
What is asserted are the structural properties the method promises: the dimensions,
the weight invariants, the admissible range, and the conservation identity.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Dimensional identities the whole method rests on.
N_EXPERT_BLOCK = 11.0
N_DISAGREEMENT_BLOCK = 13
N_CLAIM_SIDE_BLOCK = 49
N_META = 73


# ------------------------------------------------------------------ synthetic data


def test_synthetic_table_is_labelled_synthetic(synthetic_frame):
    assert "case_id" in synthetic_frame.columns
    identifiers = synthetic_frame["case_id"].astype(str)
    assert identifiers.str.startswith("SYNTH-").all()
    assert synthetic_frame["doc_region"].astype(str).str.contains("SYNTHETIC").all()


def test_synthetic_table_carries_every_item(synthetic_frame):
    from src.preprocessing.schema import AWARD_SUFFIX, CLAIM_SUFFIX, ITEMS

    for item in ITEMS:
        assert f"{item}{CLAIM_SUFFIX}" in synthetic_frame.columns
        assert f"{item}{AWARD_SUFFIX}" in synthetic_frame.columns


def test_synthetic_table_has_both_academic_years(synthetic_frame):
    years = set(synthetic_frame["doc_year"].unique().tolist())
    assert years == {2022, 2023}


def test_synthetic_awards_never_exceed_their_claims(synthetic_frame):
    from src.preprocessing.schema import AWARD_SUFFIX, CLAIM_SUFFIX, ITEMS

    claim = synthetic_frame[[f"{item}{CLAIM_SUFFIX}" for item in ITEMS]].to_numpy(float)
    award = synthetic_frame[[f"{item}{AWARD_SUFFIX}" for item in ITEMS]].to_numpy(float)
    assert np.all(award <= claim + 1e-6)
    assert np.all(award >= 0.0)


def test_synthetic_severity_column_is_a_synthetic_categorical(synthetic_frame):
    """The disability-like graded attribute is present but is not a real level."""
    assert "injury_severity_synthetic" in synthetic_frame.columns
    levels = set(synthetic_frame["injury_severity_synthetic"].astype(str))
    assert levels <= {"none", "minor", "moderate", "severe", "fatal"}


# ----------------------------------------------------- insufficient-data guards


def test_meta_row_guard_accepts_exactly_the_minimum():
    from src.pipeline import MIN_META_ROWS, assert_meta_rows_sufficient

    assert_meta_rows_sufficient(MIN_META_ROWS, MIN_META_ROWS, "test fold")


def test_meta_row_guard_rejects_one_row_below_the_minimum():
    from src.pipeline import MIN_META_ROWS, assert_meta_rows_sufficient

    with pytest.raises(RuntimeError, match="below the configured minimum"):
        assert_meta_rows_sufficient(MIN_META_ROWS - 1, MIN_META_ROWS, "test fold")


def test_meta_row_guard_names_the_offending_fold():
    from src.pipeline import assert_meta_rows_sufficient

    with pytest.raises(RuntimeError, match="fold 3 of seed 2024"):
        assert_meta_rows_sufficient(10, 100, "fold 3 of seed 2024")


def test_meta_row_guard_rejects_an_empty_matrix():
    from src.pipeline import assert_meta_rows_sufficient

    with pytest.raises(RuntimeError, match="0 complete rows"):
        assert_meta_rows_sufficient(0, 100, "test fold")


def test_demonstration_scale_clears_both_guards(synthetic_frame):
    """The shipped configuration must be large enough to be meaningful."""
    from src.utils.cli import demonstration_defaults
    from src.preprocessing.pipeline import to_item_long
    from src.preprocessing.schema import ITEMS

    defaults = demonstration_defaults()
    n_cases = defaults["max_cases"]
    n_folds = defaults["n_outer_folds"]
    rows_per_fold = int(len(synthetic_frame) * (1.0 - 1.0 / n_folds))
    units_per_case = len(to_item_long(synthetic_frame.head(n_cases), list(ITEMS))) / n_cases
    train_units = rows_per_fold * units_per_case * (n_cases / len(synthetic_frame))
    assert defaults["max_cases"] >= 100
    assert train_units >= 100, (
        "the demonstration's cross-fitted meta training fold is below the "
        "configured minimum and the run would be rejected"
    )


# ------------------------------------------------------------- end-to-end execution


@pytest.mark.slow
def test_pipeline_runs_end_to_end(test_pipeline_result):
    result = test_pipeline_result
    assert not result.unit_frame.empty
    assert not result.case_frame.empty
    assert not result.item_frame.empty
    assert not result.fold_log.empty


@pytest.mark.slow
def test_fold_count_matches_the_configuration(test_pipeline_result):
    log = test_pipeline_result.fold_log
    run = test_pipeline_result.settings["run"]
    assert set(log["seed"].unique().tolist()) == set(run["seeds"])
    assert len(log) == int(run["outer_folds"]) * len(run["seeds"])


@pytest.mark.slow
def test_every_fold_received_a_meta_training_matrix(test_pipeline_result):
    log = test_pipeline_result.fold_log
    assert (log["n_meta_train_rows"] > 0).all()
    assert (log["meta_width"] == log["meta_width"].iloc[0]).all()


@pytest.mark.slow
def test_every_expert_predicted_every_held_out_unit(test_pipeline_result):
    """A missing expert prediction would surface as a NaN cell in the meta matrix."""
    matrix = test_pipeline_result.meta_training_matrix
    assert matrix is not None
    assert np.isfinite(matrix).all()


@pytest.mark.slow
def test_meta_width_matches_the_pool_size(test_pipeline_result):
    settings = test_pipeline_result.settings
    pool_size = settings["pool_size"]
    assert settings["meta_width"] == pool_size + N_DISAGREEMENT_BLOCK + N_CLAIM_SIDE_BLOCK
    assert len(test_pipeline_result.meta_names) == settings["meta_width"]
    assert len(set(test_pipeline_result.meta_names)) == settings["meta_width"]


@pytest.mark.slow
def test_frozen_pool_meta_width_is_recorded_as_seventy_three(test_pipeline_result):
    assert test_pipeline_result.settings["frozen_pool_meta_width"] == N_META


@pytest.mark.slow
def test_meta_block_boundaries_are_the_documented_ones(test_pipeline_result):
    from src.features.meta_features import meta_block_boundaries

    pool_size = test_pipeline_result.settings["pool_size"]
    boundaries = meta_block_boundaries(pool_size)
    assert boundaries["expert_predictions"] == (0, pool_size)
    assert boundaries["disagreement"] == (pool_size, pool_size + 13)
    assert boundaries["claim_side"] == (pool_size + 13, pool_size + 13 + 49)


# ------------------------------------------------------------- weighting invariants


@pytest.mark.slow
def test_item_balanced_weights_are_balanced_in_every_fold(test_pipeline_result):
    checks = test_pipeline_result.weights_check
    assert checks["folds_checked"] > 0
    assert checks["all_balanced"] is True
    assert checks["all_positive"] is True


@pytest.mark.slow
def test_every_item_receives_one_sixteenth_of_the_training_weight(test_pipeline_result):
    checks = test_pipeline_result.weights_check
    assert checks["ideal_item_share"] == pytest.approx(1.0 / 16)
    assert checks["min_item_share"] == pytest.approx(1.0 / 16, abs=1e-9)
    assert checks["max_item_share"] == pytest.approx(1.0 / 16, abs=1e-9)


@pytest.mark.slow
def test_mean_weight_is_the_all_ones_target(test_pipeline_result):
    checks = test_pipeline_result.weights_check
    assert checks["min_mean_weight"] == pytest.approx(1.0)
    assert checks["max_mean_weight"] == pytest.approx(1.0)


@pytest.mark.slow
def test_objective_alignment_favours_the_balanced_weighting(test_pipeline_result):
    alignment = test_pipeline_result.alignment
    assert len(alignment) == 16
    assert np.allclose(alignment["alignment_ratio_item_balanced"], 1.0, atol=1e-12)
    spread = alignment["alignment_ratio_uniform"].max() / alignment["alignment_ratio_uniform"].min()
    assert spread > 1.5, "the synthetic design should be visibly unbalanced"


# ------------------------------------------------------------------ admissible range


@pytest.mark.slow
def test_every_unit_prediction_lies_inside_the_unit_interval(test_pipeline_result):
    from src.pipeline import UNIT_VARIANTS

    frame = test_pipeline_result.unit_frame
    for variant in UNIT_VARIANTS:
        values = frame[variant].to_numpy(float)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


@pytest.mark.slow
def test_every_case_prediction_lies_inside_the_unit_interval(test_pipeline_result):
    from src.pipeline import CASE_AGGREGATION_VARIANTS

    frame = test_pipeline_result.case_frame
    # The variant registry pairs each aggregated column label with the unit-level
    # variant it aggregates, so it is iterated by unpacking.
    for label, _ in CASE_AGGREGATION_VARIANTS:
        assert label in frame.columns, label
        values = frame[label].to_numpy(float)
        assert np.all(values >= 0.0), label
        assert np.all(values <= 1.0), label


@pytest.mark.slow
def test_case_target_lies_inside_the_unit_interval(test_pipeline_result):
    truth = test_pipeline_result.case_frame["y_true"].to_numpy(float)
    assert np.all(truth >= 0.0)
    assert np.all(truth <= 1.0)


# ------------------------------------------------------------------- aggregation


@pytest.mark.slow
def test_aggregation_conservation_identity_holds(test_pipeline_result):
    conservation = test_pipeline_result.conservation
    assert conservation["n_cases_checked"] > 0
    assert conservation["max_abs_deviation"] < 1e-9
    assert conservation["within_1e-12_fraction"] == 1.0


@pytest.mark.slow
def test_weight_concentration_exceeds_the_uniform_reference(test_pipeline_result):
    concentration = test_pipeline_result.concentration
    assert concentration["mean"] > concentration["uniform_reference"]
    assert concentration["max"] <= 1.0 + 1e-12


@pytest.mark.slow
def test_case_variants_all_produced_a_finite_metric(test_pipeline_result):
    assert test_pipeline_result.case_metrics
    for variant, values in test_pipeline_result.case_metrics.items():
        assert np.isfinite(values["MAE"]), variant
        assert values["MAE"] >= 0.0, variant


@pytest.mark.slow
def test_unit_metrics_cover_all_sixteen_items(test_pipeline_result):
    item_frame = test_pipeline_result.item_frame
    assert item_frame["item"].nunique() == 16
    assert item_frame["variant"].nunique() == len(test_pipeline_result.metrics)


@pytest.mark.slow
def test_variant_labels_are_the_published_ones(test_pipeline_result):
    from src.pipeline import (
        VARIANT_AGG_PROPOSED,
        VARIANT_PROPOSED,
        VARIANT_SQUARED,
        VARIANT_UNWEIGHTED,
    )

    assert VARIANT_PROPOSED == "Item-Balanced L1"
    assert VARIANT_SQUARED == "Item-Balanced L2"
    assert VARIANT_UNWEIGHTED == "Pooled L1, uniform weights"
    assert VARIANT_AGG_PROPOSED.startswith("Aggregated")


def test_internal_labels_never_appear_in_the_public_surface():
    """The internal experiment codes must not leak into the public API.

    The codes are assembled from a range rather than written out, so that this test
    does not itself introduce the strings it is checking for. A release that contains
    the codes anywhere is a release that exposes the development labelling scheme.
    """
    import src.pipeline as module

    text = "\n".join(
        value for value in vars(module).values() if isinstance(value, str)
    )
    for number in range(11, 16):
        assert f"E{number}" not in text


# ------------------------------------------------------------------ determinism


@pytest.mark.slow
def test_the_run_is_reproducible(synthetic_frame, test_pool):
    """The same configuration must produce identical predictions.

    Run at a deliberately small scale: this test is about bit-for-bit agreement
    between two runs of the same configuration, not about the size of the run. Rare
    items fall back to the constant median expert at this size, which is itself a
    deterministic path and therefore still a valid reproducibility check.
    """
    from src.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        n_outer_folds=2, n_inner_folds=2, seeds=(42,),
        pool=test_pool, max_cases=60, label="reproducibility",
    )
    first = run_pipeline(synthetic_frame, config)
    second = run_pipeline(synthetic_frame, config)
    assert np.allclose(
        first.unit_frame["Item-Balanced L1"].to_numpy(float),
        second.unit_frame["Item-Balanced L1"].to_numpy(float),
        atol=1e-12,
    )
