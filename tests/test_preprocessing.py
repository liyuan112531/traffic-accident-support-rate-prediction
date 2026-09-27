"""Tests for the schema, the loading path and the feature blocks.

The dimensional identities asserted here are the ones the method is defined by, so
they are checked as arithmetic rather than as literals taken on trust: the
request-side block is built from the claim columns, the claim-side block is the
request block plus the item indicators, and the meta representation is the expert
block plus the disagreement block plus the claim-side block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.claim_features import (
    CLAIM_TOTAL_INDEX,
    N_REQUEST_FEATURES,
    REQUEST_FEATURE_NAMES,
    assert_no_leakage,
    build_request_features,
    request_feature_block_dimension,
    request_feature_dictionary,
    request_feature_semantics,
)
from src.features.item_indicators import (
    N_ITEM_INDICATORS,
    item_indicator_matrix,
    item_indicator_names,
    item_indicator_row,
)
from src.features.meta_features import (
    DEFAULT_DISAGREEMENT_SPEC,
    META_DIMENSION,
    N_CLAIM_SIDE,
    N_DISAGREEMENT,
    N_EXPERT_PREDICTIONS,
    build_claim_side_block,
    build_disagreement_block,
    build_expert_block,
    build_meta_matrix,
    meta_block_boundaries,
    meta_dimension,
    meta_feature_names,
)
from src.preprocessing.pipeline import (
    assert_no_forbidden_columns,
    case_matrices,
    case_totals,
    item_support_rate,
    load_cases,
    structural_absence_report,
    to_item_long,
    validate_schema,
)
from src.preprocessing.schema import (
    AMOUNT_ZERO_TOL,
    AWARD_SUFFIX,
    CLAIM_SUFFIX,
    ITEMS,
    N_ITEMS,
    TARGET_NAME,
    assert_contract,
)

#: The frozen 11-expert pool, used only to check the published dimensions.
FROZEN_POOL = ("median", "ridge", "rf", "xgb", "lgb", "catboost", "c3",
               "extratrees", "histgb", "svr_rbf", "tabpfn")

#: Order of the 13 disagreement features, as the manuscript defines them.
EXPECTED_DISAGREEMENT = [
    "diff_c3_lgb", "diff_c3_xgb", "diff_c3_median",
    "pred_max", "pred_min", "pred_range", "pred_std", "agg_median",
    "disagree_xgb_lgb", "disagree_c3_catboost", "disagree_c3_ridge",
    "pred_mean_others_c3", "n_experts",
]


# ------------------------------------------------------------------------ schema


def test_schema_declares_the_sixteen_items():
    assert len(ITEMS) == N_ITEMS == 16
    assert len(set(ITEMS)) == 16
    assert ITEMS[0] == "medical"
    assert ITEMS[-1] == "other"
    assert "mental_solatium" in ITEMS
    assert TARGET_NAME == "total_support_rate"


def test_schema_contract_holds():
    assert_contract()


def test_schema_rejects_a_wrong_item_count():
    with pytest.raises(AssertionError, match="16"):
        assert_contract(ITEMS[:15])


def test_item_column_names_follow_the_suffix_convention():
    assert f"medical{CLAIM_SUFFIX}" == "medical_claim"
    assert f"medical{AWARD_SUFFIX}" == "medical_award"


# ---------------------------------------------------------------- loading a table


def test_load_cases_accepts_the_synthetic_table(synthetic_path):
    frame = load_cases(synthetic_path)
    assert len(frame) > 0
    for column in (f"{item}{CLAIM_SUFFIX}" for item in ITEMS):
        assert column in frame.columns
    validate_schema(frame)


def test_validate_schema_reports_missing_columns():
    with pytest.raises((ValueError, KeyError)):
        validate_schema(pd.DataFrame({"case_id": [1]}))


def test_assert_no_forbidden_columns_rejects_a_post_outcome_field():
    assert_no_forbidden_columns(["medical_claim", "claim_total"], "test")
    with pytest.raises(ValueError, match="forbidden post-outcome"):
        assert_no_forbidden_columns(["liability_ratio"], "test")


# ------------------------------------------------------- structural absence


def test_item_support_rate_encodes_absence_as_nan():
    claim = np.array([[0.0, 100.0]])
    award = np.array([[0.0, 40.0]])
    rate = item_support_rate(claim, award)
    assert np.isnan(rate[0, 0])
    assert rate[0, 1] == pytest.approx(0.4)


def test_item_support_rate_keeps_a_refused_claim_as_a_behavioural_zero():
    claim = np.array([[500.0]])
    award = np.array([[0.0]])
    rate = item_support_rate(claim, award)
    assert rate[0, 0] == 0.0
    assert not np.isnan(rate[0, 0])


def test_to_item_long_drops_absent_units_without_imputing():
    frame = pd.DataFrame(
        {f"{item}{CLAIM_SUFFIX}": [0.0] for item in ITEMS}
        | {f"{item}{AWARD_SUFFIX}": [0.0] for item in ITEMS}
        | {"case_id": ["SYNTH-0001"]}
    )
    frame.loc[0, "medical_claim"] = 200.0
    frame.loc[0, "medical_award"] = 150.0
    long_frame = to_item_long(frame)
    assert len(long_frame) == 1
    assert long_frame.iloc[0]["item"] == "medical"
    assert long_frame.iloc[0]["y"] == pytest.approx(0.75)


def test_structural_absence_report_counts_defined_and_absent():
    frame = pd.DataFrame(
        {f"{item}{CLAIM_SUFFIX}": [0.0, 100.0] for item in ITEMS}
        | {f"{item}{AWARD_SUFFIX}": [0.0, 50.0] for item in ITEMS}
        | {"case_id": ["SYNTH-0001", "SYNTH-0002"]}
    )
    report = structural_absence_report(frame)
    assert len(report) == N_ITEMS
    assert report["n_defined"].sum() == N_ITEMS
    assert report["n_structurally_absent"].sum() == N_ITEMS


def test_case_totals_use_the_claim_weighted_definition():
    frame = pd.DataFrame(
        {f"{item}{CLAIM_SUFFIX}": [0.0] for item in ITEMS}
        | {f"{item}{AWARD_SUFFIX}": [0.0] for item in ITEMS}
        | {"case_id": ["SYNTH-0001"]}
    )
    frame.loc[0, "medical_claim"] = 300.0
    frame.loc[0, "medical_award"] = 300.0
    frame.loc[0, "transport_claim"] = 100.0
    frame.loc[0, "transport_award"] = 0.0
    totals = case_totals(frame)
    assert totals.loc[0, "claim_total"] == pytest.approx(400.0)
    assert totals.loc[0, "award_total"] == pytest.approx(300.0)
    assert totals.loc[0, TARGET_NAME] == pytest.approx(0.75)


def test_case_matrices_have_the_canonical_shape(synthetic_frame):
    frame = synthetic_frame.head(25)
    matrices = case_matrices(frame)
    assert matrices["claim"].shape == (25, N_ITEMS)
    assert matrices["award"].shape == (25, N_ITEMS)
    assert matrices["claim_total"].shape == (25,)
    assert np.allclose(matrices["claim_total"], matrices["claim"].sum(axis=1))


# ------------------------------------------------------- request-side feature block


def test_request_block_has_thirty_three_columns():
    assert N_REQUEST_FEATURES == 33
    assert len(REQUEST_FEATURE_NAMES) == 33
    assert len(set(REQUEST_FEATURE_NAMES)) == 33


def test_request_block_decomposition_is_sixteen_plus_one_plus_sixteen():
    dimension = request_feature_block_dimension()
    assert dimension["raw_claims"] == 16
    assert dimension["claim_total"] == 1
    assert dimension["log1p_claims"] == 16
    assert dimension["total"] == 33
    assert (dimension["raw_claims"] + dimension["claim_total"]
            + dimension["log1p_claims"]) == N_REQUEST_FEATURES


def test_request_block_is_built_from_the_claim_columns(synthetic_frame):
    frame = synthetic_frame.head(20)
    features = build_request_features(frame)
    assert features.shape == (20, 33)
    raw = frame[[f"{item}{CLAIM_SUFFIX}" for item in ITEMS]].to_numpy(float)
    assert np.allclose(features.iloc[:, :16].to_numpy(float), raw)
    assert np.allclose(features.iloc[:, CLAIM_TOTAL_INDEX].to_numpy(float),
                       raw.sum(axis=1))
    assert np.allclose(features.iloc[:, 17:].to_numpy(float), np.log1p(raw))


def test_descriptive_alias_maps_every_position_to_a_readable_name():
    from src.features.claim_features import descriptive_alias

    aliases = [descriptive_alias(name) for name in REQUEST_FEATURE_NAMES]
    assert len(set(aliases)) == 33
    assert descriptive_alias("claim_0") == "medical_claim"
    assert descriptive_alias("claim_15") == "other_claim"
    assert descriptive_alias("claim_16") == "claim_total"
    assert descriptive_alias("claim_17") == "log1p_medical_claim"
    assert descriptive_alias("claim_32") == "log1p_other_claim"


def test_descriptive_alias_rejects_an_unknown_identifier():
    from src.features.claim_features import descriptive_alias

    with pytest.raises(KeyError, match="unknown request-side feature"):
        descriptive_alias("claim_33")


def test_request_feature_dictionary_documents_every_column():
    dictionary = request_feature_dictionary()
    assert set(dictionary) == set(REQUEST_FEATURE_NAMES)
    assert all(meaning.strip() for meaning in dictionary.values())
    semantics = request_feature_semantics()
    assert len(semantics) == 33
    assert [name for name, _, _ in semantics] == REQUEST_FEATURE_NAMES


def test_request_block_rejects_a_frame_that_is_missing_a_claim_column(synthetic_frame):
    frame = synthetic_frame.head(5).drop(columns=["other_claim"])
    with pytest.raises(KeyError, match="request-side columns missing"):
        build_request_features(frame)


def test_request_block_rejects_negative_claim_amounts(synthetic_frame):
    frame = synthetic_frame.head(5).copy()
    frame.loc[frame.index[0], "medical_claim"] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        build_request_features(frame)


def test_leakage_guard_rejects_a_post_outcome_name():
    assert_no_leakage(["claim_0", "claim_32"], "test")
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_leakage(["disability_level"], "test")


# --------------------------------------------------------------- item indicators


def test_item_indicators_cover_the_sixteen_items():
    assert N_ITEM_INDICATORS == 16
    names = item_indicator_names()
    assert len(names) == 16
    assert names[0] == "item_medical"
    assert names[-1] == "item_other"


def test_item_indicator_row_is_a_one_hot_vector():
    row = item_indicator_row("nursing")
    assert row.shape == (16,)
    assert row.sum() == 1.0
    assert row[ITEMS.index("nursing")] == 1.0


def test_item_indicator_matrix_has_one_active_entry_per_row():
    values = list(ITEMS) + ["medical", "other"]
    matrix = item_indicator_matrix(values)
    assert matrix.shape == (18, 16)
    assert np.allclose(matrix.sum(axis=1), 1.0)


# ----------------------------------------------------------------- claim side


def test_claim_side_block_is_request_plus_indicators():
    assert N_CLAIM_SIDE == 49
    assert N_REQUEST_FEATURES + N_ITEM_INDICATORS == 49
    request = np.ones((3, N_REQUEST_FEATURES))
    indicators = np.zeros((3, N_ITEM_INDICATORS))
    indicators[:, 0] = 1.0
    block = build_claim_side_block(request, indicators)
    assert block.shape == (3, 49)
    assert np.allclose(block[:, :33], request)
    assert np.allclose(block[:, 33:], indicators)


def test_claim_side_block_rejects_a_wrong_indicator_width():
    with pytest.raises(ValueError):
        build_claim_side_block(np.ones((3, 33)), np.ones((3, 15)))


# ------------------------------------------------------- meta representation


def test_meta_dimension_of_the_frozen_pool_is_seventy_three():
    assert META_DIMENSION == 73
    assert N_EXPERT_PREDICTIONS == 11
    assert N_DISAGREEMENT == 13
    assert N_CLAIM_SIDE == 49
    assert 11 + 13 + 49 == 73
    assert meta_dimension(11) == 73


def test_meta_block_boundaries_of_the_frozen_pool():
    boundaries = meta_block_boundaries(11)
    assert boundaries["expert_predictions"] == (0, 11)
    assert boundaries["disagreement"] == (11, 24)
    assert boundaries["claim_side"] == (24, 73)


def test_meta_feature_names_are_unique_and_ordered():
    names = meta_feature_names(FROZEN_POOL, DEFAULT_DISAGREEMENT_SPEC)
    assert len(names) == 73
    assert len(set(names)) == 73
    assert names[:11] == [f"pred_{name}" for name in FROZEN_POOL]
    assert names[11:24] == EXPECTED_DISAGREEMENT
    assert names[24:40] == [f"claim_{k}" for k in range(16)]
    assert names[40] == "claim_16", "claim_total sits at index 16 of the block"
    assert names[56] == "claim_32", "the last log1p claim closes the block"
    assert names[57:73] == [f"item_{item}" for item in ITEMS]


def test_meta_width_shrinks_with_the_pool():
    """A pool missing TabPFN yields a 72-wide representation, not a padded 73."""
    reduced = FROZEN_POOL[:-1]
    names = meta_feature_names(reduced, DEFAULT_DISAGREEMENT_SPEC)
    assert len(names) == 72
    assert meta_dimension(10) == 72
    assert meta_block_boundaries(10)["claim_side"] == (23, 72)


def test_meta_width_arithmetic_holds_for_every_pool_size():
    for size in range(2, 12):
        assert meta_dimension(size) == size + 13 + 49
        boundaries = meta_block_boundaries(size)
        assert boundaries["expert_predictions"] == (0, size)
        assert boundaries["disagreement"] == (size, size + 13)
        assert boundaries["claim_side"] == (size + 13, size + 62)


def test_meta_feature_names_reject_an_unknown_expert():
    with pytest.raises((KeyError, ValueError)):
        meta_feature_names(("median", "not_an_expert"), DEFAULT_DISAGREEMENT_SPEC)


def _fake_predictions(pool, n_units=7, seed=0):
    """One prediction per unit, as the pipeline supplies them.

    The expert block is assembled from per-unit scalars, so each expert contributes a
    vector of length ``n_units`` rather than a matrix.
    """
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(size=n_units) for name in pool}


def test_build_meta_matrix_has_the_documented_blocks():
    pool = FROZEN_POOL
    predictions = _fake_predictions(pool)
    request = np.ones((7, N_REQUEST_FEATURES))
    indicators = np.zeros((7, N_ITEM_INDICATORS))
    indicators[:, 0] = 1.0

    expert_block = build_expert_block(predictions, list(pool))
    disagreement = build_disagreement_block(predictions, list(pool),
                                            DEFAULT_DISAGREEMENT_SPEC)
    claim_side = build_claim_side_block(request, indicators)
    matrix = build_meta_matrix(predictions, request, indicators,
                               pool=list(pool), spec=DEFAULT_DISAGREEMENT_SPEC)

    assert expert_block.shape == (7, 11)
    assert disagreement.shape == (7, 13)
    assert claim_side.shape == (7, 49)
    assert matrix.shape == (7, 73)
    assert np.allclose(matrix[:, :11], expert_block)
    assert np.allclose(matrix[:, 11:24], disagreement)
    assert np.allclose(matrix[:, 24:], claim_side)
    assert np.isfinite(matrix).all()


def test_meta_matrix_block_widths_add_up():
    pool = FROZEN_POOL
    matrix = build_meta_matrix(
        _fake_predictions(pool),
        np.ones((7, N_REQUEST_FEATURES)),
        np.eye(N_ITEM_INDICATORS)[[0] * 7],
        pool=list(pool),
        spec=DEFAULT_DISAGREEMENT_SPEC,
    )
    n_pool = len(pool)
    boundaries = meta_block_boundaries(n_pool)
    assert boundaries["expert_predictions"][1] - boundaries["expert_predictions"][0] == 11
    assert boundaries["disagreement"][1] - boundaries["disagreement"][0] == 13
    assert boundaries["claim_side"][1] - boundaries["claim_side"][0] == 49
    assert matrix.shape[1] == 73
