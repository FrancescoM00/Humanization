"""Verifica che detector e semantica appartengano alle stesse riscritture."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humanizers.semantic_evasion_data import (
    DEFAULT_CATALOGUE, DEFAULT_COMMON, DEFAULT_DETECTOR_PAIRS, DEFAULT_IDS,
    _semantic_panel, build_long_table,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "results_semantic_cost_evasion/config_exp3.json").read_text())


@pytest.fixture(scope="module")
def result():
    return build_long_table(CONFIG)


def test_join_preserves_complete_sources_methods_and_detectors(result):
    frame = result["long"]
    assert len(frame) == 984
    assert frame.source_id.nunique() == 123
    assert frame.groupby(["method", "detector"]).size().eq(123).all()
    assert not frame[["source_id", "method", "detector"]].duplicated().any()
    assert np.array_equal(frame.evaded, frame.orig_detected & ~frame.rew_detected)


def test_original_asr_matches_saved_common_source_results(result):
    checks = result["baseline_check"]
    assert checks.matches_reference.all()
    expected = {"openai_roberta_base": [(45,71),(60,71),(10,71),(6,71)],
                "binoculars_falcon7b": [(17,117),(29,117),(0,117),(1,117)]}
    for detector, counts in expected.items():
        own = checks[checks.detector.eq(detector)].set_index("method")
        for method, (k,n) in zip(CONFIG["methods"], counts):
            assert own.loc[method, "evaded"] == k
            assert own.loc[method, "n_original_detected"] == n
            assert own.loc[method, "asr"] == pytest.approx(k/n)


def test_score_orientation_and_frozen_flags_are_consistent(result):
    for detector, rows in result["long"].groupby("detector"):
        threshold = rows.raw_threshold
        if detector == "openai_roberta_base":
            assert threshold.eq(.9706076979637146).all()
            np.testing.assert_array_equal(rows.orig_detected, rows.original_raw_score > threshold)
            np.testing.assert_array_equal(rows.rew_detected, rows.rewrite_raw_score > threshold)
            np.testing.assert_array_equal(rows.original_score, rows.original_raw_score)
        else:
            assert threshold.eq(.9046559856180912).all()
            np.testing.assert_array_equal(rows.orig_detected, rows.original_raw_score < threshold)
            np.testing.assert_array_equal(rows.rew_detected, rows.rewrite_raw_score < threshold)
            np.testing.assert_array_equal(rows.original_score, -rows.original_raw_score)
        np.testing.assert_array_equal(rows.score_reduction, rows.original_score-rows.rewrite_score)


def test_semantics_and_expansions_are_not_lost_or_substituted(result):
    frame = result["long"]
    catalogue = pd.read_csv(DEFAULT_CATALOGUE, float_precision="round_trip").set_index("id")
    one = frame[frame.detector.eq(CONFIG["detectors"][0])].set_index("rewrite_id")
    old = catalogue.reindex(one.index)
    np.testing.assert_array_equal(one.bs_resc_mean, old.bert_aligned_rescaled_mean)
    np.testing.assert_array_equal(one.nli_cons_mean, old.nli_bidirectional_mean)
    np.testing.assert_array_equal(one.nli_min, old.nli_bidirectional_min)
    assert one.groupby("method").apply(lambda g:int(g.len_ratio.gt(1).sum()), include_groups=False).to_dict() == {
        "qwen_humanize":25, "qwen14b_humanize":17, "dipper":4, "dipper_context3":17}
    assert not np.allclose(one.bs_resc_mean, old.bert_document_chunk_f1)


@pytest.mark.parametrize("change", ["missing", "source", "method", "score"])
def test_semantic_joins_reject_missing_or_misassigned_rows(change):
    common = pd.read_csv(DEFAULT_COMMON, float_precision="round_trip")
    catalogue = pd.read_csv(DEFAULT_CATALOGUE, float_precision="round_trip")
    ids = pd.read_csv(DEFAULT_IDS)
    if change == "missing":
        common = common.iloc[1:]
    elif change == "source":
        common.loc[0, "source_id"] = common.loc[1, "source_id"]
    elif change == "method":
        common.loc[0, "humanizer"] = "dipper"
    else:
        common.loc[0, "nli_bidirectional_mean"] = .123
    with pytest.raises(ValueError):
        _semantic_panel(CONFIG, common, catalogue, ids)


def test_rewrite_hash_mismatch_is_rejected_against_raw_detector_texts(tmp_path):
    catalogue = pd.read_csv(DEFAULT_CATALOGUE, float_precision="round_trip")
    catalogue.loc[0, "modified_sha256"] = "not-the-rewrite-hash"
    path = tmp_path / "catalogue.csv"
    catalogue.to_csv(path, index=False)
    with pytest.raises(ValueError, match="stessa riscrittura"):
        build_long_table(CONFIG, catalogue=path)


def test_changed_detector_flag_cannot_pass_the_original_asr_check(tmp_path):
    pairs = pd.read_csv(DEFAULT_DETECTOR_PAIRS, float_precision="round_trip")
    selected = pairs.in_common_success & pairs.humanizer.eq("qwen_humanize") & pairs.detector.eq("openai_roberta_base")
    row = pairs.index[selected][0]
    pairs.loc[row, "rewrite_is_ai"] = not bool(pairs.loc[row, "rewrite_is_ai"])
    path = tmp_path / "pairs.csv"
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match="decisioni detector"):
        build_long_table(CONFIG, detector_pairs=path)
