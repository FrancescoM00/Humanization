"""Verifica il catalogo descrittivo sui testi e sulle metriche archiviati."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semantics.metric_catalogue import METRIC_DEFINITIONS, evaluate_catalogue, validate_cached_table


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip"


@pytest.fixture(scope="module")
def result():
    return evaluate_catalogue(ARCHIVE)


def test_all_valid_outputs_and_expansions_are_retained(result):
    docs = result["documents"]
    assert len(docs) == 500
    assert len(result["identity"]) == 126
    assert docs.groupby("humanizer").size().to_dict() == {
        "qwen_humanize":124, "qwen14b_humanize":126, "dipper":125, "dipper_context3":125}
    coverage = result["coverage"].set_index("humanizer")
    assert coverage.n_expanded_included.to_dict() == {
        "qwen_humanize":25, "qwen14b_humanize":19, "dipper":4, "dipper_context3":17}
    common = docs[docs.in_common_sources]
    assert len(common) == 492
    assert common.source_id.nunique() == 123
    assert common.groupby("humanizer").size().eq(123).all()


def test_identity_scores_and_readability_deltas(result):
    own = result["identity"]
    np.testing.assert_allclose(own.rouge_l_f1, 1, atol=1e-12)
    np.testing.assert_allclose(own.bleu_0_100, 100, atol=1e-10)
    np.testing.assert_allclose(own.bert_document_chunk_f1, 1, atol=1e-6)
    np.testing.assert_allclose(own.embedding_cosine, 1, atol=1e-6)
    np.testing.assert_array_equal(own.flesch_delta, 0)
    np.testing.assert_array_equal(own.fk_grade_delta, 0)


def test_original_readability_is_the_same_for_all_methods_on_the_same_source(result):
    docs = result["documents"]
    assert docs.groupby("source_id").flesch_original.nunique().eq(1).all()
    assert docs.groupby("source_id").fk_grade_original.nunique().eq(1).all()
    assert docs.groupby("source_id").original_sha256.nunique().eq(1).all()


def test_recomputed_metrics_match_frozen_scores_without_new_neural_inference(result):
    validation = result["validation"]
    for key, value in validation.items():
        if key.endswith("max_absolute_difference"):
            assert value < 1e-9
    assert validation["n_cached_metric_pairs"] == 626
    assert not result["provenance"]["new_neural_model_inference"]
    assert not result["provenance"]["api_calls_executed"]


def test_common_medians_and_quartiles_match_direct_document_calculation(result):
    docs = result["documents"]
    common = docs[docs.in_common_sources]
    table = result["common_median"].set_index("humanizer")
    summary = result["summary"].query("scope == 'common_sources' and stratum == 'overall'")
    for system, own in common.groupby("humanizer"):
        for metric in METRIC_DEFINITIONS:
            q1, median, q3 = np.quantile(own[metric], [.25, .5, .75])
            row = summary[summary.humanizer.eq(system) & summary.metric.eq(metric)].iloc[0]
            assert row.n == len(own) == 123
            np.testing.assert_allclose([row.q25,row["median"],row.q75], [q1,median,q3])
            assert table.loc[system, metric] == pytest.approx(median)
    assert table.loc["qwen_humanize", "rouge_l_f1"] == pytest.approx(.56)
    assert table.loc["qwen14b_humanize", "bleu_0_100"] == pytest.approx(14.1885, abs=5e-4)


def test_flesch_delta_is_paired_before_aggregation(result):
    docs = result["documents"]
    own = docs[docs.in_common_sources & docs.humanizer.eq("qwen_humanize")]
    np.testing.assert_allclose(own.flesch_delta, own.flesch_rewrite-own.flesch_original)
    median_delta = float(own.flesch_delta.median())
    difference_of_medians = float(own.flesch_rewrite.median()-own.flesch_original.median())
    assert median_delta == pytest.approx(-5.857026, abs=1e-6)
    assert abs(median_delta-difference_of_medians) > 1


@pytest.mark.parametrize("kind", ["missing", "duplicate", "sha", "status", "stage"])
def test_incomplete_or_stale_caches_are_rejected(kind):
    frame = pd.DataFrame({"id":["a","b"], "score":[.2,.4]})
    manifest = dict(status="executed", n_pairs=2, pairs_sha256="abc", stage="bertscore")
    if kind == "missing": frame = frame.iloc[:1]
    elif kind == "duplicate": frame = pd.concat([frame,frame.iloc[:1]])
    elif kind == "sha": manifest["pairs_sha256"] = "different"
    elif kind == "status": manifest["status"] = "partial"
    else: manifest["stage"] = "embedding"
    with pytest.raises(ValueError, match="incompleta o incoerente"):
        validate_cached_table(frame, ["a","b"], manifest, "abc", "bertscore")
