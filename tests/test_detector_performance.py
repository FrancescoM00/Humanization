"""Controlla il confronto prima/dopo sugli score congelati dello studio."""
from copy import deepcopy
from pathlib import Path
import zipfile
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from humanizers.detector_performance import (
    PRIMARY_SYSTEMS, evaluate_saved_detection, load_saved_detection, prepare_score_views,
)


ARCHIVE = Path(__file__).resolve().parents[1] / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip"


@pytest.fixture(scope="module")
def data():
    return load_saved_detection(ARCHIVE)


@pytest.fixture(scope="module")
def result(data):
    return evaluate_saved_detection(data, bootstrap=0)


def test_real_panel_counts_and_identity_are_preserved(result):
    assert len(result["common_source_ids"]) == 123
    overall = result["metrics"].query("stratum == 'overall'")
    own = overall.query("scope == 'paired_available'")
    expected = dict(qwen_humanize=124, qwen14b_humanize=126, dipper=125,
                    dipper_context3=125, null_identity=126)
    for system, count in expected.items():
        assert own[own.humanizer.eq(system)].n_pairs.eq(count).all()
    assert overall.query("scope == 'common_success'").n_pairs.eq(123).all()
    identity = overall[overall.humanizer.eq("null_identity")]
    for field in ["delta_accuracy", "delta_balanced_accuracy", "delta_tpr", "delta_auroc"]:
        assert identity[field].dropna().eq(0).all()


def test_reported_metrics_match_independent_sklearn_on_saved_scores(result):
    table = result["metrics"].query("stratum == 'overall' and scope == 'common_success'")
    for row in table.itertuples():
        part = result["pairs"]
        part = part[part.detector.eq(row.detector) & part.humanizer.eq(row.humanizer) & part.in_common_success]
        y = np.r_[np.zeros(len(part)), np.ones(len(part))]
        for stage, score, prediction in [
            ("before", "original_score", "original_is_ai"),
            ("after", "rewrite_score", "rewrite_is_ai"),
        ]:
            predicted = np.r_[part.human_is_ai, part[prediction]]
            assert getattr(row, f"accuracy_{stage}") == pytest.approx(accuracy_score(y, predicted))
            assert getattr(row, f"balanced_accuracy_{stage}") == pytest.approx(balanced_accuracy_score(y, predicted))
            if row.auroc_available:
                assert getattr(row, f"auroc_{stage}") == pytest.approx(
                    roc_auc_score(y, np.r_[part.human_score, part[score]]))


def test_available_tpr_matches_the_original_experiment(result):
    with zipfile.ZipFile(ARCHIVE) as archive:
        old = pd.concat([pd.read_csv(io.BytesIO(archive.read(name)), keep_default_na=False,
                                    float_precision="round_trip") for name in
                         ["detection/detection_summary.csv", "zero_shot/zero_shot_summary.csv"]])
    old = old[old.domain.eq("overall")]
    current = result["metrics"].query("stratum == 'overall' and scope == 'paired_available'")
    merged = current.merge(old, on=["detector", "humanizer"], validate="one_to_one")
    assert len(merged) == len(current)
    np.testing.assert_allclose(merged.tpr_before, merged.paired_original_tpr, atol=1e-15)
    np.testing.assert_allclose(merged.tpr_after, merged.paired_rewrite_tpr, atol=1e-15)


def test_binary_vote_has_no_continuous_auroc(result):
    rows = result["metrics"].query("detector == 'majority_vote'")
    assert not rows.auroc_available.any()
    assert rows.auroc_before.isna().all() and rows.auroc_after.isna().all()
    assert rows.auroc_unavailable_reason.eq("binary_ensemble_no_continuous_score").all()


def test_full_cohort_accuracy_bounds_do_not_impute_scores(result):
    coverage = result["coverage"]
    np.testing.assert_allclose(coverage.accuracy_after_full_upper-coverage.accuracy_after_full_lower,
                               coverage.n_failed/(2*coverage.n_test_sources), atol=1e-15)
    perfect = coverage[coverage.n_failed.eq(0)]
    assert perfect.accuracy_after_full_lower.eq(perfect.accuracy_after_full_upper).all()


@pytest.mark.parametrize("change", ["missing", "duplicate", "hash", "metadata", "orientation", "threshold"])
def test_corrupt_inputs_are_not_treated_as_detector_evasion(data, change):
    changed = deepcopy(data)
    if change == "missing":
        changed["scores"] = changed["scores"].iloc[:-1]
    elif change == "duplicate":
        changed["scores"] = pd.concat([changed["scores"], changed["scores"].iloc[:1]])
    elif change == "hash":
        changed["scores"].loc[0, "text_sha256"] = "wrong"
    elif change == "metadata":
        changed["scores"].loc[0, "source_group_id"] = "wrong"
    elif change == "orientation":
        changed["thresholds"]["binoculars_falcon7b"]["orientation"] = "higher_is_more_ai_like"
    else:
        changed["thresholds"]["openai_roberta_base"]["threshold"] = 0
    with pytest.raises(ValueError):
        prepare_score_views(changed)


def test_calibration_humans_cannot_enter_the_test(data):
    changed = deepcopy(data)
    human = next(row["human"] for row in changed["cohort"] if row["split"] == "test")
    next(row for row in changed["cohort"] if row["split"] == "calibration")["human"] = human
    with pytest.raises(ValueError, match="calibrazione"):
        prepare_score_views(changed)


def test_failed_generations_never_receive_an_artificial_score(data):
    changed = deepcopy(data)
    for row in changed["rewrites"]:
        if row["humanizer"] == "qwen_humanize":
            row["status"] = "generation_error"
            row["modified"] = ""
    changed["scores"] = changed["scores"][~changed["scores"].humanizer.eq("qwen_humanize")]
    result = evaluate_saved_detection(changed, bootstrap=0)
    assert not result["common_source_ids"]
    rows = result["metrics"].query("humanizer == 'qwen_humanize'")
    assert rows.n_pairs.eq(0).all() and rows.auroc_after.isna().all()
    assert not result["pairs"].humanizer.eq("qwen_humanize").any()
