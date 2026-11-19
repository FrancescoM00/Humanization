"""Controlla metriche appaiate, orientamento e incertezza senza eseguire modelli."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from humanizers.detector_performance_metrics import METRICS, summarize_paired_metrics


def pairs(human=None, original=None, rewritten=None, groups=None, threshold=.5):
    human = np.asarray([.1, .2, .9] if human is None else human, dtype=float)
    original = np.asarray([.9, .2, .8] if original is None else original, dtype=float)
    rewritten = np.asarray([.2, .8, .1] if rewritten is None else rewritten, dtype=float)
    n = len(human)
    return pd.DataFrame(dict(source_id=[f"source-{i}" for i in range(n)],
        source_group_id=[f"group-{i}" for i in range(n)] if groups is None else groups,
        human_score=human, original_score=original, rewrite_score=rewritten,
        human_is_ai=human > threshold, original_is_ai=original > threshold,
        rewrite_is_ai=rewritten > threshold))


def test_known_confusion_counts_and_frozen_negative_class():
    frame = pairs()
    original = frame.copy(deep=True)
    result = summarize_paired_metrics(frame, bootstrap=0)
    assert result["n_pairs"] == result["n_human"] == result["n_ai"] == 3
    assert result["n_source_groups"] == 3
    for stage, tp, fn in [("before", 2, 1), ("after", 1, 2)]:
        assert result[f"tn_{stage}"] == 2
        assert result[f"fp_{stage}"] == 1
        assert result[f"tp_{stage}"] == tp
        assert result[f"fn_{stage}"] == fn
        assert result[f"accuracy_{stage}"] == result[f"balanced_accuracy_{stage}"]
        assert result[f"fpr_{stage}"] == 1/3
    assert result["delta_tpr"] == pytest.approx(-1/3)
    assert result["delta_accuracy"] == .5 * result["delta_tpr"]
    assert result["delta_fnr"] == -result["delta_tpr"]
    assert result["delta_fpr"] == 0
    pd.testing.assert_frame_equal(frame, original)


def test_points_match_standard_metrics():
    frame = pairs()
    result = summarize_paired_metrics(frame, bootstrap=0)
    labels = np.r_[np.zeros(len(frame)), np.ones(len(frame))]
    for stage, score, prediction in [("before", "original_score", "original_is_ai"),
                                     ("after", "rewrite_score", "rewrite_is_ai")]:
        predictions = np.r_[frame.human_is_ai, frame[prediction]]
        assert result[f"accuracy_{stage}"] == pytest.approx(accuracy_score(labels, predictions))
        assert result[f"balanced_accuracy_{stage}"] == pytest.approx(balanced_accuracy_score(labels, predictions))
        assert result[f"auroc_{stage}"] == pytest.approx(roc_auc_score(labels, np.r_[frame.human_score, frame[score]]))


def test_ties_receive_half_credit_and_threshold_ties_remain_human():
    result = summarize_paired_metrics(pairs([.5, .5], [.5, .5], [.5, .5]), bootstrap=40)
    assert result["auroc_before"] == result["auroc_after"] == .5
    assert result["tp_before"] == result["fp_before"] == 0
    assert result["accuracy_before"] == .5
    assert result["delta_auroc"] == result["delta_auroc_ci95_low"] == result["delta_auroc_ci95_high"] == 0


def test_identity_has_zero_paired_delta_and_zero_delta_intervals():
    frame = pairs()
    frame["rewrite_score"] = frame.original_score
    frame["rewrite_is_ai"] = frame.original_is_ai
    result = summarize_paired_metrics(frame, bootstrap=60, seed=4)
    for metric in METRICS:
        assert result[f"delta_{metric}"] == 0
        assert result[f"delta_{metric}_ci95_low"] == 0
        assert result[f"delta_{metric}_ci95_high"] == 0


def test_group_bootstrap_matches_weighted_auc_and_shared_class_weights():
    frame = pairs([.1, .4, .8, .2, .6], [.9, .4, .7, .3, .8], [.1, .2, .3, .4, .9],
                  groups=["linked-a", "linked-a", "b", "c", "c"])
    n_bootstrap, seed = 73, 16
    result = summarize_paired_metrics(frame, bootstrap=n_bootstrap, seed=seed)
    codes, names = pd.factorize(frame.source_group_id, sort=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(names), np.full(len(names), 1/len(names)), size=n_bootstrap)
    labels = np.r_[np.zeros(len(frame)), np.ones(len(frame))]
    boot_before, boot_after, boot_delta_accuracy = [], [], []
    for draw in draws:
        weights = draw[codes]
        sample_weights = np.r_[weights, weights]
        a = roc_auc_score(labels, np.r_[frame.human_score, frame.original_score], sample_weight=sample_weights)
        b = roc_auc_score(labels, np.r_[frame.human_score, frame.rewrite_score], sample_weight=sample_weights)
        boot_before.append(a)
        boot_after.append(b)
        tpr_delta = np.average(frame.rewrite_is_ai.astype(float)-frame.original_is_ai.astype(float), weights=weights)
        boot_delta_accuracy.append(.5*tpr_delta)
    for name, values in [("auroc_before", boot_before), ("auroc_after", boot_after),
                         ("delta_auroc", np.asarray(boot_after)-boot_before),
                         ("delta_accuracy", boot_delta_accuracy)]:
        expected = np.quantile(values, [.025, .975])
        assert result[name + "_ci95_low"] == pytest.approx(expected[0], abs=1e-14)
        assert result[name + "_ci95_high"] == pytest.approx(expected[1], abs=1e-14)
    assert result["n_pairs"] == 5 and result["n_source_groups"] == 3
    assert result["delta_fpr_ci95_low"] == result["delta_fpr_ci95_high"] == 0


def test_auc_can_decline_even_when_detection_rate_and_accuracy_increase():
    frame = pairs([.1, .2, .9], [.95, .49, .48], [.51, .52, .01])
    result = summarize_paired_metrics(frame, bootstrap=0)
    assert result["tpr_before"] == 1/3
    assert result["tpr_after"] == 2/3
    assert result["delta_accuracy"] > 0
    assert result["auroc_before"] == pytest.approx(7/9)
    assert result["auroc_after"] == pytest.approx(4/9)
    assert result["delta_auroc"] < 0


def test_negative_scores_are_supported_when_orientation_is_already_corrected():
    # Il chiamante ha già invertito il ratio Binoculars: IA crescente.
    frame = pairs([-1.4, -1.3], [-.8, -.9], [-1.6, -1.5], threshold=-1)
    result = summarize_paired_metrics(frame, bootstrap=0)
    assert result["auroc_before"] == 1
    assert result["auroc_after"] == 0
    assert result["delta_auroc"] == -1


def test_binary_ensemble_does_not_receive_a_continuous_auc():
    result = summarize_paired_metrics(pairs(), bootstrap=30, auroc=False)
    assert not result["auroc_available"]
    assert result["auroc_unavailable_reason"] == "binary_ensemble_no_continuous_score"
    for name in ["auroc_before", "auroc_after", "delta_auroc", "delta_auroc_ci95_low", "delta_auroc_ci95_high"]:
        assert np.isnan(result[name])
    assert np.isfinite(result["accuracy_before"])
    assert np.isfinite(result["delta_accuracy_ci95_low"])


def test_empty_complete_case_table_has_defined_coverage_and_undefined_metrics():
    result = summarize_paired_metrics(pd.DataFrame(), bootstrap=100)
    assert result["n_pairs"] == result["n_human"] == result["n_ai"] == 0
    assert result["n_source_groups"] == result["bootstrap_draws"] == 0
    assert result["auroc_unavailable_reason"] == "no_complete_pairs"
    for metric in METRICS:
        assert np.isnan(result[f"{metric}_before"])
        assert np.isnan(result[f"delta_{metric}_ci95_low"])


def test_one_independent_group_keeps_points_but_has_no_bootstrap_intervals():
    frame = pairs(groups=["one", "one", "one"])
    result = summarize_paired_metrics(frame, bootstrap=50)
    assert result["n_source_groups"] == 1 and result["bootstrap_draws"] == 0
    assert np.isfinite(result["delta_accuracy"])
    assert np.isnan(result["delta_accuracy_ci95_low"])
    assert np.isnan(result["auroc_before_ci95_high"])


@pytest.mark.parametrize("change", ["missing_column", "duplicate_source", "blank_group", "missing_group", "nan_score", "infinite_score", "non_boolean"])
def test_invalid_or_unpaired_inputs_are_rejected(change):
    frame = pairs()
    if change == "missing_column":
        frame = frame.drop(columns="human_score")
    elif change == "duplicate_source":
        frame.loc[1, "source_id"] = frame.loc[0, "source_id"]
    elif change == "blank_group":
        frame.loc[1, "source_group_id"] = "  "
    elif change == "missing_group":
        frame.loc[1, "source_group_id"] = None
    elif change == "nan_score":
        frame.loc[1, "rewrite_score"] = np.nan
    elif change == "infinite_score":
        frame.loc[1, "human_score"] = np.inf
    else:
        frame["human_is_ai"] = [0, 0, 1]
    with pytest.raises(ValueError):
        summarize_paired_metrics(frame, bootstrap=0)


@pytest.mark.parametrize("kwargs", [{"bootstrap": -1}, {"bootstrap": 1.5}, {"bootstrap": False}, {"seed": -1}, {"seed": True}, {"auroc": 1}])
def test_invalid_sampling_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        summarize_paired_metrics(pairs(), **kwargs)


def test_seed_reproduces_intervals():
    first = summarize_paired_metrics(pairs(), bootstrap=80, seed=91)
    second = summarize_paired_metrics(pairs(), bootstrap=80, seed=91)
    for key in first:
        if isinstance(first[key], float) and np.isnan(first[key]):
            assert np.isnan(second[key])
        else:
            assert first[key] == second[key]
