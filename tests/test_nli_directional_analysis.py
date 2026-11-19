"""Verifica direzioni, ricampionamento e distribuzioni NLI sulle fonti comuni."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import ALERT_METRICS
from semantics.nli_directional_analysis import analyze_directions, load_common_documents


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_humanizer_semantics_final"


@pytest.fixture(scope="module")
def documents():
    return load_common_documents(RESULTS / "document_scores.csv", RESULTS / "common_source_ids.csv")


@pytest.fixture(scope="module")
def analysis(documents):
    return analyze_directions(documents)


def test_real_common_panel_and_directional_quartiles(documents, analysis):
    assert len(documents) == 492
    assert documents.groupby("humanizer").size().eq(123).all()
    assert documents.groupby("source_id").size().eq(4).all()
    assert len(analysis["directional_long"]) == 16
    wide = analysis["directional_summary"].set_index("humanizer")
    assert wide.loc["dipper", "nli_reverse_min_median"] == pytest.approx(.0062064034864306)
    assert wide.loc["dipper", "nli_forward_min_median"] == pytest.approx(.8652607798576355)
    for row in analysis["directional_long"].itertuples():
        own = documents.loc[documents.humanizer.eq(row.humanizer), row.metric]
        np.testing.assert_allclose([row.q25, row.median, row.q75], own.quantile([.25, .5, .75]))


def test_spearman_matches_independent_scipy_and_preserves_source_bootstrap(documents, analysis):
    expected = {"qwen_humanize": (-.05105165733349594, .3343113953890121),
                "qwen14b_humanize": (-.0230529670730716, .3137837898753008),
                "dipper": (.4284673676359538, .6833933032880826),
                "dipper_context3": (.10699691074336484, .45823744346081546)}
    for row in analysis["correlations"].itertuples():
        own = documents[documents.humanizer.eq(row.humanizer)]
        assert row.rho == pytest.approx(spearmanr(own.nli_reverse_min, own.word_ratio).statistic)
        np.testing.assert_allclose([row.ci95_low, row.ci95_high], expected[row.humanizer], atol=1e-14)
        assert row.n_finite_bootstrap_replicates == row.bootstrap_replicates == 5000


def test_direction_that_governs_and_high_support_counts(analysis):
    weakest = analysis["weakest_direction"].set_index("humanizer")
    assert weakest.loc["dipper", "n_reverse_lower"] == 107
    assert weakest.loc["dipper", "n_forward_lower"] == 16
    assert weakest.loc["qwen_humanize", "n_ties"] == 1
    assert (weakest.n_reverse_lower + weakest.n_forward_lower + weakest.n_ties).eq(123).all()
    fractions = analysis["support_fractions"].set_index("humanizer").loc[list(PRIMARY_SYSTEMS)]
    np.testing.assert_array_equal(fractions.n_ge_0_5, [42, 41, 15, 35])
    np.testing.assert_array_equal(fractions.n_ge_0_8, [32, 34, 13, 26])


def test_bootstrap_all_pairs_and_named_focus_match_previous_results(analysis):
    new = analysis["pairwise_comparisons"]
    assert len(new) == 24 and len(analysis["focus_comparisons"]) == 8
    old = pd.read_csv(RESULTS / "paired_source_comparisons.csv")
    old = old[old.stratum.eq("overall") & old.metric.isin(new.metric)]
    keys = ["humanizer_a", "humanizer_b", "metric"]
    new, old = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
    assert new.index.equals(old.index)
    for field in ["value_a", "value_b", "difference_b_minus_a", "ci95_low", "ci95_high"]:
        np.testing.assert_allclose(new[field], old[field], atol=1e-14, rtol=0)


def test_ecdf_preserves_all_mass_and_values_below_one_thousandth(analysis):
    ecdf = analysis["ecdf_points"]
    assert ecdf.score.min() == pytest.approx(.000329600792611)
    assert ecdf.score.min() < .001
    for _, rows in ecdf.groupby("humanizer"):
        assert rows.score.is_monotonic_increasing
        assert rows.n_cumulative.is_monotonic_increasing
        assert rows.iloc[-1].n_cumulative == 123
        assert rows.iloc[-1].cumulative_fraction == 1


def synthetic_documents(documents):
    ids = sorted(documents.source_id.unique())[:3]
    result = documents[documents.source_id.isin(ids)].copy()
    for system in PRIMARY_SYSTEMS:
        mask = result.humanizer.eq(system)
        for name in ["nli_forward_min", "nli_reverse_min", "nli_bidirectional_min"]:
            result.loc[mask, name] = [.5, .8, .8]
        for name in ["nli_forward_mean", "nli_reverse_mean", "nli_bidirectional_mean"]:
            result.loc[mask, name] = [.5, .8, .8]
        for name in ALERT_METRICS:
            result.loc[mask, name] = False
    return result


def test_inclusive_cutoffs_ties_and_ecdf_duplicate_values(documents):
    result = analyze_directions(synthetic_documents(documents), bootstrap=100)
    assert result["support_fractions"].n_ge_0_5.eq(3).all()
    assert result["support_fractions"].n_ge_0_8.eq(2).all()
    assert result["weakest_direction"].n_ties.eq(3).all()
    for _, rows in result["ecdf_points"].groupby("humanizer"):
        np.testing.assert_array_equal(rows.score, [.5, .8])
        np.testing.assert_array_equal(rows.n_cumulative, [1, 3])


@pytest.mark.parametrize("change", ["missing", "duplicate", "source_group", "ratio", "score"])
def test_incoherent_source_panels_and_scores_are_rejected(documents, change):
    changed = documents.copy()
    if change == "missing":
        changed = changed.iloc[1:]
    elif change == "duplicate":
        changed = pd.concat([changed, changed.iloc[[0]]])
    elif change == "source_group":
        changed.loc[changed.index[0], "source_group_id"] = changed.iloc[1].source_group_id
    elif change == "ratio":
        changed.loc[changed.index[0], "word_ratio"] = 0
    else:
        changed.loc[changed.index[0], "nli_reverse_min"] = np.nan
    with pytest.raises(ValueError):
        analyze_directions(changed, bootstrap=10)


def test_common_id_file_must_remain_complete(documents, tmp_path):
    scores, ids = tmp_path / "scores.csv", tmp_path / "ids.csv"
    documents.to_csv(scores, index=False)
    pd.DataFrame({"source_id":sorted(documents.source_id.unique())[:-1]}).to_csv(ids, index=False)
    with pytest.raises(ValueError, match="123"):
        load_common_documents(scores, ids)
