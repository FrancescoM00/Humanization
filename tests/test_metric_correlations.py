"""Controlla il campione e le correlazioni descrittive con un riferimento indipendente."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from scripts.correlate_humanizer_metrics import METRICS, PRIMARY_SYSTEMS, correlations, load_panel


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results_humanizer_metric_catalogue/document_metrics.csv"
IDS = ROOT / "results_humanizer_semantics_final/common_source_ids.csv"


@pytest.fixture(scope="module")
def panel():
    return load_panel(SCORES, IDS)


def test_common_panel_keeps_all_four_outputs_per_source(panel):
    assert len(panel) == 492
    assert panel.source_id.nunique() == 123
    assert panel.groupby("humanizer").size().to_dict() == {s:123 for s in PRIMARY_SYSTEMS}


def test_every_matrix_matches_independent_scipy_spearman(panel):
    matrices, pairs = correlations(panel)
    assert len(matrices) == 5 and len(pairs) == 105
    for scope, matrix in matrices.items():
        own = panel if scope == "pooled_492" else panel[panel.humanizer.eq(scope)]
        expected = spearmanr(own[METRICS].to_numpy(), axis=0).statistic
        np.testing.assert_allclose(matrix.to_numpy(), expected, atol=3e-16, rtol=0)
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
        np.testing.assert_array_equal(np.diag(matrix), np.ones(7))
    assert pairs.groupby(["scope", "humanizer"], dropna=False).size().eq(21).all()


def test_method_mixing_is_visible_in_the_saved_data(panel):
    matrices, _ = correlations(panel)
    assert matrices["pooled_492"].loc["rouge_l_f1", "bleu_0_100"] == pytest.approx(.954785, abs=1e-6)
    assert matrices["pooled_492"].loc["rouge_l_f1", "flesch_delta"] > 0
    assert matrices["dipper"].loc["rouge_l_f1", "flesch_delta"] < 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "nonfinite"])
def test_incomplete_scores_are_not_silently_dropped(tmp_path, change):
    scores = pd.read_csv(SCORES)
    if change == "missing":
        scores = scores.iloc[1:]
    elif change == "duplicate":
        scores = pd.concat([scores, scores.iloc[[0]]])
    else:
        scores.loc[0, "embedding_cosine"] = np.inf
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_panel(path, IDS)


def test_undefined_constant_metric_is_explicit(panel):
    constant = panel.copy()
    constant["flesch_delta"] = 0
    with pytest.raises(ValueError, match="costante"):
        correlations(constant)
