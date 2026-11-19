"""Controlla la valutazione semantica finale e il confronto appaiato per fonte."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import (
    ALERT_METRICS, HISTORICAL_REFERENCE, NLI_METRICS, load_semantic_study,
    paired_source_comparisons, prepare_nli_documents, summarize_documents,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip"


@pytest.fixture(scope="module")
def study():
    data = load_semantic_study(ARCHIVE)
    documents, common, statuses = prepare_nli_documents(data)
    return data, documents, common, statuses


def synthetic_documents(*, identical=False):
    """Quattro metodi sugli stessi quattro testi, con allarmi sovrapposti."""
    rows = []
    for method_index, system in enumerate(PRIMARY_SYSTEMS):
        offset = 0 if identical else .02 * method_index
        for index, value in enumerate([.1, .2, .7, .9]):
            row = dict(id=f"{system}:{index}", source_id=f"source{index}",
                       source_group_id=f"group{index}", humanizer=system, domain="Essay",
                       generator="generator", in_common_sources=True,
                       original_n_words=10, modified_n_words=11 if index == 3 else 9,
                       below_historical_reference_only=False)
            row.update({name:value + offset for name in NLI_METRICS})
            row["nli_balanced_direction_mean"] = value + offset
            row.update({name:False for name in ALERT_METRICS})
            row["alert_number"] = index in {0, 1}
            row["alert_date"] = index == 0
            row["alert_entity"] = index == 2
            row["alert_hedge_removal"] = index == 2
            row["alert_any"] = any(row[name] for name in ALERT_METRICS if name != "alert_any")
            rows.append(row)
    return pd.DataFrame(rows)


def test_real_cache_covers_every_valid_rewrite_and_identity(study):
    _, documents, common, statuses = study
    assert len(documents) == 626
    assert len(common) == 123
    assert documents.groupby("humanizer").size().to_dict() == {
        "qwen_humanize": 124, "qwen14b_humanize": 126,
        "dipper": 125, "dipper_context3": 125, "identity": 126}
    primary = documents[documents.humanizer.isin(PRIMARY_SYSTEMS)]
    assert len(primary) == 500
    assert set(primary.loc[primary.in_common_sources, "source_id"]) == common
    assert primary[primary.in_common_sources].groupby("humanizer").size().eq(123).all()
    assert len(statuses) == 630
    assert statuses.status.eq("ok").sum() == 626
    assert documents.nli_forward_coverage.eq(1).all()
    assert documents.nli_reverse_coverage.eq(1).all()


def test_final_primary_min_matches_previously_saved_scores(study):
    _, documents, _, _ = study
    old = pd.read_csv(ROOT / "results_kaggle_v2_study/nli_primary_scores.csv")
    old = old[old.primary_status.eq("scored")].set_index("id")
    current = documents.set_index("id").reindex(old.index)
    assert set(documents.id) == set(old.index)
    for field in ("source_id", "source_group_id", "humanizer", "domain", "generator"):
        assert current[field].equals(old[field])
    np.testing.assert_allclose(current.nli_bidirectional_min, old.primary_score, atol=1e-15, rtol=0)
    np.testing.assert_allclose(current.nli_forward_min, old.primary_forward, atol=1e-15, rtol=0)
    np.testing.assert_allclose(current.nli_reverse_min, old.primary_reverse, atol=1e-15, rtol=0)


def test_bidirectional_mean_preserves_the_existing_worst_direction_definition(study):
    _, documents, _, _ = study
    expected = np.minimum(documents.nli_forward_mean, documents.nli_reverse_mean)
    np.testing.assert_array_equal(documents.nli_bidirectional_mean, expected)
    balanced = .5 * (documents.nli_forward_mean + documents.nli_reverse_mean)
    np.testing.assert_array_equal(documents.nli_balanced_direction_mean, balanced)
    assert (documents.nli_balanced_direction_mean > documents.nli_bidirectional_mean).any()
    assert (documents.nli_bidirectional_mean >= documents.nli_bidirectional_min).all()


def test_valid_expansions_remain_in_the_nli_panel(study):
    _, documents, common, _ = study
    with zipfile.ZipFile(ARCHIVE) as archive:
        controls = [json.loads(line) for line in archive.read("null_controls/rewrites.jsonl").splitlines()]
    expected_counts = dict(zip(PRIMARY_SYSTEMS, [25, 19, 4, 17]))
    common_counts = dict(zip(PRIMARY_SYSTEMS, [25, 17, 4, 17]))
    for system in PRIMARY_SYSTEMS:
        expanded = {row["source_id"] for row in controls
                    if row["humanizer"] == "null_length__" + system
                    and row["metadata"].get("length_match_status") == "target_longer_than_original"}
        available = set(documents.loc[documents.humanizer.eq(system), "source_id"])
        assert len(expanded) == expected_counts[system]
        assert expanded <= available
        assert len(expanded & common) == common_counts[system]


@pytest.mark.parametrize("corruption", ["missing_unit", "alignment_hash", "aggregate", "manifest_hash"])
def test_incomplete_or_mismatched_cache_is_rejected(study, corruption):
    original, _, _, _ = study
    data = deepcopy(original)
    if corruption == "missing_unit":
        data["units"].pop()
    elif corruption == "alignment_hash":
        identifier = data["pairs"][0]["id"]
        data["alignment"][identifier]["modified_sha256"] = "different"
    elif corruption == "aggregate":
        index = data["nli"].index[data["nli"]["mode"].eq("window2")][0]
        data["nli"].loc[index, "mean"] = .01
    else:
        data["nli_run"]["pairs_sha256"] = "different"
    with pytest.raises(ValueError):
        prepare_nli_documents(data)


def test_summary_reports_quartiles_union_of_alerts_and_descriptive_reference():
    documents = synthetic_documents(identical=True)
    # Un solo valore viene posto sotto il riferimento storico: nessuna etichetta di errore.
    documents.loc[documents.source_id.eq("source0"), "below_historical_reference_only"] = True
    result = summarize_documents(documents)
    distributions = result["distributions"]
    row = distributions[(distributions.scope == "common_sources")
                        & (distributions.stratum == "overall")
                        & (distributions.humanizer == PRIMARY_SYSTEMS[0])
                        & (distributions.metric == "nli_bidirectional_min")].iloc[0]
    assert row.n_documents == 4 and row.n_expanded == 1
    assert row.q25 == pytest.approx(.175)
    assert row["median"] == pytest.approx(.45)
    assert row.q75 == pytest.approx(.75)
    assert row.iqr == pytest.approx(.575)
    alerts = result["alerts"]
    alerts = alerts[(alerts.scope == "common_sources") & (alerts.stratum == "overall")
                    & (alerts.humanizer == PRIMARY_SYSTEMS[0])].set_index("alert")
    assert alerts.loc["alert_any", "n_flagged"] == 3
    assert alerts.loc["alert_any", "fraction"] == .75
    assert alerts.loc["alert_number", "fraction"] == .5
    assert alerts.loc["alert_date", "fraction"] == .25
    zero = alerts.loc["alert_negation"]
    assert zero.wilson95_low == 0 and zero.wilson95_high > 0
    historical = result["historical"]
    assert historical.n_below_historical_reference.eq(1).all()
    assert historical.historical_reference.eq(HISTORICAL_REFERENCE).all()
    assert historical.decision_threshold.eq("none").all()
    assert not any("failure" in column or "violation" in column for column in historical.columns)


def test_identical_methods_have_zero_deltas_and_flag_degenerate_bootstrap():
    result = paired_source_comparisons(synthetic_documents(identical=True), bootstrap=200, seed=12)
    assert result.difference_b_minus_a.eq(0).all()
    assert result.bootstrap_degenerate.all()
    assert result.ci95_low.isna().all() and result.ci95_high.isna().all()
    assert result.interval_status.eq("no_observed_bootstrap_variation").all()


def test_paired_bootstrap_preserves_source_alignment_and_is_reproducible():
    documents = synthetic_documents()
    first = paired_source_comparisons(documents, bootstrap=300, seed=7)
    second = paired_source_comparisons(documents.sample(frac=1, random_state=42), bootstrap=300, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.n_sources.eq(4).all()
    nli = first[first.metric.isin(NLI_METRICS)]
    for row in nli.itertuples():
        expected = .02 * (PRIMARY_SYSTEMS.index(row.humanizer_b) - PRIMARY_SYSTEMS.index(row.humanizer_a))
        assert row.difference_b_minus_a == pytest.approx(expected)
    common = documents.copy()
    common.loc[common.source_id.eq("source3"), "in_common_sources"] = False
    excluded = paired_source_comparisons(common, bootstrap=100, seed=7)
    assert excluded.n_sources.eq(3).all()


def test_unpaired_sources_cannot_enter_the_comparison():
    documents = synthetic_documents()
    documents = documents.drop(documents.index[0])
    with pytest.raises(ValueError, match="stesse fonti"):
        paired_source_comparisons(documents, bootstrap=100)


def test_real_qwen_median_difference_uses_the_same_123_sources(study):
    _, documents, _, _ = study
    # Il confronto richiede le colonne delle guardie, qui non oggetto del test NLI.
    documents = documents.assign(**{metric:False for metric in ALERT_METRICS})
    result = paired_source_comparisons(documents, bootstrap=300, seed=42)
    row = result[(result.stratum == "overall") & (result.metric == "nli_bidirectional_min")
                 & (result.humanizer_a == "qwen_humanize")
                 & (result.humanizer_b == "qwen14b_humanize")].iloc[0]
    assert row.n_sources == 123
    assert row.value_a == pytest.approx(.10971000790596008)
    assert row.value_b == pytest.approx(.10487779229879379)
    assert row.difference_b_minus_a == pytest.approx(row.value_b-row.value_a)
    assert row.ci95_low <= row.ci95_high


@pytest.mark.parametrize("bootstrap,seed", [(0, 1), (2, -1)])
def test_invalid_bootstrap_settings_are_rejected(bootstrap, seed):
    with pytest.raises(ValueError):
        paired_source_comparisons(synthetic_documents(), bootstrap=bootstrap, seed=seed)
