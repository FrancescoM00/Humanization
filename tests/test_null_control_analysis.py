"""Verifica i pannelli dei controlli usando gli score archiviati dello studio."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humanizers.detector_performance import PRIMARY_SYSTEMS
from humanizers.null_control_analysis import CONTROLS, analyze_null_controls, load_null_study


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip"


@pytest.fixture(scope="module")
def study():
    data = load_null_study(ARCHIVE)
    return data, analyze_null_controls(data)


def select_summary(result, *, panel="all126", scope="eligible_per_humanizer", kind="length"):
    frame = result["summary"]
    return frame[frame.panel.eq(panel) & frame.scope.eq(scope) & frame.control_kind.eq(kind)]


def test_legacy_length_counts_and_point_estimates_are_reproduced(study):
    _, result = study
    old = pd.read_csv(ROOT / "results_kaggle_v2_study/null_comparisons.csv", keep_default_na=False)
    old = old[old.panel.isin(["all126", "unseen84"])
              & old.condition.str.startswith(("null_length__", "null_length_noise__"))]
    new = result["summary"]
    new = new[new.scope.eq("eligible_per_humanizer")
              & new.control_kind.isin(["length", "length_and_lexical_noise"])]
    assert len(new) == len(old) == 80
    keys = ["panel", "detector", "condition", "reference_humanizer"]
    new = new.rename(columns={"humanizer": "reference_humanizer"}).set_index(keys).sort_index()
    old = old.set_index(keys).sort_index()
    assert new.index.equals(old.index)
    for field in ("total_sources", "eligible_sources", "eligible_original_detected", "control_successes",
                  "reference_successes", "control_only_successes", "reference_only_successes",
                  "control_asr", "reference_asr", "delta_asr_reference_minus_control",
                  "eligible_original_tpr", "eligible_control_tpr", "eligible_reference_tpr"):
        np.testing.assert_allclose(new[field].to_numpy(float), old[field].to_numpy(float), atol=1e-14, rtol=0)
    np.testing.assert_array_equal(new.excluded_sources, old.ineligible_sources)


def test_common_panels_use_the_same_sources_for_every_humanizer(study):
    _, result = study
    assert result["panel_sizes"] == {"all126": 126, "unseen84": 84}
    expected = {"length": 78, "lexical_noise": 69, "length_and_lexical_noise": 33}
    memberships = result["memberships"]
    for panel in ("all126", "unseen84"):
        for scope in ("common_eligible", "common_success_for_control", "common_all_controls"):
            part = memberships[memberships.panel.eq(panel) & memberships.scope.eq(scope)]
            for kind, rows in part.groupby("control_kind"):
                sets = [set(rows.loc[rows.humanizer.eq(system), "source_id"]) for system in PRIMARY_SYSTEMS]
                assert all(sources == sets[0] for sources in sets)
                if panel == "all126" and scope == "common_eligible":
                    assert len(sets[0]) == expected[kind]
            if scope == "common_all_controls":
                sets = [set(rows.source_id) for _, rows in part.groupby(["control_kind", "humanizer"])]
                assert len(sets) == 12
                assert all(sources == sets[0] for sources in sets)
                if panel == "all126":
                    assert len(sets[0]) == 33


def test_common_lengths_are_an_actual_intersection_of_eligibility(study):
    _, result = study
    eligible = result["eligibility"]
    eligible = eligible[eligible.control_kind.eq("length") & eligible.strict_eligible]
    sets = [set(eligible.loc[eligible.humanizer.eq(system), "source_id"]) for system in PRIMARY_SYSTEMS]
    common = set.intersection(*sets)
    assert len(common) == 78
    assert dict(zip(PRIMARY_SYSTEMS, map(len, sets))) == {
        "qwen_humanize": 99, "qwen14b_humanize": 107, "dipper": 121, "dipper_context3": 108}
    memberships = result["memberships"]
    selected = memberships[memberships.panel.eq("all126") & memberships.scope.eq("common_eligible")
                           & memberships.control_kind.eq("length")]
    assert set(selected.source_id) == common


def test_expansion_sensitivity_includes_successful_longer_rewrites(study):
    data, result = study
    expected = {"qwen_humanize": 124, "qwen14b_humanize": 126, "dipper": 125, "dipper_context3": 125}
    summary = select_summary(result, scope="all_success_per_humanizer")
    for system, n in expected.items():
        assert set(summary.loc[summary.humanizer.eq(system), "eligible_sources"]) == {n}
        successes = {r["source_id"] for r in data["rewrites"]
                     if r["humanizer"] == system and r["status"] == "ok"}
        memberships = result["memberships"]
        actual = memberships[memberships.panel.eq("all126")
                             & memberships.scope.eq("all_success_per_humanizer")
                             & memberships.control_kind.eq("length") & memberships.humanizer.eq(system)]
        assert set(actual.source_id) == successes
    common = select_summary(result, scope="common_success_for_control")
    assert set(common.eligible_sources) == {123}


def test_expanded_length_controls_keep_the_original_instead_of_padding(study):
    _, result = study
    pairs = result["pairs"]
    expanded = pairs[pairs.control_kind.eq("length") & pairs.rewrite_longer]
    assert len(expanded) > 0
    assert expanded.capped_control_not_length_matched.all()
    assert not expanded.strict_eligible.any()
    assert expanded.capped_eligible.all()
    np.testing.assert_array_equal(expanded.original_is_ai, expanded.control_is_ai)
    np.testing.assert_array_equal(expanded.original_raw_score, expanded.control_raw_score)
    assert (expanded.control_n_words == expanded.original_n_words).all()
    assert (expanded.reference_n_words > expanded.control_n_words).all()


def test_exclusions_separate_generation_failures_from_expansion_and_missing_noise(study):
    _, result = study
    eligibility = result["eligibility"]
    qwen = eligibility[eligibility.humanizer.eq("qwen_humanize") & eligibility.control_kind.eq("length")]
    assert qwen.rewrite_longer.sum() == 25
    assert qwen.reference_status.ne("ok").sum() == 2
    assert qwen.strict_eligible.sum() == 99
    noise = eligibility[eligibility.control_kind.eq("lexical_noise")]
    # La possibilità di inserire rumore nell'originale non dipende dal riscrittore.
    assert noise.groupby("source_id").noise_applied.nunique().eq(1).all()
    assert noise.groupby("source_id").noise_applied.first().sum() == 70
    assert (~noise.groupby("source_id").noise_applied.first()).sum() == 56


def test_wilson_zero_success_intervals_and_multiplicity_families(study):
    _, result = study
    summary = result["summary"]
    for label in ("control", "reference"):
        selected = summary[summary[f"{label}_successes"].eq(0) & summary.eligible_original_detected.gt(0)]
        assert len(selected) > 0
        assert selected[f"{label}_asr_ci95_low"].eq(0).all()
        assert selected[f"{label}_asr_ci95_high"].gt(0).all()
    assert summary.mcnemar_p_holm_20_family_size.eq(20).all()
    assert summary.mcnemar_p_holm_60_family_size.eq(60).all()
    assert summary.groupby(["panel", "scope", "control_kind"]).size().eq(20).all()
    assert summary.groupby(["panel", "scope"]).size().eq(60).all()
    assert (summary.mcnemar_p_holm_20 >= summary.mcnemar_p).all()
    assert (summary.mcnemar_p_holm_60 >= summary.mcnemar_p_holm_20).all()


def test_exact_corrected_results_change_with_the_common_source_panel(study):
    _, result = study
    legacy = select_summary(result)
    common = select_summary(result, scope="common_eligible")
    expected_legacy = {("openai_roberta_base", "qwen_humanize"),
                       ("openai_roberta_base", "qwen14b_humanize"),
                       ("binoculars_falcon7b", "qwen_humanize"),
                       ("binoculars_falcon7b", "qwen14b_humanize"),
                       ("radar_vicuna", "dipper_context3")}
    expected_common = expected_legacy - {("binoculars_falcon7b", "qwen_humanize"),
                                         ("radar_vicuna", "dipper_context3")}
    assert set(zip(legacy.loc[legacy.greater_than_control_holm20, "detector"],
                   legacy.loc[legacy.greater_than_control_holm20, "humanizer"])) == expected_legacy
    assert set(zip(common.loc[common.greater_than_control_holm20, "detector"],
                   common.loc[common.greater_than_control_holm20, "humanizer"])) == expected_common
    qwen_binoculars = common[common.detector.eq("binoculars_falcon7b")
                             & common.humanizer.eq("qwen_humanize")].iloc[0]
    assert qwen_binoculars.mcnemar_p_holm_20 == pytest.approx(.06640625)


def test_conditional_asr_does_not_equal_the_net_change_in_tpr(study):
    _, result = study
    summary = select_summary(result, panel="unseen84")
    row = summary[summary.detector.eq("radar_vicuna") & summary.humanizer.eq("qwen_humanize")].iloc[0]
    assert row.eligible_sources == 64
    assert row.eligible_original_detected == 49
    assert row.control_successes == 0 and row.reference_successes == 3
    assert row.delta_asr_reference_minus_control == pytest.approx(3 / 49)
    assert row.eligible_control_detected == 53
    assert row.eligible_reference_detected == 52
    assert row.reference_minus_control_tpr == pytest.approx(-1 / 64)
    assert row.eligible_reference_tpr > row.eligible_original_tpr


def test_joint_comparison_requires_all_three_controls_and_the_declared_source_panel(study):
    _, result = study
    summary, joint = result["summary"], result["three_controls"]
    assert set(summary.control_kind) == set(CONTROLS)
    assert not joint[joint.scope.ne("common_all_controls")].same_sources_all_controls.any()
    assert joint[joint.scope.eq("common_all_controls")].same_sources_all_controls.all()
    for row in joint.to_dict("records"):
        selected = summary
        for field in ("panel", "scope", "detector", "humanizer"):
            selected = selected[selected[field].eq(row[field])]
        assert len(selected) == 3
        assert row["greater_than_all_three_holm60"] == bool(selected.greater_than_control_holm60.all())
