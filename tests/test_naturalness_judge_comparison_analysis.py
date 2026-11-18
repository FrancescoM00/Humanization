"""I giudici si confrontano sugli stessi originali, senza usare chiamate API."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from humanizers.naturalness_panel import make_panel_plan
from humanizers.naturalness_judge_comparison_analysis import analyze_judge_comparison


SYSTEMS = ["small", "large", "passage", "windows"]
META = ["request_id", "source_id", "source_group_id", "domain", "generator", "order_id",
        "repeat_index", "system_order", "original_word_count"]


def plans(shared_group=False):
    panels = [dict(source_id=f"s{i}", source_group_id="shared" if shared_group else f"g{i}",
        domain="Essay", generator="Claude", original_word_count=100, systems=SYSTEMS,
        texts={system: f"A complete passage for {system}." for system in SYSTEMS}) for i in range(3)]
    return make_panel_plan(panels, "qwen/test"), make_panel_plan(panels, "llama/test")


def rows(plan, values=(1, 2, 4, 5)):
    quality = dict(zip(SYSTEMS, values))
    result = []
    for item in plan["requests"]:
        row = {name: deepcopy(item[name]) for name in META}
        row.update(status="ok", label_source="LLM", requested_model=plan["model"],
            response_model=plan["model"], judgment={"ratings": {
                f"T{i}": quality[system] for i, system in enumerate(item["system_order"], 1)}})
        result.append(row)
    return result


def test_identical_scores_have_perfect_descriptive_agreement_and_zero_shifts(tmp_path):
    ref, cand = plans()
    summary = analyze_judge_comparison(ref, rows(ref), cand, rows(cand), tmp_path, bootstrap=40)
    assert summary["n_common_complete_sources"] == 3
    assert summary["n_paired_ratings"] == 3*12*4
    agreement = pd.read_csv(tmp_path/"ordinal_agreement.csv")
    assert np.allclose(agreement.exact_agreement, 1)
    overall = agreement[agreement.scope == "overall"].iloc[0]
    assert overall.spearman == pytest.approx(1)
    assert overall.quadratic_weighted_kappa == pytest.approx(1)
    # Per il singolo metodo tutti i voti sono costanti: correlazione e kappa
    # non sono definiti, anche quando l'accordo esatto è del 100%.
    assert agreement.loc[agreement.scope == "system", "spearman"].isna().all()
    assert agreement.loc[agreement.scope == "system", "quadratic_weighted_kappa"].isna().all()
    shifts = pd.read_csv(tmp_path/"system_shifts.csv")
    assert np.allclose(shifts.mean_shift_candidate_minus_reference, 0)
    assert np.allclose(shifts.ci95_low, 0) and np.allclose(shifts.ci95_high, 0)
    contrasts = pd.read_csv(tmp_path/"humanizer_contrast_shifts.csv")
    assert np.allclose(contrasts.contrast_shift_candidate_minus_reference, 0)
    assert (tmp_path/"judge_system_means.png").is_file()


def test_partial_sources_from_either_judge_are_excluded_but_raw_records_remain(tmp_path):
    ref, cand = plans()
    ref_rows, cand_rows = rows(ref), rows(cand)
    ref_rows.pop(next(i for i, x in enumerate(ref_rows) if x["source_id"] == "s0"))
    cand_rows.pop(next(i for i, x in enumerate(cand_rows) if x["source_id"] == "s1"))
    summary = analyze_judge_comparison(ref, ref_rows, cand, cand_rows, tmp_path, bootstrap=40)
    assert summary["common_complete_source_ids"] == ["s2"]
    for role in ["reference", "candidate"]:
        matched = pd.read_csv(tmp_path/f"matched_{role}"/"source_system_scores.csv")
        assert set(matched.source_id) == {"s2"}
        raw = pd.read_csv(tmp_path/f"raw_{role}_ratings.csv")
        assert set(raw.source_id) == {"s0", "s1", "s2"}
    shifts = pd.read_csv(tmp_path/"system_shifts.csv")
    assert set(shifts.n_sources) == {1}
    assert shifts.ci95_low.isna().all()


@pytest.mark.parametrize("change", ["prompt", "order", "temperature", "group", "slot"])
def test_changed_material_order_or_other_parameters_are_rejected(tmp_path, change):
    ref, cand = plans()
    item = cand["requests"][0]
    if change == "prompt":
        item["payload"]["messages"][0]["content"] += " Different instruction."
    elif change == "order":
        item["system_order"] = list(reversed(item["system_order"]))
    elif change == "temperature":
        item["payload"]["temperature"] = .5
    elif change == "group":
        item["source_group_id"] = "another-original"
    else:
        item["repeat_index"] = 99
    with pytest.raises(ValueError):
        analyze_judge_comparison(ref, rows(ref), cand, rows(cand), tmp_path, bootstrap=40)


def test_no_completed_candidate_produces_empty_primary_files_without_fabricated_scores(tmp_path):
    ref, cand = plans()
    summary = analyze_judge_comparison(ref, rows(ref), cand, [], tmp_path, bootstrap=40)
    assert summary["n_common_complete_sources"] == summary["n_paired_ratings"] == 0
    for name in ["system_shifts.csv", "humanizer_contrast_shifts.csv", "ordinal_agreement.csv",
                 "matched_ratings.csv", "system_summary_by_judge.csv"]:
        assert pd.read_csv(tmp_path/name).empty
    assert (tmp_path/"reference_judgments.json").is_file()
    assert not list(tmp_path.glob("*.png"))
    assert not any(path.suffix in {".md", ".txt", ".pdf"} for path in tmp_path.rglob("*"))


def test_shared_group_preserves_source_weight_and_has_no_false_independent_interval(tmp_path):
    ref, cand = plans(shared_group=True)
    summary = analyze_judge_comparison(ref, rows(ref), cand, rows(cand, (2, 3, 4, 5)),
                                      tmp_path, bootstrap=40)
    assert summary["n_common_source_groups"] == 1
    shifts = pd.read_csv(tmp_path/"system_shifts.csv").set_index("system")
    assert shifts.loc["small", "mean_shift_candidate_minus_reference"] == 1
    assert shifts.loc["passage", "mean_shift_candidate_minus_reference"] == 0
    assert shifts.ci95_low.isna().all()
    contrasts = pd.read_csv(tmp_path/"humanizer_contrast_shifts.csv")
    item = contrasts[(contrasts.system_a == "small") & (contrasts.system_b == "passage")].iloc[0]
    assert item.contrast_shift_candidate_minus_reference == 1


def test_model_specific_reasoning_option_may_differ_but_response_provenance_must_match(tmp_path):
    ref, cand = plans()
    cand.pop("reasoning_effort")
    for item in cand["requests"]:
        item["payload"].pop("reasoning_effort")
    analyze_judge_comparison(ref, rows(ref), cand, rows(cand), tmp_path/"allowed", bootstrap=40)
    incorrect = rows(cand)
    incorrect[0]["response_model"] = ref["model"]
    with pytest.raises(ValueError):
        analyze_judge_comparison(ref, rows(ref), cand, incorrect, tmp_path/"wrong-origin", bootstrap=40)
