"""Le unità di analisi sono le fonti complete, non le singole chiamate ripetute."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from humanizers.naturalness_panel import make_panel_plan
from humanizers.naturalness_panel_analysis import analyze_panel


MODEL = "qwen/qwen3.8-27b"
SYSTEMS = ["small", "large", "passage", "windows"]
METADATA = ["request_id", "source_id", "source_group_id", "domain", "generator",
            "order_id", "repeat_index", "system_order", "original_word_count"]


def panels(shared_group=False):
    return [dict(source_id=f"source-{i}", source_group_id="shared" if shared_group else f"group-{i}",
                 domain="Essay" if i < 2 else "Reuters", generator="Claude",
                 original_word_count=[100, 200, 400][i], systems=SYSTEMS,
                 texts={name: f"A natural passage with wording {j}." for j, name in enumerate(SYSTEMS)})
            for i in range(3)]


def judged(plan, score):
    rows = []
    for item in plan["requests"]:
        row = {name: deepcopy(item[name]) for name in METADATA}
        row.update(status="ok", label_source="LLM", requested_model=MODEL, response_model=MODEL,
                   judgment={"ratings": {f"T{position}": score(item, position, system)
                             for position, system in enumerate(item["system_order"], 1)}})
        rows.append(row)
    return rows


def test_method_scores_are_restored_without_artificial_position_effect(tmp_path):
    plan = make_panel_plan(panels(), MODEL)
    quality = dict(zip(SYSTEMS, [1, 2, 4, 5]))
    rows = judged(plan, lambda item, position, system: quality[system])
    summary = analyze_panel(plan, rows, tmp_path, bootstrap=40)
    scores = pd.read_csv(tmp_path / "source_system_scores.csv")
    assert summary["n_complete_sources"] == 3
    assert len(scores) == 12 and set(scores.n_ratings) == {12}
    assert all(row.mean_rating == quality[row.system] for row in scores.itertuples())
    system = pd.read_csv(tmp_path / "system_summary.csv").set_index("system")
    assert system.mean_rating.to_dict() == quality
    assert set(system.n_sources) == {3}
    assert set(system.n_source_groups) == {3}
    effects = pd.read_csv(tmp_path / "position_effects.csv")
    assert np.allclose(effects.mean_centered_rating, 0)
    repeat = pd.read_csv(tmp_path / "repeatability.csv")
    assert repeat.exact_agreement.all() and np.allclose(repeat.sd_rating, 0)
    order = pd.read_csv(tmp_path / "order_sensitivity.csv")
    assert np.allclose(order.order_mean_range, 0)
    pairs = pd.read_csv(tmp_path / "paired_differences.csv")
    for row in pairs.itertuples():
        assert row.mean_difference == quality[row.system_a] - quality[row.system_b]


def test_position_only_bias_is_visible_and_balanced_method_means_are_equal(tmp_path):
    plan = make_panel_plan(panels(), MODEL)
    ratings_by_position = {1: 1, 2: 2, 3: 4, 4: 5}
    rows = judged(plan, lambda item, position, system: ratings_by_position[position])
    analyze_panel(plan, rows, tmp_path, bootstrap=40)
    system = pd.read_csv(tmp_path / "system_summary.csv")
    assert np.allclose(system.mean_rating, 3)
    effects = pd.read_csv(tmp_path / "position_effects.csv")
    overall = effects[(effects.scope == "overall") & (effects.stratum == "all")].set_index("position")
    assert np.allclose(overall.loc[[1, 2, 3, 4], "mean_centered_rating"], [-2, -1, 1, 2])
    repeat = pd.read_csv(tmp_path / "repeatability.csv")
    assert np.allclose(repeat.sd_rating, 0)
    sensitivity = pd.read_csv(tmp_path / "order_sensitivity.csv")
    assert np.allclose(sensitivity.order_mean_range, 4)
    assert np.allclose(sensitivity.within_order_sd_rms, 0)


def test_repeat_noise_is_distinguished_from_order_sensitivity(tmp_path):
    plan = make_panel_plan(panels(), MODEL)
    rows = judged(plan, lambda item, position, system: [1, 3, 5][item["repeat_index"]])
    analyze_panel(plan, rows, tmp_path, bootstrap=40)
    repeat = pd.read_csv(tmp_path / "repeatability.csv")
    assert not repeat.exact_agreement.any()
    assert np.allclose(repeat.rating_range, 4)
    sensitivity = pd.read_csv(tmp_path / "order_sensitivity.csv")
    assert np.allclose(sensitivity.order_mean_range, 0)
    assert (sensitivity.within_order_sd_rms > 0).all()


def test_incomplete_sources_are_visible_but_do_not_enter_primary_scores(tmp_path):
    selected = panels()[:2]
    plan = make_panel_plan(selected, MODEL)
    rows = judged(plan, lambda item, position, system: 4)
    missing = next(r for r in rows if r["source_id"] == "source-0")
    rows.remove(missing)
    eligibility = dict(selected_source_ids=["source-0", "source-1", "generation-failed"],
                       systems=SYSTEMS, failures=[dict(source_id="generation-failed",
                       source_group_id="failed-group", domain="Reuters", generator="Claude",
                       original_word_count=400)])
    summary = analyze_panel(plan, rows, tmp_path, eligibility=eligibility, bootstrap=40)
    assert summary["n_selected_sources"] == 3
    assert summary["n_eligible_sources"] == 2
    assert summary["n_complete_sources"] == 1
    raw = pd.read_csv(tmp_path / "ratings.csv")
    assert len(raw[raw.source_id == "source-0"]) == 11 * 4
    assert not raw[raw.source_id == "source-0"].complete_panel.any()
    scores = pd.read_csv(tmp_path / "source_system_scores.csv")
    assert set(scores.source_id) == {"source-1"}
    coverage = pd.read_csv(tmp_path / "source_coverage.csv").set_index("source_id")
    assert coverage.loc["source-0", "status"] == "partial"
    assert coverage.loc["source-0", "missing_requests"] == 1
    assert coverage.loc["generation-failed", "status"] == "generation_ineligible"


def test_shared_original_group_is_not_counted_as_several_independent_clusters(tmp_path):
    plan = make_panel_plan(panels(shared_group=True), MODEL)
    rows = judged(plan, lambda item, position, system: 3)
    analyze_panel(plan, rows, tmp_path, bootstrap=40)
    system = pd.read_csv(tmp_path / "system_summary.csv")
    assert set(system.n_sources) == {3}
    assert set(system.n_source_groups) == {1}
    assert system.ci95_low.isna().all() and system.ci95_high.isna().all()
    paired = pd.read_csv(tmp_path / "paired_differences.csv")
    assert set(paired.n_source_groups) == {1}
    assert paired.ci95_low.isna().all() and paired.ci95_high.isna().all()


def test_planned_analysis_does_not_fabricate_ratings_or_primary_results(tmp_path):
    plan = make_panel_plan(panels(), MODEL)
    summary = analyze_panel(plan, [], tmp_path, bootstrap=40)
    assert summary["n_complete_sources"] == summary["n_valid_requests"] == 0
    assert pd.read_csv(tmp_path / "ratings.csv").empty
    assert pd.read_csv(tmp_path / "source_system_scores.csv").empty
    assert pd.read_csv(tmp_path / "system_summary.csv").empty
    assert not any(p.suffix.lower() in {".md", ".txt", ".pdf"} for p in tmp_path.rglob("*") if p.is_file())


def test_duplicate_judgments_and_changed_metadata_are_rejected(tmp_path):
    plan = make_panel_plan(panels(), MODEL)
    rows = judged(plan, lambda item, position, system: 3)
    with pytest.raises(ValueError):
        analyze_panel(plan, rows + [rows[0]], tmp_path / "duplicate", bootstrap=40)
    rows[0]["domain"] = "Another genre"
    with pytest.raises(ValueError):
        analyze_panel(plan, rows, tmp_path / "changed", bootstrap=40)
