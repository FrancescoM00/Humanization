#!/usr/bin/env python3
"""Ricalcola i controlli nulli dagli score salvati, con Wilson, McNemar e Holm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_null_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_null_cache"))

from humanizers.detector_performance import PRIMARY_SYSTEMS
from humanizers.null_control_analysis import CONTROLS, SCOPES, analyze_null_controls, load_null_study
from humanizers.null_control_statistics import wilson_interval


def _figures(summary, output):
    """Mostra gli ASR sul campione comune e la sensibilità agli allungamenti."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    detectors = ("openai_roberta_base", "fakespot_roberta_base_raw", "radar_vicuna", "binoculars_falcon7b")
    labels = ("OpenAI RoBERTa", "Fakespot", "RADAR", "Binoculars")
    system_labels = ("Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, detector, label in zip(axes.flat, detectors, labels):
        part = summary[(summary.panel == "all126") & (summary.scope == "common_eligible")
                       & (summary.control_kind == "length") & (summary.detector == detector)].set_index("humanizer").loc[list(PRIMARY_SYSTEMS)]
        x = np.arange(4)
        for offset, field, name, color in [(-.17, "control", "Solo prefisso", "#7c9296"),
                                          (.17, "reference", "Riscrittura", "#155e63")]:
            y = 100 * part[field + "_asr"].to_numpy()
            low = 100 * part[field + "_asr_ci95_low"].to_numpy()
            high = 100 * part[field + "_asr_ci95_high"].to_numpy()
            ax.bar(x + offset, y, .32, color=color, label=name)
            ax.errorbar(x + offset, y, yerr=[np.maximum(0, y-low), np.maximum(0, high-y)],
                        fmt="none", ecolor="black", capsize=3, lw=1)
        for i, (_, row) in enumerate(part.iterrows()):
            if row.greater_than_control_holm60:
                ax.text(i, min(103, 100 * row.reference_asr_ci95_high + 3), "*", ha="center", fontsize=16)
        ax.set(ylim=(0, 109), title=label, ylabel="ASR (%)")
        ax.set_xticks(x, system_labels, rotation=12, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Controllo di lunghezza: 78 fonti comuni ai quattro riscrittori\n"
                 "IC Wilson 95%; * McNemar bilaterale, Holm sui 60 confronti dei tre controlli", fontsize=11)
    fig.savefig(output / "length_common_sources.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, detector, label in zip(axes.flat, detectors, labels):
        for scope, style, name in [("eligible_per_humanizer", "o-", "Eleggibili del singolo metodo"),
                                   ("common_eligible", "s-", "78 fonti comuni"),
                                   ("all_success_per_humanizer", "^-", "Tutti gli output validi, inclusi allungamenti")]:
            part = summary[(summary.panel == "all126") & (summary.scope == scope)
                           & (summary.control_kind == "length") & (summary.detector == detector)].set_index("humanizer").loc[list(PRIMARY_SYSTEMS)]
            ax.plot(range(4), 100 * part.reference_asr, style, label=name, lw=1)
        ax.set(ylim=(-3, 103), title=label, ylabel="ASR della riscrittura (%)")
        ax.set_xticks(range(4), system_labels, rotation=12, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Sensibilità dell'ASR alla selezione delle fonti\n"
                 "Campioni diversi: grafico descrittivo, nessun confronto fra metodi fuori dal pannello comune", fontsize=11)
    fig.savefig(output / "expansion_sensitivity.png", dpi=170)
    plt.close(fig)
    return ["length_common_sources.png", "expansion_sensitivity.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results_null_controls_robustness")
    args = parser.parse_args()
    data = load_null_study(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("input_sha256") != data["input_sha256"]:
        raise ValueError("La cartella contiene input diversi: scegliere un nuovo output")
    result = analyze_null_controls(data)
    summary = result["summary"]
    common_length = summary[summary.scope.eq("common_eligible") & summary.control_kind.eq("length")]
    sensitivity = summary[summary.control_kind.eq("length") & summary.scope.isin(
        ["eligible_per_humanizer", "common_eligible", "all_success_per_humanizer", "common_success_for_control"])]
    joint = summary[summary.scope.eq("common_all_controls")]
    files = {
        "comparisons.csv": summary,
        "length_common_sources.csv": common_length,
        "all_three_same_sources.csv": joint,
        "expansion_sensitivity.csv": sensitivity,
        "eligibility.csv": result["eligibility"],
        "coverage.csv": result["coverage"],
        "paired_scores.csv": result["pairs"],
        "panel_membership.csv": result["memberships"],
        "three_control_evidence.csv": result["three_controls"],
    }
    for name, table in files.items():
        table.to_csv(args.output / name, index=False)
    figures = _figures(summary, args.output)
    manifest = dict(protocol="null_controls_paired_robustness_v2",
        created_at=datetime.now(timezone.utc).isoformat(), input_sha256=data["input_sha256"],
        panel_sizes=result["panel_sizes"], systems=list(PRIMARY_SYSTEMS), controls=list(CONTROLS), scopes=list(SCOPES),
        primary_length_table="all126 / common_eligible / length: same 78 sources for all four systems",
        primary_three_control_table="all126 / common_all_controls: same 33 sources for all systems and controls",
        analysis_status="Retrospective robustness analysis of inspected study data; not a new confirmatory holdout",
        inference="Two-sided exact McNemar on control-only vs rewrite-only evasions among baseline-detected originals",
        multiplicity="Holm20 within panel/scope/control (five detectors x four writers), plus Holm60 across all three controls within panel/scope; Holm60 used for primary evidence",
        scope_multiplicity="Panels and scopes are not independent confirmations; no selection of a favorable scope. Alternative scopes are sensitivity analyses, not extra discovery families",
        asr_intervals="Two-sided Wilson 95%, one independent source per trial; pointwise descriptive intervals, not simultaneous intervals",
        zero_success_example=dict(successes=0, trials=95, wilson95_upper=wilson_interval(0, 95)[1], note="3.1% is the exact one-sided upper limit, not the two-sided 95% interval"),
        delta_asr="reference rewrite ASR minus control ASR; positive means more evasion for the rewrite",
        tpr="TPR includes every source in the selected panel; descriptive net change can differ from conditional ASR",
        scopes_description={
            "eligible_per_humanizer": "Original strict eligibility, different sources by writer; reproduces the old diagnostic counts",
            "common_eligible": "Intersection of strict eligibility over all four writers, separately for each control",
            "all_success_per_humanizer": "Include expanded valid outputs using already saved capped prefixes; noise must still be applicable",
            "common_success_for_control": "Intersection of valid/capped-applicable outputs for all writers, separately for each control; length uses 123, noise can use fewer",
            "common_all_controls": "Strict eligibility intersection over all four writers and all three controls; same sources everywhere"},
        expansion_policy="For expanded rewrites, saved length control equals the unchanged original; flagged as NOT length matched. This sensitivity does not identify a causal length effect",
        failure_policy="No score imputation; unsuccessful or truncated generations remain excluded and counted",
        noise_policy="Frozen dictionary, at most one replacement with POS/NER restrictions; unchanged texts are not counted as noise. No eligible word in 56/126 original texts; prefixes can remove eligible words",
        evidence_rule="Positive ASR delta and Holm60 p<0.05 for each of all three controls; common_all_controls is the comparable joint panel. Failing this rule is insufficient evidence, not proof of no effect",
        majority_vote="Historical vote of the three supervised classifiers; not independent of them",
        calibration_policy="Frozen thresholds and directions; calibration humans disjoint from test; no refitting",
        source_of_truth="New CSVs supersede bootstrap ASR intervals and unadjusted effect claims in results_kaggle_v2_study/null_comparisons.csv; old files retained unchanged",
        model_inference_executed=False, api_calls_executed=False, files=list(files), figures=figures)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    display = common_length[common_length.panel.eq("all126")]
    print(display[["detector", "humanizer", "eligible_sources", "eligible_original_detected", "control_successes",
                   "reference_successes", "mcnemar_p_holm_20", "mcnemar_p_holm_60"]].to_string(index=False))
    print("Risultati:", args.output)
    print("Nessuna nuova inferenza o chiamata API.")


if __name__ == "__main__":
    main()
