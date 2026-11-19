#!/usr/bin/env python3
"""Calcola accuratezza, AUROC e variazioni prima/dopo dagli score già disponibili."""
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
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_detector_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_detector_cache"))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from humanizers.detector_performance import PRIMARY_SYSTEMS, evaluate_saved_detection, load_saved_detection


def _figures(result, output):
    """Mostra i delta e le curve ROC sullo stesso pannello comune ai riscrittori."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    table = result["metrics"]
    common = table[(table.scope == "common_success") & (table.stratum == "overall")
                   & table.humanizer.isin(PRIMARY_SYSTEMS) & table.auroc_available]
    detectors = list(dict.fromkeys(common.detector))
    system_labels = dict(zip(PRIMARY_SYSTEMS, ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]))
    detector_labels = {"openai_roberta_base": "OpenAI RoBERTa", "fakespot_roberta_base_raw": "Fakespot",
                       "radar_vicuna": "RADAR", "binoculars_falcon7b": "Binoculars"}
    files = []
    if not len(common):
        for name in ("detector_performance_deltas.png", "detector_roc_before_after.png"):
            (output / name).unlink(missing_ok=True)
        return files
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, column, scale, title in zip(axes, ["delta_accuracy", "delta_auroc"], [100, 1],
                                      ["Delta accuratezza (punti percentuali)", "Delta AUROC"]):
        values = common.pivot(index="detector", columns="humanizer", values=column).reindex(
            index=detectors, columns=PRIMARY_SYSTEMS) * scale
        limit = max(.01, float(np.nanmax(np.abs(values.to_numpy()))))
        plot = ax.imshow(values, vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
        ax.set_xticks(range(len(PRIMARY_SYSTEMS)), [system_labels[s] for s in PRIMARY_SYSTEMS], rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(detectors)), [detector_labels.get(d, d) for d in detectors], fontsize=8)
        for row in range(len(detectors)):
            for column_index in range(len(PRIMARY_SYSTEMS)):
                value = values.iloc[row, column_index]
                label = f"{value:+.2f}" if scale == 100 else f"{value:+.3f}"
                ax.text(column_index, row, label, color="white" if abs(value) > .6*limit else "black",
                        ha="center", va="center", fontsize=9)
        ax.set_title(title + "\n(dopo − prima)", fontsize=10)
        fig.colorbar(plot, ax=ax, fraction=.04)
    n = int(common.n_pairs.iloc[0])
    fig.suptitle(f"Confronto sul pannello comune: {n} coppie umano/IA")
    fig.savefig(output / "detector_performance_deltas.png", dpi=170)
    plt.close(fig)
    files.append("detector_performance_deltas.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    pairs = result["pairs"]
    for ax, name in zip(axes.flat, detectors):
        panel = pairs[(pairs.detector == name) & pairs.in_common_success]
        for index, system in enumerate(PRIMARY_SYSTEMS):
            part = panel[panel.humanizer == system]
            if not len(part):
                continue
            y = np.r_[np.zeros(len(part)), np.ones(len(part))]
            if index == 0:
                fpr, tpr, _ = roc_curve(y, np.r_[part.human_score, part.original_score])
                ax.plot(fpr, tpr, color="black", ls="--", label=f"Originali ({auc(fpr,tpr):.3f})")
            fpr, tpr, _ = roc_curve(y, np.r_[part.human_score, part.rewrite_score])
            ax.plot(fpr, tpr, label=f"{system_labels[system]} ({auc(fpr,tpr):.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=.7, alpha=.6)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Falsi positivi sugli umani", ylabel="Veri positivi IA",
               title=detector_labels.get(name, name))
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ROC prima/dopo: stessi umani e stessi originali per tutti i riscrittori")
    fig.savefig(output / "detector_roc_before_after.png", dpi=170)
    plt.close(fig)
    files.append("detector_roc_before_after.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results_detector_performance")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--humanizer", action="append", help="Condizione da analizzare, ripetibile")
    args = parser.parse_args()
    if args.bootstrap < 0 or args.seed < 0 or args.threads < 1:
        parser.error("Bootstrap e seme non negativi; almeno un thread")
    data = load_saved_detection(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("input_sha256") != data["input_sha256"]:
            raise ValueError("La cartella contiene un'altra esecuzione: usare un nuovo output")
    with threadpool_limits(limits=args.threads):
        result = evaluate_saved_detection(data, bootstrap=args.bootstrap, seed=args.seed, systems=args.humanizer)
    metrics = result["metrics"]
    overall = metrics[metrics.stratum.eq("overall")].copy()
    primary = overall[overall.scope.eq("common_success") & overall.humanizer.isin(PRIMARY_SYSTEMS)
                      & overall.auroc_available].copy()
    concise = ["detector", "humanizer", "scope", "n_pairs", "n_failed_for_system",
               "accuracy_before", "accuracy_after", "delta_accuracy", "delta_accuracy_ci95_low", "delta_accuracy_ci95_high",
               "auroc_before", "auroc_after", "delta_auroc", "delta_auroc_ci95_low", "delta_auroc_ci95_high",
               "tpr_before", "tpr_after", "delta_tpr", "fpr_before", "fpr_after", "threshold"]
    files = {
        "detector_performance.csv": metrics,
        "performance_overall.csv": overall,
        "common_success_comparison.csv": primary[concise],
        "baseline_test_performance.csv": result["baseline"],
        "generation_coverage_and_accuracy_bounds.csv": result["coverage"],
        "paired_test_scores.csv": result["pairs"],
    }
    for name, table in files.items():
        table.to_csv(args.output / name, index=False)
    figures = _figures(result, args.output)
    calibration = [row for row in data["cohort"] if row["split"] == "calibration"]
    tests = [row for row in data["cohort"] if row["split"] == "test"]
    manifest = dict(
        protocol="detector_before_after_paired_test_v1", created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=data["input_sha256"], bootstrap=args.bootstrap, seed=args.seed,
        n_calibration_humans=len(calibration), n_test_sources=len(tests),
        n_common_success_sources=len(result["common_source_ids"]), common_source_ids=result["common_source_ids"],
        systems=result["systems"], score_directions=result["directions"], frozen_thresholds=data["thresholds"],
        delta_definition="after minus before; negative accuracy/AUROC/TPR means worse detector performance",
        labels="human=0; original machine=1; humanized machine=1 according to generation provenance",
        paired_available="Same human/original/rewrite sources with valid output, separately for each humanizer",
        common_success="Intersection of successful outputs from the four primary humanizers; same sources for all comparisons",
        calibration_policy="Use saved human-only thresholds without recalibration; calibration humans never enter test metrics",
        failure_policy="No missing score imputation; report complete-pair coverage and full-cohort accuracy bounds",
        majority_auroc="Not reported: AUROC of the binary historical vote equals balanced accuracy; no independent continuous ranking score",
        intervals="Pointwise percentile 95%; paired bootstrap of source groups, same draws for both classes and both stages",
        metric_dependence="On the balanced paired panel, accuracy equals balanced accuracy and delta_accuracy = 0.5 * delta_TPR",
        interpretation="Retrospective descriptive additions; these do not replace the existing primary TPR endpoint or conditional ASR",
        files=list(files), figures=figures, model_inference_executed=False,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    identity = overall[overall.humanizer.eq("null_identity")]
    for field in ("delta_accuracy", "delta_tpr", "delta_auroc"):
        if not identity[field].dropna().eq(0).all():
            raise AssertionError("Il controllo identità deve avere tutti i delta esattamente nulli")
    print(primary[["detector", "humanizer", "n_pairs", "delta_accuracy", "delta_auroc", "delta_tpr"]].to_string(index=False))
    print("Risultati:", args.output)
    print("Score già salvati: nessun detector eseguito nuovamente.")


if __name__ == "__main__":
    main()
