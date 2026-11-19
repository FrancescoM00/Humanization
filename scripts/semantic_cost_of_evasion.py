#!/usr/bin/env python3
"""Unisce evasione dei detector e supporto semantico sulla stessa riscrittura."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_exp3_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_exp3_cache"))

from threadpoolctl import threadpool_limits

from humanizers.semantic_evasion_data import build_long_table
from humanizers.semantic_evasion_analysis import analyze


def figures(result, config, output):
    """Due pannelli per metrica; le bande sono puntuali e ricampionano le fonti."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = dict(zip(config["methods"], ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]))
    colors = dict(zip(config["methods"], ["#2864a0", "#d77b14", "#208378", "#a74891"]))
    detector_labels = {"openai_roberta_base":"OpenAI RoBERTa-base", "binoculars_falcon7b":"Binoculars"}
    files = []
    for profile, title, filename in [
            ("nli_mean", "Supporto NLI: media conservativa", "asr_curve_nli.png"),
            ("bertscore_rescaled_mean", "BERTScore allineato: media riscalata", "asr_curve_bertscore.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), constrained_layout=True, sharey=True)
        for ax, detector in zip(axes, config["detectors"]):
            for method in config["methods"]:
                part = result["curves"]
                part = part[part.detector.eq(detector) & part.method.eq(method) & part.curve.eq(profile)].sort_values("tau")
                ax.plot(part.tau, part.asr, label=labels[method], color=colors[method], lw=1.8)
                ax.fill_between(part.tau, part.bootstrap_ci95_low, part.bootstrap_ci95_high,
                                color=colors[method], alpha=.13)
            for tau in config["sensitivity_taus"]:
                primary = tau == config["moderate"]["nli_cons_mean_min"]
                ax.axvline(tau, color="#444444" if primary else "#888888",
                           ls="--" if primary else ":", lw=1.2 if primary else .8)
            n = int(result["main_table"].loc[result["main_table"].detector.eq(detector), "n_original_detected"].iloc[0])
            ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Requisito di supporto: score ≥ τ",
                   title=f"{detector_labels[detector]} — denominatore {n}")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.grid(alpha=.15)
        axes[0].set_ylabel("Evasioni con requisito semantico / originali rilevati")
        axes[1].legend(loc="upper right", fontsize=8)
        repetitions = config["bootstrap_replicates"]
        fig.suptitle(title + f"\nDenominatore fisso; bootstrap per fonte, {repetitions:,} repliche; τ = 0,7 / 0,8 / 0,9".replace(",000", ".000"), fontsize=11)
        fig.supxlabel("Proxy automatici: stesso τ tra metriche non equivale alla stessa fedeltà. Bande puntuali; zero eventi può dare bande degeneri.", fontsize=8)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        files.append(filename)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "results_semantic_cost_evasion/config_exp3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results_semantic_cost_evasion")
    args = parser.parse_args()
    body = args.config.read_bytes()
    config = json.loads(body)
    config_sha = hashlib.sha256(body).hexdigest()
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("config_sha256") != config_sha:
        raise ValueError("La cartella contiene una configurazione diversa: scegliere un nuovo output")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "config_exp3.json"
    if frozen.exists() and frozen.read_bytes() != body:
        raise ValueError("Non sostituire una configurazione congelata: usare una nuova cartella")
    if not frozen.exists():
        frozen.write_bytes(body)
    # La verifica delle etichette e dell'ASR precede ogni calcolo semantico condizionato.
    data = build_long_table(config)
    print("Join e ASR di riferimento verificati:", flush=True)
    print(data["baseline_check"].to_string(index=False), flush=True)
    with threadpool_limits(limits=2):
        result = analyze(data["long"], config)
    tables = {
        "evasion_semantics_long.csv": data["long"],
        "baseline_asr_check.csv": data["baseline_check"],
        "main_results.csv": result["main_table"],
        "asr_by_threshold.csv": result["curves"],
        "sensitivity_thresholds.csv": result["sensitivity"],
        "evaded_vs_not.csv": result["evaded_vs_not"],
        "score_reduction_correlations.csv": result["score_correlations"],
        "method_comparisons_mcnemar.csv": result["method_comparisons"],
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False)
    images = figures(result, config, args.output)
    manifest = dict(protocol="semantic_cost_of_evasion_v1", created_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=config_sha, config_definition_commit="efb7d32",
        provenance=data["provenance"], baseline_check="All eight common-panel ASRs checked before conditional semantic analysis",
        n_sources=123, n_rewrites=492, n_detector_rows=len(data["long"]),
        analysis_status=config["analysis_status"], thresholds_blind=False,
        denominator=config["asr_denominator"], bootstrap_unit=config["bootstrap_unit"],
        bootstrap_replicates=config["bootstrap_replicates"], bootstrap_seed=config["bootstrap_seed"],
        interval_policy=config["interval_policy"], group_population=config["evaded_vs_not_population"],
        definitions="Moderate and stringent are operational proxy-score requirements, not human-verified preservation labels",
        score_comparison="BERTScore and NLI thresholds use the same numerical grid, not matched calibration or identical evidence contexts",
        empty_groups="No imputation; unavailable comparisons or resamples are explicitly counted",
        causal_interpretation="None. Association between evasion and proxy support does not identify meaning lost or a causal cost",
        selection_limit="The common panel already excludes incomplete generations; semantic scores can therefore be more favorable than for all attempts",
        analysis_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            "humanizers/semantic_evasion_data.py", "humanizers/semantic_evasion_analysis.py", "scripts/semantic_cost_of_evasion.py"]},
        runtime_versions={name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "matplotlib"]},
        model_inference_executed=False, api_calls_executed=False,
        files=[*tables, "config_exp3.json"], figures=images)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["main_table"].to_string(index=False))
    print("Risultati:", args.output)


if __name__ == "__main__":
    main()
