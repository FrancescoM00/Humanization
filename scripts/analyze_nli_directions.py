#!/usr/bin/env python3
"""Separa forward e reverse e mostra la distribuzione completa dell'NLI salvato."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_nli_directions_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_nli_directions_cache"))

import numpy as np
from threadpoolctl import threadpool_limits

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import HISTORICAL_REFERENCE
from semantics.nli_directional_analysis import analyze_directions, load_common_documents


def figures(result, output):
    """L'ECDF conserva ogni osservazione, anche quelle inferiori a 0,001."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = dict(zip(PRIMARY_SYSTEMS, ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]))
    colors = dict(zip(PRIMARY_SYSTEMS, ["#2864a0", "#d77b14", "#208378", "#a74891"]))
    documents, points = result["documents"], result["ecdf_points"]
    minimum = float(documents.nli_bidirectional_min.min())
    if minimum <= 0:
        raise ValueError("La scala logaritmica richiede valori positivi: nessuno zero può essere sostituito o escluso")
    lower = 10. ** math.floor(math.log10(minimum))
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for system in PRIMARY_SYSTEMS:
        part = points[points.humanizer.eq(system)].sort_values("score")
        x = np.r_[lower, part.score.to_numpy(), 1.]
        y = np.r_[0., part.cumulative_fraction.to_numpy(), 1.]
        ax.step(x, y, where="post", label=labels[system], color=colors[system], lw=1.9)
    ax.axvline(HISTORICAL_REFERENCE, color="#444444", ls="--", lw=1.2,
               label="0,04075: riferimento storico")
    ax.set(xscale="log", xlim=(lower, 1), ylim=(0, 1.01),
           xlabel="Minimo NLI bidirezionale (scala logaritmica)",
           ylabel="Quota di documenti con minimo ≤ x",
           title="Distribuzione cumulativa completa: 123 fonti comuni\n"
                 "Ogni curva comprende tutti i documenti; il riferimento storico non è una soglia decisionale")
    ticks = [10. ** k for k in range(int(math.log10(lower)), 1)]
    ax.set_xticks(ticks, [format(value, "g").replace(".", ",") for value in ticks])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(which="major", alpha=.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    files = []
    for extension in ("png", "svg"):
        name = "nli_min_ecdf_log." + extension
        fig.savefig(output / name, dpi=200)
        files.append(name)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True, sharey=True)
    for ax, system in zip(axes.flat, PRIMARY_SYSTEMS):
        own = documents[documents.humanizer.eq(system)]
        correlation = result["correlations"].set_index("humanizer").loc[system]
        ax.scatter(own.word_ratio, own.nli_reverse_min, s=18, alpha=.65, color=colors[system])
        ax.axvline(1, color="gray", ls=":", lw=.8)
        ax.set(yscale="log", ylim=(lower, 1.05), xlabel="Parole riscrittura / parole originale",
               ylabel="Minimo NLI reverse", title=f"{labels[system]} — Spearman ρ = {correlation['rho']:.3f}")
        ax.grid(alpha=.15)
    fig.suptitle("Lunghezza e supporto reverse: associazioni descrittive, non effetti causali", fontsize=11)
    fig.savefig(output / "reverse_min_vs_length_ratio.png", dpi=180)
    plt.close(fig)
    files.append("reverse_min_vs_length_ratio.png")
    return files, lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "results_humanizer_semantics_final/document_scores.csv")
    parser.add_argument("--common-ids", type=Path, default=ROOT / "results_humanizer_semantics_final/common_source_ids.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results_humanizer_semantics_final/directional_analysis")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.seed < 0:
        parser.error("Almeno una replica bootstrap e seme non negativo")
    input_sha = {"document_scores.csv":hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                 "common_source_ids.csv":hashlib.sha256(args.common_ids.read_bytes()).hexdigest()}
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("input_sha256") != input_sha:
        raise ValueError("La cartella contiene input diversi: scegliere un nuovo output")
    documents = load_common_documents(args.scores, args.common_ids)
    with threadpool_limits(limits=2):
        result = analyze_directions(documents, bootstrap=args.bootstrap, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = ["id", "source_id", "source_group_id", "humanizer", "domain", "generator",
               "nli_forward_min", "nli_reverse_min", "nli_forward_mean", "nli_reverse_mean",
               "nli_bidirectional_min", "original_n_words", "modified_n_words", "word_ratio"]
    tables = {
        "common_document_scores.csv": result["documents"][columns],
        "directional_summary.csv": result["directional_summary"],
        "directional_summary_long.csv": result["directional_long"],
        "correlations_length.csv": result["correlations"],
        "weakest_direction.csv": result["weakest_direction"],
        "support_fractions.csv": result["support_fractions"],
        "ecdf_points.csv": result["ecdf_points"],
        "paired_directional_comparisons.csv": result["pairwise_comparisons"],
        "focused_paired_comparisons.csv": result["focus_comparisons"],
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False)
    images, lower = figures(result, args.output)
    manifest = dict(protocol="nli_directions_full_distribution_v1", created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=input_sha, n_sources=int(documents.source_id.nunique()), n_documents=len(documents),
        systems=list(PRIMARY_SYSTEMS), bootstrap_replicates=args.bootstrap, seed=args.seed,
        scores="Saved complete window2 NLI; no new model inference and no source selection based on support or length",
        direction_interpretation="Forward can indicate unsupported additions/changes; reverse can indicate omissions. These are model-based diagnostics, not confirmed human labels",
        quartiles="Document median and linear Q1/Q3, separately for forward_min,reverse_min,forward_mean,reverse_mean",
        bootstrap="Same sorted source IDs and original seed derivation; paired source percentile95 of B minus A document medians; all four outputs kept together",
        correlations="Spearman(reverse_min, rewrite/original alphabetic word ratio); same source bootstrap draws, recompute average ranks inside every replicate; no causal interpretation",
        intervals="Pointwise descriptive95; no familywise significance claims; no observed bootstrap variation is marked rather than reported as certain equality",
        focus="Two pairs named explicitly: DIPPER three sentences versus whole passage, Qwen14B versus3B. All six humanizer pairs retained",
        weakest_direction="Strict reverse_min < forward_min or forward_min < reverse_min; exact ties reported separately",
        ecdf="Right-continuous empirical CDF P(score<=x), complete common panel; no clipping, no smoothing or formal bimodality claim",
        log_x_min=lower, observed_minimum=float(documents.nli_bidirectional_min.min()),
        historical_reference=dict(value=HISTORICAL_REFERENCE, role="Reference line only, never a semantic decision threshold"),
        fractions="Descriptive shares with score >=0.5 and >=0.8, inclusive; not calibrated success or failure rates",
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            "semantics/nli_directional_analysis.py", "scripts/analyze_nli_directions.py"]},
        runtime_versions={name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy"]},
        model_inference_executed=False, api_calls_executed=False, files=list(tables), figures=images)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["directional_summary"].to_string(index=False))
    print(result["correlations"].to_string(index=False))
    print(result["support_fractions"].to_string(index=False))
    print("Risultati:", args.output)


if __name__ == "__main__":
    main()
