#!/usr/bin/env python3
"""Completa il catalogo descrittivo di similarità e leggibilità dei riscrittori."""
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
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_catalogue_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_catalogue_cache"))

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.metric_catalogue import METRIC_DEFINITIONS, evaluate_catalogue


def figures(documents, output):
    """Mostra distribuzioni descrittive sulle stesse fonti, senza graduatorie."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = documents[documents.in_common_sources]
    labels = ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]
    n = common.source_id.nunique()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    for ax, metric, label, unit in zip(axes.flat,
            ["rouge_l_f1", "bleu_0_100", "bert_document_chunk_f1", "embedding_cosine"],
            ["ROUGE-L F1", "BLEU per documento", "BERTScore F1 a chunk", "Coseno MiniLM"],
            ["0–1", "0–100", "F1 grezzo, senza riscalatura", "similarità coseno"]):
        values = [common.loc[common.humanizer.eq(s), metric].to_numpy() for s in PRIMARY_SYSTEMS]
        ax.boxplot(values, tick_labels=labels, showfliers=False, whis=(0, 100))
        if metric == "bleu_0_100":
            ax.set_ylim(-2, 102)
        else:
            ax.set_ylim(min(-.02, min(float(v.min()) for v in values)-.02), 1.02)
        ax.set(title=label, ylabel=unit)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
    fig.suptitle(f"Catalogo descrittivo: {n} fonti comuni\n"
                 "Scatole Q1–Q3, linea mediana, baffi intera distribuzione; non misure di umanità", fontsize=11)
    fig.savefig(output / "catalogue_similarity.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    original = common[common.humanizer.eq(PRIMARY_SYSTEMS[0])].flesch_original.to_numpy()
    values = [original] + [common.loc[common.humanizer.eq(s), "flesch_rewrite"].to_numpy() for s in PRIMARY_SYSTEMS]
    axes[0].boxplot(values, tick_labels=["Originali"] + labels, showfliers=False, whis=(0, 100))
    axes[0].set(title="Flesch Reading Ease", ylabel="Punti: più alto = lettura stimata più facile")
    delta = [common.loc[common.humanizer.eq(s), "flesch_delta"].to_numpy() for s in PRIMARY_SYSTEMS]
    axes[1].boxplot(delta, tick_labels=labels, showfliers=False, whis=(0, 100))
    axes[1].axhline(0, ls="--", color="gray", lw=.8)
    axes[1].set(title="Variazione rispetto allo stesso originale", ylabel="Flesch riscrittura − Flesch originale")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle(f"Leggibilità su {n} fonti comuni: descrizione, non giudizio di naturalezza", fontsize=11)
    fig.savefig(output / "catalogue_readability.png", dpi=170)
    plt.close(fig)
    return ["catalogue_similarity.png", "catalogue_readability.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results_humanizer_metric_catalogue")
    args = parser.parse_args()
    result = evaluate_catalogue(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("provenance", {}).get("input_sha256") != result["provenance"].get("input_sha256"):
            raise ValueError("La cartella contiene input diversi: scegliere un nuovo output")
    tables = {
        "document_metrics.csv": result["documents"],
        "summary_descriptive.csv": result["summary"],
        "summary_common_median.csv": result["common_median"],
        "summary_common_mean.csv": result["common_mean"],
        "coverage.csv": result["coverage"],
        "identity_control.csv": result["identity"],
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False)
    (args.output / "metric_definitions.json").write_text(json.dumps(METRIC_DEFINITIONS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    images = figures(result["documents"], args.output)
    manifest = dict(protocol="humanizer_complete_metric_catalogue_v1", created_at=datetime.now(timezone.utc).isoformat(),
        provenance=result["provenance"], validation=result["validation"],
        runtime_versions={name:importlib.metadata.version(name) for name in
                          ["rouge-score", "sacrebleu", "textstat", "nltk", "numpy", "pandas"]},
        role="Descriptive catalogue on real rewrites; no combined score, thresholds, p-values or ranking of humanization quality",
        comparison_scope="Common sources for cross-humanizer tables; all valid outputs retained in per-method summaries; expansions included",
        lexical_execution="ROUGE-L and document-level sentence BLEU recomputed on CPU and checked against archived scores",
        readability_execution="Flesch Reading Ease and Flesch-Kincaid Grade recomputed with textstat en_US and local cmudict, compared with archived text features",
        neural_execution="Exact saved BERTScore and MiniLM scores reused; final window2 NLI included as a separate existing diagnostic; no new model inference",
        bertscore_protocol="Primary document F1 is the existing chunk-matched approximation, not standard BERTScore on a truncated document. Additional aligned/rescaled metrics use a separate protocol",
        readability_note="Flesch is readability, not semantic fidelity or naturalness. Delta is calculated per source before aggregation; median(delta) need not equal difference of medians",
        summary_note="Medians and quartiles describe documents; means/SD also saved. BLEU 0–100, other similarity scores unscaled; Flesch not clipped to 0–100",
        thresholds="none", model_inference_executed=False, api_calls_executed=False,
        analysis_code_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            "semantics/metric_catalogue.py", "scripts/summarize_humanizer_metrics.py"]},
        files=[*tables, "metric_definitions.json"], figures=images)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["humanizer", "n_documents", "rouge_l_f1", "bleu_0_100", "bert_document_chunk_f1",
               "embedding_cosine", "flesch_original", "flesch_rewrite", "flesch_delta"]
    print(result["common_median"][columns].to_string(index=False))
    print("Risultati:", args.output)
    print("Nessuna nuova inferenza di modelli o chiamata API.")


if __name__ == "__main__":
    main()
