#!/usr/bin/env python3
"""Correlazioni descrittive di Spearman fra metriche documentali sui quattro metodi."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_correlations_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_correlations_cache"))

import numpy as np
import pandas as pd

from humanizers.detector_performance import PRIMARY_SYSTEMS


METRICS = ["rouge_l_f1", "bleu_0_100", "bert_document_chunk_f1", "embedding_cosine",
           "nli_bidirectional_min", "nli_bidirectional_mean", "flesch_delta"]
LABELS = ["ROUGE-L", "BLEU", "BERTScore", "Coseno", "NLI minimo", "NLI media", "Δ Flesch"]


def load_panel(scores_path, common_ids_path):
    """Mantiene esattamente 123 fonti per quattro metodi, senza eliminazioni per coppia."""
    ids = pd.read_csv(common_ids_path, keep_default_na=False)
    if "source_id" not in ids or len(ids) != 123 or ids.source_id.duplicated().any():
        raise ValueError("Occorrono i 123 identificativi univoci del campione comune")
    data = pd.read_csv(scores_path, keep_default_na=False, float_precision="round_trip")
    required = {"id", "source_id", "source_group_id", "humanizer", "domain", "generator", *METRICS}
    if not required <= set(data):
        raise ValueError("Mancano metadati o metriche nel catalogo documentale")
    data = data[data.source_id.isin(set(ids.source_id))].copy()
    if (len(data) != 492 or data.id.duplicated().any()
            or data[["source_id", "humanizer"]].duplicated().any()
            or set(data.humanizer) != set(PRIMARY_SYSTEMS)):
        raise ValueError("Il campione deve avere 492 coppie complete, senza duplicati")
    for system in PRIMARY_SYSTEMS:
        if set(data.loc[data.humanizer.eq(system), "source_id"]) != set(ids.source_id):
            raise ValueError("I quattro metodi devono avere esattamente le stesse fonti")
    if (not data.groupby("source_id").source_group_id.nunique().eq(1).all()
            or not data.groupby("source_group_id").source_id.nunique().eq(1).all()):
        raise ValueError("Le fonti devono mantenere gli stessi gruppi nei quattro metodi")
    if not np.isfinite(data[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("Una metrica mancante non può essere eliminata selettivamente dalla correlazione")
    return data[["id", "source_id", "source_group_id", "humanizer", "domain", "generator", *METRICS]]


def correlations(data):
    """Una matrice sulle 492 coppie e quattro matrici entro metodo, senza p-value."""
    panels = {"pooled_492": data}
    panels.update({name:data[data.humanizer.eq(name)] for name in PRIMARY_SYSTEMS})
    matrices, rows = {}, []
    for scope, part in panels.items():
        if not np.isfinite(part[METRICS].to_numpy(dtype=float)).all():
            raise ValueError("Ogni correlazione deve usare tutte le osservazioni del pannello")
        if (part[METRICS].nunique() < 2).any():
            raise ValueError("Spearman non è definito per una metrica costante")
        matrix = part[METRICS].corr(method="spearman")
        matrices[scope] = matrix
        for i, left in enumerate(METRICS):
            for right in METRICS[i+1:]:
                rows.append(dict(scope="pooled" if scope == "pooled_492" else "within_humanizer",
                    humanizer="" if scope == "pooled_492" else scope,
                    metric_a=left, metric_b=right, n_documents=len(part),
                    n_sources=part.source_group_id.nunique(), spearman_rho=float(matrix.loc[left, right])))
    return matrices, pd.DataFrame(rows)


def figures(matrices, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, matrix, title):
        values = matrix.to_numpy()
        im = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(7), LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(7), LABELS, fontsize=8)
        ax.set_title(title, fontsize=10)
        for row in range(7):
            for col in range(7):
                ax.text(col, row, f"{values[row,col]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(values[row,col]) > .6 else "black")
        return im
    fig, ax = plt.subplots(figsize=(8, 6.7), constrained_layout=True)
    im = draw(ax, matrices["pooled_492"], "Spearman su 492 coppie: 123 fonti × quattro metodi")
    fig.colorbar(im, ax=ax, label="ρ di Spearman", fraction=.045)
    fig.suptitle("Correlazioni descrittive; NLI media = media della direzione più debole", fontsize=10)
    fig.savefig(output / "spearman_pooled_heatmap.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)
    for ax, system, title in zip(axes.flat, PRIMARY_SYSTEMS,
                                ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]):
        im = draw(ax, matrices[system], title + " — 123 coppie")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="ρ di Spearman", fraction=.025)
    fig.suptitle("Correlazioni entro ciascun metodo, sulle stesse fonti; nessun test di significatività", fontsize=11)
    fig.savefig(output / "spearman_within_humanizers.png", dpi=180)
    plt.close(fig)
    return ["spearman_pooled_heatmap.png", "spearman_within_humanizers.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "results_humanizer_metric_catalogue/document_metrics.csv")
    parser.add_argument("--common-ids", type=Path, default=ROOT / "results_humanizer_semantics_final/common_source_ids.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results_humanizer_metric_catalogue/correlations")
    args = parser.parse_args()
    input_sha = {"document_metrics.csv":hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                 "common_source_ids.csv":hashlib.sha256(args.common_ids.read_bytes()).hexdigest()}
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("input_sha256") != input_sha:
        raise ValueError("La cartella contiene input diversi: scegliere un nuovo output")
    panel = load_panel(args.scores, args.common_ids)
    matrices, pairs = correlations(panel)
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for scope, matrix in matrices.items():
        name = "spearman_" + scope + ".csv"
        matrix.to_csv(args.output / name, index_label="metric")
        files.append(name)
    panel.to_csv(args.output / "common_document_metrics.csv", index=False)
    pairs.to_csv(args.output / "spearman_pairs.csv", index=False)
    images = figures(matrices, args.output)
    manifest = dict(protocol="humanizer_document_metric_correlations_v1",
        created_at=datetime.now(timezone.utc).isoformat(), input_sha256=input_sha,
        method="Spearman with average ranks for ties", metrics=METRICS,
        n_documents=492, n_sources=123, documents_per_humanizer=123,
        n_distinct_metric_pairs_per_scope=21,
        aggregation="One matrix pooling all four rewrites per source, plus one matrix separately for each humanizer",
        missing_policy="All seven metrics finite in all selected documents; no selective pairwise row deletion",
        nli_mean_definition="min(forward_mean, reverse_mean), not arithmetic mean of directions",
        bertscore_definition="Raw document chunk-matched F1 approximation used in the catalogue",
        delta_flesch_definition="Flesch Reading Ease rewrite minus original, per document",
        interpretation="Descriptive associations, not metric validity or accuracy; pooled correlations can reflect differences among methods and domains",
        repeated_sources="The 492 rows share 123 original sources. They are not 492 independent sources; no iid p-values or confidence intervals are reported",
        statistical_tests="none", model_inference_executed=False, api_calls_executed=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=files + ["common_document_metrics.csv", "spearman_pairs.csv"], figures=images)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(matrices["pooled_492"].round(3).to_string())
    print("Risultati:", args.output)


if __name__ == "__main__":
    main()
