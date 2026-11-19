#!/usr/bin/env python3
"""Valutatore semantico finale: NLI window2 salvato, guardie lessicali e bootstrap."""
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
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_semantics_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "humanization_semantics_cache"))

import numpy as np
import pandas as pd
import spacy
from threadpoolctl import threadpool_limits

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import (
    ALERT_METRICS, HISTORICAL_REFERENCE, load_semantic_study, paired_source_comparisons,
    prepare_nli_documents, summarize_documents,
)
from semantics.humanizer_guards import definitions, score_pairs


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _figures(documents, alerts, output):
    """Rappresenta distribuzioni e quote sul medesimo campione di fonti."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = documents[documents.in_common_sources]
    labels = ["Qwen 3B", "Qwen 14B NF4", "DIPPER intero", "DIPPER tre frasi"]
    n = common.source_id.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["nli_bidirectional_min", "nli_bidirectional_mean"],
                                 ["Minimo bidirezionale", "Media della direzione più debole"]):
        values = [common.loc[common.humanizer.eq(s), metric].to_numpy() for s in PRIMARY_SYSTEMS]
        ax.boxplot(values, tick_labels=labels, showfliers=False, whis=(0, 100))
        ax.set(ylim=(-.025, 1.025), ylabel="Supporto NLI", title=title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle(f"NLI con premesse fino a due frasi: {n} fonti comuni\n"
                 "Scatola: Q1–Q3; linea: mediana; baffi: intera distribuzione; nessuna soglia decisionale", fontsize=10)
    fig.savefig(output / "nli_distributions_common.png", dpi=170)
    plt.close(fig)
    data = alerts.query("scope == 'common_sources' and stratum == 'overall'")
    values = data.pivot(index="humanizer", columns="alert", values="fraction").reindex(
        index=PRIMARY_SYSTEMS, columns=ALERT_METRICS).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(11, 4.1), constrained_layout=True)
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(4), labels)
    ax.set_xticks(range(7), ["Numeri", "Date", "Entità", "Negazioni", "Quantificatori", "Cautele ridotte", "Almeno una"], rotation=15, ha="right")
    for row in range(4):
        for col in range(7):
            ax.text(col, row, f"{values[row,col]:.1f}%", ha="center", va="center",
                    color="white" if values[row,col] > 60 else "black")
    ax.set_title(f"Segnalazioni lessicali deterministiche sulle stesse {n} fonti\n"
                 "Differenze da verificare: non sono percentuali di errori semantici confermati", fontsize=10)
    fig.colorbar(im, ax=ax, label="Documenti con segnalazione (%)")
    fig.savefig(output / "lexical_alert_rates_common.png", dpi=170)
    plt.close(fig)
    return ["nli_distributions_common.png", "lexical_alert_rates_common.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results_humanizer_semantics_final")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.seed < 0 or args.threads < 1:
        parser.error("Almeno una replica, un thread e seme non negativo")
    data = load_semantic_study(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("input_sha256") != data["input_sha256"]:
        raise ValueError("La cartella contiene input diversi: scegliere un nuovo output")
    documents, common, status = prepare_nli_documents(data)
    nlp = spacy.load("en_core_web_sm")
    def progress(done, total):
        print(f"Inventari lessicali: {done}/{total}", flush=True)
    with threadpool_limits(limits=args.threads):
        guards = score_pairs(data["pairs"], nlp=nlp, progress_callback=progress)
        documents = documents.merge(guards, on="id", validate="one_to_one")
        if len(documents) != len(data["pairs"]):
            raise ValueError("Le guardie devono coprire tutte le coppie NLI valide")
        summaries = summarize_documents(documents)
        comparisons = paired_source_comparisons(documents, bootstrap=args.bootstrap, seed=args.seed)
    identity = documents[documents.humanizer.eq("identity")]
    if identity.alert_any.any():
        raise AssertionError("Il confronto identità non deve produrre segnalazioni lessicali")
    primary = documents[documents.humanizer.isin(PRIMARY_SYSTEMS)].copy()
    coverage = []
    for system in PRIMARY_SYSTEMS:
        attempted = status[status.humanizer.eq(system)]
        own = primary[primary.humanizer.eq(system)]
        coverage.append(dict(humanizer=system, n_attempted=len(attempted), n_valid=len(own),
            n_failed=int(attempted.status.ne("ok").sum()), n_complete_bidirectional_nli=len(own),
            n_expanded_included=int(own.modified_n_words.gt(own.original_n_words).sum()),
            n_common_sources=len(common), n_excluded_from_common=len(attempted)-len(common)))
    files = {
        "document_scores.csv": primary,
        "nli_distributions.csv": summaries["distributions"],
        "lexical_alert_rates.csv": summaries["alerts"],
        "historical_reference_only.csv": summaries["historical"],
        "paired_source_comparisons.csv": comparisons,
        "coverage.csv": pd.DataFrame(coverage),
        "generation_status.csv": status,
        "identity_control.csv": identity,
        "common_source_ids.csv": pd.DataFrame({"source_id":sorted(common)}),
    }
    for name, table in files.items():
        serial = table.copy()
        for column in serial:
            if serial[column].map(lambda value:isinstance(value, (dict, list))).any():
                serial[column] = serial[column].map(lambda value:json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value)
        serial.to_csv(args.output / name, index=False)
    pairs = {r["id"]:r for r in data["pairs"]}
    with (args.output / "lexical_alert_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in primary[primary.alert_any].to_dict("records"):
            row.update(original=pairs[row["id"]]["original"], modified=pairs[row["id"]]["modified"])
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")
    figures = _figures(primary, summaries["alerts"], args.output)
    manifest = dict(protocol="humanizer_final_semantics_window2_lexical_v1", created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=data["input_sha256"], n_attempts_per_system=126, n_valid_outputs=len(primary),
        n_common_sources=len(common), common_source_ids=sorted(common), coverage=coverage,
        nli_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        nli_configured_revision=data["config"]["semantic_models"]["nli"], historical_nli_run=data["nli_run"],
        nli_cache_policy="Reuse complete window2 scores for exact saved texts; recompute document aggregates from saved units; no new NLI inference",
        nli_window_definition="Native contiguous windows of one or two aligned sentence/chunk units; same original hypothesis units, <=160 NLI tokens per chunk; max384 tokens per pair; no selective coverage averaging",
        forward="Original as premise supports rewritten hypotheses; weak support can suggest additions or altered claims",
        reverse="Rewrite as premise supports original hypotheses; weak support can suggest lost information",
        document_min="Minimum over hypothesis supports in each direction; bidirectional_min = min(forward_min, reverse_min)",
        document_mean="Mean over hypotheses in each direction; bidirectional_mean = min(forward_mean, reverse_mean), matching the existing suite",
        balanced_direction_mean="Additional explicitly separate diagnostic: 0.5*(forward_mean+reverse_mean)",
        distributions="Median, Q1 and Q3 across documents; IQR = Q3-Q1; all_valid per method and common_sources; no decision threshold",
        historical_reference=dict(value=HISTORICAL_REFERENCE, role="Descriptive count only, not a calibrated failure threshold for humanizer outputs"),
        lexical_rules=definitions(), spacy_version=spacy.__version__, spacy_model=nlp.meta,
        alerts_interpretation="Potential changes for review, not confirmed semantic violations; numeric/date format normalization is limited; NER aliases/coreference and scope can produce false alarms",
        alert_rate_denominator="All valid documents in the declared scope, including those with no original item of that type; added items may still trigger a warning",
        comparison="Same common source IDs for all four humanizers; resample sources jointly; difference B minus A of document medians for NLI, rates for alerts",
        bootstrap_replicates=args.bootstrap, seed=args.seed,
        intervals="Pointwise paired-source percentile95, descriptive, no corrected significance claims; degenerate bootstrap intervals left empty with status, not interpreted as zero uncertainty",
        alert_rate_intervals="Marginal Wilson two-sided95 intervals, including zero events; not simultaneous intervals",
        generation_failure_policy="Explicit coverage and status; no fabricated NLI score or semantic label for incomplete generations",
        scope="Semantic diagnostic application to real paraphrasers, not a new gold-labelled validation; human annotation remains separate",
        historical_code_provenance="Recorded historical NLI code hash retained; current analysis code does not replace historical inference provenance",
        analysis_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            "semantics/humanizer_evaluation.py", "semantics/humanizer_guards.py", "scripts/evaluate_humanizer_semantics.py"]},
        nli_model_inference_executed=False, spacy_inference_executed=True, api_calls_executed=False,
        files=[*files, "lexical_alert_evidence.jsonl"], figures=figures)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    print(summaries["distributions"].query("scope == 'common_sources' and stratum == 'overall' and metric in ['nli_bidirectional_min','nli_bidirectional_mean']")[["humanizer", "metric", "n_documents", "median", "q25", "q75"]].to_string(index=False))
    print(summaries["alerts"].query("scope == 'common_sources' and stratum == 'overall'")[["humanizer", "alert", "n_flagged", "n_documents", "fraction"]].to_string(index=False))
    print("Risultati:", args.output)


if __name__ == "__main__":
    main()
