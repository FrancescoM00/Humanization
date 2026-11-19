#!/usr/bin/env python3
"""Prepara e valuta il pilota OpAI riutilizzando i detector dello studio MGTBench."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/humanization_opai_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/humanization_opai_cache")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from humanizers.null_control_statistics import wilson_interval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(output, config):
    """Rifiuta pannelli cambiati dopo il campionamento congelato."""
    manifest = json.loads((output / "selection.json").read_text())
    if manifest["config_sha256"] != sha(output / "config_pilot.json"):
        raise ValueError("Configurazione diversa da quella del pannello")
    if manifest["panel_sha256"] != sha(output / "panel.jsonl"):
        raise ValueError("I testi selezionati sono cambiati")
    rows = [json.loads(s) for s in (output / "panel.jsonl").read_text().splitlines()]
    if len(rows) != 3 * 4 * config["n_documents_per_domain"]:
        raise ValueError("Pannello incompleto")
    return rows


def prepare(config, output, source):
    """Una traiettoria completa per documento; nessuna selezione basata sui punteggi."""
    import pyarrow.parquet as pq
    from semantics.opai import _normalize
    provenance = json.loads((source.parent / "records.provenance.json").read_text())
    if provenance["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("La revisione dei dati non coincide col piano")
    trajectories = defaultdict(dict)
    inputs = []
    excluded = set(config["excluded_document_ids"])
    for item in provenance["source_files"]:
        path = source / Path(item["repository_path"]).name
        if sha(path) != item["sha256"]:
            raise ValueError(f"Parquet ufficiale cambiato: {path}")
        inputs.append({"file":path.name, "sha256":item["sha256"]})
        parquet = pq.ParquetFile(path)
        columns = ["record_id", "document_hash_id", "domain", "generator", "split", "subset", "version_index", "text"]
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise ValueError("Schema ufficiale non supportato: non indovinare le colonne")
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            for raw in batch.to_pylist():
                if raw["split"] != config["split"] or raw["subset"] != config["subset"]:
                    continue
                if int(raw["version_index"]) not in config["versions"]:
                    continue
                row = _normalize(raw)
                if row["document_id"] in excluded:
                    continue
                key = (row["document_id"], row["domain"], row["generator"])
                version = row["version"]
                if version in trajectories[key] and trajectories[key][version]["text"] != row["text"]:
                    raise ValueError("Versione ambigua della stessa traiettoria")
                trajectories[key][version] = row
    available = defaultdict(lambda:defaultdict(list))
    for (doc, domain, editor), versions in sorted(trajectories.items()):
        if set(versions) == set(config["versions"]):
            available[domain][doc].append(editor)
    rng = random.Random(config["seed"])
    rows, pool_counts = [], {}
    for domain in sorted(config["domains"]):
        pool = available[domain]
        pool_counts[domain] = len(pool)
        if len(pool) < config["n_documents_per_domain"]:
            raise ValueError(f"{domain}: solo {len(pool)} documenti eleggibili")
        documents = rng.sample(sorted(pool), config["n_documents_per_domain"])
        for doc in documents:
            editor = rng.choice(sorted(pool[doc]))
            for version in config["versions"]:
                row = trajectories[doc, domain, editor][version]
                text = row["text"]
                rows.append(dict(doc_id=doc, group_id=doc, domain=domain, version=f"v{version}",
                    editor=editor, trajectory_id=f"{doc}|{domain}|{editor}",
                    record_id=row["record_id"], n_words=len(text.split()), text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if {r["doc_id"] for r in rows} & excluded:
        raise ValueError("Sovrapposizione con esperimenti precedenti")
    body = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    panel = output / "panel.jsonl"
    if panel.exists() and panel.read_text() != body:
        raise ValueError("Non sostituire un pannello già congelato")
    panel.write_text(body)
    pd.DataFrame([{k:v for k,v in r.items() if k != "text"} for r in rows]).to_csv(output / "panel_metadata.csv",index=False)
    manifest = dict(config_sha256=sha(output / "config_pilot.json"), panel_sha256=sha(panel),
        n_documents=len(rows)//3, n_texts=len(rows), available_documents_by_domain=pool_counts,
        n_excluded_documents=len(excluded), overlap_with_previous_sources=0,
        dataset_revision=config["dataset_revision"], source_files=inputs,
        selection_without_detector_scores=True)
    (output / "selection.json").write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def score(config, output, detector_name, device, observer_device, performer_device):
    """Chiama le funzioni originali: niente nuova normalizzazione o ricalibrazione."""
    rows = read_panel(output,config)
    detector = next(d for d in config["detectors"] if d["name"]==detector_name)
    if detector_name == "openai_roberta_base":
        import torch
        torch.set_num_threads(2)
        from humanizers.detection import score_detector
        frame, metadata = score_detector(rows,detector,output,device=device)
    else:
        from humanizers.zero_shot import score_binoculars
        frame, metadata = score_binoculars(rows,detector,output,
            observer_device=observer_device,performer_device=performer_device)
    frozen = config["frozen_thresholds"][detector_name]
    frame = frame.rename(columns={"score":"raw_score"})
    frame["detected"] = (frame.raw_score > frozen["raw_threshold"] if frozen["detected_if"]=="greater_than"
                         else frame.raw_score < frozen["raw_threshold"])
    frame.to_csv(output / f"scores_{detector_name}.csv",index=False)
    metadata.update(config_sha256=sha(output / "config_pilot.json"),panel_sha256=sha(output / "panel.jsonl"),
                    frozen_threshold=frozen,recalibration=False)
    (output / f"run_{detector_name}.json").write_text(json.dumps(metadata,indent=2)+'\n')


def summarize(config, output):
    """Tassi descrittivi e AUROC: nessun test statistico o etichetta umana aggiunta."""
    panel = pd.DataFrame(read_panel(output,config)).drop(columns="text")
    frames, complete, pending = [], [], []
    for detector in config["detectors"]:
        name = detector["name"]
        path = output / f"scores_{name}.csv"
        if not path.exists():
            pending.append(name)
            continue
        run = json.loads((output / f"run_{name}.json").read_text())
        if run["config_sha256"] != sha(output / "config_pilot.json") or run["panel_sha256"] != sha(output / "panel.jsonl"):
            raise ValueError("Punteggi appartenenti a un altro piano/pannello")
        scores = pd.read_csv(path,keep_default_na=False,float_precision="round_trip")
        keys = ["doc_id","version"]
        if len(scores)!=len(panel) or scores[keys].duplicated().any() or not scores.detector.eq(name).all():
            raise ValueError("Score incompleti o duplicati")
        matched = scores.merge(panel,on=keys,validate="one_to_one",suffixes=("","_panel"))
        if len(matched) != len(panel):
            raise ValueError("Gli score non coprono gli stessi documenti e versioni")
        for field in ["group_id","domain","editor","trajectory_id","text_sha256","n_words"]:
            if not matched[field].eq(matched[field+"_panel"]).all():
                raise ValueError("Score e testi non coincidono")
        if not np.isfinite(scores.raw_score).all():
            raise ValueError("Score non finiti")
        rule = config["frozen_thresholds"][name]
        expected = scores.raw_score > rule["raw_threshold"] if rule["detected_if"]=="greater_than" else scores.raw_score < rule["raw_threshold"]
        if not scores.detected.eq(expected).all():
            raise ValueError("Decisione non conforme alla soglia congelata")
        frames.append(scores)
        complete.append(name)
    summary, auc, decisions = [], [], []
    long = pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=[
        "doc_id","group_id","domain","version","editor","n_words","detector","raw_score","detected"])
    long.to_csv(output / "scores_long.csv",index=False)
    if frames:
        for name, own in long.groupby("detector"):
            for domain in [*config["domains"],"overall"]:
                part = own if domain=="overall" else own[own.domain.eq(domain)]
                for version, group in part.groupby("version"):
                    k,n = int(group.detected.sum()),len(group)
                    low,high = wilson_interval(k,n)
                    summary.append(dict(dataset="OpAI-Bench",detector=name,domain=domain,version=version,
                        k=k,n=n,rate=k/n,wilson95_low=low,wilson95_high=high,
                        median_words=group.n_words.median(),rate_type="FPR" if version=="v0" else "edited_text_detection_rate"))
                    if domain=="overall" and version!="v0":
                        decisions.append(dict(detector=name,version=version,k=k,n=n,rate=k/n,
                            minimum_fraction=config["decision_rule"]["minimum_detection_fraction"],
                            proceed=k/n>=config["decision_rule"]["minimum_detection_fraction"]))
                for version in ["v5","v8"]:
                    pair = part[part.version.isin(["v0",version])]
                    oriented = -pair.raw_score if name=="binoculars_falcon7b" else pair.raw_score
                    auc.append(dict(detector=name,domain=domain,comparison="v0_vs_"+version,
                        n_documents=len(pair)//2,auroc=roc_auc_score(pair.version.ne("v0"),oriented)))
    plot_rates(pd.DataFrame(summary, columns=["detector","domain","version","rate","wilson95_low","wilson95_high"]),output)
    for name,k in [("openai_roberta_base",71),("binoculars_falcon7b",117)]:
        for version,a,n,kind in [("original_machine",k,123,"machine_detection_rate"),("human_calibration",15,300,"calibration_FPR_not_test")]:
            low,high=wilson_interval(a,n)
            summary.append(dict(dataset="MGTBench_reference",detector=name,domain="overall",version=version,
                k=a,n=n,rate=a/n,wilson95_low=low,wilson95_high=high,median_words=np.nan,rate_type=kind))
    pd.DataFrame(summary).to_csv(output / "summary_rates.csv",index=False)
    pd.DataFrame(auc,columns=["detector","domain","comparison","n_documents","auroc"]).to_csv(output / "auroc.csv",index=False)
    pd.DataFrame(decisions,columns=["detector","version","k","n","rate","minimum_fraction","proceed"]).to_csv(output / "feasibility.csv",index=False)
    status = dict(status="completed" if not pending else "awaiting_detector_scoring",complete_detectors=complete,
        pending_detectors=pending,n_expected_scores=len(panel)*2,n_available_scores=sum(map(len,frames)),
        config_sha256=sha(output / "config_pilot.json"),panel_sha256=sha(output / "panel.jsonl"),
        human_annotation_produced=False,recalibration=False)
    (output / "experiment.json").write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


def plot_rates(frame,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True,constrained_layout=True)
    for ax,name in zip(axes,["openai_roberta_base","binoculars_falcon7b"]):
        own=frame[frame.detector.eq(name)]
        ax.set_title("OpenAI RoBERTa-base" if name.startswith("openai") else "Binoculars")
        if own.empty:
            ax.text(.5,.5,"Scoring in attesa",transform=ax.transAxes,ha="center")
        else:
            domains=["essays","news","reports","abstracts"]
            for j,(version,color) in enumerate(zip(["v0","v5","v8"],["#5b8e82","#337ab7","#d28532"])):
                p=own[own.version.eq(version)].set_index("domain").loc[domains]
                ax.bar(np.arange(4)+(j-1)*.24,p.rate,width=.24,label=version,color=color,
                    yerr=np.vstack([np.maximum(0,p.rate-p.wilson95_low),np.maximum(0,p.wilson95_high-p.rate)]),capsize=3)
            ax.set_xticks(range(4),["Essay","News","Government\nreports","arXiv\nabstracts"])
            ax.legend()
        ax.axhline(.05,color="gray",ls=":",lw=1,label="FPR riferimento5%")
        ax.set_ylim(0,1.03);ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Quota rilevata; Wilson95%")
    fig.suptitle("OpAI: v0 umano; v5/v8 cumulativamente rieditati da LLM")
    fig.savefig(output / "detection_rates.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["prepare","score","summarize"],required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"results_opai_detectability_pilot")
    parser.add_argument("--source",type=Path,default=ROOT/"data/opai/official_parquet")
    parser.add_argument("--only",choices=["openai_roberta_base","binoculars_falcon7b"])
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--observer-device",default="cuda:0")
    parser.add_argument("--performer-device",default="cuda:1")
    args=parser.parse_args()
    config=json.loads((args.output/"config_pilot.json").read_text())
    if args.stage=="prepare":prepare(config,args.output,args.source)
    elif args.stage=="summarize":summarize(config,args.output)
    else:
        if not args.only:parser.error("--only è obbligatorio per lo scoring")
        score(config,args.output,args.only,args.device,args.observer_device,args.performer_device)
        summarize(config,args.output)


if __name__=="__main__":main()
