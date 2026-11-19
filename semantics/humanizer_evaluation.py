"""Valuta supporto NLI e segnalazioni lessicali sulle riscritture dello studio."""
from __future__ import annotations

import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from humanizers.detection import validate_inputs
from humanizers.detector_performance import PRIMARY_SYSTEMS
from humanizers.null_control_statistics import wilson_interval


HISTORICAL_REFERENCE = .04075
NLI_METRICS = ["nli_bidirectional_min", "nli_bidirectional_mean", "nli_forward_min",
               "nli_reverse_min", "nli_forward_mean", "nli_reverse_mean"]
ALERT_METRICS = ["alert_number", "alert_date", "alert_entity", "alert_negation",
                 "alert_quantifier", "alert_hedge_removal", "alert_any"]
MEMBERS = ("config.json", "cohorts/study.jsonl", "rewriting/rewrites.jsonl",
           "semantic_context/pairs.jsonl", "semantic_context/alignment.json",
           "semantic_context/contextual_nli.csv", "semantic_context/contextual_nli_units.jsonl",
           "semantic_context/contextual_nli_run.json")


def load_semantic_study(path):
    """Legge gli input e la cache NLI senza importare o rieseguire i modelli."""
    path = Path(path)
    if path.is_dir():
        raw = {name:(path / name).read_bytes() for name in MEMBERS}
    else:
        with zipfile.ZipFile(path) as archive:
            if any(archive.namelist().count(name) != 1 for name in MEMBERS):
                raise ValueError("Lo studio deve contenere una sola copia di ogni input richiesto")
            raw = {name:archive.read(name) for name in MEMBERS}
    jsonl = lambda name: [json.loads(line) for line in raw[name].splitlines() if line.strip()]
    return dict(config=json.loads(raw["config.json"]), cohort=jsonl("cohorts/study.jsonl"),
                rewrites=jsonl("rewriting/rewrites.jsonl"), pairs=jsonl("semantic_context/pairs.jsonl"),
                alignment=json.loads(raw["semantic_context/alignment.json"]),
                nli=pd.read_csv(io.BytesIO(raw["semantic_context/contextual_nli.csv"]), float_precision="round_trip"),
                units=[r for r in jsonl("semantic_context/contextual_nli_units.jsonl") if r["mode"] == "window2"],
                nli_run=json.loads(raw["semantic_context/contextual_nli_run.json"]),
                input_sha256={name:hashlib.sha256(content).hexdigest() for name,content in raw.items()})


def prepare_nli_documents(data):
    """Controlla copertura e corrispondenza dei testi, poi ricostruisce minimo e media."""
    sources = validate_inputs(data["cohort"], data["rewrites"])
    tests = [r for r in data["cohort"] if r["split"] == "test"]
    if len({r["source_group_id"] for r in tests}) != len(tests):
        raise ValueError("Il confronto richiede una sola fonte indipendente per gruppo")
    expected = {r["id"]:r for r in data["rewrites"] if r["status"] == "ok"}
    pairs = {r["id"]:r for r in data["pairs"]}
    if len(pairs) != len(data["pairs"]) or set(pairs) != set(expected):
        raise ValueError("La cache deve coprire esattamente tutte le riscritture valide")
    run = data["nli_run"]
    if (run.get("status") != "executed" or "window2" not in run.get("modes", [])
            or run.get("pairs_sha256") != data["input_sha256"]["semantic_context/pairs.jsonl"]):
        raise ValueError("Manifest NLI incoerente con i testi valutati")
    summary = data["nli"].query("mode == 'window2'").set_index(["id", "direction"])
    if summary.index.duplicated().any() or set(summary.index) != {(i,d) for i in pairs for d in ("forward", "reverse")}:
        raise ValueError("Mancano aggregati NLI bidirezionali univoci")
    units = {}
    for row in data["units"]:
        if row["id"] not in pairs or row["direction"] not in {"forward", "reverse"}:
            raise ValueError("Unità NLI estranea alle coppie dello studio")
        units.setdefault((row["id"], row["direction"]), []).append(row)
    rows = []
    for identifier, pair in pairs.items():
        reference = expected[identifier]
        for field in ("original", "modified", "source_id", "source_group_id", "humanizer", "domain", "generator"):
            if pair[field] != reference[field]:
                raise ValueError("Testo o metadati NLI diversi dalla riscrittura valida")
        alignment = data["alignment"][identifier]
        for field in ("original", "modified"):
            if alignment[field + "_sha256"] != hashlib.sha256(pair[field].encode()).hexdigest():
                raise ValueError("Hash dell'allineamento diverso dal testo")
        row = {k:pair[k] for k in ("id", "source_id", "source_group_id", "humanizer", "domain", "generator")}
        row.update(status="ok", original_sha256=alignment["original_sha256"], modified_sha256=alignment["modified_sha256"])
        for direction in ("forward", "reverse"):
            hypotheses = alignment[("modified" if direction == "forward" else "original") + "_chunks"]
            part = sorted(units.get((identifier, direction), []), key=lambda r:r["hypothesis_index"])
            if [r["hypothesis_index"] for r in part] != list(range(len(hypotheses))) or not part:
                raise ValueError("Copertura NLI incompleta o duplicata: nessun valore viene imputato")
            values = np.asarray([r["entailment"] for r in part], dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Supporto NLI non valido")
            for unit, text in zip(part, hypotheses):
                if not unit["eligible"] or unit["hypothesis"] != text or unit["max_pair_tokens"] != run["max_pair_tokens"]:
                    raise ValueError("Ipotesi NLI o copertura diverse dall'allineamento")
            saved = summary.loc[identifier, direction]
            if (not bool(saved["complete"]) or saved["n_hypotheses"] != len(values)
                    or saved["n_eligible_hypotheses"] != len(values)
                    or not np.allclose([values.min(), values.mean()], [saved["min"], saved["mean"]], atol=1e-7, rtol=0)):
                raise ValueError("Le unità NLI non riproducono gli aggregati salvati")
            prefix = "nli_" + direction
            worst = part[int(values.argmin())]
            row.update({prefix + "_min": float(values.min()), prefix + "_mean": float(values.mean()),
                        prefix + "_n_hypotheses": len(values), prefix + "_coverage": 1.,
                        prefix + "_worst_hypothesis": worst["hypothesis"],
                        prefix + "_best_premise_for_worst": worst["best_premise"]})
        row["nli_bidirectional_min"] = min(row["nli_forward_min"], row["nli_reverse_min"])
        # Si conserva la definizione della suite: è la direzione più debole.
        row["nli_bidirectional_mean"] = min(row["nli_forward_mean"], row["nli_reverse_mean"])
        row["nli_balanced_direction_mean"] = .5 * (row["nli_forward_mean"] + row["nli_reverse_mean"])
        row["below_historical_reference_only"] = row["nli_bidirectional_min"] < HISTORICAL_REFERENCE
        rows.append(row)
    documents = pd.DataFrame(rows)
    available = {s:set(documents.loc[documents.humanizer.eq(s), "source_id"]) for s in PRIMARY_SYSTEMS}
    if any(not value for value in available.values()):
        raise ValueError("Manca un riscrittore primario")
    common = set.intersection(*available.values())
    documents["in_common_sources"] = documents.source_id.isin(common)
    status = pd.DataFrame([{k:r[k] for k in ("id", "source_id", "source_group_id", "humanizer", "domain", "generator", "status")}
                           for r in data["rewrites"]])
    return documents, common, status


def _strata(frame):
    yield "overall", "all", frame
    for domain, part in frame.groupby("domain", sort=True):
        yield "domain", domain, part


def summarize_documents(documents):
    """Distribuzioni per documento e quote di segnalazioni, senza soglie decisionali."""
    summaries, alerts, reference = [], [], []
    primary = documents[documents.humanizer.isin(PRIMARY_SYSTEMS)]
    for scope, frame in [("all_valid", primary), ("common_sources", primary[primary.in_common_sources])]:
        for stratum, value, part in _strata(frame):
            for system, own in part.groupby("humanizer", sort=False):
                info = dict(scope=scope, stratum=stratum, stratum_value=value, humanizer=system,
                            n_documents=len(own), n_expanded=int(own.modified_n_words.gt(own.original_n_words).sum()))
                for metric in [*NLI_METRICS, "nli_balanced_direction_mean"]:
                    values = own[metric].to_numpy(dtype=float)
                    q1, median, q3 = np.quantile(values, [.25, .5, .75])
                    summaries.append(dict(**info, metric=metric, median=float(median), q25=float(q1),
                                          q75=float(q3), iqr=float(q3-q1), mean=float(values.mean())))
                for metric in ALERT_METRICS:
                    n = int(own[metric].sum())
                    low, high = wilson_interval(n, len(own))
                    alerts.append(dict(**info, alert=metric, n_flagged=n, fraction=n/len(own),
                                       wilson95_low=low, wilson95_high=high,
                                       interpretation="deterministic warning, not a confirmed semantic error"))
                reference.append(dict(**info, n_below_historical_reference=int(own.below_historical_reference_only.sum()),
                                      historical_reference=HISTORICAL_REFERENCE, decision_threshold="none",
                                      interpretation="descriptive historical reference only; not a semantic failure label"))
    return dict(distributions=pd.DataFrame(summaries), alerts=pd.DataFrame(alerts), historical=pd.DataFrame(reference))


def paired_source_comparisons(documents, *, bootstrap=5000, seed=42):
    """Ricampiona gli stessi ID per tutti i metodi; riporta differenze B meno A."""
    if bootstrap < 1 or seed < 0:
        raise ValueError("Occorrono almeno una replica bootstrap e un seme non negativo")
    common = documents[documents.humanizer.isin(PRIMARY_SYSTEMS) & documents.in_common_sources]
    rows = []
    for stratum, value, part in _strata(common):
        sources = sorted(part.source_id.unique())
        if part.groupby("humanizer").size().to_dict() != {s:len(sources) for s in PRIMARY_SYSTEMS}:
            raise ValueError("Il confronto appaiato richiede le stesse fonti per tutti i metodi")
        digest = hashlib.sha256(json.dumps([seed, stratum, value]).encode()).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        draws = rng.integers(0, len(sources), size=(bootstrap, len(sources)))
        for metric in [*NLI_METRICS, *ALERT_METRICS]:
            values = part.pivot(index="source_id", columns="humanizer", values=metric).reindex(
                index=sources, columns=PRIMARY_SYSTEMS).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Uno score mancante non può entrare nel bootstrap appaiato")
            statistic = np.mean if metric in ALERT_METRICS else np.median
            point = statistic(values, axis=0)
            samples = statistic(values[draws], axis=1)
            for a, b in combinations(range(len(PRIMARY_SYSTEMS)), 2):
                differences = samples[:, b] - samples[:, a]
                low, high = np.quantile(differences, [.025, .975])
                degenerate = bool(np.ptp(differences) < 1e-15)
                rows.append(dict(stratum=stratum, stratum_value=value, metric=metric,
                    statistic="alert_rate" if metric in ALERT_METRICS else "document_median",
                    humanizer_a=PRIMARY_SYSTEMS[a], humanizer_b=PRIMARY_SYSTEMS[b], n_sources=len(sources),
                    value_a=float(point[a]), value_b=float(point[b]), difference_b_minus_a=float(point[b]-point[a]),
                    ci95_low=float(low) if not degenerate else np.nan,
                    ci95_high=float(high) if not degenerate else np.nan,
                    bootstrap_replicates=bootstrap, bootstrap_degenerate=degenerate,
                    interval_status="no_observed_bootstrap_variation" if degenerate else "pointwise_paired_source_percentile",
                    inference="descriptive interval; no multiple-comparison significance claim"))
    return pd.DataFrame(rows)
