"""Unisce evasione e semantica sulla stessa riscrittura del campione comune."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .detector_performance import load_saved_detection, prepare_score_views


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "exports/kaggle_v2_study/humanization_v2_study_protocol_v2.zip"
DEFAULT_COMMON = ROOT / "results_humanizer_metric_catalogue/correlations/common_document_metrics.csv"
DEFAULT_CATALOGUE = ROOT / "results_humanizer_metric_catalogue/document_metrics.csv"
DEFAULT_IDS = ROOT / "results_humanizer_semantics_final/common_source_ids.csv"
DEFAULT_DETECTOR_PAIRS = ROOT / "results_detector_performance/paired_test_scores.csv"
META = ["source_id", "source_group_id", "domain", "generator", "humanizer"]


def _csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _boolean(value):
    """Non interpreta la stringa 'False' come un valore vero."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ("True", "False"):
        return value == "True"
    raise ValueError("Un flag detector deve essere un booleano esplicito")


def _semantic_panel(config, common, catalogue, identifiers):
    """Verifica prima i testi semantici, senza osservare le etichette dei detector."""
    n = config["population"]["n_common_sources"]
    methods = config["methods"]
    required_common = {"id", *META, "nli_bidirectional_mean", "nli_bidirectional_min"}
    required_catalogue = required_common | {
        "original_sha256", "modified_sha256", "status", "original_n_words", "modified_n_words",
        "bert_aligned_rescaled_mean", "nli_forward_min", "nli_reverse_min",
        "nli_forward_mean", "nli_reverse_mean"}
    if not required_common <= set(common) or not required_catalogue <= set(catalogue):
        raise ValueError("Mancano colonne necessarie nel catalogo semantico")
    if ("source_id" not in identifiers or len(identifiers) != n
            or identifiers.source_id.duplicated().any()
            or not identifiers.source_id.map(lambda x:isinstance(x, str) and bool(x.strip())).all()):
        raise ValueError("Gli ID comuni devono essere univoci e rispettare la numerosità fissata")
    if catalogue.id.duplicated().any() or catalogue[["source_id", "humanizer"]].duplicated().any():
        raise ValueError("Il catalogo contiene riscritture duplicate")
    ids = set(identifiers.source_id)
    selected = catalogue[catalogue.source_id.isin(ids) & catalogue.humanizer.isin(methods)].copy()
    if (len(common) != n * len(methods) or common.id.duplicated().any()
            or common[["source_id", "humanizer"]].duplicated().any()
            or len(selected) != len(common) or set(common.id) != set(selected.id)
            or set(common.humanizer) != set(methods) or not selected.status.eq("ok").all()):
        raise ValueError("Il campione semantico deve avere un output valido per fonte e metodo")
    for method in methods:
        if set(common.loc[common.humanizer.eq(method), "source_id"]) != ids:
            raise ValueError("Ogni metodo deve usare esattamente le fonti comuni")
    if (not common.groupby("source_id").source_group_id.nunique().eq(1).all()
            or not common.groupby("source_group_id").source_id.nunique().eq(1).all()):
        raise ValueError("Il bootstrap richiede una sola fonte indipendente per gruppo")
    left, right = common.set_index("id"), selected.set_index("id")
    right = right.reindex(left.index)
    if not left[META].equals(right[META]):
        raise ValueError("Gli ID semantici non corrispondono agli stessi testi e metodi")
    numeric_shared = [name for name in common if name in selected and name not in {"id", *META}]
    for column in numeric_shared:
        a, b = left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=1e-12, rtol=0):
            raise ValueError("Il catalogo e il campione comune hanno punteggi semantici diversi")
    for suffix in ("min", "mean"):
        values = right[["nli_forward_"+suffix, "nli_reverse_"+suffix]].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
                or not np.allclose(values.min(axis=1), right["nli_bidirectional_"+suffix], atol=1e-12, rtol=0)):
            raise ValueError("La convenzione NLI bidirezionale non coincide con quella della suite")
    lengths = right[["original_n_words", "modified_n_words"]].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Le lunghezze devono essere finite e positive")
    if not np.isfinite(right.bert_aligned_rescaled_mean.to_numpy(dtype=float)).all():
        raise ValueError("Manca il BERTScore allineato riscalato")
    if not right.groupby("source_id").original_sha256.nunique().eq(1).all():
        raise ValueError("Lo stesso originale non può cambiare fra metodi")
    panel = right.reset_index()[["id", *META, "original_sha256", "modified_sha256",
        "original_n_words", "modified_n_words", "nli_bidirectional_mean", "nli_bidirectional_min",
        "bert_aligned_rescaled_mean"]].rename(columns={"id":"rewrite_id", "humanizer":"method",
        "nli_bidirectional_mean":"nli_cons_mean", "nli_bidirectional_min":"nli_min",
        "bert_aligned_rescaled_mean":"bs_resc_mean"})
    panel["len_ratio"] = panel.modified_n_words / panel.original_n_words
    return panel


def build_long_table(config, *, archive=DEFAULT_ARCHIVE, common_metrics=DEFAULT_COMMON,
                     catalogue=DEFAULT_CATALOGUE, common_ids=DEFAULT_IDS,
                     detector_pairs=DEFAULT_DETECTOR_PAIRS):
    """Join esatto fonte/metodo/ID e verifica ASR prima dell'analisi dei vincoli semantici."""
    methods, detectors = config["methods"], config["detectors"]
    if len(set(methods)) != 4 or len(set(detectors)) != 2:
        raise ValueError("La configurazione richiede quattro metodi e due detector distinti")
    panel = _semantic_panel(config, _csv(common_metrics), _csv(catalogue), _csv(common_ids))
    data = load_saved_detection(archive)
    _, views, directions = prepare_score_views(data)
    rewrites = {(r["humanizer"], str(r["source_id"])):r for r in data["rewrites"]}
    records = []
    for detector in detectors:
        if detector not in views or data["thresholds"][detector].get("target_fpr") != .05:
            raise ValueError("Manca un detector con soglia congelata al FPR obiettivo del 5%")
        for semantic in panel.to_dict("records"):
            source, method = semantic["source_id"], semantic["method"]
            rewrite = rewrites[method, source]
            if (rewrite["status"] != "ok" or rewrite["id"] != semantic["rewrite_id"]
                    or _sha(rewrite["original"]) != semantic["original_sha256"]
                    or _sha(rewrite["modified"]) != semantic["modified_sha256"]
                    or any(rewrite[k] != semantic[k] for k in ("source_group_id", "domain", "generator"))):
                raise ValueError("La semantica non appartiene alla stessa riscrittura del detector")
            original = views[detector][source, "machine", ""]
            modified = views[detector][source, "rewrite", method]
            before, after = bool(original["is_ai"]), bool(modified["is_ai"])
            records.append(dict(**semantic, detector=detector, orig_detected=before, rew_detected=after,
                evaded=before and not after, original_raw_score=original["score"],
                rewrite_raw_score=modified["score"], original_score=original["oriented_score"],
                rewrite_score=modified["oriented_score"],
                score_reduction=original["oriented_score"]-modified["oriented_score"],
                raw_threshold=data["thresholds"][detector]["threshold"], score_orientation=directions[detector]))
    long = pd.DataFrame(records)
    keys = ["source_id", "method", "detector"]
    if long[keys].duplicated().any() or len(long) != config["population"]["n_rewrites"] * len(detectors):
        raise ValueError("Il join detector/semantica non produce il pannello completo")
    reference = _csv(detector_pairs).rename(columns={"humanizer":"method"})
    reference = reference[reference.source_id.isin(set(panel.source_id))
                          & reference.method.isin(methods) & reference.detector.isin(detectors)].copy()
    if (len(reference) != len(long) or reference[keys].duplicated().any()
            or not reference.in_common_success.map(_boolean).all()):
        raise ValueError("Il confronto ASR precedente non copre lo stesso campione comune")
    actual = long.set_index(keys).sort_index()
    saved = reference.set_index(keys).sort_index()
    if not actual.index.equals(saved.index):
        raise ValueError("Il riferimento detector usa fonti o riscrittori diversi")
    for a, b in [("orig_detected", "original_is_ai"), ("rew_detected", "rewrite_is_ai")]:
        if not np.array_equal(actual[a].to_numpy(dtype=bool), saved[b].map(_boolean).to_numpy()):
            raise ValueError("Il join non riproduce le decisioni detector precedenti")
    for column in ("source_group_id", "domain", "generator", "score_orientation"):
        if not actual[column].equals(saved[column]):
            raise ValueError("I metadati del riferimento detector non coincidono")
    for column in ("original_score", "rewrite_score", "original_raw_score", "rewrite_raw_score", "raw_threshold"):
        if not np.allclose(actual[column], saved[column], atol=1e-12, rtol=0):
            raise ValueError("I punteggi detector differiscono dal riferimento precedente")
    checks = []
    for detector in detectors:
        for method in methods:
            own = long[long.detector.eq(detector) & long.method.eq(method)]
            old = reference[reference.detector.eq(detector) & reference.method.eq(method)]
            n = int(own.orig_detected.sum())
            k = int(own.evaded.sum())
            old_original = old.original_is_ai.map(_boolean)
            n_ref = int(old_original.sum())
            k_ref = int((old_original & ~old.rewrite_is_ai.map(_boolean)).sum())
            matches = n == n_ref and k == k_ref
            if not matches:
                raise ValueError("L'ASR semplice non coincide con il riferimento: interrompere l'analisi")
            checks.append(dict(detector=detector, method=method, n_sources=len(own),
                n_original_detected=n, evaded=k, asr=k/n if n else np.nan,
                reference_n_original_detected=n_ref, reference_evaded=k_ref,
                asr_reference=k_ref/n_ref if n_ref else np.nan, matches_reference=matches,
                source_population="same 123 common sources, successful outputs from all four methods"))
    paths = {"common_document_metrics.csv":common_metrics, "catalogue_document_metrics.csv":catalogue,
             "common_source_ids.csv":common_ids, "reference_detector_pairs.csv":detector_pairs}
    provenance = dict(input_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()},
        archive_input_sha256=data["input_sha256"], frozen_detector_thresholds={d:data["thresholds"][d] for d in detectors},
        reference_population="Common123 only: full126 and historical pilot ASR have different denominators",
        join="source_id + method + rewrite_id, text hashes and metadata, then detector; row-by-row check against saved paired detector scores",
        semantic_mean="min(forward_mean,reverse_mean)",
        bertscore="aligned_bert_f1_rescaled_bidirectional_mean; supplemented from catalogue, not the raw document F1",
        length_ratio="modified_n_words / original_n_words; no expansion excluded",
        calibration="Original frozen thresholds; target5%FPR on separate calibration humans; no recalibration",
        model_inference_executed=False, api_calls_executed=False)
    ordered = ["source_id", "source_group_id", "domain", "generator", "method", "detector",
               "orig_detected", "rew_detected", "evaded", "nli_cons_mean", "nli_min", "bs_resc_mean", "len_ratio"]
    long = long[ordered + [c for c in long if c not in ordered]]
    return dict(long=long, baseline_check=pd.DataFrame(checks), provenance=provenance)
