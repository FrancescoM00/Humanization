"""Catalogo descrittivo delle metriche sulle riscritture reali dello studio."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import load_semantic_study, prepare_nli_documents
from semantics.metrics import lexical_scores
from stylometry.features import _readability


METRIC_DEFINITIONS = {
    "rouge_l_f1": dict(label="ROUGE-L F1", units="0–1", family="overlap",
                       meaning="Sottosequenza comune più lunga, F1 documentale con Porter stemming; originale come riferimento."),
    "bleu_0_100": dict(label="BLEU", units="0–100", family="overlap",
                       meaning="SacreBLEU sentence-level sull'intero passaggio, tokenizer 13a, smoothing exp, effective order, unico riferimento originale; non corpus BLEU."),
    "bert_document_chunk_f1": dict(label="BERTScore F1 documentale approssimato", units="0–1", family="contextual_similarity",
                       meaning="RoBERTa-base layer10, IDF e riscalatura disattivati; migliori corrispondenze fra chunk, precision/recall pesate per token e media armonica finale."),
    "embedding_cosine": dict(label="Coseno MiniLM documentale", units="−1–1", family="embedding_similarity",
                       meaning="Coseno fra medie pesate per token di embedding di frase/chunk normalizzati; nessun troncamento documentale."),
    "flesch_original": dict(label="Flesch Reading Ease originale", units="indice FRE, senza clipping", family="readability",
                       meaning="textstat en_US sul testo originale; valori più alti indicano maggiore facilità strutturale di lettura."),
    "flesch_rewrite": dict(label="Flesch Reading Ease riscrittura", units="indice FRE, senza clipping", family="readability",
                       meaning="textstat en_US sul testo riscritto; non è una misura di naturalezza o conservazione semantica."),
    "flesch_delta": dict(label="Delta Flesch Reading Ease", units="punti FRE", family="readability",
                       meaning="FRE della riscrittura meno FRE dell'originale, calcolato per coppia prima dell'aggregazione."),
    "fk_grade_original": dict(label="Flesch–Kincaid Grade originale", units="anni scolastici USA stimati", family="readability",
                       meaning="Indice di leggibilità textstat en_US, aggiuntivo rispetto al Flesch Reading Ease."),
    "fk_grade_rewrite": dict(label="Flesch–Kincaid Grade riscrittura", units="anni scolastici USA stimati", family="readability",
                       meaning="Indice di leggibilità della riscrittura; un aumento indica maggiore complessità stimata."),
    "fk_grade_delta": dict(label="Delta Flesch–Kincaid Grade", units="anni scolastici stimati", family="readability",
                       meaning="Grade della riscrittura meno grade dell'originale, calcolato per coppia."),
    "bert_aligned_rescaled_mean": dict(label="BERTScore allineato riscalato, media", units="scala riscalata, valori negativi possibili", family="aligned_contextual_similarity",
                       meaning="Massimo F1 per unità su tutte le unità dell'altro testo; riscalatura con baseline ufficiale 0,814463; media delle unità e direzione peggiore."),
    "bert_aligned_rescaled_min": dict(label="BERTScore allineato riscalato, minimo", units="scala riscalata, valori negativi possibili", family="aligned_contextual_similarity",
                       meaning="Minimo delle corrispondenze massime per unità, nelle due direzioni; protocollo distinto dall'approssimazione documentale."),
    "nli_bidirectional_min": dict(label="NLI bidirezionale, minimo", units="0–1", family="nli",
                       meaning="Premesse a finestre di una o due unità; minimo fra supporti minimi forward e reverse."),
    "nli_bidirectional_mean": dict(label="NLI bidirezionale, media conservativa", units="0–1", family="nli",
                       meaning="Minimo fra le medie dei supporti forward e reverse; non media aritmetica delle due direzioni."),
}

STAGES = ("lexical", "bertscore", "embedding", "aligned_bertscore")
EXTRA_MEMBERS = tuple(
    name for stage in STAGES for name in
    (f"semantic/scores/{stage}.csv", f"semantic/scores/{stage}_run.json")
) + ("style/features.csv", "style/style_manifest.json", "semantic/alignment.json")


def _read_extra(path):
    path = Path(path)
    if path.is_dir():
        return {name:(path/name).read_bytes() for name in EXTRA_MEMBERS}
    with zipfile.ZipFile(path) as archive:
        if any(archive.namelist().count(name) != 1 for name in EXTRA_MEMBERS):
            raise ValueError("L'archivio deve contenere una sola copia di ogni metrica richiesta")
        return {name:archive.read(name) for name in EXTRA_MEMBERS}


def validate_cached_table(frame, identifiers, manifest, pairs_sha256, stage):
    """Una cache mancante, duplicata o riferita ad altri testi non viene riutilizzata."""
    if ("id" not in frame or frame.id.duplicated().any() or set(frame.id) != set(identifiers)
            or manifest.get("status") != "executed" or manifest.get("n_pairs") != len(identifiers)
            or manifest.get("pairs_sha256") != pairs_sha256 or manifest.get("stage") != stage):
        raise ValueError(f"Cache {stage} incompleta o incoerente con le coppie originali")


def _max_difference(actual, cached, label, tolerance=1e-9):
    a, b = np.asarray(actual, dtype=float), np.asarray(cached, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Valori mancanti o non finiti: {label}")
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    if difference > tolerance:
        raise ValueError(f"La ricalcolazione differisce dalla cache: {label}, massimo {difference}")
    return difference


def _summary(documents):
    rows = []
    for scope, frame in [("all_valid", documents), ("common_sources", documents[documents.in_common_sources])]:
        strata = [("overall", "all", frame)] + [("domain", d, p) for d,p in frame.groupby("domain", sort=True)]
        for stratum, value, part in strata:
            for system in PRIMARY_SYSTEMS:
                own = part[part.humanizer.eq(system)]
                for metric in METRIC_DEFINITIONS:
                    values = own[metric].to_numpy(dtype=float)
                    if not len(values) or not np.isfinite(values).all():
                        raise ValueError("Un riepilogo richiede tutti i documenti validi del campione")
                    q1, median, q3 = np.quantile(values, [.25, .5, .75])
                    rows.append(dict(scope=scope, stratum=stratum, stratum_value=value, humanizer=system,
                                     metric=metric, n=len(values), mean=float(values.mean()),
                                     sd=float(values.std(ddof=1)), median=float(median),
                                     q25=float(q1), q75=float(q3), iqr=float(q3-q1)))
    return pd.DataFrame(rows)


def evaluate_catalogue(input_path):
    """Ricalcola overlap e leggibilità e recupera i punteggi neurali esatti già salvati."""
    data = load_semantic_study(input_path)
    nli, common, statuses = prepare_nli_documents(data)
    pairs = data["pairs"]
    pairs_lookup = {r["id"]:r for r in pairs}
    ids = list(nli.id)
    raw = _read_extra(input_path)
    input_sha = dict(data["input_sha256"])
    input_sha.update({name:hashlib.sha256(content).hexdigest() for name,content in raw.items()})
    tables, manifests = {}, {}
    pairs_sha = data["input_sha256"]["semantic_context/pairs.jsonl"]
    for stage in STAGES:
        tables[stage] = pd.read_csv(io.BytesIO(raw[f"semantic/scores/{stage}.csv"]), float_precision="round_trip")
        manifests[stage] = json.loads(raw[f"semantic/scores/{stage}_run.json"])
        validate_cached_table(tables[stage], ids, manifests[stage], pairs_sha, stage)
        tables[stage] = tables[stage].set_index("id").loc[ids]
    if manifests["aligned_bertscore"]["alignment_sha256"] != input_sha["semantic/alignment.json"]:
        raise ValueError("BERTScore allineato riferito a un allineamento diverso")
    alignment = json.loads(raw["semantic/alignment.json"])
    for identifier in ids:
        for field in ("original", "modified"):
            if alignment[identifier][field + "_sha256"] != hashlib.sha256(pairs_lookup[identifier][field].encode()).hexdigest():
                raise ValueError("Testi dell'allineamento BERTScore diversi dalle coppie valide")

    recomputed = lexical_scores(pairs).set_index("id").loc[ids]
    validation = {
        "rouge_l_max_absolute_difference": _max_difference(recomputed.lexical_rouge_l_f1, tables["lexical"].lexical_rouge_l_f1, "ROUGE-L"),
        "bleu_0_1_max_absolute_difference": _max_difference(recomputed.lexical_bleu, tables["lexical"].lexical_bleu, "BLEU"),
    }
    features = pd.read_csv(io.BytesIO(raw["style/features.csv"]), float_precision="round_trip")
    if features.text_sha256.duplicated().any() or any(
            hashlib.sha256(str(r.text).encode()).hexdigest() != r.text_sha256 for r in features.itertuples()):
        raise ValueError("Gli hash della cache stilometrica non corrispondono ai testi")
    style_manifest = json.loads(raw["style/style_manifest.json"])
    if (style_manifest.get("pairs_sha256") != pairs_sha
            or style_manifest.get("cohort_sha256") != data["input_sha256"]["cohorts/study.jsonl"]):
        raise ValueError("La cache di leggibilità appartiene a un altro pannello")
    features = features.set_index("text_sha256")
    readability = _readability()
    unique_texts = {text for pair in pairs for text in (pair["original"], pair["modified"])}
    readability_by_sha = {}
    fre_diff, grade_diff = [], []
    for text in sorted(unique_texts):
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha not in features.index:
            raise ValueError("Manca un testo nella cache di leggibilità")
        fre, grade = float(readability.flesch_reading_ease(text)), float(readability.flesch_kincaid_grade(text))
        row = features.loc[sha]
        fre_diff.append(_max_difference([fre], [row.flesch_reading_ease], "Flesch Reading Ease"))
        grade_diff.append(_max_difference([grade], [row.flesch_kincaid_grade], "Flesch–Kincaid Grade"))
        readability_by_sha[sha] = (fre, grade, float(row.n_words))
    validation.update(flesch_max_absolute_difference=max(fre_diff), fk_grade_max_absolute_difference=max(grade_diff),
                      n_unique_texts_recomputed=len(unique_texts), n_cached_metric_pairs=len(ids))

    frame = nli.copy().set_index("id")
    frame["rouge_l_f1"] = recomputed.lexical_rouge_l_f1
    frame["bleu_0_100"] = recomputed.lexical_bleu * 100
    frame["bert_document_chunk_f1"] = tables["bertscore"].bert_chunk_f1
    frame["embedding_cosine"] = tables["embedding"].embedding_cosine
    bert = tables["bertscore"]
    if (not bert.bert_num_layers.eq(10).all() or bert.bert_idf.any() or bert.bert_rescale_with_baseline.any()):
        raise ValueError("Il protocollo BERTScore documentale non coincide con quello dichiarato")
    aligned = tables["aligned_bertscore"]
    baseline = aligned.aligned_bert_baseline_f1.to_numpy()
    if not np.allclose(baseline, .814463, atol=1e-12, rtol=0) or not aligned.aligned_bert_num_layers.eq(10).all():
        raise ValueError("Baseline o layer BERTScore allineato inattesi")
    for aggregate in ("mean", "min"):
        values = aligned[f"aligned_bert_f1_rescaled_bidirectional_{aggregate}"]
        expected = (aligned[f"aligned_bert_f1_bidirectional_{aggregate}"].to_numpy()-baseline)/(1-baseline)
        validation[f"aligned_bert_rescale_{aggregate}_max_absolute_difference"] = _max_difference(values, expected, "riscalatura BERTScore allineato")
        frame[f"bert_aligned_rescaled_{aggregate}"] = values
    for field, suffix in (("original", "original"), ("modified", "rewrite")):
        values = [readability_by_sha[hashlib.sha256(pairs_lookup[i][field].encode()).hexdigest()]
                  for i in frame.index]
        frame[f"flesch_{suffix}"] = [v[0] for v in values]
        frame[f"fk_grade_{suffix}"] = [v[1] for v in values]
        frame["original_n_words" if field == "original" else "modified_n_words"] = [v[2] for v in values]
    frame["flesch_delta"] = frame.flesch_rewrite-frame.flesch_original
    frame["fk_grade_delta"] = frame.fk_grade_rewrite-frame.fk_grade_original
    if not np.isfinite(frame[list(METRIC_DEFINITIONS)].to_numpy(dtype=float)).all():
        raise ValueError("Il catalogo non può ignorare metriche mancanti o non finite")
    frame = frame.reset_index()
    documents = frame[frame.humanizer.isin(PRIMARY_SYSTEMS)].copy()
    identity = frame[frame.humanizer.eq("identity")].copy()
    summary = _summary(documents)
    wide = {}
    for statistic in ("median", "mean"):
        part = summary.query("scope == 'common_sources' and stratum == 'overall'")
        out = part.pivot(index="humanizer", columns="metric", values=statistic).reindex(PRIMARY_SYSTEMS)
        out.insert(0, "n_documents", len(common))
        wide[statistic] = out.reset_index().rename_axis(columns=None)
    coverage = []
    for system in PRIMARY_SYSTEMS:
        own = documents[documents.humanizer.eq(system)]
        attempts = statuses[statuses.humanizer.eq(system)]
        coverage.append(dict(humanizer=system, n_attempted=len(attempts), n_valid=len(own),
            n_failed=int(attempts.status.ne("ok").sum()),
            n_expanded_included=int(own.modified_n_words.gt(own.original_n_words).sum()),
            n_common_sources=len(common)))
    provenance = dict(input_sha256=input_sha, cached_stage_manifests=manifests,
        nli_historical_run=data["nli_run"], model_revisions=data["config"]["semantic_models"],
        bert_model="FacebookAI/roberta-base", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        bert_aligned_baseline_f1=.814463, historical_style_manifest=style_manifest,
        recomputed_metrics=["ROUGE-L", "sentence BLEU", "Flesch Reading Ease", "Flesch–Kincaid Grade"],
        reused_metrics=["document chunk-matched BERTScore", "document MiniLM cosine", "aligned rescaled BERTScore", "final window2 NLI"],
        new_neural_model_inference=False, api_calls_executed=False, automatic_downloads=False,
        descriptions=METRIC_DEFINITIONS)
    return dict(documents=documents, identity=identity, summary=summary, common_median=wide["median"],
                common_mean=wide["mean"], coverage=pd.DataFrame(coverage), validation=validation, provenance=provenance)
