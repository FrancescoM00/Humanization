"""Separa le direzioni NLI e descrive la distribuzione delle fonti comuni."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from humanizers.detector_performance import PRIMARY_SYSTEMS
from semantics.humanizer_evaluation import (
    ALERT_METRICS, NLI_METRICS, paired_source_comparisons,
)


DIRECTIONAL_METRICS = ("nli_forward_min", "nli_reverse_min",
                       "nli_forward_mean", "nli_reverse_mean")
FOCUS_PAIRS = (("dipper", "dipper_context3"),
               ("qwen_humanize", "qwen14b_humanize"))


def _validate_documents(documents):
    """Ogni fonte deve avere un solo output valido per ciascuno dei quattro metodi."""
    required = {"source_id", "source_group_id", "humanizer", "domain", "status",
                "original_n_words", "modified_n_words", "word_ratio",
                *NLI_METRICS, *ALERT_METRICS}
    if not required <= set(documents):
        raise ValueError("Mancano colonne richieste dagli score semantici")
    frame = documents.copy()
    if frame.empty or set(frame.humanizer) != set(PRIMARY_SYSTEMS):
        raise ValueError("Occorrono tutti e quattro i riscrittori primari")
    if frame[["source_id", "humanizer"]].duplicated().any() or not frame.status.eq("ok").all():
        raise ValueError("Output duplicati o non validi nel campione comune")
    for column in ("source_id", "source_group_id"):
        if not frame[column].map(lambda value:isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("Le fonti e i gruppi devono avere identificativi non vuoti")
    sources = set(frame.source_id)
    if any(set(frame.loc[frame.humanizer.eq(s), "source_id"]) != sources for s in PRIMARY_SYSTEMS):
        raise ValueError("I quattro riscrittori devono usare le stesse fonti")
    if (not frame.groupby("source_id").source_group_id.nunique().eq(1).all()
            or not frame.groupby("source_group_id").source_id.nunique().eq(1).all()):
        raise ValueError("Il bootstrap richiede una sola fonte indipendente per gruppo")
    if not frame.groupby("source_id").domain.nunique().eq(1).all():
        raise ValueError("Una stessa fonte non può cambiare dominio")
    values = frame[list(NLI_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Gli score NLI devono essere finiti e compresi fra zero e uno")
    if not np.allclose(frame.nli_bidirectional_min,
                       np.minimum(frame.nli_forward_min, frame.nli_reverse_min), atol=1e-12, rtol=0):
        raise ValueError("Il minimo bidirezionale non coincide con il minimo delle direzioni")
    lengths = frame[["original_n_words", "modified_n_words", "word_ratio"]].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Lunghezze e rapporto devono essere finiti e positivi")
    if not np.allclose(frame.word_ratio, frame.modified_n_words / frame.original_n_words,
                       atol=1e-12, rtol=1e-12):
        raise ValueError("Il rapporto non coincide con le lunghezze dichiarate")
    if not frame.groupby("source_id").original_n_words.nunique().eq(1).all():
        raise ValueError("Il testo originale deve avere la stessa lunghezza per tutti i metodi")
    frame["in_common_sources"] = True
    return frame


def load_common_documents(scores_path, common_ids_path):
    """Filtra esplicitamente i 123 ID congelati, mantenendo anche gli allungamenti."""
    identifiers = pd.read_csv(common_ids_path, keep_default_na=False)
    if ("source_id" not in identifiers or len(identifiers) != 123
            or identifiers.source_id.duplicated().any()
            or not identifiers.source_id.map(lambda x:isinstance(x, str) and bool(x.strip())).all()):
        raise ValueError("Il file deve contenere esattamente 123 ID di fonte univoci")
    documents = pd.read_csv(scores_path, keep_default_na=False, float_precision="round_trip")
    if "source_id" not in documents:
        raise ValueError("Mancano gli identificativi di fonte negli score")
    selected = documents[documents.source_id.isin(set(identifiers.source_id))]
    if set(selected.source_id) != set(identifiers.source_id) or len(selected) != 492:
        raise ValueError("Mancano output per una o più delle 123 fonti comuni")
    return _validate_documents(selected)


def _rank_correlation(left, right):
    """Spearman con ranghi medi dei pareggi, anche per matrici di ricampionamenti."""
    a, b = rankdata(left, axis=-1), rankdata(right, axis=-1)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a*a).sum(axis=-1) * (b*b).sum(axis=-1))
    return np.divide((a*b).sum(axis=-1), denominator,
                     out=np.full(np.shape(denominator), np.nan), where=denominator > 0)


def analyze_directions(documents, *, bootstrap=5000, seed=42):
    """Quartili, associazioni con la lunghezza, ECDF e bootstrap appaiato già definito."""
    if (isinstance(bootstrap, bool) or not isinstance(bootstrap, Integral) or bootstrap < 1
            or isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("Repliche intere positive e seme intero non negativo richiesti")
    documents = _validate_documents(documents)
    sources = sorted(documents.source_id.unique())
    n = len(sources)
    # Stesse estrazioni della funzione storica: cambiano gli aggregati, non le fonti ricampionate.
    digest = hashlib.sha256(json.dumps([int(seed), "overall", "all"]).encode()).hexdigest()
    draws = np.random.default_rng(int(digest[:8], 16)).integers(0, n, size=(int(bootstrap), n))
    wide, long, correlations, weakest, fractions, ecdf = [], [], [], [], [], []
    for system in PRIMARY_SYSTEMS:
        own = documents[documents.humanizer.eq(system)].set_index("source_id").loc[sources]
        info = dict(humanizer=system, n_sources=n)
        row = dict(info)
        for metric in DIRECTIONAL_METRICS:
            q1, median, q3 = np.quantile(own[metric].to_numpy(dtype=float), [.25, .5, .75])
            row.update({metric + "_median": float(median), metric + "_q25": float(q1), metric + "_q75": float(q3)})
            long.append(dict(**info, metric=metric, median=float(median), q25=float(q1), q75=float(q3), iqr=float(q3-q1)))
        wide.append(row)
        reverse, ratio = own.nli_reverse_min.to_numpy(dtype=float), own.word_ratio.to_numpy(dtype=float)
        rho = float(_rank_correlation(reverse, ratio))
        sampled = _rank_correlation(reverse[draws], ratio[draws])
        finite = sampled[np.isfinite(sampled)]
        low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        degenerate = len(finite) > 0 and np.ptp(finite) < 1e-15
        status = ("constant_input" if not np.isfinite(rho) else
                  "no_observed_bootstrap_variation" if degenerate else
                  "some_undefined_resamples" if len(finite) != bootstrap else "pointwise_source_percentile")
        correlations.append(dict(**info, rho=rho, ci95_low=float(low) if not degenerate else np.nan,
            ci95_high=float(high) if not degenerate else np.nan, bootstrap_replicates=int(bootstrap),
            n_finite_bootstrap_replicates=len(finite), interval_status=status,
            interpretation="associazione descrittiva, non effetto causale della lunghezza"))
        forward = own.nli_forward_min.to_numpy(dtype=float)
        nr, nf, nt = int((reverse < forward).sum()), int((forward < reverse).sum()), int((reverse == forward).sum())
        weakest.append(dict(**info, n_reverse_lower=nr, n_forward_lower=nf, n_ties=nt,
                            fraction_reverse_lower=nr/n, fraction_forward_lower=nf/n, fraction_ties=nt/n))
        scores = own.nli_bidirectional_min.to_numpy(dtype=float)
        n5, n8 = int((scores >= .5).sum()), int((scores >= .8).sum())
        fractions.append(dict(**info, n_ge_0_5=n5, fraction_ge_0_5=n5/n,
                              n_ge_0_8=n8, fraction_ge_0_8=n8/n))
        values, counts = np.unique(scores, return_counts=True)
        for score, cumulative in zip(values, counts.cumsum()):
            ecdf.append(dict(**info, score=float(score), n_cumulative=int(cumulative), cumulative_fraction=float(cumulative/n)))
    comparisons = paired_source_comparisons(documents, bootstrap=int(bootstrap), seed=int(seed))
    comparisons = comparisons[comparisons.stratum.eq("overall") & comparisons.metric.isin(DIRECTIONAL_METRICS)].reset_index(drop=True)
    focus = comparisons[[tuple(x) in FOCUS_PAIRS for x in comparisons[["humanizer_a", "humanizer_b"]].to_numpy()]].copy()
    return dict(documents=documents, directional_summary=pd.DataFrame(wide), directional_long=pd.DataFrame(long),
                correlations=pd.DataFrame(correlations), weakest_direction=pd.DataFrame(weakest),
                support_fractions=pd.DataFrame(fractions), ecdf_points=pd.DataFrame(ecdf),
                pairwise_comparisons=comparisons, focus_comparisons=focus.reset_index(drop=True))
