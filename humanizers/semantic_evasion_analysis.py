"""Unisce evasione e requisiti semantici con denominatori e bootstrap appaiati."""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .null_control_statistics import holm_adjust, mcnemar_exact, wilson_interval


def _validate(frame, config):
    required = {"source_id", "source_group_id", "method", "detector", "orig_detected",
                "rew_detected", "evaded", "nli_cons_mean", "nli_min", "bs_resc_mean",
                "len_ratio", "score_reduction"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("Mancano colonne o coppie dell'esperimento")
    if frame[["source_id", "method", "detector"]].duplicated().any():
        raise ValueError("Una riscrittura non può comparire due volte per detector")
    if set(frame.method) != set(config["methods"]) or set(frame.detector) != set(config["detectors"]):
        raise ValueError("Metodi e detector devono corrispondere alla configurazione fissata")
    sources = sorted(frame.source_id.unique())
    for _, own in frame.groupby(["method", "detector"]):
        if set(own.source_id) != set(sources):
            raise ValueError("Ogni confronto richiede le stesse fonti")
    if (not frame.groupby("source_id").source_group_id.nunique().eq(1).all()
            or not frame.groupby("source_group_id").source_id.nunique().eq(1).all()):
        raise ValueError("Il bootstrap richiede una sola fonte per gruppo")
    for field in ("orig_detected", "rew_detected", "evaded"):
        if not frame[field].map(lambda v:isinstance(v, (bool, np.bool_))).all():
            raise ValueError("I flag di rilevazione devono essere booleani")
    if not np.array_equal(frame.evaded, frame.orig_detected & ~frame.rew_detected):
        raise ValueError("Il flag evasione non corrisponde alle decisioni del detector")
    if not frame.groupby(["source_id", "detector"]).orig_detected.nunique().eq(1).all():
        raise ValueError("La rilevazione dell'originale non può cambiare fra metodi")
    numeric = frame[["nli_cons_mean", "nli_min", "bs_resc_mean", "len_ratio", "score_reduction"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame.len_ratio <= 0).any():
        raise ValueError("Punteggi e rapporti devono essere finiti, senza imputazione")
    if ((frame[["nli_cons_mean", "nli_min"]] < 0).any().any()
            or (frame[["nli_cons_mean", "nli_min"]] > 1).any().any()):
        raise ValueError("I supporti NLI devono appartenere all'intervallo zero–uno")
    return sources


def _proportion(k, n):
    low, high = wilson_interval(int(k), int(n))
    return dict(asr=float(k/n) if n else np.nan, wilson95_low=low, wilson95_high=high)


def _group_summary(values):
    if not len(values):
        return dict(median=np.nan, q25=np.nan, q75=np.nan)
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return dict(median=float(median), q25=float(q1), q75=float(q3))


def _within_method(own, draws, metrics):
    """Confronta esclusivamente originali rilevati, senza confondere i non eleggibili."""
    original = own.orig_detected.to_numpy(bool)
    evaded = own.evaded.to_numpy(bool)
    non_evaded = original & ~evaded
    n_e, n_n = int(evaded.sum()), int(non_evaded.sum())
    rows = []
    for metric in metrics:
        values = own[metric].to_numpy(float)
        summary = {"evaded_" + k:v for k,v in _group_summary(values[evaded]).items()}
        summary.update({"not_evaded_" + k:v for k,v in _group_summary(values[non_evaded]).items()})
        difference, low, high, valid, degenerate = np.nan, np.nan, np.nan, 0, False
        if n_e and n_n:
            difference = summary["evaded_median"] - summary["not_evaded_median"]
            sampled = values[draws]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                left = np.nanmedian(np.where(evaded[draws], sampled, np.nan), axis=1)
                right = np.nanmedian(np.where(non_evaded[draws], sampled, np.nan), axis=1)
            differences = left - right
            differences = differences[np.isfinite(differences)]
            valid = len(differences)
            if valid:
                low, high = np.quantile(differences, [.025, .975])
                degenerate = bool(np.ptp(differences) < 1e-15)
            if degenerate:
                low, high = np.nan, np.nan
            status = ("single_source_group_exploratory" if min(n_e, n_n) == 1 else
                      "no_observed_bootstrap_variation" if degenerate else
                      "some_undefined_resamples" if valid < len(draws) else "pointwise_source_percentile")
        else:
            status = "empty_evaded_group" if not n_e else "empty_not_evaded_group"
        rows.append(dict(metric=metric, n_original_detected=int(original.sum()), n_evaded=n_e,
            n_not_evaded=n_n, **summary, difference_evaded_minus_not=float(difference),
            ci95_low=float(low), ci95_high=float(high), bootstrap_n_valid=valid,
            bootstrap_n_undefined=len(draws)-valid, bootstrap_degenerate=degenerate,
            interval_status=status))
    return rows


def analyze(long_df, config):
    """Calcola tabelle e curve definite nella configurazione, senza adattarne le soglie."""
    sources = _validate(long_df, config)
    bootstrap, seed = config["bootstrap_replicates"], config["bootstrap_seed"]
    if (isinstance(bootstrap, bool) or not isinstance(bootstrap, Integral) or bootstrap < 1
            or isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("Repliche positive e seme intero non negativo richiesti")
    n_sources = len(sources)
    draws = np.random.default_rng(int(seed)).integers(0, n_sources, size=(int(bootstrap), n_sources))
    # Il medesimo ricampionamento conserva insieme tutti i metodi e i detector.
    weights = np.zeros((bootstrap, n_sources), dtype=float)
    np.add.at(weights, (np.arange(bootstrap)[:, None], draws), 1)
    grid = config["curve_grid"]
    count = int(round((grid["stop"]-grid["start"]) / grid["step"]))
    taus = np.array([round(grid["start"] + i * grid["step"], 12)
                     for i in range(count + int(grid.get("include_stop", True)))])
    main, curves, groups, score_correlations, stringent_by_detector = [], [], [], [], {}
    for detector in config["detectors"]:
        stringent_by_detector[detector] = {}
        for method in config["methods"]:
            own = long_df[long_df.detector.eq(detector) & long_df.method.eq(method)].set_index("source_id").loc[sources]
            original = own.orig_detected.to_numpy(bool)
            evaded = own.evaded.to_numpy(bool)
            denominator = int(original.sum())
            boot_denominator = weights @ original.astype(float)
            valid = boot_denominator > 0
            mean, minimum = own.nli_cons_mean.to_numpy(float), own.nli_min.to_numpy(float)
            moderate = evaded & (mean >= config["moderate"]["nli_cons_mean_min"])
            stringent = evaded & (mean >= config["stringent"]["nli_cons_mean_min"]) & (minimum >= config["stringent"]["nli_min_min"])
            stringent_by_detector[detector][method] = (original, stringent)
            base = dict(detector=detector, method=method, n_sources=n_sources, n_original_detected=denominator)
            row = dict(base)
            for name, successes in [("simple", evaded), ("moderate", moderate), ("stringent", stringent)]:
                k = int(successes.sum())
                row[name + "_k"] = k
                row.update({name + "_" + key:value for key,value in _proportion(k, denominator).items()})
            expanded = evaded & (own.len_ratio.to_numpy(float) > 1)
            row.update(evaded_expanded=int(expanded.sum()),
                       evaded_expanded_fraction=float(expanded.sum()/evaded.sum()) if evaded.sum() else np.nan,
                       n_expanded=int((own.len_ratio > 1).sum()))
            row.update(moderate_retained_fraction=float(moderate.sum()/evaded.sum()) if evaded.sum() else np.nan,
                       stringent_retained_fraction=float(stringent.sum()/evaded.sum()) if evaded.sum() else np.nan)
            main.append(row)
            for profile in config["curves"]:
                score = own[profile["score"]].to_numpy(float)
                qualifies = score[:, None] >= taus[None, :]
                if profile["additional_nli_min"] is not None:
                    qualifies &= minimum[:, None] >= profile["additional_nli_min"]
                success = evaded[:, None] & qualifies
                counts = success.sum(axis=0)
                boot_counts = weights[valid] @ success.astype(float)
                boot_asr = boot_counts / boot_denominator[valid, None]
                low, high = np.quantile(boot_asr, [.025, .975], axis=0) if valid.any() else (np.full(len(taus), np.nan),)*2
                degenerate = np.ptp(boot_asr, axis=0) < 1e-15 if valid.any() else np.ones(len(taus), bool)
                qualified_originals = (original[:, None] & qualifies).sum(axis=0)
                for j, tau in enumerate(taus):
                    point = _proportion(int(counts[j]), denominator)
                    semantic_fraction = float(qualified_originals[j]/denominator) if denominator else np.nan
                    curves.append(dict(**base, curve=profile["name"], tau=float(tau), k=int(counts[j]), **point,
                        bootstrap_ci95_low=float(low[j]), bootstrap_ci95_high=float(high[j]),
                        bootstrap_n_valid=int(valid.sum()), bootstrap_n_undefined=int((~valid).sum()),
                        bootstrap_degenerate=bool(degenerate[j]), zero_events=int(counts[j]) == 0,
                        simple_asr=row["simple_asr"], semantic_qualified_among_originals=int(qualified_originals[j]),
                        semantic_qualified_fraction_among_originals=semantic_fraction,
                        independence_asr=row["simple_asr"] * semantic_fraction))
            for item in _within_method(own, draws, config["evaded_vs_not_statistics"]):
                groups.append(dict(detector=detector, method=method, **item))
            reduction = own.score_reduction.to_numpy(float)
            constant = np.unique(reduction).size < 2 or np.unique(mean).size < 2
            rho = float(spearmanr(reduction, mean).statistic) if not constant else np.nan
            score_correlations.append(dict(**base, rho=rho,
                statistic="Spearman(score_reduction,nli_cons_mean), all common sources",
                status="constant_input" if constant else "descriptive_no_p_value"))
    comparisons = []
    for detector, methods in stringent_by_detector.items():
        for a, b in combinations(config["methods"], 2):
            original_a, values_a = methods[a]
            original_b, values_b = methods[b]
            if not np.array_equal(original_a, original_b):
                raise ValueError("I metodi devono avere la stessa eleggibilità originale")
            a_only = int((values_a & ~values_b).sum())
            b_only = int((values_b & ~values_a).sum())
            n = int(original_a.sum())
            comparisons.append(dict(detector=detector, method_a=a, method_b=b, n_original_detected=n,
                stringent_k_a=int(values_a.sum()), stringent_k_b=int(values_b.sum()),
                a_only=a_only, b_only=b_only, mcnemar_p=mcnemar_exact(a_only, b_only),
                delta_stringent_asr_b_minus_a=float((values_b.sum()-values_a.sum())/n) if n else np.nan))
    comparison = pd.DataFrame(comparisons)
    comparison["mcnemar_p_holm"] = holm_adjust(comparison.mcnemar_p.tolist())
    comparison["holm_family_size"] = len(comparison)
    curve_frame = pd.DataFrame(curves)
    sensitivity = curve_frame[np.isclose(curve_frame.tau.to_numpy()[:,None], config["sensitivity_taus"]).any(axis=1)].copy()
    return dict(main_table=pd.DataFrame(main), curves=curve_frame, sensitivity=sensitivity,
                evaded_vs_not=pd.DataFrame(groups), score_correlations=pd.DataFrame(score_correlations),
                method_comparisons=comparison)
