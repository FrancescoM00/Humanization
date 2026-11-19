"""Misura i cambiamenti dei detector sulle stesse triple umano/originale/riscrittura."""
from __future__ import annotations

import numpy as np
import pandas as pd


METRICS = ("accuracy", "balanced_accuracy", "auroc", "tpr", "fnr", "fpr")
REQUIRED_COLUMNS = (
    "source_id", "source_group_id", "human_score", "original_score", "rewrite_score",
    "human_is_ai", "original_is_ai", "rewrite_is_ai",
)


def _empty_result(bootstrap, seed, auroc):
    result = dict(
        n_pairs=0, n_source_groups=0, n_human=0, n_ai=0,
        bootstrap_requested_draws=bootstrap, bootstrap_draws=0, bootstrap_seed=seed,
        auroc_available=False,
        auroc_unavailable_reason="no_complete_pairs" if auroc else "binary_ensemble_no_continuous_score",
    )
    for stage in ("before", "after"):
        for count in ("tn", "fp", "fn", "tp"):
            result[f"{count}_{stage}"] = 0
    for metric in METRICS:
        for name in (f"{metric}_before", f"{metric}_after", f"delta_{metric}"):
            result[name] = float("nan")
            result[name + "_ci95_low"] = float("nan")
            result[name + "_ci95_high"] = float("nan")
    return result


def _validated_arrays(pairs):
    missing = set(REQUIRED_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Mancano colonne nelle triple appaiate: {sorted(missing)}")
    if pairs.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Le triple appaiate non possono contenere valori mancanti")
    ids = pairs.source_id.map(lambda value: str(value).strip())
    groups = pairs.source_group_id.map(lambda value: str(value).strip())
    if ids.eq("").any() or groups.eq("").any() or ids.duplicated().any():
        raise ValueError("Gli ID devono essere non vuoti e le fonti devono essere uniche")
    scores = {}
    for name in ("human_score", "original_score", "rewrite_score"):
        try:
            values = pairs[name].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Gli score dei detector devono essere numerici e finiti") from exc
        if not np.isfinite(values).all():
            raise ValueError("Gli score dei detector devono essere numerici e finiti")
        scores[name] = values
    decisions = {}
    for name in ("human_is_ai", "original_is_ai", "rewrite_is_ai"):
        if any(not isinstance(value, (bool, np.bool_)) for value in pairs[name].tolist()):
            raise ValueError("Le decisioni congelate dei detector devono essere booleane")
        decisions[name] = pairs[name].to_numpy(dtype=bool)
    codes, names = pd.factorize(groups, sort=True)
    return scores, decisions, codes, len(names)


def _auc_kernel(positive, negative):
    """Confronta ogni IA con ogni umano; un pareggio vale mezzo confronto."""
    return ((positive[:, None] > negative[None, :]).astype(float)
            + .5 * (positive[:, None] == negative[None, :]))


def _weighted_auc(kernel, weights, mass):
    """I due testi della stessa coppia ricevono sempre lo stesso peso."""
    return ((weights @ kernel) * weights).sum(axis=1) / mass**2


def summarize_paired_metrics(pairs: pd.DataFrame, *, bootstrap=5000, seed=42, auroc=True) -> dict:
    """Calcola prima, dopo e differenza sulle sole triple complete fornite.

    Gli score devono essere già orientati verso IA crescente; le decisioni
    booleane provengono dalle soglie congelate. Nessuna calibrazione, inferenza
    o imputazione viene effettuata qui. Ogni coppia contiene un umano e un
    testo IA, quindi accuratezza e accuratezza bilanciata coincidono.

    Il bootstrap estrae gruppi di fonte interi: umano, originale e riscrittura
    viaggiano insieme e i pesi restano identici prima e dopo. Gli intervalli
    sono percentili puntuali al 95%, condizionati a score e soglie salvati.
    Con un solo gruppo o zero ricampionamenti gli intervalli sono indefiniti.
    """
    if not isinstance(pairs, pd.DataFrame):
        raise TypeError("Le triple appaiate devono essere una tabella pandas")
    if (isinstance(bootstrap, (bool, np.bool_)) or not isinstance(bootstrap, (int, np.integer))
            or bootstrap < 0):
        raise ValueError("Il numero di ricampionamenti deve essere un intero non negativo")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Il seme deve essere un intero non negativo")
    if not isinstance(auroc, (bool, np.bool_)):
        raise ValueError("L'opzione AUROC deve essere booleana")
    bootstrap, seed, auroc = int(bootstrap), int(seed), bool(auroc)
    result = _empty_result(bootstrap, seed, auroc)
    if pairs.empty:
        return result
    scores, decisions, codes, n_groups = _validated_arrays(pairs)
    n = len(pairs)
    human = decisions["human_is_ai"]
    original = decisions["original_is_ai"]
    rewritten = decisions["rewrite_is_ai"]
    result.update(n_pairs=n, n_source_groups=n_groups, n_human=n, n_ai=n,
                  auroc_available=auroc,
                  auroc_unavailable_reason="" if auroc else "binary_ensemble_no_continuous_score")
    tn, fp = int((~human).sum()), int(human.sum())
    before, after = {}, {}
    for stage, prediction, values in (("before", original, before), ("after", rewritten, after)):
        tp = int(prediction.sum())
        result.update({f"tn_{stage}": tn, f"fp_{stage}": fp,
                       f"fn_{stage}": n-tp, f"tp_{stage}": tp})
        values.update(accuracy=(tn+tp)/(2*n), tpr=tp/n, fnr=1-tp/n, fpr=fp/n)
        values["balanced_accuracy"] = values["accuracy"]
    kernels = None
    if auroc:
        kernels = (_auc_kernel(scores["original_score"], scores["human_score"]),
                   _auc_kernel(scores["rewrite_score"], scores["human_score"]))
        before["auroc"], after["auroc"] = (float(kernel.mean()) for kernel in kernels)
    else:
        before["auroc"] = after["auroc"] = float("nan")
    delta = {name: after[name]-before[name] for name in METRICS}
    # Queste identità derivano dalla stessa classe negativa, invariata, e dal bilanciamento.
    delta["accuracy"] = delta["balanced_accuracy"] = .5 * delta["tpr"]
    delta["fnr"] = -delta["tpr"]
    delta["fpr"] = 0.0
    for metric in METRICS:
        result.update({f"{metric}_before": before[metric], f"{metric}_after": after[metric],
                       f"delta_{metric}": delta[metric]})
    if not bootstrap or n_groups < 2:
        return result

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1/n_groups), size=bootstrap)
    weights = counts[:, codes].astype(float)
    mass = weights.sum(axis=1)
    tnr = weights @ (~human).astype(float) / mass
    original_tpr = weights @ original.astype(float) / mass
    rewritten_tpr = weights @ rewritten.astype(float) / mass
    fpr = 1-tnr
    boots_before = dict(accuracy=.5*(tnr+original_tpr), balanced_accuracy=.5*(tnr+original_tpr),
                        tpr=original_tpr, fnr=1-original_tpr, fpr=fpr)
    boots_after = dict(accuracy=.5*(tnr+rewritten_tpr), balanced_accuracy=.5*(tnr+rewritten_tpr),
                       tpr=rewritten_tpr, fnr=1-rewritten_tpr, fpr=fpr)
    if auroc:
        boots_before["auroc"] = _weighted_auc(kernels[0], weights, mass)
        boots_after["auroc"] = _weighted_auc(kernels[1], weights, mass)
    result["bootstrap_draws"] = bootstrap
    for metric in METRICS:
        if metric == "auroc" and not auroc:
            continue
        values_before, values_after = boots_before[metric], boots_after[metric]
        difference = values_after-values_before
        if metric in {"accuracy", "balanced_accuracy"}:
            difference = .5*(rewritten_tpr-original_tpr)
        elif metric == "fnr":
            difference = -(rewritten_tpr-original_tpr)
        for name, values in ((f"{metric}_before", values_before),
                             (f"{metric}_after", values_after),
                             (f"delta_{metric}", difference)):
            low, high = np.quantile(values, [.025, .975])
            result[name + "_ci95_low"] = float(low)
            result[name + "_ci95_high"] = float(high)
    return result
