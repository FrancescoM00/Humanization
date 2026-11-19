"""Intervalli e confronti appaiati per i controlli dei detector."""
from __future__ import annotations

import math
from numbers import Integral, Real

from scipy.stats import binomtest, norm


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name}: atteso un intero non negativo")
    return int(value)


def wilson_interval(successes, total, confidence=.95):
    """Intervallo bilaterale di Wilson; nessuna osservazione restituisce NaN.

    L'intervallo descrive una proporzione binomiale: ogni fonte deve comparire
    una sola volta. Anche con zero successi il limite superiore resta positivo.
    """
    successes = _count(successes, "successes")
    total = _count(total, "total")
    if successes > total:
        raise ValueError("I successi non possono superare le osservazioni")
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("La confidenza deve essere un numero compreso fra zero e uno")
    if total == 0:
        return math.nan, math.nan
    # isf evita l'arrotondamento a uno di (1 + confidence) / 2.
    z = float(norm.isf((1 - float(confidence)) / 2))
    p = successes / total
    z2_n = z * z / total
    center = (p + z2_n / 2) / (1 + z2_n)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z2_n)
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return low, high


def mcnemar_exact(control_only, reference_only):
    """P-value bilaterale esatto sulle sole coppie discordanti.

    I due conteggi indicano successi esclusivi del controllo e del riscrittore
    sulle stesse fonti. In assenza di discordanti il p-value è uno.
    """
    control_only = _count(control_only, "control_only")
    reference_only = _count(reference_only, "reference_only")
    discordant = control_only + reference_only
    if discordant == 0:
        return 1.0
    return float(binomtest(control_only, discordant, p=.5, alternative="two-sided").pvalue)


def holm_adjust(p_values):
    """Correzione di Holm per una famiglia fissata, nell'ordine originale."""
    p_values = list(p_values)
    for value in p_values:
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Ogni p-value deve essere finito e compreso fra zero e uno")
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted
