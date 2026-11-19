"""Segnalazioni lessicali deterministiche per riscritture reali.

Le differenze rilevate richiedono ispezione: non sono errori semantici accertati.
Le regole non dipendono dai punteggi NLI e non modificano le guardie storiche.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import re
import unicodedata

from .lexical_guards import (
    NEGATIONS, QUANTIFIERS, definitions as historical_definitions,
    numeric_values, text_inventory,
)


PROTOCOL = "humanizer_lexical_guards_v2"
FAMILIES = ("number", "date", "entity", "negation", "quantifier", "hedge")
ALERT_COLUMNS = ("alert_number", "alert_date", "alert_entity", "alert_negation",
                 "alert_quantifier", "alert_hedge_removal", "alert_any")
EXCLUDED_ENTITY_LABELS = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"}
MONTHS = {name: index for index, name in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTHS.update({name[:3]: value for name, value in list(MONTHS.items())})
MONTH_PATTERN = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
DAY_PATTERN = r"\d{1,2}(?:st|nd|rd|th)?"
NAMED_DATES = re.compile(
    rf"\b(?:{DAY_PATTERN}\s*{MONTH_PATTERN}(?:\s*,?\s*\d{{4}})?|"
    rf"{MONTH_PATTERN}\s*{DAY_PATTERN}(?:\s*,?\s*\d{{4}})?(?!\d)|"
    rf"{MONTH_PATTERN}\s*\d{{4}})\b", re.I)
ISO_DATES = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
BOUND_PHRASES = {
    ("no", "more", "than"): "upper_inclusive",
    ("not", "more", "than"): "upper_inclusive",
    ("up", "to"): "upper_inclusive",
    ("at", "most"): "upper_inclusive",
    ("no", "fewer", "than"): "lower_inclusive",
    ("not", "less", "than"): "lower_inclusive",
    ("at", "least"): "lower_inclusive",
    ("more", "than"): "lower_exclusive",
    ("greater", "than"): "lower_exclusive",
    ("over",): "lower_exclusive",
    ("less", "than"): "upper_exclusive",
    ("fewer", "than"): "upper_exclusive",
    ("under",): "upper_exclusive",
    ("below",): "upper_exclusive",
}


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())


def _canonical_date(text):
    """Normalizza solo date esplicite; le forme ambigue restano testuali."""
    text = _normal(text)
    clean = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text)
    clean = re.sub(r"[,\.]", "", clean)
    clean = re.sub(r"^(?:the\s+)", "", clean)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean):
        try:
            return "calendar:" + date.fromisoformat(clean).isoformat()
        except ValueError:
            return "surface:" + text
    month_first = re.fullmatch(r"([a-z]+)\s*(\d{1,2})(?:\s*(\d{4}))?", clean)
    day_first = re.fullmatch(r"(\d{1,2})\s*([a-z]+)(?:\s*(\d{4}))?", clean)
    if month_first or day_first:
        match = month_first or day_first
        month, day = ((match[1], match[2]) if month_first else (match[2], match[1]))
        if month in MONTHS:
            try:
                value = date(int(match[3] or 2000), MONTHS[month], int(day))
            except ValueError:
                return "surface:" + text
            return ("calendar:" + value.isoformat() if match[3] else
                    f"month-day:{value.month:02d}-{value.day:02d}")
    month_year = re.fullmatch(r"([a-z]+)\s*(\d{4})", clean)
    if month_year and month_year[1] in MONTHS:
        return f"year-month:{int(month_year[2]):04d}-{MONTHS[month_year[1]]:02d}"
    if re.fullmatch(r"\d{4}", clean):
        return "year:" + clean
    if clean in MONTHS:
        return f"month:{MONTHS[clean]:02d}"
    return "surface:" + clean


def _date_inventory(doc):
    """Le date esplicite hanno priorità sugli span NER sovrapposti."""
    candidates = sorted([(m.start(), m.end(), m.group()) for pattern in (ISO_DATES, NAMED_DATES)
                         for m in pattern.finditer(doc.text)], key=lambda x: (x[0], -(x[1]-x[0])))
    accepted = []
    for start, end, text in candidates:
        if not any(start < b and a < end for a, b, _ in accepted):
            accepted.append((start, end, text))
    for entity in doc.ents:
        if entity.label_ == "DATE" and not any(entity.start_char < b and a < entity.end_char
                                               for a, b, _ in accepted):
            accepted.append((entity.start_char, entity.end_char, entity.text))
    return Counter(_canonical_date(text) for _, _, text in accepted), accepted


def _entity_surface(text):
    text = re.sub(r"(?:'s|’s)$", "", _normal(text))
    text = text.replace(".", "").replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", "", text).split())


def _quantifiers(doc):
    """Associa i limiti a un valore vicino; non interpreta 'over' spaziale."""
    tokens = list(doc)
    words = [_normal(token.text) for token in tokens]
    counts, consumed = Counter(), set()
    for start in range(len(tokens)):
        if start in consumed:
            continue
        for phrase in sorted(BOUND_PHRASES, key=len, reverse=True):
            end = start + len(phrase)
            if tuple(words[start:end]) != phrase:
                continue
            number_index = end
            while number_index < len(tokens) and words[number_index] in {
                    "about", "approximately", "roughly", "$", "£", "€", "usd", "gbp", "eur"}:
                number_index += 1
            if number_index >= len(tokens) or not tokens[number_index].like_num:
                continue
            values = numeric_values(tokens[number_index:number_index+5])
            value = next(iter(values), "surface:" + words[number_index])
            counts[BOUND_PHRASES[phrase] + "|" + value] += 1
            consumed.update(range(start, end))
            break
    for index, word in enumerate(words):
        if index not in consumed and word in QUANTIFIERS:
            counts[QUANTIFIERS[word]] += 1
    return counts, consumed


def extract_inventory(doc):
    """Estrae inventari confrontabili e metadati, tutti serializzabili in JSON."""
    if not doc.has_annotation("ENT_IOB"):
        raise ValueError("Le guardie sulle entità richiedono spaCy con NER attivo")
    old = text_inventory(doc)
    dates, date_spans = _date_inventory(doc)
    numeric_tokens = [token for token in doc if not any(
        token.idx < end and start < token.idx + len(token) for start, end, _ in date_spans)]
    entities = sorted({_entity_surface(entity.text) for entity in doc.ents
                       if entity.label_ not in EXCLUDED_ENTITY_LABELS and _entity_surface(entity.text)})
    quantifiers, consumed = _quantifiers(doc)
    negations = Counter(old["negation"])
    # Il 'no' di 'no more than' appartiene al limite, non a una negazione separata.
    for token in doc:
        word = _normal(token.text)
        if token.i in consumed and word in NEGATIONS:
            key = NEGATIONS[word]
            negations[key] -= 1
            if not negations[key]:
                del negations[key]
    return {"number": dict(numeric_values(numeric_tokens)), "date": dict(dates),
            "entity": dict.fromkeys(entities, 1), "negation": dict(negations),
            "quantifier": dict(quantifiers), "hedge": dict(old["hedge"]),
            "n_words": sum(bool(token.is_alpha) for token in doc),
            "entity_spans": [{"text": e.text, "label": e.label_} for e in doc.ents
                             if e.label_ not in EXCLUDED_ENTITY_LABELS],
            "date_spans": [{"text": text, "start": start, "end": end,
                            "canonical": _canonical_date(text)} for start, end, text in sorted(date_spans)]}


def compare_inventories(left, right):
    """Segnala cambiamenti d'inventario; per le cautele conta soltanto il calo totale."""
    result = {}
    for family in FAMILIES:
        before, after = Counter(left[family]), Counter(right[family])
        removed, added = before - after, after - before
        n_before, n_after = sum(before.values()), sum(after.values())
        prefix = "guard_" + family
        result.update({prefix + "_original_count": n_before, prefix + "_modified_count": n_after,
                       prefix + "_signed_count_delta": n_after-n_before,
                       prefix + "_lost_items": sum(removed.values()), prefix + "_gained_items": sum(added.values()),
                       prefix + "_lost": bool(removed), prefix + "_introduced": bool(added),
                       prefix + "_removed": dict(removed), prefix + "_added": dict(added),
                       prefix + "_original_inventory": dict(before), prefix + "_modified_inventory": dict(after)})
        alert = n_after < n_before if family == "hedge" else bool(removed or added)
        result["alert_" + ("hedge_removal" if family == "hedge" else family)] = alert
    result["alert_any"] = any(result[name] for name in ALERT_COLUMNS if name != "alert_any")
    result["original_n_words"], result["modified_n_words"] = left["n_words"], right["n_words"]
    result["word_ratio"] = right["n_words"] / left["n_words"] if left["n_words"] else float("nan")
    for name, inventory in (("original", left), ("modified", right)):
        result["guard_" + name + "_entity_spans"] = inventory["entity_spans"]
        result["guard_" + name + "_date_spans"] = inventory["date_spans"]
    return result


def score_pairs(pairs, nlp=None, progress_callback=None):
    """Elabora ogni testo una volta e restituisce conteggi, evidenze e segnalazioni."""
    import pandas as pd
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    records = list(pairs)
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Gli identificativi delle coppie devono essere univoci")
    texts = list(dict.fromkeys(text for row in records for text in (row["original"], row["modified"])))
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Le guardie richiedono testi non vuoti")
    inventories = {}
    for index, (text, doc) in enumerate(zip(texts, nlp.pipe(texts, batch_size=32)), 1):
        inventories[hashlib.sha256(text.encode()).hexdigest()] = extract_inventory(doc)
        if progress_callback and (index % 100 == 0 or index == len(texts)):
            progress_callback(index, len(texts))
    rows = []
    for row in records:
        left, right = [inventories[hashlib.sha256(row[name].encode()).hexdigest()]
                       for name in ("original", "modified")]
        rows.append({"id": row["id"], **compare_inventories(left, right)})
    return pd.DataFrame(rows)


def definitions():
    """Documenta inventari e limiti senza calibrare soglie sui risultati."""
    return {"protocol": PROTOCOL, "alerts": list(ALERT_COLUMNS),
            "historical_rules_reused": historical_definitions(),
            "number_policy": "Multinsieme normalizzato storico, esclusi gli span riconosciuti come date; cambi di valore/unità e frequenza di menzione.",
            "date_policy": "Date NER e pattern calendario espliciti; normalizzazione di mese/giorno/anno e ISO, nessuna interpretazione fuzzy di date ambigue.",
            "entity_policy": "Presenza di superfici NER normalizzate, senza pesare ripetizioni o differenze di etichetta; esclusi tipi numerici/date. Nessuna risoluzione di alias.",
            "entity_labels_excluded": sorted(EXCLUDED_ENTITY_LABELS),
            "quantifier_phrases": {" ".join(k):v for k,v in BOUND_PHRASES.items()},
            "quantifier_policy": "Limiti multi-parola con numero vicino e valore normalizzato; priorità ai limiti lunghi, senza doppi conteggi. Altri quantificatori dal lessico storico.",
            "hedge_policy": "Calo del totale di cautele modali MD e lessicali storiche; may/might e sinonimi non generano allarme se il totale resta uguale.",
            "negation_policy": "Conteggi e inventario normalizzati; le negazioni dentro limiti numerici sono conteggiate solo come quantificatori.",
            "warnings_are_confirmed_semantic_errors": False,
            "limits": ["Errori NER e alias, pronomi o riferimenti impliciti possono produrre falsi allarmi.",
                       "Inventari globali non individuano necessariamente entità o numeri riassegnati a fatti diversi.",
                       "Ripetizioni di numeri/date e cambi di forma non normalizzati possono produrre segnalazioni.",
                       "Abilità, permesso e incertezza dei modali non sono distinti; lessico delle cautele incompleto.",
                       "Non vengono risolti portata della negazione, coreferenza, conversioni generali di unità o equivalenza semantica."]}
