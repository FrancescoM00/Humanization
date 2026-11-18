"""Congela il confronto Google sui soli pannelli Qwen completi."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

from .naturalness import _digest
from .naturalness_judge_comparison import (
    REFERENCE_MODEL, _checkpoint_bytes, _json_bytes, _jsonl_bytes, restrict_plan,
)
from .naturalness_panel import parse_panel_judgment
from .naturalness_panel_analysis import _read_ratings
from .naturalness_panel_runner import ITEM_METADATA, _validate_plan, _validate_stored


GOOGLE_MODEL = "gemma-4-31b-it"
GOOGLE_ADAPTER_PROTOCOL = "google_native_minimal_v1"
GOOGLE_COMPARISON_PROTOCOL = "google_cross_family_naturalness_v1"
PILOT_SOURCE_COUNT = 10


def native_payload_for_request(payload):
    """Traduce i due messaggi congelati senza cambiare il loro testo.

    Il requisito di risposta JSON resta nel messaggio di sistema e nel parser.
    Non si dichiara supporto nativo a responseMimeType o a uno schema JSON.
    ``minimal`` è il livello richiesto, non una garanzia di pensiero disattivato.
    """
    fields = {"model", "temperature", "response_format", "max_completion_tokens", "messages"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Il payload canonico Google contiene parametri mancanti o inattesi")
    if payload["model"] != GOOGLE_MODEL:
        raise ValueError("Il piano fissa il solo modello Gemma previsto; nessun fallback")
    if (type(payload["temperature"]) not in {int, float} or payload["temperature"] != 0
            or payload["response_format"] != {"type": "json_object"}
            or type(payload["max_completion_tokens"]) is not int
            or payload["max_completion_tokens"] != 256):
        raise ValueError("Temperatura, formato o limite di output differiscono dal riferimento")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Sono richiesti i due messaggi originali, di sistema e utente")
    for message, role in zip(messages, ["system", "user"]):
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != role or not isinstance(message["content"], str)
                or not message["content"].strip()):
            raise ValueError("I ruoli e i contenuti devono coincidere con il formato congelato")
    return {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }


def validate_google_request(item, model=GOOGLE_MODEL):
    """Verifica che il payload realmente inviato derivi dai messaggi canonici."""
    if model != GOOGLE_MODEL:
        raise ValueError("Il piano Google richiede il modello Gemma esplicito")
    if not isinstance(item, dict) or any(name not in item for name in ITEM_METADATA):
        raise ValueError("La richiesta Google non contiene tutti i metadati previsti")
    if item.get("native_payload") != native_payload_for_request(item.get("payload")):
        raise ValueError("Il payload nativo cambia testi, ruoli o configurazione congelata")


def _google_request_id(protocol, item):
    return _digest([
        protocol, GOOGLE_ADAPTER_PROTOCOL, "google", GOOGLE_MODEL,
        item["source_id"], item["source_group_id"], item["order_id"], item["repeat_index"],
        item["payload"], item["native_payload"],
    ])


def validate_google_plan(plan):
    """Valida il piano completo; il pilota non ne riduce le richieste."""
    if (plan.get("provider") != "google" or plan.get("model") != GOOGLE_MODEL
            or plan.get("adapter_protocol") != GOOGLE_ADAPTER_PROTOCOL
            or plan.get("reasoning_control") != "google_minimal_verified"
            or plan.get("thinking_level") != "minimal"
            or "reasoning_effort" not in plan or plan["reasoning_effort"] is not None):
        raise ValueError("Provider, modello o livello di pensiero Google non coincidono con il piano")
    systems = plan.get("systems")
    if (not isinstance(systems, list) or len(systems) != 4 or len(set(systems)) != 4
            or plan.get("repetitions") != 3 or plan.get("n_orders_per_source") != 4
            or plan.get("max_completion_tokens") != 256 or plan.get("temperature") != 0):
        raise ValueError("Il piano Google richiede quattro metodi e quattro ordini ripetuti tre volte")
    requests = plan.get("requests")
    if not isinstance(requests, list) or plan.get("expected_completion_requests") != len(requests):
        raise ValueError("Il numero delle richieste Google non coincide con il piano")
    lookup, slots, counts, positions, source_order = {}, set(), Counter(), Counter(), []
    for item in requests:
        validate_google_request(item, plan["model"])
        identifier = item["request_id"]
        if (not isinstance(identifier, str) or not identifier or identifier in lookup
                or identifier != _google_request_id(plan["protocol"], item)):
            raise ValueError("Gli identificatori Google sono duplicati o diversi dal payload congelato")
        order = item["system_order"]
        if not isinstance(order, list) or sorted(order) != sorted(systems):
            raise ValueError("L'ordine Google non contiene esattamente i quattro metodi")
        if item["order_id"] not in {f"order_{i}" for i in range(4)}:
            raise ValueError("Identificatore di ordine Google inatteso")
        if type(item["repeat_index"]) is not int or item["repeat_index"] not in {0, 1, 2}:
            raise ValueError("Identificatore di ripetizione Google inatteso")
        slot = (item["source_id"], item["order_id"], item["repeat_index"])
        if slot in slots:
            raise ValueError("Una fonte ripete lo stesso slot Google")
        slots.add(slot)
        if item["source_id"] not in counts:
            source_order.append(item["source_id"])
        counts[item["source_id"]] += 1
        for position, system in enumerate(order):
            positions[item["source_id"], system, position] += 1
        lookup[identifier] = item
    if plan.get("n_sources") != len(counts) or any(n != 12 for n in counts.values()):
        raise ValueError("Ogni fonte Google deve avere tutte le dodici richieste pianificate")
    if any(n != 3 for n in positions.values()) or len(positions) != len(counts) * 16:
        raise ValueError("Le posizioni non sono bilanciate entro ogni fonte Google")
    if plan.get("pilot_source_ids") != source_order[:PILOT_SOURCE_COUNT]:
        raise ValueError("Il pilota deve usare le prime fonti del piano congelato")
    # Il lettore comune verifica anche metadati e sequenze coerenti entro fonte e ordine.
    _read_ratings(plan, [])
    return lookup


def prepare_google_comparison(reference, output, *, model=GOOGLE_MODEL):
    """Fissa tutti i completi Qwen e un pilota operativo sui primi dieci.

    Il campione non dipende dai voti. Passare dal pilota allo studio non cambia
    il piano, il campione, gli ID delle richieste o il registro dei tentativi.
    """
    if model != GOOGLE_MODEL:
        raise ValueError(f"Il confronto fissa esplicitamente {GOOGLE_MODEL}; nessun fallback")
    raw = _checkpoint_bytes(reference)
    original = json.loads(raw["judge_plan.json"])
    rows = [json.loads(line) for line in raw["judgments.jsonl"].splitlines() if line.strip()]
    attempts = [json.loads(line) for line in raw["api_attempts.jsonl"].splitlines() if line.strip()]
    if original.get("model") != REFERENCE_MODEL:
        raise ValueError("Il riferimento deve essere il precedente giudice Qwen")
    lookup = _validate_plan(original)
    _validate_stored(original, rows, attempts, lookup, parse_panel_judgment)
    _, metadata, expected, valid, invalid, complete, _ = _read_ratings(original, rows)
    if (original.get("repetitions") != 3 or original.get("n_orders_per_source") != 4
            or len(original.get("systems", [])) != 4 or any(n != 12 for n in expected.values())):
        raise ValueError("Il confronto richiede quattro metodi e quattro ordini ripetuti tre volte")
    if not complete:
        raise ValueError("Il checkpoint non contiene alcun pannello con dodici risposte valide")
    reference_plan = restrict_plan(original, complete)
    reference_rows = [row for row in rows if row["source_id"] in complete]
    plan = deepcopy(reference_plan)
    plan.update(provider="google", model=model, adapter_protocol=GOOGLE_ADAPTER_PROTOCOL,
                reasoning_control="google_minimal_verified", reasoning_effort=None,
                thinking_level="minimal")
    source_order = []
    for item in plan["requests"]:
        if item["source_id"] not in source_order:
            source_order.append(item["source_id"])
        item["payload"]["model"] = model
        item["payload"].pop("reasoning_effort", None)
        item["native_payload"] = native_payload_for_request(item["payload"])
        item["request_id"] = _google_request_id(plan["protocol"], item)
    plan["pilot_source_ids"] = source_order[:PILOT_SOURCE_COUNT]
    validate_google_plan(plan)
    selected = sorted(complete)
    domains = Counter(metadata[source]["domain"] for source in selected)
    partial = sorted(source for source in expected if source not in complete
                     and (valid.get(source, 0) or invalid.get(source, 0)))
    setup = dict(
        protocol=GOOGLE_COMPARISON_PROTOCOL, provider="google", reference_model=REFERENCE_MODEL,
        candidate_model=model, adapter_protocol=GOOGLE_ADAPTER_PROTOCOL,
        thinking_level="minimal", reasoning_control="google_minimal_verified",
        reference_plan_sha256=_digest(original),
        reference_files_sha256={name: hashlib.sha256(content).hexdigest()
                                for name, content in raw.items()},
        reference_candidate_sources=len(expected), n_complete_reference_sources=len(selected),
        n_excluded_partial_sources=len(partial), excluded_partial_source_ids=partial,
        n_excluded_not_started_sources=len(expected)-len(complete)-len(partial),
        complete_source_ids=selected, complete_sources_by_domain=dict(sorted(domains.items())),
        expected_candidate_requests=len(plan["requests"]),
        pilot_source_ids=list(plan["pilot_source_ids"]),
        n_pilot_sources=len(plan["pilot_source_ids"]),
        expected_pilot_requests=12 * len(plan["pilot_source_ids"]),
        pilot_selection_rule="First complete sources in the existing frozen request schedule; no selection by score",
        reference_valid_requests=sum(valid.values()),
        reference_valid_requests_retained=len(reference_rows),
        reference_valid_requests_excluded=sum(valid.values())-len(reference_rows),
        selection_rule="All and only sources with all 12 reference requests valid; no selection by score",
        comparison_rule="Only sources complete for both judges; same passages, prompts, orders and repeats",
        model_control="Qwen reasoning_effort=none; Google thinkingLevel=minimal requires per-response token verification",
        json_control="The original system prompt requires JSON; native responseMimeType is not assumed supported",
        provider_parameter_scope="Canonical payload preserves scientific settings; native_payload freezes actual Google request",
        human_annotation_produced=False,
    )
    table = io.StringIO()
    fields = ["source_id", "domain", "valid_requests", "invalid_requests", "missing_requests",
              "included_in_comparison", "included_in_pilot", "status"]
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in sorted(expected):
        writer.writerow(dict(source_id=source, domain=metadata[source]["domain"],
            valid_requests=valid.get(source, 0), invalid_requests=invalid.get(source, 0),
            missing_requests=expected[source]-valid.get(source, 0)-invalid.get(source, 0),
            included_in_comparison=source in complete,
            included_in_pilot=source in plan["pilot_source_ids"],
            status="complete" if source in complete else "partial" if source in partial else "not_started"))
    files = {
        "comparison_setup.json": _json_bytes(setup),
        "reference/judge_plan.json": _json_bytes(reference_plan),
        "reference/judgments.jsonl": _jsonl_bytes(reference_rows),
        "reference/source_selection.csv": table.getvalue().encode(),
        "judge_plan.json": _json_bytes(plan),
    }
    output = Path(output)
    for name, content in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"{name} differisce dal piano congelato: usare un'altra cartella")
    for name, content in files.items():
        target = output / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return plan, reference_plan, reference_rows, setup
