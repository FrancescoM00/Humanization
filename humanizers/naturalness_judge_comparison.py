"""Fissa i pannelli completi del primo giudice senza cambiare testi o ordini."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .naturalness import _digest
from .naturalness_panel import parse_panel_judgment
from .naturalness_panel_analysis import _read_ratings
from .naturalness_panel_runner import _validate_plan, _validate_stored


REFERENCE_MODEL = "qwen/qwen3.8-27b"
SECOND_MODEL = "llama-3.3-70b-versatile"
CORE_FILES = ("judge_plan.json", "judgments.jsonl", "api_attempts.jsonl")


def _checkpoint_bytes(path):
    """Legge soltanto i tre file necessari, anche da un Input già estratto."""
    path = Path(path)
    if path.is_dir():
        if not (path / CORE_FILES[0]).is_file():
            matches = [p.parent for p in path.rglob(CORE_FILES[0])
                       if all((p.parent / name).is_file() for name in CORE_FILES)]
            if len(matches) != 1:
                raise ValueError("Indicare una sola cartella con il checkpoint Qwen")
            path = matches[0]
        if any(not (path / name).is_file() for name in CORE_FILES):
            raise ValueError("Il checkpoint Qwen non contiene piano, risposte e tentativi")
        if sum((path / name).stat().st_size for name in CORE_FILES) > 128 * 2**20:
            raise ValueError("Il checkpoint supera la dimensione prevista")
        return {name: (path / name).read_bytes() for name in CORE_FILES}
    with zipfile.ZipFile(path) as archive:
        if any(archive.namelist().count(name) != 1 for name in CORE_FILES):
            raise ValueError("Lo ZIP deve contenere una sola copia dei tre file alla radice")
        if sum(archive.getinfo(name).file_size for name in CORE_FILES) > 128 * 2**20:
            raise ValueError("Il checkpoint supera la dimensione prevista")
        return {name: archive.read(name) for name in CORE_FILES}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def restrict_plan(plan, sources):
    """Filtra soltanto le fonti, mantenendo la sequenza delle richieste congelate."""
    result = deepcopy(plan)
    result["requests"] = [item for item in result["requests"] if item["source_id"] in sources]
    result["n_sources"] = len({item["source_id"] for item in result["requests"]})
    result["expected_completion_requests"] = len(result["requests"])
    return result


def prepare_judge_comparison(reference, output, *, model=SECOND_MODEL):
    """Esclude ogni pannello incompleto prima di pianificare il secondo giudice.

    La scelta usa esclusivamente la completezza delle dodici risposte, mai i
    voti. Il nuovo registro dei tentativi parte separatamente da quello Qwen.
    """
    if model != SECOND_MODEL:
        raise ValueError(f"Il confronto fissa esplicitamente {SECOND_MODEL}; nessun fallback")
    raw = _checkpoint_bytes(reference)
    original = json.loads(raw["judge_plan.json"])
    rows = [json.loads(line) for line in raw["judgments.jsonl"].splitlines() if line.strip()]
    attempts = [json.loads(line) for line in raw["api_attempts.jsonl"].splitlines() if line.strip()]
    if original.get("model") != REFERENCE_MODEL:
        raise ValueError("Il riferimento deve essere il precedente giudice Qwen")
    lookup = _validate_plan(original)
    _validate_stored(original, rows, attempts, lookup, parse_panel_judgment)
    ratings, metadata, expected, valid, invalid, complete, _ = _read_ratings(original, rows)
    if (original.get("repetitions") != 3 or original.get("n_orders_per_source") != 4
            or len(original.get("systems", [])) != 4 or any(n != 12 for n in expected.values())):
        raise ValueError("Il confronto richiede quattro metodi e quattro ordini ripetuti tre volte")
    if not complete:
        raise ValueError("Il checkpoint non contiene alcun pannello con dodici risposte valide")
    reference_plan = restrict_plan(original, complete)
    reference_rows = [row for row in rows if row["source_id"] in complete]
    plan = deepcopy(reference_plan)
    plan.update(model=model, reasoning_effort=None, reasoning_control="native_non_reasoning")
    for item in plan["requests"]:
        payload = item["payload"]
        payload["model"] = model
        payload.pop("reasoning_effort", None)
        item["request_id"] = _digest([
            plan["protocol"], item["source_id"], item["source_group_id"],
            item["order_id"], item["repeat_index"], payload,
        ])
    _validate_plan(plan)
    selected = sorted(complete)
    domains = Counter(metadata[source]["domain"] for source in selected)
    partial = sorted(source for source in expected if source not in complete
                     and (valid.get(source, 0) or invalid.get(source, 0)))
    setup = dict(
        protocol="groq_cross_family_naturalness_v1", reference_model=REFERENCE_MODEL,
        candidate_model=model, reference_plan_sha256=_digest(original),
        reference_files_sha256={name: hashlib.sha256(content).hexdigest()
                                for name, content in raw.items()},
        reference_candidate_sources=len(expected), n_complete_reference_sources=len(selected),
        n_excluded_partial_sources=len(partial), excluded_partial_source_ids=partial,
        n_excluded_not_started_sources=len(expected)-len(complete)-len(partial),
        complete_source_ids=selected, complete_sources_by_domain=dict(sorted(domains.items())),
        expected_candidate_requests=len(plan["requests"]),
        reference_valid_requests=sum(valid.values()),
        reference_valid_requests_retained=len(reference_rows),
        reference_valid_requests_excluded=sum(valid.values())-len(reference_rows),
        selection_rule="All and only sources with all 12 reference requests valid; no selection by score",
        comparison_rule="Only sources complete for both judges; same passages, prompts, orders and repeats",
        model_control="Qwen reasoning_effort=none; Llama is native non-reasoning and omits that API option",
        human_annotation_produced=False,
    )
    table = io.StringIO()
    fields = ["source_id", "domain", "valid_requests", "invalid_requests", "missing_requests",
              "included_in_comparison", "status"]
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in sorted(expected):
        writer.writerow(dict(source_id=source, domain=metadata[source]["domain"],
            valid_requests=valid.get(source, 0), invalid_requests=invalid.get(source, 0),
            missing_requests=expected[source]-valid.get(source, 0)-invalid.get(source, 0),
            included_in_comparison=source in complete,
            status="complete" if source in complete else "partial" if source in partial else "not_started"))
    files = {
        "comparison_setup.json": _json_bytes(setup),
        "reference/judge_plan.json": _json_bytes(reference_plan),
        "reference/judgments.jsonl": _jsonl_bytes(reference_rows),
        "reference/source_selection.csv": table.getvalue().encode(),
        "judge_plan.json": _json_bytes(plan),
    }
    output = Path(output)
    # Si verificano tutti i conflitti prima di scrivere: nessun run viene sovrascritto.
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
