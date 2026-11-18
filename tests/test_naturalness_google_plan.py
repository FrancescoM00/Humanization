"""Controlla il piano Google congelato senza usare credenziali o servizi."""
from copy import deepcopy
import csv
import io
import json
import zipfile

import pytest

from humanizers.naturalness import read_jsonl
from humanizers.naturalness_google_plan import (
    GOOGLE_MODEL, native_payload_for_request, prepare_google_comparison,
    validate_google_plan, validate_google_request,
)
from humanizers.naturalness_panel import make_panel_plan
from humanizers.naturalness_panel_runner import ITEM_METADATA


SYSTEMS = ["qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3"]
QWEN = "qwen/qwen3.8-27b"


def _reference(path, n_complete=2):
    """Aggiunge una fonte parziale a un numero fissato di fonti complete."""
    panels = [dict(source_id=f"source-{i}", source_group_id=f"source-{i}",
                   domain=["Essay", "Reuters", "WritingPrompts"][i % 3], generator="ChatGPT",
                   original_word_count=150, systems=SYSTEMS,
                   texts={name: f"An English paragraph for source {i} and method {j}."
                          for j, name in enumerate(SYSTEMS)})
              for i in range(n_complete + 1)]
    original = make_panel_plan(panels, QWEN)
    rows, attempts = [], []
    for item in original["requests"]:
        partial = item["source_id"] == f"source-{n_complete}"
        if partial and sum(row["source_id"] == item["source_id"] for row in rows) == 5:
            continue
        row = {name: deepcopy(item[name]) for name in ITEM_METADATA}
        row.update(label_source="LLM", requested_model=QWEN, response_model=QWEN, status="ok",
                   judgment={"ratings": {f"T{i}": 4 for i in range(1, 5)}})
        rows.append(row)
        attempts.append(dict(request_id=item["request_id"], attempt_number=len(attempts)+1))
    path.mkdir(parents=True)
    (path/"judge_plan.json").write_text(json.dumps(original))
    for name, value in [("judgments.jsonl", rows), ("api_attempts.jsonl", attempts)]:
        (path/name).write_text("".join(json.dumps(row)+"\n" for row in value))
    return original, rows


def _snapshot(path):
    return {str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("as_zip", [False, True])
def test_google_plan_keeps_complete_sources_and_native_materials(tmp_path, as_zip):
    source = tmp_path/"input"
    original, rows = _reference(source)
    reference = source
    if as_zip:
        reference = tmp_path/"reference.zip"
        with zipfile.ZipFile(reference, "w") as archive:
            for p in source.iterdir():
                archive.write(p, p.name)
    before = _snapshot(source)
    output = tmp_path/"google"
    plan, ref_plan, ref_rows, setup = prepare_google_comparison(reference, output)
    assert _snapshot(source) == before
    assert setup["n_complete_reference_sources"] == plan["n_sources"] == 2
    assert setup["n_excluded_partial_sources"] == 1
    assert setup["expected_candidate_requests"] == 24
    assert len(plan["requests"]) == len(ref_rows) == 24
    assert not (output/"judgments.jsonl").exists()
    assert not (output/"api_attempts.jsonl").exists()
    assert plan["provider"] == "google"
    assert plan["thinking_level"] == "minimal"
    assert plan["reasoning_effort"] is None
    assert plan["pilot_source_ids"] == [
        source for source in dict.fromkeys(item["source_id"] for item in plan["requests"])
    ]
    for old, new in zip(ref_plan["requests"], plan["requests"]):
        assert old["request_id"] != new["request_id"]
        assert old["system_order"] == new["system_order"]
        assert old["payload"]["messages"] == new["payload"]["messages"]
        assert new["native_payload"]["systemInstruction"]["parts"][0]["text"] == old["payload"]["messages"][0]["content"]
        assert new["native_payload"]["contents"][0]["parts"][0]["text"] == old["payload"]["messages"][1]["content"]
        assert new["native_payload"]["generationConfig"] == {
            "temperature": 0, "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        }
        assert "reasoning_effort" not in new["payload"]
        assert "responseMimeType" not in new["native_payload"]["generationConfig"]
    selection = list(csv.DictReader(io.StringIO((output/"reference/source_selection.csv").read_text())))
    assert next(x for x in selection if x["source_id"] == "source-2")["included_in_comparison"] == "False"
    assert len(validate_google_plan(plan)) == 24


def test_pilot_does_not_shrink_or_reorder_the_frozen_full_plan(tmp_path):
    source = tmp_path/"reference"
    _reference(source, n_complete=12)
    plan, _, _, setup = prepare_google_comparison(source, tmp_path/"google")
    source_order = list(dict.fromkeys(item["source_id"] for item in plan["requests"]))
    assert plan["pilot_source_ids"] == source_order[:10]
    assert setup["n_pilot_sources"] == 10
    assert setup["expected_pilot_requests"] == 120
    assert plan["expected_completion_requests"] == 144
    assert len(plan["requests"]) == 144
    first = _snapshot(tmp_path/"google")
    assert prepare_google_comparison(source, tmp_path/"google")[0] == plan
    assert _snapshot(tmp_path/"google") == first


@pytest.mark.parametrize("mutation", ["message", "native_message", "temperature", "thinking", "model", "pilot", "id"])
def test_changed_materials_or_profile_are_rejected(tmp_path, mutation):
    source = tmp_path/"reference"
    _reference(source)
    plan = prepare_google_comparison(source, tmp_path/"google")[0]
    item = plan["requests"][0]
    if mutation == "message":
        item["payload"]["messages"][1]["content"] += " Changed."
    elif mutation == "native_message":
        item["native_payload"]["contents"][0]["parts"][0]["text"] += " Changed."
    elif mutation == "temperature":
        item["payload"]["temperature"] = .5
    elif mutation == "thinking":
        item["native_payload"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] = "high"
    elif mutation == "model":
        plan["model"] = "another-model"
    elif mutation == "pilot":
        plan["pilot_source_ids"] = list(reversed(plan["pilot_source_ids"]))
    else:
        item["request_id"] = "changed-id"
    with pytest.raises(ValueError):
        validate_google_plan(plan)


@pytest.mark.parametrize("mutation", ["extra_parameter", "thinking_option", "role", "missing_user"])
def test_native_adapter_rejects_unreviewed_parameters(tmp_path, mutation):
    source = tmp_path/"reference"
    _reference(source)
    payload = prepare_google_comparison(source, tmp_path/"google")[0]["requests"][0]["payload"]
    if mutation == "extra_parameter":
        payload["tools"] = []
    elif mutation == "thinking_option":
        payload["reasoning_effort"] = "none"
    elif mutation == "role":
        payload["messages"][0]["role"] = "user"
    else:
        payload["messages"].pop()
    with pytest.raises(ValueError):
        native_payload_for_request(payload)


def test_changed_reference_is_rejected_before_output_overwrite(tmp_path):
    source, output = tmp_path/"reference", tmp_path/"google"
    _reference(source)
    prepare_google_comparison(source, output)
    before = _snapshot(output)
    rows = read_jsonl(source/"judgments.jsonl")
    rows[0]["judgment"]["ratings"]["T1"] = 5
    (source/"judgments.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="piano congelato"):
        prepare_google_comparison(source, output)
    assert _snapshot(output) == before


def test_a_different_model_does_not_create_a_run(tmp_path):
    source = tmp_path/"reference"
    _reference(source)
    with pytest.raises(ValueError, match="nessun fallback"):
        prepare_google_comparison(source, tmp_path/"google", model="gemma-something-else")
    assert not (tmp_path/"google").exists()
