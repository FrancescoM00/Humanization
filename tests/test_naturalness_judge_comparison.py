"""Controlla il confronto fra giudici su soli pannelli completi, senza API."""
from copy import deepcopy
import csv
import io
import json
import zipfile

import pytest

from humanizers.naturalness import read_jsonl
from humanizers.naturalness_judge_comparison import (
    REFERENCE_MODEL, SECOND_MODEL, prepare_judge_comparison,
)
from humanizers.naturalness_panel import make_panel_plan
from humanizers.naturalness_panel_runner import ITEM_METADATA, run_panel_plan


SYSTEMS = ["qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3"]
COMPLETE = {"essay-complete", "reuters-complete"}
PARTIAL = "writing-partial"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def reference_checkpoint(directory):
    """Crea due fonti con dodici voti e una con cinque, con ledger valido."""
    panels = []
    for source, domain in [("essay-complete", "Essay"),
                           ("reuters-complete", "Reuters"),
                           (PARTIAL, "WritingPrompts")]:
        panels.append(dict(
            source_id=source, source_group_id=source, domain=domain, generator="ChatGPT",
            original_word_count=150, systems=SYSTEMS,
            texts={system: f"An English passage for {source}, rewritten by method {i}."
                   for i, system in enumerate(SYSTEMS)},
        ))
    plan = make_panel_plan(panels, REFERENCE_MODEL, repetitions=3, seed=42)
    rows, attempts = [], []
    partial_seen = 0
    for item in plan["requests"]:
        if item["source_id"] == PARTIAL:
            partial_seen += 1
            if partial_seen > 5:
                continue
        row = {key: deepcopy(item[key]) for key in ITEM_METADATA}
        row.update(label_source="LLM", requested_model=REFERENCE_MODEL,
                   response_model=REFERENCE_MODEL, status="ok",
                   judgment={"ratings": {f"T{i}": 4 for i in range(1, 5)}},
                   usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        rows.append(row)
        attempts.append(dict(request_id=item["request_id"], attempt_number=len(attempts) + 1))
    directory.mkdir(parents=True)
    files = {
        "judge_plan.json": json_bytes(plan),
        "judgments.jsonl": jsonl_bytes(rows),
        "api_attempts.jsonl": jsonl_bytes(attempts),
        "runtime_info.json": json_bytes({"settings": {"model": REFERENCE_MODEL}}),
    }
    for name, content in files.items():
        (directory / name).write_bytes(content)
    return plan, rows, attempts


def snapshot(path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode) for p in files}


def pack(directory, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(directory).as_posix())
    return destination


@pytest.mark.parametrize("kind", ["directory", "nested_directory", "zip"])
def test_only_complete_sources_are_replanned_without_touching_the_reference(tmp_path, kind):
    root = tmp_path / "input"
    source = root / "dataset" / "checkpoint"
    old_plan, old_rows, _ = reference_checkpoint(source)
    if kind == "zip":
        reference = pack(source, root / "checkpoint.zip")
    elif kind == "nested_directory":
        reference = root / "dataset"
    else:
        reference = source
    before = snapshot(root)
    output = tmp_path / "new-judge"

    plan, reference_plan, reference_rows, setup = prepare_judge_comparison(reference, output)

    assert snapshot(root) == before
    assert setup["n_complete_reference_sources"] == 2
    assert setup["n_excluded_partial_sources"] == 1
    assert setup["excluded_partial_source_ids"] == [PARTIAL]
    assert setup["reference_valid_requests"] == 29
    assert setup["reference_valid_requests_retained"] == 24
    assert setup["reference_valid_requests_excluded"] == 5
    assert setup["complete_sources_by_domain"] == {"Essay": 1, "Reuters": 1}
    assert plan["n_sources"] == reference_plan["n_sources"] == 2
    assert plan["expected_completion_requests"] == 24
    assert set(item["source_id"] for item in plan["requests"]) == COMPLETE
    assert len(reference_rows) == 24
    assert reference_rows == [row for row in old_rows if row["source_id"] in COMPLETE]
    old_requests = [item for item in old_plan["requests"] if item["source_id"] in COMPLETE]
    assert reference_plan["requests"] == old_requests
    assert plan["model"] == SECOND_MODEL
    assert plan["reasoning_control"] == "native_non_reasoning"
    assert plan["reasoning_effort"] is None
    for old, new in zip(old_requests, plan["requests"]):
        assert new["request_id"] != old["request_id"]
        assert {key: old[key] for key in ITEM_METADATA if key != "request_id"} == {
            key: new[key] for key in ITEM_METADATA if key != "request_id"}
        old_payload, new_payload = deepcopy(old["payload"]), deepcopy(new["payload"])
        assert old_payload.pop("model") == REFERENCE_MODEL
        assert new_payload.pop("model") == SECOND_MODEL
        assert old_payload.pop("reasoning_effort") == "none"
        assert "reasoning_effort" not in new_payload
        assert new_payload == old_payload
    for source_id in COMPLETE:
        slots = [(item["order_id"], item["repeat_index"])
                 for item in plan["requests"] if item["source_id"] == source_id]
        assert len(slots) == len(set(slots)) == 12
        assert {order for order, _ in slots} == {f"order_{i}" for i in range(4)}
        assert {repeat for _, repeat in slots} == {0, 1, 2}
    assert not (output / "judgments.jsonl").exists()
    assert not (output / "api_attempts.jsonl").exists()
    assert json.loads((output / "reference" / "judge_plan.json").read_text()) == reference_plan
    assert read_jsonl(output / "reference" / "judgments.jsonl") == reference_rows
    selection = list(csv.DictReader(io.StringIO((output / "reference" / "source_selection.csv").read_text())))
    partial = next(row for row in selection if row["source_id"] == PARTIAL)
    assert partial["included_in_comparison"] == "False"
    assert partial["valid_requests"] == "5" and partial["missing_requests"] == "7"


def test_repeated_preparation_is_identical_and_does_not_rewrite_existing_files(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    reference_checkpoint(source)
    first = prepare_judge_comparison(source, output)
    before = snapshot(output)
    second = prepare_judge_comparison(source, output)
    assert second == first
    assert snapshot(output) == before


def test_twelve_stored_responses_with_one_invalid_response_are_not_a_complete_panel(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    plan, rows, attempts = reference_checkpoint(source)
    present = {row["request_id"] for row in rows}
    missing = [item for item in plan["requests"]
               if item["source_id"] == PARTIAL and item["request_id"] not in present]
    for index, item in enumerate(missing):
        row = {key: deepcopy(item[key]) for key in ITEM_METADATA}
        row.update(label_source="LLM", requested_model=REFERENCE_MODEL,
                   response_model=REFERENCE_MODEL, status="ok",
                   judgment={"ratings": {f"T{i}": 4 for i in range(1, 5)}})
        if index == len(missing) - 1:
            row["status"] = "invalid_response"
            row.pop("judgment")
            row["error"] = "Judge JSON must contain exactly the ratings field"
        rows.append(row)
        attempts.append(dict(request_id=item["request_id"], attempt_number=len(attempts) + 1))
    (source / "judgments.jsonl").write_bytes(jsonl_bytes(rows))
    (source / "api_attempts.jsonl").write_bytes(jsonl_bytes(attempts))
    candidate, _, _, setup = prepare_judge_comparison(source, output)
    assert setup["n_complete_reference_sources"] == 2
    assert setup["excluded_partial_source_ids"] == [PARTIAL]
    assert setup["reference_valid_requests_excluded"] == 11
    assert {item["source_id"] for item in candidate["requests"]} == COMPLETE


def test_no_complete_reference_panel_stops_before_creating_a_candidate_run(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    _, rows, _ = reference_checkpoint(source)
    rows = [row for row in rows if row["source_id"] == PARTIAL]
    attempts = [dict(request_id=row["request_id"], attempt_number=index)
                for index, row in enumerate(rows, 1)]
    (source / "judgments.jsonl").write_bytes(jsonl_bytes(rows))
    (source / "api_attempts.jsonl").write_bytes(jsonl_bytes(attempts))
    with pytest.raises(ValueError, match="alcun pannello"):
        prepare_judge_comparison(source, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["ratings", "source_metadata", "passage", "partial_ratings"])
def test_preparation_rejects_a_changed_reference_before_overwriting_any_output(tmp_path, mutation):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    reference_checkpoint(source)
    prepare_judge_comparison(source, output)
    before = snapshot(output)
    plan = json.loads((source / "judge_plan.json").read_text())
    rows = read_jsonl(source / "judgments.jsonl")
    if mutation in {"ratings", "partial_ratings"}:
        row = next(row for row in rows if (row["source_id"] == PARTIAL) == (mutation == "partial_ratings"))
        row["judgment"]["ratings"]["T1"] = 5
        (source / "judgments.jsonl").write_bytes(jsonl_bytes(rows))
    elif mutation == "source_metadata":
        for item in plan["requests"]:
            if item["source_id"] == "essay-complete":
                item["generator"] = "A different generator"
        for row in rows:
            if row["source_id"] == "essay-complete":
                row["generator"] = "A different generator"
        (source / "judge_plan.json").write_bytes(json_bytes(plan))
        (source / "judgments.jsonl").write_bytes(jsonl_bytes(rows))
    else:
        plan["requests"][0]["payload"]["messages"][1]["content"] += " "
        (source / "judge_plan.json").write_bytes(json_bytes(plan))
    with pytest.raises(ValueError):
        prepare_judge_comparison(source, output)
    assert snapshot(output) == before


def test_second_model_cannot_be_silently_changed_in_an_existing_comparison(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    reference_checkpoint(source)
    prepare_judge_comparison(source, output)
    before = snapshot(output)
    with pytest.raises(ValueError, match="nessun fallback"):
        prepare_judge_comparison(source, output, model="a-different-judge")
    assert snapshot(output) == before


def test_reference_requires_the_exact_qwen_model(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    reference_checkpoint(source)
    plan_path = source / "judge_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["model"] = "qwen-another-model"
    plan_path.write_bytes(json_bytes(plan))
    with pytest.raises(ValueError, match="precedente giudice Qwen"):
        prepare_judge_comparison(source, output)
    assert not output.exists()


class FakeNativeClient:
    def __init__(self):
        self.posts = []

    def check_model(self, model):
        assert model == SECOND_MODEL
        return dict(requested_model=model, present_in_authenticated_catalog=True,
                    reasoning_control="native_non_reasoning")

    def request(self, method, endpoint, payload):
        assert method == "POST" and endpoint == "/chat/completions"
        assert payload["model"] == SECOND_MODEL and "reasoning_effort" not in payload
        self.posts.append(deepcopy(payload))
        return dict(model=SECOND_MODEL, choices=[dict(finish_reason="stop", message=dict(
            content=json.dumps({"ratings": {f"T{i}": 3 for i in range(1, 5)}})))],
            usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


def test_candidate_budget_and_resumed_results_are_separate_from_the_reference(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "candidate"
    _, _, reference_attempts = reference_checkpoint(source)
    original = snapshot(source)
    plan, _, _, _ = prepare_judge_comparison(source, output)
    reference_copies = snapshot(output / "reference")
    first = FakeNativeClient()
    result = run_panel_plan(plan, output, execute=True, client=first, max_requests=3)
    assert len(reference_attempts) == 29
    assert result["generation_post_attempts"] == result["completed_valid_requests"] == 3
    assert len(first.posts) == 3
    first_ids = {row["request_id"] for row in read_jsonl(output / "judgments.jsonl")}
    prepare_judge_comparison(source, output)
    second = FakeNativeClient()
    result = run_panel_plan(plan, output, execute=True, client=second, max_requests=24)
    assert result["status"] == "completed"
    assert result["generation_post_attempts"] == result["completed_valid_requests"] == 24
    assert result["n_complete_source_panels"] == 2
    assert len(second.posts) == 21
    rows = read_jsonl(output / "judgments.jsonl")
    assert first_ids <= {row["request_id"] for row in rows}
    assert len({row["request_id"] for row in rows}) == 24
    assert all(row["requested_model"] == SECOND_MODEL for row in rows)
    assert snapshot(source) == original
    assert snapshot(output / "reference") == reference_copies
