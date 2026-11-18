"""Verifica pilota, retry, provenienza e ripresa Google con risposte locali."""
from copy import deepcopy
import json

import pytest

from humanizers.naturalness import APIError, read_jsonl
from humanizers.naturalness_google import GOOGLE_MODEL
from humanizers.naturalness_google_plan import (
    GOOGLE_ADAPTER_PROTOCOL, _google_request_id, native_payload_for_request, validate_google_plan,
)
from humanizers.naturalness_panel import make_panel_plan
import humanizers.naturalness_google_runner as runner


SYSTEMS = ["small", "large", "passage", "windows"]


def plan(n_sources=11):
    panels = [dict(source_id=f"source-{number}", source_group_id=f"source-{number}",
                   domain="Essay", generator="Claude", original_word_count=150, systems=SYSTEMS,
                   texts={system: f"A natural passage, version {number}-{index}."
                          for index, system in enumerate(SYSTEMS)}) for number in range(n_sources)]
    result = make_panel_plan(panels, "qwen/qwen3.8-27b")
    result.update(provider="google", model=GOOGLE_MODEL, adapter_protocol=GOOGLE_ADAPTER_PROTOCOL,
                  reasoning_control="google_minimal_verified", reasoning_effort=None, thinking_level="minimal")
    source_order = []
    for item in result["requests"]:
        if item["source_id"] not in source_order:
            source_order.append(item["source_id"])
        item["payload"]["model"] = GOOGLE_MODEL
        item["payload"].pop("reasoning_effort")
        item["native_payload"] = native_payload_for_request(item["payload"])
        item["request_id"] = _google_request_id(result["protocol"], item)
    result["pilot_source_ids"] = source_order[:10]
    validate_google_plan(result)
    return result


def response(*, text=None, thoughts=0, version=GOOGLE_MODEL, usage=None, thought_parts=False):
    raw = dict(
        modelVersion=version,
        candidates=[dict(content=dict(role="model", parts=[dict(
            text=json.dumps({"ratings": {f"T{i}": 4 for i in range(1, 5)}}) if text is None else text
        )]), finishReason="STOP")],
        usageMetadata=dict(promptTokenCount=100, candidatesTokenCount=20,
                           totalTokenCount=120+thoughts, thoughtsTokenCount=thoughts) if usage is None else usage,
    )
    if thought_parts:
        raw["candidates"][0]["content"]["parts"].insert(0, dict(text="A thought.", thought=True))
    return raw


class FakeClient:
    def __init__(self, replies=None, *, output=None, check_error=None):
        self.replies = list(replies or [])
        self.posts = []
        self.gets = 0
        self.output = output
        self.check_error = check_error
        self._key = "test-only-google-secret"

    def check_model(self, model):
        assert model == GOOGLE_MODEL
        self.gets += 1
        if self.check_error:
            raise self.check_error
        return dict(provider="google", requested_model=model, present_in_authenticated_catalog=True)

    def request(self, method, endpoint, payload=None):
        assert method == "POST" and endpoint == "/chat/completions"
        if self.output is not None:
            ledger = read_jsonl(self.output/"api_attempts.jsonl")
            assert ledger and ledger[-1]["attempt_number"] == len(ledger)
        self.posts.append(deepcopy(payload))
        item = self.replies.pop(0) if self.replies else response()
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)


def no_client(*args, **kwargs):
    raise AssertionError("Non devono essere lette credenziali o effettuate richieste")


def run(p, output, **kwargs):
    kwargs.setdefault("min_request_interval", 0)
    kwargs.setdefault("sleep", lambda seconds: None)
    return runner.run_google_panel_plan(p, output, **kwargs)


def test_offline_preparation_never_constructs_a_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    result = run(plan(), tmp_path)
    assert result["status"] == "planned" and result["generation_post_attempts"] == 0
    assert result["expected_completion_requests"] == 132 and result["stage_expected_requests"] == 120
    assert not (tmp_path/"api_attempts.jsonl").exists()
    pilot = json.loads((tmp_path/"pilot_check.json").read_text())
    assert pilot["status"] == "incomplete" and not pilot["passed"]


def test_pilot_stops_at_ten_panels_and_promotes_without_changing_plan(tmp_path):
    p = plan()
    api = FakeClient(output=tmp_path)
    initial = run(p, tmp_path, execute=True, client=api)
    original_plan = (tmp_path/"judge_plan.json").read_bytes()
    assert len(api.posts) == 120
    assert initial["status"] == "pilot_complete" and initial["pilot_passed"]
    assert initial["n_complete_source_panels"] == 10
    assert initial["completed_valid_requests"] == 120 and initial["expected_completion_requests"] == 132
    rows_before = (tmp_path/"judgments.jsonl").read_bytes()
    second = FakeClient(output=tmp_path)
    complete = run(p, tmp_path, execute=True, stage="full", client=second)
    assert len(second.posts) == 12 and complete["status"] == "completed"
    assert complete["generation_post_attempts"] == complete["completed_valid_requests"] == 132
    assert complete["n_complete_source_panels"] == 11
    assert (tmp_path/"judge_plan.json").read_bytes() == original_plan
    assert (tmp_path/"judgments.jsonl").read_bytes().startswith(rows_before)
    assert complete["provider_model_version_pinned"] == GOOGLE_MODEL


def test_full_stage_is_blocked_before_any_client_until_pilot_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    result = run(plan(), tmp_path, execute=True, stage="full")
    assert result["status"] == "partial" and result["blocked_reason"] == "pilot_not_passed"
    assert result["generation_post_attempts"] == 0


def test_post_budget_counts_503_retries_and_is_recorded_before_sending(tmp_path):
    api = FakeClient([APIError(503, "busy")]*3 + [response()], output=tmp_path)
    sleeps = []
    result = run(plan(), tmp_path, execute=True, client=api, max_requests=4, sleep=sleeps.append)
    assert len(api.posts) == 4 and result["generation_post_attempts"] == 4
    assert result["completed_valid_requests"] == 1
    assert result["blocked_reason"] == "request_budget_exhausted"
    assert sleeps == [20, 40, 60]
    assert result["http_error_counts"] == {"503": 3}
    errors = read_jsonl(tmp_path/"api_errors.jsonl")
    assert [item["attempt_number"] for item in errors] == [1, 2, 3]
    assert all(item["status_code"] == 503 for item in errors)


def test_retry_limit_stops_after_five_attempts_and_never_sleeps_over_sixty(tmp_path):
    api = FakeClient([APIError(503, "busy")]*5)
    sleeps = []
    result = run(plan(), tmp_path, execute=True, client=api, sleep=sleeps.append)
    assert len(api.posts) == result["generation_post_attempts"] == 5
    assert result["blocked_reason"] == "provider_http_503"
    assert sleeps == [20, 40, 60, 60]
    assert not (tmp_path/"judgments.jsonl").exists()


def test_long_retry_after_preserves_checkpoint_without_a_blocking_wait(tmp_path):
    api = FakeClient([APIError(429, "quota", "61")])
    sleeps = []
    result = run(plan(), tmp_path, execute=True, client=api, sleep=sleeps.append)
    assert sleeps == [] and len(api.posts) == 1
    assert result["blocked_reason"] == "provider_http_429"
    assert result["generation_post_attempts"] == 1


def test_regular_request_pacing_remains_active_after_retries(tmp_path, monkeypatch):
    clock = [0.0]
    starts = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    class PacedClient(FakeClient):
        def request(self, method, endpoint, payload=None):
            starts.append(clock[0])
            return super().request(method, endpoint, payload)

    def sleep(seconds):
        assert 0 <= seconds <= 60
        clock[0] += seconds

    api = PacedClient([APIError(503, "busy"), response()])
    result = run(plan(), tmp_path, execute=True, client=api, max_requests=4,
                 min_request_interval=8, sleep=sleep)
    assert starts == [0, 20, 28, 36] and result["generation_post_attempts"] == 4


def test_budget_and_valid_answers_survive_resume(tmp_path, monkeypatch):
    p = plan()
    first = run(p, tmp_path, execute=True, client=FakeClient(), max_requests=2)
    answers = read_jsonl(tmp_path/"judgments.jsonl")
    before = (tmp_path/"judgments.jsonl").read_bytes()
    second_api = FakeClient()
    second = run(p, tmp_path, execute=True, client=second_api, max_requests=5)
    assert first["generation_post_attempts"] == 2
    assert second["generation_post_attempts"] == second["completed_valid_requests"] == 5
    assert len(second_api.posts) == 3
    assert (tmp_path/"judgments.jsonl").read_bytes().startswith(before)
    assert len({row["request_id"] for row in read_jsonl(tmp_path/"judgments.jsonl")}) == 5
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    blocked = run(p, tmp_path, execute=True, max_requests=5)
    assert blocked["blocked_reason"] == "request_budget_exhausted"


def test_malformed_json_is_preserved_and_never_retried_until_favorable(tmp_path):
    p = plan()
    raw = response(text="This is not JSON.")
    api = FakeClient([raw])
    first = run(p, tmp_path, execute=True, client=api)
    assert len(api.posts) == 1 and first["invalid_response_requests"] == 1
    bad_row = read_jsonl(tmp_path/"judgments.jsonl")[0]
    assert bad_row["raw_response"] == raw and bad_row["usage_metadata"] == raw["usageMetadata"]
    assert bad_row["status"] == "invalid_response" and bad_row["validation_reason"] == "invalid_model_response"
    second_api = FakeClient()
    second = run(p, tmp_path, execute=True, client=second_api)
    assert len(second_api.posts) == 119
    assert second["status"] == "pilot_complete" and second["pilot_passed"]
    assert second["n_complete_source_panels"] == 9
    rows = read_jsonl(tmp_path/"judgments.jsonl")
    assert sum(row["request_id"] == bad_row["request_id"] for row in rows) == 1


@pytest.mark.parametrize("raw,reason", [
    (response(thoughts=1), "positive_thinking_tokens_or_parts"),
    (response(usage={}), "unverified_thinking_tokens"),
    (response(thought_parts=True), "positive_thinking_tokens_or_parts"),
])
def test_positive_or_unknown_thinking_stops_now_and_every_resume(tmp_path, monkeypatch, raw, reason):
    p = plan()
    api = FakeClient([raw])
    first = run(p, tmp_path, execute=True, client=api)
    assert len(api.posts) == 1 and first["blocked_reason"] == reason
    saved = read_jsonl(tmp_path/"judgments.jsonl")[0]
    assert saved["raw_response"] == raw and saved["status"] == "invalid_response"
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    second = run(p, tmp_path, execute=True)
    assert second["blocked_reason"] == reason and second["generation_post_attempts"] == 1
    pilot = json.loads((tmp_path/"pilot_check.json").read_text())
    assert not pilot["passed"]


def test_explicit_zero_and_usage_balance_zero_are_reported_separately(tmp_path):
    inferred = response(usage=dict(promptTokenCount=100, candidatesTokenCount=20, totalTokenCount=120))
    result = run(plan(), tmp_path, execute=True, client=FakeClient([inferred]), max_requests=2)
    usage = result["token_usage"]
    assert usage["thoughts_tokens"] == 0 and usage["responses_without_verified_zero_thoughts"] == 0
    assert usage["responses_with_explicit_zero_thought_count"] == 1
    assert usage["responses_with_zero_inferred_from_usage_balance"] == 1


@pytest.mark.parametrize("version", [None, "other-model", GOOGLE_MODEL + "-"])
def test_missing_or_unrelated_model_version_is_a_permanent_provenance_failure(tmp_path, monkeypatch, version):
    p = plan()
    first = run(p, tmp_path, execute=True, client=FakeClient([response(version=version)]))
    assert first["blocked_reason"] == "provider_model_mismatch"
    row = read_jsonl(tmp_path/"judgments.jsonl")[0]
    assert row["validation_error_kind"] == "provider_model_version" and row["raw_response"]["modelVersion"] == version
    # La verifica deriva dalla risposta grezza anche se il flag dell'errore manca.
    row.pop("validation_reason")
    (tmp_path/"judgments.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    resumed = run(p, tmp_path, execute=True)
    assert resumed["blocked_reason"] == "provider_model_mismatch"
    assert resumed["generation_post_attempts"] == 1


def test_model_version_change_is_retained_and_blocks_future_requests(tmp_path, monkeypatch):
    p = plan()
    first = run(p, tmp_path, execute=True, client=FakeClient([
        response(version=GOOGLE_MODEL + "-first"), response(version=GOOGLE_MODEL + "-second")]))
    assert first["generation_post_attempts"] == 2
    assert first["completed_valid_requests"] == first["invalid_response_requests"] == 1
    assert first["blocked_reason"] == "provider_model_version_changed"
    rows = read_jsonl(tmp_path/"judgments.jsonl")
    assert rows[1]["raw_response"]["modelVersion"] == GOOGLE_MODEL + "-second"
    rows[1].pop("validation_reason")
    (tmp_path/"judgments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    resumed = run(p, tmp_path, execute=True)
    assert resumed["blocked_reason"] == "provider_model_version_changed"


def test_pilot_rejects_five_percent_invalid_even_with_verified_zero_thoughts(tmp_path):
    p = plan()
    # Sei invalidi su 120 sono il 5%; ogni invocazione si ferma al primo invalido.
    for _ in range(6):
        run(p, tmp_path, execute=True, client=FakeClient([response(text="invalid JSON")]))
    result = run(p, tmp_path, execute=True, client=FakeClient())
    pilot = json.loads((tmp_path/"pilot_check.json").read_text())
    assert pilot["stored_responses"] == 120 and pilot["invalid_fraction"] == .05
    assert result["status"] == "pilot_failed" and not pilot["passed"]


@pytest.mark.parametrize("mutation", ["judgment", "usage", "evidence", "provider", "attempt_sequence"])
def test_checkpoint_tampering_is_rejected_before_new_calls(tmp_path, monkeypatch, mutation):
    p = plan()
    run(p, tmp_path, execute=True, client=FakeClient(), max_requests=1)
    if mutation == "attempt_sequence":
        row = read_jsonl(tmp_path/"api_attempts.jsonl")[0]
        row["attempt_number"] = 8
        (tmp_path/"api_attempts.jsonl").write_text(json.dumps(row) + "\n")
    else:
        row = read_jsonl(tmp_path/"judgments.jsonl")[0]
        if mutation == "judgment":
            row["judgment"]["ratings"]["T1"] = 1
        elif mutation == "usage":
            row["usage"]["prompt_tokens"] = 999
        elif mutation == "evidence":
            row["google_thinking_evidence"]["thoughts_evidence"] = "invented"
        else:
            row["provider"] = "another-provider"
        (tmp_path/"judgments.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    with pytest.raises(ValueError):
        run(p, tmp_path, execute=True)


def test_plan_change_is_refused_before_calls(tmp_path, monkeypatch):
    p = plan()
    run(p, tmp_path)
    saved = json.loads((tmp_path/"judge_plan.json").read_text())
    saved["seed"] = 99
    (tmp_path/"judge_plan.json").write_text(json.dumps(saved))
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    with pytest.raises(ValueError, match="diverso"):
        run(p, tmp_path, execute=True)


def test_model_catalog_failure_does_not_spend_a_post(tmp_path):
    api = FakeClient(check_error=ValueError("Unavailable model"))
    result = run(plan(), tmp_path, execute=True, client=api)
    assert result["blocked_reason"] == "requested_model_unavailable"
    assert result["generation_post_attempts"] == 0 and api.posts == []


@pytest.mark.parametrize("kwargs", [dict(stage="other"), dict(max_requests=0), dict(max_requests=True),
                                    dict(max_retries=-1), dict(min_request_interval=float("nan"))])
def test_invalid_runtime_options_fail_without_a_client(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    with pytest.raises(ValueError):
        run(plan(), tmp_path, execute=True, **kwargs)
