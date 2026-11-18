"""Verifica profili dei giudici, payload e ripresa senza chiamare servizi."""
from copy import deepcopy
import json

import pytest

from humanizers.naturalness import APIError, GroqClient, _digest, read_jsonl
from humanizers.naturalness_panel import make_panel_plan
from humanizers.naturalness_judges import (
    NativeNonReasoningGroqClient, client_for_plan, validate_judge_payload,
)
import humanizers.naturalness_judges as judges
import humanizers.naturalness_panel_runner as runner


QWEN = "qwen/qwen3.8-27b"
LLAMA = "llama-3.3-70b-versatile"
SYSTEMS = ["small", "large", "passage", "windows"]


def legacy_plan():
    panel = dict(
        source_id="one", source_group_id="one", domain="Essay", generator="Claude",
        original_word_count=150, systems=SYSTEMS,
        texts={name: f"A natural passage with wording {i}." for i, name in enumerate(SYSTEMS)},
    )
    return make_panel_plan([panel], QWEN)


def native_plan():
    plan = legacy_plan()
    plan.update(model=LLAMA, reasoning_control="native_non_reasoning", reasoning_effort=None)
    for item in plan["requests"]:
        item["payload"]["model"] = LLAMA
        del item["payload"]["reasoning_effort"]
        item["request_id"] = _digest([item["request_id"], item["payload"]])
    return plan


def response(*, reasoning=False):
    message = dict(content=json.dumps({"ratings": {f"T{i}": 4 for i in range(1, 5)}}))
    if reasoning:
        message["reasoning_content"] = "A reasoning field is not an accepted rating."
    return dict(model=LLAMA, choices=[dict(finish_reason="stop", message=message)],
                usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


class FakeNativeClient(NativeNonReasoningGroqClient):
    def __init__(self, *, replies=None, models=None):
        super().__init__(api_key="test-only-not-a-real-key")
        self.replies = list(replies or [])
        self.models = [LLAMA] if models is None else models
        self.posts = []
        self.gets = 0

    def request(self, method, endpoint, payload=None):
        if method == "GET":
            assert endpoint == "/models"
            self.gets += 1
            return {"data": [{"id": model, "active": True} for model in self.models]}
        assert method == "POST" and endpoint == "/chat/completions"
        self.posts.append(deepcopy(payload))
        item = self.replies.pop(0) if self.replies else response()
        if isinstance(item, Exception):
            raise item
        return item


def no_client(*args, **kwargs):
    raise AssertionError("Non devono essere lette credenziali o costruiti client")


def test_legacy_validation_leaves_the_frozen_plan_and_payloads_identical():
    plan = legacy_plan()
    before = deepcopy(plan)
    runner._validate_plan(plan)
    for item in plan["requests"]:
        validate_judge_payload(plan, item["payload"])
    assert plan == before
    assert "reasoning_control" not in plan
    assert all(item["payload"]["reasoning_effort"] == "none" for item in plan["requests"])


def test_native_validation_accepts_only_explicit_omission_without_mutation():
    plan = native_plan()
    before = deepcopy(plan)
    runner._validate_plan(plan)
    assert plan == before
    assert all("reasoning_effort" not in item["payload"] for item in plan["requests"])


@pytest.mark.parametrize("mutation", ["model", "effort", "missing_effort", "control", "payload_model", "payload_effort"])
def test_native_profile_rejects_unapproved_or_inconsistent_options(mutation):
    plan = native_plan()
    if mutation == "model":
        plan["model"] = QWEN
        for item in plan["requests"]:
            item["payload"]["model"] = QWEN
    elif mutation == "effort":
        plan["reasoning_effort"] = "none"
    elif mutation == "missing_effort":
        del plan["reasoning_effort"]
    elif mutation == "control":
        plan["reasoning_control"] = "automatic_fallback"
    elif mutation == "payload_model":
        plan["requests"][0]["payload"]["model"] = QWEN
    elif mutation == "payload_effort":
        plan["requests"][0]["payload"]["reasoning_effort"] = None
    with pytest.raises(ValueError):
        runner._validate_plan(plan)


def test_native_model_name_alone_does_not_enable_the_native_profile():
    plan = native_plan()
    del plan["reasoning_control"]
    with pytest.raises(ValueError):
        runner._validate_plan(plan)


def test_client_factory_uses_only_the_explicit_profile(monkeypatch):
    native, legacy = object(), object()
    monkeypatch.setattr(judges, "NativeNonReasoningGroqClient", lambda: native)
    monkeypatch.setattr(judges, "GroqClient", lambda: legacy)
    assert client_for_plan(native_plan()) is native
    assert client_for_plan(legacy_plan()) is legacy
    bad = native_plan()
    bad["model"] = "some-other-model"
    with pytest.raises(ValueError):
        client_for_plan(bad)


def test_native_catalog_reports_the_intrinsic_control_without_fallback():
    client = FakeNativeClient()
    result = client.check_model(LLAMA)
    assert client.gets == 1 and client.posts == []
    assert result["present_in_authenticated_catalog"] is True
    assert result["reasoning_control"] == "native_non_reasoning"
    assert "non viene inviato" in result["no_thinking_option"]
    with pytest.raises(ValueError):
        client.check_model(QWEN)
    assert client.gets == 1


def test_offline_native_plan_does_not_construct_a_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    monkeypatch.setattr(runner, "client_for_plan", no_client)
    result = runner.run_panel_plan(native_plan(), tmp_path)
    assert result["status"] == "planned"
    assert result["reasoning_effort"] is None
    assert result["reasoning_control"] == "native_non_reasoning"
    assert result["generation_post_attempts"] == 0


def test_default_native_client_preserves_the_payload_and_resume_budget(tmp_path, monkeypatch):
    first = FakeNativeClient()
    second = FakeNativeClient()
    available = iter([first, second])
    monkeypatch.setattr(runner, "client_for_plan", lambda plan: next(available))
    monkeypatch.setattr(runner, "GroqClient", no_client)
    plan = native_plan()
    initial = runner.run_panel_plan(plan, tmp_path, execute=True, max_requests=2)
    assert initial["completed_valid_requests"] == initial["generation_post_attempts"] == 2
    assert initial["reasoning_effort"] is None
    assert initial["model_check"]["reasoning_control"] == "native_non_reasoning"
    assert first.posts == [item["payload"] for item in plan["requests"][:2]]
    final = runner.run_panel_plan(plan, tmp_path, execute=True, max_requests=12)
    assert final["status"] == "completed"
    assert final["completed_valid_requests"] == final["generation_post_attempts"] == 12
    assert second.posts == [item["payload"] for item in plan["requests"][2:]]
    assert len(read_jsonl(tmp_path / "judgments.jsonl")) == 12
    assert json.loads((tmp_path / "judge_plan.json").read_text()) == plan


def test_missing_native_model_stops_before_generation(tmp_path):
    client = FakeNativeClient(models=[])
    result = runner.run_panel_plan(native_plan(), tmp_path, execute=True, client=client)
    assert result["blocked_reason"] == "requested_model_unavailable"
    assert result["generation_post_attempts"] == 0
    assert client.gets == 1 and client.posts == []


def test_native_http400_does_not_retry_with_different_options(tmp_path):
    client = FakeNativeClient(replies=[APIError(400, "unsupported request option")])
    plan = native_plan()
    result = runner.run_panel_plan(plan, tmp_path, execute=True, client=client)
    assert result["blocked_reason"] == "provider_http_400"
    assert result["generation_post_attempts"] == 1
    assert client.posts == [plan["requests"][0]["payload"]]
    assert "reasoning_effort" not in client.posts[0]


def test_native_reasoning_response_is_invalid_and_not_resampled(tmp_path):
    client = FakeNativeClient(replies=[response(reasoning=True)])
    result = runner.run_panel_plan(native_plan(), tmp_path, execute=True, client=client)
    assert result["blocked_reason"] == "invalid_model_response"
    assert result["invalid_response_requests"] == 1
    assert result["n_primary_sources"] == 0
    assert len(client.posts) == 1


def test_other_judge_checkpoint_is_rejected_without_new_calls(tmp_path):
    runner.run_panel_plan(legacy_plan(), tmp_path)
    client = FakeNativeClient()
    with pytest.raises(ValueError):
        runner.run_panel_plan(native_plan(), tmp_path, execute=True, client=client)
    assert client.gets == 0 and client.posts == []
