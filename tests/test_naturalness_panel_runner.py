"""Il runner conserva budget, risposte e provenienza attraverso le riprese."""
from copy import deepcopy
import json

import pytest

from humanizers.naturalness import APIError, GroqClient, read_jsonl
from humanizers.naturalness_panel import make_panel_plan
import humanizers.naturalness_panel_runner as runner


MODEL = "qwen/qwen3.8-27b"
SYSTEMS = ["small", "large", "passage", "windows"]


def plan():
    p = dict(source_id="one", source_group_id="one", domain="Essay", generator="Claude",
             systems=SYSTEMS, texts={s: f"A natural passage with wording {i}." for i, s in enumerate(SYSTEMS)},
             original_word_count=150)
    return make_panel_plan([p], MODEL)


def response(scores=None):
    return dict(model=MODEL, choices=[dict(finish_reason="stop", message=dict(
        content=json.dumps({"ratings": scores or {f"T{i}": 4 for i in range(1, 5)}})))],
        usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


class FakeClient(GroqClient):
    def __init__(self, *, replies=None, models=None, attempts_path=None):
        super().__init__(api_key="test-only-not-a-real-key")
        self.replies = list(replies or [])
        self.models = [MODEL] if models is None else models
        self.attempts_path = attempts_path
        self.posts = []
        self.gets = 0

    def request(self, method, endpoint, payload=None):
        if method == "GET":
            assert endpoint == "/models"
            self.gets += 1
            return {"data": [{"id": m, "active": True} for m in self.models]}
        assert method == "POST" and endpoint == "/chat/completions"
        if self.attempts_path is not None:
            attempts = read_jsonl(self.attempts_path)
            assert attempts and attempts[-1]["attempt_number"] == len(attempts)
        self.posts.append(payload)
        item = self.replies.pop(0) if self.replies else response()
        if isinstance(item, Exception):
            raise item
        return item


def no_client(*args, **kwargs):
    raise AssertionError("Il planning non deve leggere credenziali o creare un client")


def test_planning_and_empty_execution_do_not_construct_a_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    result = runner.run_panel_plan(plan(), tmp_path / "dry")
    assert result["status"] == "planned"
    assert result["generation_post_attempts"] == 0
    assert not (tmp_path / "dry" / "api_attempts.jsonl").exists()
    empty = make_panel_plan([], MODEL)
    result = runner.run_panel_plan(empty, tmp_path / "empty", execute=True)
    assert result["status"] == "not_judged_no_eligible_panels"
    assert result["generation_post_attempts"] == 0


def test_attempt_budget_survives_resume_and_answers_are_not_repeated(tmp_path):
    p = plan()
    first = FakeClient(attempts_path=tmp_path / "api_attempts.jsonl")
    initial = runner.run_panel_plan(p, tmp_path, execute=True, client=first, max_requests=2)
    assert len(first.posts) == 2 and first.gets == 1
    assert initial["status"] == "partial"
    assert initial["generation_post_attempts"] == 2
    assert initial["n_primary_sources"] == 0
    answered = {r["request_id"] for r in read_jsonl(tmp_path / "judgments.jsonl")}
    second = FakeClient(attempts_path=tmp_path / "api_attempts.jsonl")
    final = runner.run_panel_plan(p, tmp_path, execute=True, client=second, max_requests=12)
    assert len(second.posts) == 10
    assert final["status"] == "completed"
    assert final["generation_post_attempts"] == final["completed_valid_requests"] == 12
    assert final["n_primary_sources"] == 1
    rows = read_jsonl(tmp_path / "judgments.jsonl")
    assert len({r["request_id"] for r in rows}) == 12
    assert answered <= {r["request_id"] for r in rows}
    assert final["token_usage"] == dict(prompt_tokens=1200, completion_tokens=240, total_tokens=1440)


def test_finished_and_already_spent_runs_do_not_contact_the_provider(tmp_path, monkeypatch):
    p = plan()
    client = FakeClient()
    runner.run_panel_plan(p, tmp_path, execute=True, client=client, max_requests=1)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    limited = runner.run_panel_plan(p, tmp_path, execute=True, max_requests=1)
    assert limited["blocked_reason"] == "request_budget_exhausted"
    assert limited["generation_post_attempts"] == 1
    runner.run_panel_plan(p, tmp_path, execute=True, client=FakeClient(), max_requests=12)
    finished = runner.run_panel_plan(p, tmp_path, execute=True, max_requests=12)
    assert finished["status"] == "completed"


def test_retries_count_as_attempts_and_are_persisted_before_post(tmp_path):
    client = FakeClient(replies=[APIError(429, "rate limit", "0"), response()],
                        attempts_path=tmp_path / "api_attempts.jsonl")
    sleeps = []
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=client,
                                   max_requests=2, sleep=sleeps.append)
    assert sleeps == [0.0]
    assert len(client.posts) == 2
    assert result["generation_post_attempts"] == 2
    assert result["completed_valid_requests"] == 1
    attempts = read_jsonl(tmp_path / "api_attempts.jsonl")
    assert attempts[0]["request_id"] == attempts[1]["request_id"]


def test_long_rate_limit_stops_instead_of_blocking_the_notebook(tmp_path):
    client = FakeClient(replies=[APIError(429, "daily quota", "3600")])
    sleeps = []
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=client,
                                   max_requests=20, sleep=sleeps.append)
    assert sleeps == [] and len(client.posts) == 1
    assert result["status"] == "partial"
    assert result["blocked_reason"] == "provider_http_429"


def test_request_pacing_applies_between_post_attempts_including_retries(tmp_path, monkeypatch):
    clock = [0.0]
    starts = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    class PacedClient(FakeClient):
        def request(self, method, endpoint, payload=None):
            if method == "POST":
                starts.append(clock[0])
            return super().request(method, endpoint, payload)

    def sleep(seconds):
        assert 0 <= seconds <= 60
        clock[0] += seconds

    client = PacedClient(replies=[APIError(429, "brief quota", "0"), response()])
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=client,
                                   max_requests=3, min_request_interval=4, sleep=sleep)
    assert starts == [0.0, 4.0, 8.0]
    assert result["generation_post_attempts"] == 3
    assert result["completed_valid_requests"] == 2


def test_catalog_failure_prevents_generation(tmp_path):
    client = FakeClient(models=[])
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=client)
    assert client.gets == 1 and client.posts == []
    assert result["generation_post_attempts"] == 0
    assert result["status"] == "partial"


def test_invalid_ratings_are_retained_without_resampling(tmp_path):
    invalid = response()
    invalid["choices"][0]["message"]["content"] = '{"ratings":{"T1":6}}'
    first = FakeClient(replies=[invalid])
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=first, max_requests=20)
    assert len(first.posts) == 1
    assert result["invalid_response_requests"] == 1
    assert result["blocked_reason"] == "invalid_model_response"
    invalid_id = read_jsonl(tmp_path / "judgments.jsonl")[0]["request_id"]
    second = FakeClient()
    resumed = runner.run_panel_plan(plan(), tmp_path, execute=True, client=second, max_requests=20)
    assert len(second.posts) == 11
    assert resumed["completed_valid_requests"] == 11
    assert resumed["invalid_response_requests"] == 1
    assert resumed["n_primary_sources"] == 0
    assert sum(r["request_id"] == invalid_id for r in read_jsonl(tmp_path / "judgments.jsonl")) == 1


def test_changed_plan_is_rejected_before_any_new_request(tmp_path):
    p = plan()
    runner.run_panel_plan(p, tmp_path)
    changed = deepcopy(p)
    changed["requests"][0]["payload"]["temperature"] = .2
    client = FakeClient()
    with pytest.raises(ValueError):
        runner.run_panel_plan(changed, tmp_path, execute=True, client=client)
    assert not client.posts and not client.gets


@pytest.mark.parametrize("alter", ["metadata", "foreign_attempt", "attempt_number", "rating"])
def test_tampered_saved_results_and_ledgers_are_rejected(tmp_path, alter):
    p = plan()
    runner.run_panel_plan(p, tmp_path, execute=True, client=FakeClient(), max_requests=1)
    if alter in {"metadata", "rating"}:
        file = tmp_path / "judgments.jsonl"
        rows = read_jsonl(file)
        if alter == "metadata":
            rows[0]["system_order"] = rows[0]["system_order"][::-1]
        else:
            rows[0]["judgment"]["ratings"]["T1"] = True
    else:
        file = tmp_path / "api_attempts.jsonl"
        rows = read_jsonl(file)
        rows[0]["request_id" if alter == "foreign_attempt" else "attempt_number"] = "foreign" if alter == "foreign_attempt" else 7
    file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    client = FakeClient()
    with pytest.raises(ValueError):
        runner.run_panel_plan(p, tmp_path, execute=True, client=client, max_requests=12)
    assert client.gets == 0 and client.posts == []


def test_provider_errors_are_redacted_in_the_saved_manifest(tmp_path):
    client = FakeClient(replies=[APIError(403, "Bearer secret-token gsk_abc123 test-only-not-a-real-key")])
    result = runner.run_panel_plan(plan(), tmp_path, execute=True, client=client)
    value = json.dumps(result)
    assert "test-only-not-a-real-key" not in value
    assert "gsk_abc123" not in value and "secret-token" not in value
