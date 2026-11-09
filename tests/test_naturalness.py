"""Checks of order handling, budgets, model identity and reference provenance."""
import csv
import json

import pytest

from humanizers.naturalness import (
    APIError, GroqClient, canonical_preference, load_pairs, make_plan,
    parse_judgment, run_plan, summarize_orders, validate_ratings,
)


MODEL = "qwen/qwen3.8-27b"


def pair(source="source-1"):
    return dict(source_id=source, source_group_id=source, domain="Essay", generator="Claude",
                system_a="qwen_humanize", system_b="dipper", text_a="First passage.", text_b="Second passage.")


def response(preference="A", model=MODEL):
    return dict(model=model, choices=[dict(finish_reason="stop", message=dict(content=json.dumps(
        dict(preference=preference, naturalness_A=4, naturalness_B=3))))],
        usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


class FakeClient(GroqClient):
    def __init__(self, responses=None, models=None):
        super().__init__(api_key="test-only-not-a-real-key")
        self.models = [MODEL] if models is None else models
        self.responses = list(responses or [response("A"), response("B")])
        self.posts = []
        self.gets = 0

    def request(self, method, endpoint, payload=None):
        if method == "GET":
            self.gets += 1
            assert endpoint == "/models"
            return {"data": [{"id": name, "active": True} for name in self.models]}
        assert endpoint == "/chat/completions"
        self.posts.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_both_orders_have_same_texts_and_disable_thinking_explicitly():
    plan = make_plan([pair()], MODEL)
    first, second = plan["requests"]
    assert first["order"] == "AB" and second["order"] == "BA"
    before = json.loads(first["payload"]["messages"][1]["content"])
    after = json.loads(second["payload"]["messages"][1]["content"])
    assert before["passage_A"] == after["passage_B"]
    assert before["passage_B"] == after["passage_A"]
    assert first["payload"]["reasoning_effort"] == "none"
    assert first["payload"]["max_completion_tokens"] <= 200
    assert "reasoning_format" not in first["payload"]
    assert canonical_preference("A", "AB") == canonical_preference("B", "BA") == "system_a"
    with pytest.raises(ValueError, match="reasoning_effort"):
        make_plan([pair()], MODEL, reasoning_effort="hidden")


def test_load_pairs_checks_same_original_and_does_not_filter_failed_outputs(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    a = dict(id="a", source_id="source-1", source_group_id="source-1", humanizer="a",
             domain="Essay", generator="Claude", status="ok", original="Original.", modified="Rewritten.")
    b = dict(a, id="b", humanizer="b", original="A different source.")
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="same original"):
        load_pairs([path], "a", "b")
    b.update(original="Original.", status="generation_error", modified="")
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="failed generation"):
        load_pairs([path], "a", "b")


def test_dry_plan_makes_no_api_call_and_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = FakeClient()
    manifest = run_plan(make_plan([pair()], MODEL), tmp_path, client=client)
    assert manifest["status"] == "planned"
    assert client.gets == 0 and client.posts == []
    assert not (tmp_path / "judgments.jsonl").exists()


def test_unknown_model_is_rejected_before_any_generation(tmp_path):
    client = FakeClient(models=["some-other-model"])
    with pytest.raises(ValueError, match="absent.*no fallback"):
        run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client)
    assert client.gets == 1 and not client.posts


def test_resume_and_global_budget_keep_completed_requests_and_order_disagreement(tmp_path):
    plan = make_plan([pair()], MODEL)
    client = FakeClient(responses=[response("A"), response("A")])
    first = run_plan(plan, tmp_path, execute=True, client=client, max_requests=1)
    assert first["status"] == "partial" and first["generation_post_attempts"] == 1
    second = run_plan(plan, tmp_path, execute=True, client=client, max_requests=2)
    assert second["generation_post_attempts"] == 2 and len(client.posts) == 2
    summary = json.loads((tmp_path / "pairwise_summary.json").read_text())
    assert not summary[0]["order_consistent"] and summary[0]["preference"] is None
    run_plan(plan, tmp_path, execute=True, client=client, max_requests=2)
    assert len(client.posts) == 2
    with pytest.raises(ValueError, match="plan differs"):
        run_plan(make_plan([pair("changed")], MODEL), tmp_path)


def test_rate_limit_retries_honor_delay_and_count_toward_budget(tmp_path):
    client = FakeClient(responses=[APIError(429, "rate limited", "2"), response("A")])
    sleeps = []
    result = run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client,
                      max_requests=2, sleep=sleeps.append)
    assert sleeps == [2.0]
    assert result["generation_post_attempts"] == 2 and result["completed_valid_orders"] == 1
    assert result["blocked_reason"] == "request_budget_exhausted"


def test_long_retry_after_defers_without_sleep_or_unsupported_reasoning_fallback(tmp_path):
    client = FakeClient(responses=[APIError(429, "rate limited", "120")])
    result = run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client,
                      max_requests=6, sleep=lambda _: pytest.fail("Do not block for two minutes"))
    assert result["status"] == "partial" and len(client.posts) == 1
    assert result["blocked_reason"] == "provider_http_429"
    other = FakeClient(responses=[APIError(400, "unsupported no-thinking option")])
    result = run_plan(make_plan([pair()], MODEL), tmp_path / "unsupported", execute=True, client=other)
    assert len(other.posts) == 1
    assert other.posts[0]["reasoning_effort"] == "none"
    assert result["blocked_reason"] == "provider_http_400"


def test_invalid_model_reasoning_or_truncation_is_not_an_accepted_judgment():
    with pytest.raises(ValueError, match="different model"):
        parse_judgment(response(model="fallback"), MODEL)
    item = response()
    item["choices"][0]["message"]["reasoning"] = "Hidden thoughts"
    with pytest.raises(ValueError, match="reasoning"):
        parse_judgment(item, MODEL)
    item = response()
    item["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="truncated"):
        parse_judgment(item, MODEL)


def test_rating_validation_distinguishes_assistant_proxy_and_empty_human_forms(tmp_path):
    ratings = tmp_path / "ratings.csv"
    key = tmp_path / "key.csv"
    write_csv(ratings, [dict(review_id="N1", naturalness_1_to_5=""),
                        dict(review_id="N2", naturalness_1_to_5="")])
    write_csv(key, [dict(review_id="N1", source_id="source-1", role="qwen_humanize"),
                    dict(review_id="N2", source_id="source-1", role="dipper")])
    plan = make_plan([pair()], MODEL)
    rows = [dict(source_id="source-1", order="AB", status="ok", judgment={"preference": "A"}),
            dict(source_id="source-1", order="BA", status="ok", judgment={"preference": "B"})]
    empty = validate_ratings(plan, rows, ratings, key, label_source="human")
    assert empty["n_compared"] == 0 and empty["agreement"] is None
    write_csv(ratings, [dict(review_id="N1", naturalness_1_to_5="5"),
                        dict(review_id="N2", naturalness_1_to_5="3")])
    proxy = validate_ratings(plan, rows, ratings, key, label_source="assistant_proxy")
    assert proxy["n_compared"] == 1 and proxy["agreement"] == 1
    assert not proxy["reference_is_human"] and proxy["cohen_kappa"] is None
