"""Controlla pensieri, provenienza e risposte Google senza effettuare richieste."""
from copy import deepcopy
import json

import pytest

from humanizers.naturalness_google_results import inspect_google_thinking, normalize_google_response
from humanizers.naturalness_panel import parse_panel_judgment


MODEL = "gemma-4-31b-it"
ABSENT = object()


def response(*, thoughts=0):
    usage = dict(promptTokenCount=100, candidatesTokenCount=20, totalTokenCount=120)
    if thoughts is not ABSENT:
        usage["thoughtsTokenCount"] = thoughts
        if type(thoughts) is int and thoughts >= 0:
            usage["totalTokenCount"] += thoughts
    return dict(
        modelVersion=MODEL,
        candidates=[dict(finishReason="STOP", content=dict(role="model", parts=[dict(
            text=json.dumps({"ratings": {f"T{i}": 4 for i in range(1, 5)}}))]))],
        usageMetadata=usage,
    )


def test_explicit_zero_is_distinct_from_a_zero_inferred_from_usage():
    explicit = inspect_google_thinking(response())
    inferred = inspect_google_thinking(response(thoughts=ABSENT))
    assert explicit == dict(thoughts_token_count=0, thoughts_field_present=True,
        thoughts_evidence="explicit_zero", zero_thoughts_verified=True, has_thought_parts=False)
    assert inferred == dict(thoughts_token_count=0, thoughts_field_present=False,
        thoughts_evidence="usage_balance_zero", zero_thoughts_verified=True, has_thought_parts=False)


def test_positive_thoughts_remain_visible_and_are_never_accepted():
    raw = response(thoughts=121)
    evidence = inspect_google_thinking(raw)
    assert evidence["thoughts_token_count"] == 121
    assert evidence["thoughts_evidence"] == "explicit_positive"
    assert not evidence["zero_thoughts_verified"]
    with pytest.raises(ValueError, match="pensiero"):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("usage", [ABSENT, None, {}, [], {"totalTokenCount": 120}])
def test_missing_usage_or_missing_balance_is_unknown(usage):
    raw = response(thoughts=ABSENT)
    if usage is ABSENT:
        del raw["usageMetadata"]
    else:
        raw["usageMetadata"] = usage
    evidence = inspect_google_thinking(raw)
    assert evidence["thoughts_token_count"] is None
    assert evidence["thoughts_evidence"] == "unknown"
    assert not evidence["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("value", [None, True, False, -1, 0.0, "0", float("nan"), float("inf")])
def test_thought_count_requires_an_actual_nonnegative_integer(value):
    raw = response(thoughts=value)
    evidence = inspect_google_thinking(raw)
    assert evidence["thoughts_field_present"]
    assert evidence["thoughts_evidence"] == "unknown"
    assert evidence["thoughts_token_count"] is None
    assert not evidence["zero_thoughts_verified"]


@pytest.mark.parametrize("key", ["promptTokenCount", "candidatesTokenCount", "totalTokenCount", "toolUsePromptTokenCount"])
@pytest.mark.parametrize("value", [None, True, -1, "0", 0.0])
def test_malformed_usage_counters_cannot_certify_zero(key, value):
    raw = response()
    raw["usageMetadata"][key] = value
    assert not inspect_google_thinking(raw)["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


def test_tool_tokens_are_included_in_the_balance_when_reported():
    raw = response(thoughts=ABSENT)
    raw["usageMetadata"].update(toolUsePromptTokenCount=7, totalTokenCount=127)
    evidence = inspect_google_thinking(raw)
    assert evidence["zero_thoughts_verified"]
    assert evidence["thoughts_evidence"] == "usage_balance_zero"
    assert normalize_google_response(raw, MODEL)["usage"]["tool_use_prompt_tokens"] == 7


@pytest.mark.parametrize("thoughts", [0, ABSENT])
@pytest.mark.parametrize("total", [119, 121, 131])
def test_contradictory_token_balance_does_not_verify_zero(thoughts, total):
    raw = response(thoughts=thoughts)
    raw["usageMetadata"]["totalTokenCount"] = total
    assert not inspect_google_thinking(raw)["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("text", ["", "A thought even though the counter says zero."])
def test_thought_parts_override_a_zero_counter_even_if_the_part_is_empty(text):
    raw = response()
    raw["candidates"][0]["content"]["parts"].insert(0, {"thought": True, "text": text})
    evidence = inspect_google_thinking(raw)
    assert evidence["has_thought_parts"]
    assert not evidence["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


def test_normalization_preserves_version_and_accepts_existing_strict_parser():
    raw = response()
    raw["modelVersion"] = MODEL + "-001"
    original = deepcopy(raw)
    normalized = normalize_google_response(raw, MODEL)
    assert raw == original
    assert normalized["model"] == MODEL
    assert normalized["provider_model_version"] == MODEL + "-001"
    assert normalized["provider_model_match_rule"] == "configured_prefix_suffix"
    assert normalized["usage"] == dict(prompt_tokens=100, completion_tokens=20,
        total_tokens=120, thoughts_tokens=0, tool_use_prompt_tokens=0)
    assert parse_panel_judgment(normalized, MODEL, [f"T{i}" for i in range(1, 5)]) == {
        "ratings": {f"T{i}": 4 for i in range(1, 5)}}


@pytest.mark.parametrize("version", [None, "", "gemini-4", "gemma-4-31b-itx", MODEL + "-"])
def test_missing_or_unrelated_model_versions_are_rejected(version):
    raw = response()
    if version is None:
        raw.pop("modelVersion")
    else:
        raw["modelVersion"] = version
    with pytest.raises(ValueError, match="modelVersion|versione"):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("finish", [None, "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"])
def test_only_a_natural_complete_stop_is_accepted(finish):
    raw = response()
    raw["candidates"][0]["finishReason"] = finish
    with pytest.raises(ValueError, match="troncata"):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("mutation", ["no_candidates", "two_candidates", "blocked_prompt", "blocked_rating",
    "no_content", "no_parts", "empty_text", "not_text", "tool_call", "refusal"])
def test_blocked_empty_or_nontext_candidates_are_invalid(mutation):
    raw = response()
    candidate = raw["candidates"][0]
    if mutation == "no_candidates":
        raw["candidates"] = []
    elif mutation == "two_candidates":
        raw["candidates"].append(deepcopy(candidate))
    elif mutation == "blocked_prompt":
        raw["promptFeedback"] = {"blockReason": "SAFETY"}
    elif mutation == "blocked_rating":
        candidate["safetyRatings"] = [{"blocked": True}]
    elif mutation == "no_content":
        del candidate["content"]
    elif mutation == "no_parts":
        candidate["content"]["parts"] = []
    elif mutation == "empty_text":
        candidate["content"]["parts"] = [{"text": "  "}]
    elif mutation == "not_text":
        candidate["content"]["parts"] = [{"text": None}]
    elif mutation == "tool_call":
        candidate["content"]["parts"] = [{"functionCall": {"name": "judge"}}]
    elif mutation == "refusal":
        candidate["refusal"] = "A refusal."
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


def test_multiple_text_parts_are_concatenated_without_rewriting_json():
    raw = response(thoughts=ABSENT)
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    raw["candidates"][0]["content"]["parts"] = [{"text": text[:15]}, {"text": text[15:]}]
    normalized = normalize_google_response(raw, MODEL)
    assert normalized["choices"][0]["message"]["content"] == text
    assert parse_panel_judgment(normalized, MODEL, [f"T{i}" for i in range(1, 5)])


@pytest.mark.parametrize("text", ['{"ratings":{"T1":4}}', '```json\n{"ratings":{}}\n```',
    '{"ratings":{"T1":true,"T2":4,"T3":4,"T4":4}}',
    '{"ratings":{"T1":4,"T2":4,"T3":4,"T4":4},"explanation":"A reason"}'])
def test_normalization_does_not_repair_invalid_rating_json(text):
    raw = response()
    raw["candidates"][0]["content"]["parts"] = [{"text": text}]
    normalized = normalize_google_response(raw, MODEL)
    with pytest.raises(ValueError):
        parse_panel_judgment(normalized, MODEL, [f"T{i}" for i in range(1, 5)])
