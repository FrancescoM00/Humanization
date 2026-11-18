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
        thoughts_evidence="explicit_zero", zero_thoughts_verified=True, has_thought_parts=False,
        has_nonempty_thought_parts=False, empty_thought_part_count=0, nonempty_thought_part_count=0)
    assert inferred == dict(thoughts_token_count=0, thoughts_field_present=False,
        thoughts_evidence="usage_balance_zero", zero_thoughts_verified=True, has_thought_parts=False,
        has_nonempty_thought_parts=False, empty_thought_part_count=0, nonempty_thought_part_count=0)


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


def test_nonempty_thought_parts_override_a_zero_counter():
    raw = response()
    raw["candidates"][0]["content"]["parts"].insert(
        0, {"thought": True, "text": "A thought even though the counter says zero."})
    evidence = inspect_google_thinking(raw)
    assert evidence["has_thought_parts"]
    assert evidence["has_nonempty_thought_parts"]
    assert evidence["nonempty_thought_part_count"] == 1
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
    assert normalized["response_text_normalization"] == "none"
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


def test_observed_empty_thought_and_fenced_ratings_are_accepted_without_changing_raw():
    raw = dict(
        modelVersion=MODEL,
        candidates=[dict(finishReason="STOP", content=dict(role="model", parts=[
            {"thought": True, "text": ""},
            {"text": '```json\n{\n  "ratings": {\n    "T1": 5,\n    "T2": 4,\n    "T3": 3,\n    "T4": 4\n  }\n}\n```'},
        ]))],
        usageMetadata=dict(promptTokenCount=980, candidatesTokenCount=52, totalTokenCount=1032,
            promptTokensDetails=[dict(modality="TEXT", tokenCount=980)], serviceTier="standard"),
    )
    original = deepcopy(raw)
    normalized = normalize_google_response(raw, MODEL)
    assert raw == original
    assert normalized["response_text_normalization"] == "single_json_code_fence"
    evidence = normalized["google_thinking_evidence"]
    assert evidence["has_thought_parts"] and not evidence["has_nonempty_thought_parts"]
    assert evidence["empty_thought_part_count"] == 1
    assert evidence["nonempty_thought_part_count"] == 0
    assert evidence["thoughts_evidence"] == "usage_balance_zero"
    assert evidence["zero_thoughts_verified"]
    assert parse_panel_judgment(normalized, MODEL, ["T1", "T2", "T3", "T4"]) == {
        "ratings": {"T1": 5, "T2": 4, "T3": 3, "T4": 4}}


@pytest.mark.parametrize("empty", ["", " ", "\n\t"])
@pytest.mark.parametrize("thoughts", [0, ABSENT])
def test_empty_text_thought_markers_require_verified_zero_tokens(empty, thoughts):
    raw = response(thoughts=thoughts)
    raw["candidates"][0]["content"]["parts"].insert(0, {"thought": True, "text": empty})
    normalized = normalize_google_response(raw, MODEL)
    assert normalized["google_thinking_evidence"]["empty_thought_part_count"] == 1
    assert normalized["google_thinking_evidence"]["zero_thoughts_verified"]
    assert parse_panel_judgment(normalized, MODEL, ["T1", "T2", "T3", "T4"])


@pytest.mark.parametrize("thoughts", [121, ABSENT])
def test_empty_marker_does_not_hide_positive_or_unknown_thinking(thoughts):
    raw = response(thoughts=thoughts)
    raw["candidates"][0]["content"]["parts"].insert(0, {"thought": True, "text": ""})
    if thoughts is ABSENT:
        del raw["usageMetadata"]
    assert not inspect_google_thinking(raw)["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("part", [
    {"thought": True}, {"thought": True, "text": None},
    {"thought": True, "text": [], "thoughtSignature": "opaque"},
    {"thought": True, "text": "", "functionCall": {"name": "judge"}},
    {"thought": True, "text": "", "inlineData": {"mimeType": "image/png", "data": "data"}},
    {"thought": True, "text": "", "unknownPayload": {"data": "data"}},
])
def test_malformed_or_nontext_thought_parts_remain_blocked(part):
    raw = response()
    raw["candidates"][0]["content"]["parts"].insert(0, part)
    evidence = inspect_google_thinking(raw)
    assert evidence["has_thought_parts"] and evidence["has_nonempty_thought_parts"]
    assert not evidence["zero_thoughts_verified"]
    with pytest.raises(ValueError):
        normalize_google_response(raw, MODEL)


@pytest.mark.parametrize("tag", ["json", "JSON", ""])
def test_one_whole_json_code_fence_is_only_a_format_wrapper(tag):
    raw = response()
    json_text = raw["candidates"][0]["content"]["parts"][0]["text"]
    raw["candidates"][0]["content"]["parts"][0]["text"] = f"  ```{tag}\n{json_text}\n```\n"
    original = deepcopy(raw)
    normalized = normalize_google_response(raw, MODEL)
    assert raw == original
    assert normalized["choices"][0]["message"]["content"] == json_text
    assert normalized["response_text_normalization"] == "single_json_code_fence"
    assert parse_panel_judgment(normalized, MODEL, ["T1", "T2", "T3", "T4"])


@pytest.mark.parametrize("wrapper", [
    'Here are the ratings:\n```json\n{body}\n```',
    '```json\n{body}\n```\nA further explanation.',
    '```json\n{body}\n```\n```json\n{body}\n```',
    '```python\n{body}\n```',
    '```json {body}```',
])
def test_prose_multiple_fences_or_other_languages_are_not_repaired(wrapper):
    raw = response()
    json_text = raw["candidates"][0]["content"]["parts"][0]["text"]
    raw["candidates"][0]["content"]["parts"][0]["text"] = wrapper.replace("{body}", json_text)
    normalized = normalize_google_response(raw, MODEL)
    assert normalized["response_text_normalization"] == "none"
    with pytest.raises(ValueError):
        parse_panel_judgment(normalized, MODEL, ["T1", "T2", "T3", "T4"])


@pytest.mark.parametrize("json_text", [
    '{"ratings":{"T1":5,"T1":4,"T2":4,"T3":3,"T4":4}}',
    '{"ratings":{"T1":true,"T2":4,"T3":3,"T4":4}}',
    '{"ratings":{"T1":6,"T2":4,"T3":3,"T4":4}}',
    '{"ratings":{"T1":5,"T2":4,"T3":3}}',
    '{"ratings":{"T1":5,"T2":4,"T3":3,"T4":4},"explanation":"reason"}',
])
def test_fenced_json_keeps_strict_duplicate_key_and_rating_checks(json_text):
    raw = response()
    raw["candidates"][0]["content"]["parts"][0]["text"] = f"```json\n{json_text}\n```"
    normalized = normalize_google_response(raw, MODEL)
    assert normalized["response_text_normalization"] == "single_json_code_fence"
    with pytest.raises(ValueError):
        parse_panel_judgment(normalized, MODEL, ["T1", "T2", "T3", "T4"])
