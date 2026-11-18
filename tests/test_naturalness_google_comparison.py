"""Verifica il confronto Google e la CLI sui dati congelati, senza inferenza."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from humanizers.naturalness_google_plan import prepare_google_comparison
from humanizers.naturalness_google_results import normalize_google_response
from humanizers.naturalness_judge_comparison_analysis import analyze_judge_comparison
from humanizers.naturalness_panel import parse_panel_judgment
from humanizers.naturalness_panel_runner import ITEM_METADATA


REFERENCE = Path(__file__).resolve().parents[1] / "exports/naturalness_judge_comparison/qwen_reference_20261118.zip"


def _prepared(tmp_path):
    return prepare_google_comparison(REFERENCE, tmp_path / "google")


def _candidate_rows(plan, reference_rows):
    """Due fonti complete e una parziale, con voti sintetici uguali al riferimento."""
    originals = {(r["source_id"], r["order_id"], r["repeat_index"]): r for r in reference_rows}
    sources = list(dict.fromkeys(item["source_id"] for item in plan["requests"]))[:3]
    rows = []
    for item in plan["requests"]:
        if item["source_id"] not in sources:
            continue
        if item["source_id"] == sources[-1] and sum(r["source_id"] == sources[-1] for r in rows) == 4:
            continue
        baseline = originals[(item["source_id"], item["order_id"], item["repeat_index"])]
        raw = dict(modelVersion=plan["model"], responseId="offline-test-only",
            candidates=[dict(finishReason="STOP", content=dict(role="model", parts=[dict(
                text=json.dumps(baseline["judgment"]))]))],
            usageMetadata=dict(promptTokenCount=100, candidatesTokenCount=20,
                               totalTokenCount=120, thoughtsTokenCount=0))
        normalized = normalize_google_response(raw, plan["model"])
        row = {key: item[key] for key in ITEM_METADATA}
        row.update(provider="google", requested_model=plan["model"], response_model=plan["model"],
                   provider_model_version=raw["modelVersion"], label_source="LLM", status="ok",
                   raw_response=raw, judgment=parse_panel_judgment(normalized, plan["model"], ["T1", "T2", "T3", "T4"]))
        rows.append(row)
    return rows


def test_google_comparison_uses_only_common_complete_sources(tmp_path):
    plan, reference_plan, reference_rows, _ = _prepared(tmp_path)
    rows = _candidate_rows(plan, reference_rows)
    output = tmp_path / "comparison"
    result = analyze_judge_comparison(reference_plan, reference_rows, plan, rows,
                                      output, bootstrap=5)
    assert result["n_complete_reference"] == 61
    assert result["n_complete_candidate"] == result["n_common_complete_sources"] == 2
    assert result["n_paired_ratings"] == 2 * 12 * 4
    assert result["ordinal_agreement"]["exact_agreement"] == 1
    assert result["candidate_provider"] == "google"
    shifts = pd.read_csv(output / "system_shifts.csv")
    assert shifts.n_sources.eq(2).all()
    assert shifts.mean_shift_candidate_minus_reference.eq(0).all()
    assert len(pd.read_csv(output / "raw_candidate_ratings.csv")) == len(rows) * 4


@pytest.mark.parametrize("change", ["native_text", "thoughts", "actual_version"])
def test_google_comparison_checks_actual_materials_and_response(tmp_path, change):
    plan, reference_plan, reference_rows, _ = _prepared(tmp_path)
    rows = _candidate_rows(plan, reference_rows)
    if change == "native_text":
        plan["requests"][0]["native_payload"]["contents"][0]["parts"][0]["text"] += "Changed"
    elif change == "thoughts":
        rows[0]["raw_response"]["usageMetadata"].update(thoughtsTokenCount=121, totalTokenCount=241)
    else:
        rows[0]["provider_model_version"] = "another-version"
    with pytest.raises(ValueError):
        analyze_judge_comparison(reference_plan, reference_rows, plan, rows,
                                  tmp_path / "comparison", bootstrap=5)


def test_cli_offline_google_pilot_does_not_read_secret_or_infer(tmp_path, monkeypatch):
    from scripts import compare_naturalness_google as cli
    import humanizers.naturalness_google_runner as runner

    def no_client(*args, **kwargs):
        raise AssertionError("La preparazione offline non deve creare un client")

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GoogleGemmaClient", no_client)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["compare_naturalness_google.py", "--output", str(output), "--bootstrap", "5"])
    cli.main()
    manifest = json.loads((output / "judge_run.json").read_text())
    pilot = json.loads((output / "pilot_check.json").read_text())
    comparison = json.loads((output / "comparison/summary.json").read_text())
    assert manifest["generation_post_attempts"] == 0
    assert manifest["expected_completion_requests"] == 732
    assert pilot["expected_requests"] == 120 and not pilot["passed"]
    assert comparison["n_common_complete_sources"] == 0
    assert not (output / "judgments.jsonl").exists()
    assert not list(output.rglob("*.md")) and not list(output.rglob("*.pdf"))
