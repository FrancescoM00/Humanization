#!/usr/bin/env python3
"""Esegue il pilota Gemma e il confronto con Qwen tramite API native Google."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_google_matplotlib"))

from humanizers.naturalness import read_jsonl
from humanizers.naturalness_google_plan import GOOGLE_MODEL, prepare_google_comparison
from humanizers.naturalness_google_runner import run_google_panel_plan
from humanizers.naturalness_judge_comparison_analysis import analyze_judge_comparison
from humanizers.naturalness_panel_analysis import analyze_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT /
        "exports/naturalness_judge_comparison/qwen_reference_20261118.zip")
    parser.add_argument("--provider", choices=["google"], default="google")
    parser.add_argument("--model", choices=[GOOGLE_MODEL], default=GOOGLE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--max-requests", type=int, default=4000,
                        help="Tetto totale POST, inclusi 429, 503, retry e riprese")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--min-request-interval", type=float, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.bootstrap < 2 or args.max_requests < 1 or args.max_retries < 0
            or not 0 <= args.min_request_interval <= 60):
        parser.error("Bootstrap almeno due, budget positivo e pausa fra zero e sessanta secondi")
    plan, reference_plan, reference_rows, setup = prepare_google_comparison(
        args.reference, args.output, model=args.model)
    manifest = run_google_panel_plan(plan, args.output, execute=args.execute, stage=args.stage,
        max_requests=args.max_requests, max_retries=args.max_retries,
        min_request_interval=args.min_request_interval)
    path = args.output / "judgments.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    analyze_panel(reference_plan, reference_rows, args.output / "reference",
                  bootstrap=args.bootstrap, seed=args.seed)
    panel_summary = analyze_panel(plan, rows, args.output, bootstrap=args.bootstrap, seed=args.seed)
    panel_summary.update(provider="google", thinking_level="minimal",
        thinking_scope="minimal is not labelled as off; see pilot_check.json and per-response token evidence",
        method_family_scope="Gemma and Qwen belong to different model families; this is not human validation")
    (args.output / "summary.json").write_text(
        json.dumps(panel_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    comparison = analyze_judge_comparison(reference_plan, reference_rows, plan, rows,
        args.output / "comparison", bootstrap=args.bootstrap, seed=args.seed)
    pilot_path = args.output / "pilot_check.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    console = dict(provider="google", reference_model=reference_plan["model"],
        candidate_model=plan["model"], thinking_level="minimal", stage=args.stage,
        reference_complete_sources=setup["n_complete_reference_sources"],
        reference_excluded_partial_sources=setup["n_excluded_partial_sources"],
        expected_candidate_requests=plan["expected_completion_requests"],
        pilot_sources=len(plan["pilot_source_ids"]), expected_pilot_requests=setup["expected_pilot_requests"],
        status=manifest["status"], completed_valid_requests=manifest["completed_valid_requests"],
        generation_post_attempts=manifest["generation_post_attempts"],
        blocked_reason=manifest.get("blocked_reason"), pilot_check=pilot,
        n_common_complete_sources=comparison["n_common_complete_sources"],
        human_annotation_produced=False)
    if manifest.get("last_provider_error"):
        console["last_provider_error"] = manifest["last_provider_error"]
    print(json.dumps(console, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
