#!/usr/bin/env python3
"""Valuta con Llama gli stessi pannelli completi già giudicati da Qwen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_naturalness_matplotlib"))

from humanizers.naturalness import read_jsonl
from humanizers.naturalness_judge_comparison import prepare_judge_comparison, SECOND_MODEL
from humanizers.naturalness_judge_comparison_analysis import analyze_judge_comparison
from humanizers.naturalness_panel_analysis import analyze_panel
from humanizers.naturalness_panel_runner import run_panel_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT /
        "exports/naturalness_judge_comparison/qwen_reference_20261118.zip")
    parser.add_argument("--model", default=SECOND_MODEL, choices=[SECOND_MODEL])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=900)
    parser.add_argument("--min-request-interval", type=float, default=8.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 2 or args.max_requests < 1 or not 0 <= args.min_request_interval <= 60:
        parser.error("Bootstrap almeno due, budget positivo e pausa fra zero e sessanta secondi")
    plan, reference_plan, reference_rows, setup = prepare_judge_comparison(
        args.reference, args.output, model=args.model)
    manifest = run_panel_plan(plan, args.output, execute=args.execute,
        max_requests=args.max_requests, min_request_interval=args.min_request_interval)
    path = args.output / "judgments.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    analyze_panel(reference_plan, reference_rows, args.output / "reference",
                  bootstrap=args.bootstrap, seed=args.seed)
    analyze_panel(plan, rows, args.output, bootstrap=args.bootstrap, seed=args.seed)
    comparison = analyze_judge_comparison(reference_plan, reference_rows, plan, rows,
        args.output / "comparison", bootstrap=args.bootstrap, seed=args.seed)
    console = dict(reference_model=reference_plan["model"], candidate_model=plan["model"],
        reference_complete_sources=setup["n_complete_reference_sources"],
        reference_excluded_partial_sources=setup["n_excluded_partial_sources"],
        reference_complete_sources_by_domain=setup["complete_sources_by_domain"],
        expected_candidate_requests=plan["expected_completion_requests"],
        status=manifest["status"], valid_requests=manifest["completed_valid_requests"],
        completed_candidate_sources=manifest["n_complete_source_panels"],
        post_attempts=manifest["generation_post_attempts"],
        blocked_reason=manifest.get("blocked_reason"),
        comparison_output=str(args.output / "comparison"), human_annotation_produced=False)
    if manifest.get("last_provider_error"):
        console["last_provider_error"] = manifest["last_provider_error"]
    print(json.dumps(console, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
