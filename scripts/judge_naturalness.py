#!/usr/bin/env python3
"""Plan a bounded Groq AB/BA naturalness evaluation; API calls require --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanizers.naturalness import GroqClient, load_pairs, make_plan, read_jsonl, run_plan, validate_ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrites", type=Path, action="append", help="Repeat for each complete rewriting file")
    parser.add_argument("--humanizer-a", default="qwen_humanize")
    parser.add_argument("--humanizer-b", default="dipper")
    parser.add_argument("--model", default="qwen/qwen3.8-27b", help="Exact user-requested Groq model; no fallback")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-key", type=Path, help="Optional source_id CSV, e.g. paired rating key with six sources")
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=6, help="Total POST attempts across resumes, including retries")
    parser.add_argument("--max-completion-tokens", type=int, default=200)
    parser.add_argument("--max-input-characters", type=int, default=30000)
    parser.add_argument("--reasoning-effort", choices=["none"], default="none",
                        help="Must be accepted by the selected model; hiding reasoning is not disabling it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="Perform bounded authenticated API requests")
    parser.add_argument("--check-model", action="store_true", help="Authenticated GET /models only, no generation")
    parser.add_argument("--ratings", type=Path, help="Optional filled single-text 1–5 ratings CSV")
    parser.add_argument("--rating-key", type=Path)
    parser.add_argument("--label-source", choices=["human", "assistant_proxy", "other_proxy"])
    args = parser.parse_args()
    if args.check_model:
        print(json.dumps(GroqClient().check_model(args.model), indent=2))
        return
    if not args.rewrites or not args.output:
        parser.error("--rewrites and --output are required unless using --check-model")
    if bool(args.ratings) != bool(args.rating_key) or args.ratings and not args.label_source:
        parser.error("Validation requires --ratings, --rating-key and explicit --label-source")
    pairs = load_pairs(args.rewrites, args.humanizer_a, args.humanizer_b,
                       source_key=args.source_key, max_pairs=args.max_pairs, seed=args.seed)
    plan = make_plan(pairs, args.model, reasoning_effort=args.reasoning_effort,
                     max_completion_tokens=args.max_completion_tokens,
                     max_input_characters=args.max_input_characters)
    manifest = run_plan(plan, args.output, execute=args.execute, max_requests=args.max_requests)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.ratings:
        path = args.output / "judgments.jsonl"
        judgments = read_jsonl(path) if path.exists() else []
        validation = validate_ratings(plan, judgments, args.ratings, args.rating_key, label_source=args.label_source)
        (args.output / "rating_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
