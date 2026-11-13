#!/usr/bin/env python3
"""Prepara o esegue valutazioni 1–5 con ordini bilanciati e ripetizioni."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humanization_naturalness_matplotlib"))

from humanizers.naturalness import GroqClient, read_jsonl
from humanizers.naturalness_panel import load_panels, make_panel_plan
from humanizers.naturalness_panel_analysis import analyze_panel
from humanizers.naturalness_panel_runner import run_panel_plan


DEFAULT_SYSTEMS = ["qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3"]


def _fingerprint(path):
    return dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrites", type=Path, action="append",
                        help="File JSONL delle riscritture, ripetibile")
    parser.add_argument("--humanizer", dest="humanizers", action="append",
                        help="Metodo da includere; ripetere per tre o quattro metodi")
    parser.add_argument("--source-key", type=Path,
                        help="CSV facoltativo con la colonna source_id")
    parser.add_argument("--max-sources", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3,
                        help="Ripetizioni identiche di ciascun ordine bilanciato")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-requests", type=int, default=36,
                        help="Limite totale di tentativi POST, comprese riprese e retry")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-input-characters", type=int, default=30000)
    parser.add_argument("--min-request-interval", type=float, default=0,
                        help="Intervallo minimo in secondi tra gli inizi delle richieste")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--execute", action="store_true",
                        help="Invia effettivamente le richieste al giudice Groq")
    parser.add_argument("--check-model", action="store_true",
                        help="Controlla soltanto il catalogo autenticato dei modelli")
    args = parser.parse_args()

    if args.check_model:
        print(json.dumps(GroqClient().check_model(args.model), indent=2))
        return
    if not args.rewrites or args.output is None:
        parser.error("Servono --rewrites e --output")
    if args.max_sources < 1 or args.repetitions < 1 or args.bootstrap < 2:
        parser.error("Fonti e ripetizioni devono essere positive; bootstrap almeno 2")
    if args.max_requests < 1 or not 0 <= args.min_request_interval <= 60:
        parser.error("Budget POST positivo e intervallo richieste compreso fra 0 e 60 secondi")
    systems = args.humanizers if args.humanizers is not None else DEFAULT_SYSTEMS
    eligibility = {}
    panels = load_panels(args.rewrites, systems, max_sources=args.max_sources,
                         seed=args.seed, source_key=args.source_key,
                         selection_report=eligibility)
    eligibility["input_rewrites"] = [_fingerprint(path) for path in args.rewrites]
    if args.source_key is not None:
        eligibility["source_key"] = _fingerprint(args.source_key)
    plan = make_panel_plan(panels, args.model, repetitions=args.repetitions,
                           seed=args.seed, max_completion_tokens=args.max_completion_tokens,
                           max_input_characters=args.max_input_characters)
    if not panels:
        plan["systems"] = list(systems)

    args.output.mkdir(parents=True, exist_ok=True)
    eligibility_path = args.output / "judge_eligibility.json"
    plan_path = args.output / "judge_plan.json"
    for path, value in [(eligibility_path, eligibility), (plan_path, plan)]:
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"{path.name} differisce dal run salvato: usare una nuova cartella")
    eligibility_path.write_text(json.dumps(eligibility, ensure_ascii=False, indent=2) + "\n")
    manifest = run_panel_plan(plan, args.output, execute=args.execute,
                              max_requests=args.max_requests,
                              min_request_interval=args.min_request_interval)
    result_path = args.output / "judgments.jsonl"
    judgments = read_jsonl(result_path) if result_path.exists() else []
    analyze_panel(plan, judgments, args.output, eligibility=eligibility,
                  bootstrap=args.bootstrap, seed=args.seed)

    console = dict(protocol=plan["protocol"], model=plan["model"],
                   status=manifest["status"],
                   selected_sources=eligibility["n_selected_sources"],
                   eligible_sources=eligibility["n_eligible_panels"],
                   generation_failure_sources=eligibility["n_generation_failure_panels"],
                   planned_calls=plan["expected_completion_requests"],
                   valid_calls=manifest.get("completed_valid_requests", 0),
                   post_attempts=manifest.get("generation_post_attempts", 0),
                   complete_source_panels=manifest.get("n_complete_source_panels", 0),
                   max_post_attempts=args.max_requests,
                   output=str(args.output),
                   human_annotation_produced=False)
    if manifest.get("blocked_reason"):
        console["blocked_reason"] = manifest["blocked_reason"]
    if manifest.get("last_provider_error"):
        console["last_provider_error"] = manifest["last_provider_error"]
    print(json.dumps(console, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
