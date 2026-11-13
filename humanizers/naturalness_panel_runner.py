"""Esegue un piano di rating già fissato, con ripresa e budget espliciti."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from .naturalness import APIError, GroqClient, _digest, _retry_seconds, read_jsonl


ITEM_METADATA = (
    "request_id", "source_id", "source_group_id", "domain", "generator",
    "order_id", "repeat_index", "system_order", "original_word_count",
)
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _safe_error(error, client=None):
    value = str(error)
    key = getattr(client, "_key", None) if client is not None else None
    if isinstance(key, str) and key:
        value = value.replace(key, "[REDACTED]")
    value = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
    return " ".join(value.split())[:1000]


def _validate_plan(plan):
    requests = plan.get("requests")
    if not isinstance(requests, list) or plan.get("expected_completion_requests") != len(requests):
        raise ValueError("Il numero di richieste non coincide con il piano")
    if not isinstance(plan.get("model"), str) or not plan["model"].strip():
        raise ValueError("Il piano deve fissare un modello esplicito")
    systems = plan.get("systems")
    allowed_sizes = {3, 4} if requests else {0, 3, 4}
    if not isinstance(systems, list) or len(systems) not in allowed_sizes or len(set(systems)) != len(systems):
        raise ValueError("Il piano richiede tre o quattro sistemi distinti")
    by_request, slots = {}, set()
    for item in requests:
        if any(name not in item for name in ITEM_METADATA):
            raise ValueError("Una richiesta non contiene tutti i metadati previsti")
        if not isinstance(item["request_id"], str) or not item["request_id"] or item["request_id"] in by_request:
            raise ValueError("Gli identificatori delle richieste devono essere unici")
        if not isinstance(item["system_order"], list) or sorted(item["system_order"]) != sorted(systems):
            raise ValueError("L'ordine della richiesta non contiene tutti i sistemi")
        slot = (item["source_id"], item["order_id"], item["repeat_index"])
        if slot in slots:
            raise ValueError("Una fonte ripete lo stesso slot di ordine e ripetizione")
        slots.add(slot)
        payload = item.get("payload", {})
        if payload.get("model") != plan["model"] or payload.get("reasoning_effort") != "none":
            raise ValueError("Il modello o l'opzione no-thinking non coincide con il piano")
        by_request[item["request_id"]] = item
    if len({item["source_id"] for item in requests}) != plan.get("n_sources"):
        raise ValueError("Il numero di fonti non coincide con il piano")
    return by_request


def _validate_stored(plan, rows, attempts, by_request, parse_judgment):
    attempted = set()
    for index, attempt in enumerate(attempts, 1):
        number = attempt.get("attempt_number")
        if isinstance(number, bool) or not isinstance(number, int) or number != index:
            raise ValueError("La numerazione del registro dei tentativi non è sequenziale")
        if attempt.get("request_id") not in by_request:
            raise ValueError("Il registro contiene un tentativo estraneo al piano")
        attempted.add(attempt["request_id"])
    done = set()
    for row in rows:
        request_id = row.get("request_id")
        if request_id not in by_request or request_id in done:
            raise ValueError("Le risposte salvate sono duplicate o estranee al piano")
        if request_id not in attempted:
            raise ValueError("Una risposta salvata non ha un tentativo registrato")
        item = by_request[request_id]
        if any(row.get(key) != item[key] for key in ITEM_METADATA):
            raise ValueError("I metadati della risposta salvata differiscono dalla richiesta")
        if row.get("requested_model") != plan["model"] or row.get("label_source") != "LLM":
            raise ValueError("La provenienza della risposta salvata non coincide con il piano")
        if row.get("status") not in {"ok", "invalid_response"}:
            raise ValueError("Una risposta salvata ha uno stato non riconosciuto")
        if row["status"] == "ok":
            response = dict(model=row.get("response_model"), choices=[dict(
                finish_reason="stop", message=dict(content=json.dumps(row.get("judgment"))))])
            ids = [f"T{i + 1}" for i in range(len(item["system_order"]))]
            parse_judgment(response, plan["model"], ids)
        done.add(request_id)
    return done


def _source_coverage(plan, rows):
    expected, valid = defaultdict(set), defaultdict(set)
    information = {}
    for item in plan["requests"]:
        expected[item["source_id"]].add(item["request_id"])
        information[item["source_id"]] = item
    for row in rows:
        if row["status"] == "ok":
            valid[row["source_id"]].add(row["request_id"])
    complete = sorted(source for source, ids in expected.items() if valid[source] == ids)
    incomplete = sorted(set(expected) - set(complete))
    by_domain = {}
    for source, item in information.items():
        domain = item["domain"]
        entry = by_domain.setdefault(domain, dict(n_eligible_sources=0, n_complete_sources=0))
        entry["n_eligible_sources"] += 1
        entry["n_complete_sources"] += source in complete
    for entry in by_domain.values():
        entry["complete_panel_coverage"] = entry["n_complete_sources"] / entry["n_eligible_sources"]
    return dict(
        n_eligible_source_panels=len(expected),
        n_complete_source_panels=len(complete),
        n_primary_sources=len(complete),
        n_complete_source_groups=len({information[s]["source_group_id"] for s in complete}),
        eligible_source_coverage=len(complete) / len(expected) if expected else None,
        complete_source_ids=complete,
        incomplete_source_ids=incomplete,
        complete_panel_coverage_by_domain=by_domain,
    )


def _token_usage(rows):
    totals = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total = 0
        for row in rows:
            value = row.get("usage", {}).get(field, 0)
            try:
                total += max(0, int(value))
            except (TypeError, ValueError):
                pass
        totals[field] = total
    return totals


def run_panel_plan(plan, output, *, execute=False, client=None, max_requests=36,
                   max_retries=2, max_retry_wait=30, min_request_interval=0.0,
                   sleep=time.sleep):
    """Riprende il piano senza azzerare i tentativi già spesi, inclusi i retry."""
    from .naturalness_panel import parse_panel_judgment

    if (isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1
            or isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0
            or not 0 <= max_retry_wait <= 60
            or isinstance(min_request_interval, bool)
            or not isinstance(min_request_interval, (int, float))
            or not 0 <= min_request_interval <= 60):
        raise ValueError("Budget esplicito o limiti di retry non validi")
    started = time.monotonic()
    by_request = _validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "judge_plan.json"
    if plan_path.exists():
        if _digest(json.loads(plan_path.read_text())) != _digest(plan):
            raise ValueError("Il piano differisce da quello salvato: usare un'altra directory")
    else:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results_path, attempts_path = output / "judgments.jsonl", output / "api_attempts.jsonl"
    rows = read_jsonl(results_path) if results_path.exists() else []
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    done = _validate_stored(plan, rows, attempts, by_request, parse_panel_judgment)
    old_path = output / "judge_run.json"
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    model_check = previous.get("model_check")
    status, blocked_reason, provider_error = "planned", None, None
    pending = [item for item in plan["requests"] if item["request_id"] not in done]

    if not plan["requests"]:
        status = "not_judged_no_eligible_panels"
    elif not pending:
        status = "completed" if all(row["status"] == "ok" for row in rows) else "partial"
        if status == "partial":
            blocked_reason = "stored_invalid_or_incomplete_responses"
    elif execute and len(attempts) >= max_requests:
        status, blocked_reason = "partial", "request_budget_exhausted"
    elif execute:
        client = client if client is not None else GroqClient()
        model_check = None
        try:
            model_check = client.check_model(plan["model"])
        except APIError as exc:
            status, blocked_reason = "partial", f"model_check_http_{exc.status}"
            provider_error = _safe_error(exc, client)
        except ValueError as exc:
            status, blocked_reason = "partial", "requested_model_unavailable"
            provider_error = _safe_error(exc, client)
        else:
            status = "completed"
            stop = False
            last_post_started = None
            for item in pending:
                for retry in range(max_retries + 1):
                    if len(attempts) >= max_requests:
                        status, blocked_reason, stop = "partial", "request_budget_exhausted", True
                        break
                    if last_post_started is not None and min_request_interval:
                        wait = min_request_interval - (time.monotonic() - last_post_started)
                        if wait > 0:
                            sleep(wait)
                    attempt = dict(request_id=item["request_id"], attempt_number=len(attempts) + 1,
                                   created_at=datetime.now(timezone.utc).isoformat())
                    _append_jsonl(attempts_path, attempt)
                    attempts.append(attempt)
                    sent_at = time.monotonic()
                    last_post_started = sent_at
                    try:
                        response = client.request("POST", "/chat/completions", item["payload"])
                    except APIError as exc:
                        wait = _retry_seconds(exc.retry_after)
                        if (exc.status in RETRYABLE_HTTP and retry < max_retries
                                and wait <= max_retry_wait and len(attempts) < max_requests):
                            sleep(wait)
                            continue
                        status, blocked_reason, stop = "partial", f"provider_http_{exc.status}", True
                        provider_error = _safe_error(exc, client)
                        break
                    row = {name: item[name] for name in ITEM_METADATA}
                    raw = response if isinstance(response, dict) else {}
                    usage = raw.get("usage", {})
                    row.update(
                        label_source="LLM", requested_model=plan["model"], response_model=raw.get("model"),
                        usage=usage if isinstance(usage, dict) else {}, status="ok",
                        elapsed_seconds=time.monotonic() - sent_at,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    ids = [f"T{i + 1}" for i in range(len(item["system_order"]))]
                    try:
                        if not isinstance(response, dict):
                            raise ValueError("Il provider non ha restituito un oggetto JSON")
                        row["judgment"] = parse_panel_judgment(response, plan["model"], ids)
                    except (ValueError, TypeError, KeyError) as exc:
                        row.update(status="invalid_response", error=_safe_error(exc, client))
                        status, blocked_reason, stop = "partial", "invalid_model_response", True
                    _append_jsonl(results_path, row)
                    rows.append(row)
                    done.add(item["request_id"])
                    valid = sum(r["status"] == "ok" for r in rows)
                    print(f"Giudizi validi: {valid}/{len(plan['requests'])}; "
                          f"tentativi POST: {len(attempts)}/{max_requests}; {row['status']}", flush=True)
                    break
                if stop:
                    break

    valid_count = sum(row["status"] == "ok" for row in rows)
    invalid_count = len(rows) - valid_count
    if not execute and (attempts or rows) and plan["requests"]:
        status = "completed" if valid_count == len(plan["requests"]) else "partial"
        if status == "partial":
            blocked_reason = previous.get("blocked_reason") or "stored_invalid_or_incomplete_responses"
            provider_error = previous.get("last_provider_error")
    elif status == "completed" and valid_count != len(plan["requests"]):
        status, blocked_reason = "partial", "stored_invalid_or_incomplete_responses"
    manifest = dict(
        protocol=plan["protocol"], plan_sha256=_digest(plan), status=status,
        blocked_reason=blocked_reason, last_provider_error=provider_error,
        api_execution_requested=bool(execute), generation_api_executed=bool(attempts),
        expected_completion_requests=len(plan["requests"]), completed_valid_requests=valid_count,
        invalid_response_requests=invalid_count, stored_responses=len(rows),
        max_requests=max_requests, generation_post_attempts=len(attempts),
        min_request_interval_seconds=min_request_interval,
        remaining_request_budget=max(0, max_requests - len(attempts)),
        model_check=model_check, requested_model=plan["model"], reasoning_effort="none",
        human_annotation_produced=False, label_source="LLM",
        no_implicit_model_or_reasoning_fallback=True,
        token_usage=_token_usage(rows), elapsed_seconds_this_run=time.monotonic() - started,
        api_response_elapsed_seconds_total=sum(row.get("elapsed_seconds", 0) for row in rows),
        client_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        http_transport="requests; redirects disabled",
        budget_note="Tutti i tentativi POST, inclusi retry e riprese, consumano il budget; costo monetario non stimato.",
        primary_scope="Pannelli con tutte le richieste valide; le risposte parziali restano disponibili.",
        **_source_coverage(plan, rows),
    )
    (output / "judge_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
