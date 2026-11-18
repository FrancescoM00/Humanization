"""Esegue il pilota Google e conserva ogni tentativo, risposta e controllo."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from .naturalness import APIError, _digest, _retry_seconds, read_jsonl
from .naturalness_google import GoogleGemmaClient
from .naturalness_google_plan import validate_google_plan
from .naturalness_google_results import inspect_google_thinking, normalize_google_response
from .naturalness_panel import parse_panel_judgment
from .naturalness_panel_runner import ITEM_METADATA, _append_jsonl, _safe_error, _source_coverage


RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _read_rows(path):
    return read_jsonl(path) if path.is_file() else []


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _google_usage(raw, evidence):
    """Un dato mancante non diventa una prova di zero token di pensiero."""
    meta = raw.get("usageMetadata", {}) if isinstance(raw, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    result = {}
    for target, source in (("prompt_tokens", "promptTokenCount"),
                           ("completion_tokens", "candidatesTokenCount"),
                           ("total_tokens", "totalTokenCount")):
        value = meta.get(source)
        result[target] = value if type(value) is int and value >= 0 else None
    result["thoughts_tokens"] = evidence.get("thoughts_token_count")
    if evidence.get("zero_thoughts_verified"):
        result["thoughts_tokens"] = 0
    return result


def _thinking_problem(evidence):
    if evidence.get("has_thought_parts") or (type(evidence.get("thoughts_token_count")) is int
                                             and evidence["thoughts_token_count"] > 0):
        return "positive_thinking_tokens_or_parts"
    if not evidence.get("zero_thoughts_verified"):
        return "unverified_thinking_tokens"
    return None


def _validate_stored(plan, rows, attempts, errors, by_request):
    """Rivalida le risposte grezze e non si fida dei soli flag del checkpoint."""
    attempted = set()
    for number, item in enumerate(attempts, 1):
        if type(item.get("attempt_number")) is not int or item["attempt_number"] != number:
            raise ValueError("Il registro Google dei tentativi non è sequenziale")
        if item.get("request_id") not in by_request:
            raise ValueError("Un tentativo Google non appartiene al piano congelato")
        if item.get("provider", "google") != "google" or item.get("requested_model", plan["model"]) != plan["model"]:
            raise ValueError("Il registro dei tentativi appartiene a un altro provider o modello")
        attempted.add(item["request_id"])
    error_attempts = set()
    for error in errors:
        number = error.get("attempt_number")
        if type(number) is not int or not 1 <= number <= len(attempts) or number in error_attempts:
            raise ValueError("Il registro degli errori Google non corrisponde ai tentativi")
        if error.get("request_id") != attempts[number-1]["request_id"]:
            raise ValueError("Un errore Google si riferisce a una richiesta diversa")
        error_attempts.add(number)
    done, versions, first_observed_version = set(), set(), None
    for row in rows:
        identifier = row.get("request_id")
        if identifier not in by_request or identifier in done or identifier not in attempted:
            raise ValueError("Una risposta Google è duplicata, estranea al piano o senza tentativo")
        item = by_request[identifier]
        if any(row.get(name) != item[name] for name in ITEM_METADATA):
            raise ValueError("I metadati Google differiscono dal piano congelato")
        if (row.get("provider") != "google" or row.get("requested_model") != plan["model"]
                or row.get("label_source") != "LLM" or row.get("thinking_level") != "minimal"):
            raise ValueError("La provenienza o la configurazione della risposta Google è diversa")
        if row.get("status") not in {"ok", "invalid_response"}:
            raise ValueError("Una risposta Google ha uno stato sconosciuto")
        raw = row.get("raw_response")
        if not isinstance(raw, dict):
            raise ValueError("Il checkpoint Google deve conservare la risposta grezza")
        evidence = inspect_google_thinking(raw)
        raw_version = raw.get("modelVersion")
        if first_observed_version is None and _version_matches(raw_version, plan["model"]):
            first_observed_version = raw_version
        if row.get("google_thinking_evidence") != evidence:
            raise ValueError("Il controllo dei token di pensiero non coincide con la risposta grezza")
        usage_meta = raw.get("usageMetadata", {})
        if row.get("usage_metadata") != usage_meta:
            raise ValueError("I metadati dei token differiscono dalla risposta Google grezza")
        if row.get("usage") != _google_usage(raw, evidence):
            raise ValueError("Il conteggio dei token non coincide con la risposta Google grezza")
        if row["status"] == "ok":
            normalized = normalize_google_response(raw, plan["model"])
            ids = [f"T{i+1}" for i in range(len(item["system_order"]))]
            judgment = parse_panel_judgment(normalized, plan["model"], ids)
            if _thinking_problem(evidence) or row.get("judgment") != judgment:
                raise ValueError("Un giudizio Google valido non supera la rivalidazione grezza")
            if (row.get("response_model") != plan["model"]
                    or row.get("provider_model_version") != normalized.get("provider_model_version")
                    or row.get("provider_model_match_rule") != normalized.get("provider_model_match_rule")):
                raise ValueError("Il modello effettivo della risposta Google non coincide")
            version = normalized.get("provider_model_version")
            if not isinstance(version, str) or not version:
                raise ValueError("Un giudizio Google valido deve conservare modelVersion")
            versions.add(version)
        done.add(identifier)
    if len(versions) > 1:
        raise ValueError("Le risposte Google valide contengono versioni di modello differenti")
    return done, next(iter(versions), first_observed_version)


def _version_matches(version, requested):
    return isinstance(version, str) and (
        version == requested or version.startswith(requested + "-") and len(version) > len(requested)+1)


def _stored_problem(rows):
    versions = set()
    for row in rows:
        raw_version = row["raw_response"].get("modelVersion")
        if not _version_matches(raw_version, row["requested_model"]):
            return "provider_model_mismatch"
        versions.add(raw_version)
        problem = _thinking_problem(row["google_thinking_evidence"])
        if problem:
            return problem
        if row.get("validation_reason") in {"provider_model_version_changed", "provider_model_mismatch"}:
            return row["validation_reason"]
    if len(versions) > 1:
        return "provider_model_version_changed"
    return None


def _pilot_check(plan, rows, attempts, errors):
    """Il pilota si chiude solo quando tutti i suoi slot hanno una risposta."""
    sources = set(plan["pilot_source_ids"])
    expected = {item["request_id"] for item in plan["requests"] if item["source_id"] in sources}
    selected = [row for row in rows if row["request_id"] in expected]
    valid = [row for row in selected if row["status"] == "ok"]
    invalid = len(selected)-len(valid)
    positive = sum(_thinking_problem(row["google_thinking_evidence"]) == "positive_thinking_tokens_or_parts"
                   for row in selected)
    unknown = sum(_thinking_problem(row["google_thinking_evidence"]) == "unverified_thinking_tokens"
                  for row in selected)
    fraction = invalid/len(selected) if selected else None
    expected_sources = Counter(item["source_id"] for item in plan["requests"] if item["source_id"] in sources)
    valid_sources = Counter(row["source_id"] for row in valid)
    complete = sorted(source for source, count in expected_sources.items() if valid_sources[source] == count)
    all_accounted = len(selected) == len(expected)
    passed = bool(expected and all_accounted and not positive and not unknown
                  and fraction is not None and fraction < .05 and complete)
    observed_versions = {row["raw_response"].get("modelVersion") for row in selected
                         if _version_matches(row["raw_response"].get("modelVersion"), plan["model"])}
    version_changes = len(observed_versions) > 1 or any(
        not _version_matches(row["raw_response"].get("modelVersion"), plan["model"])
        for row in selected)
    if version_changes:
        passed = False
    pilot_errors = [error for error in errors if error["request_id"] in expected]
    error_counts = Counter(str(error.get("status_code")) for error in pilot_errors)
    warning = []
    if error_counts["429"]:
        warning.append("Sono presenti HTTP 429: verificare ritmo e quote prima dello studio completo")
    if invalid:
        warning.append("I pannelli con risposte non valide sono esclusi dai confronti primari")
    return dict(
        status="passed" if passed else "failed" if all_accounted or positive or unknown or version_changes else "incomplete",
        passed=passed, provider="google", requested_model=plan["model"], thinking_level="minimal",
        n_pilot_sources=len(sources), pilot_source_ids=sorted(sources),
        expected_requests=len(expected), stored_responses=len(selected), valid_responses=len(valid),
        invalid_responses=invalid, invalid_fraction=fraction,
        all_planned_requests_answered=all_accounted, n_complete_source_panels=len(complete),
        complete_source_ids=complete, positive_thinking_responses=positive,
        unverified_thinking_responses=unknown, model_version_changed=version_changes,
        observed_provider_model_versions=sorted(observed_versions),
        thinking_evidence_counts=dict(Counter(row["google_thinking_evidence"].get("thoughts_evidence", "unknown")
                                             for row in selected)),
        post_attempts=sum(item["request_id"] in expected for item in attempts),
        http_429_count=error_counts["429"], http_503_count=error_counts["503"], warnings=warning,
        pass_rule="All pilot slots answered; invalid fraction below 5%; at least one complete panel; "
                  "all responses verified zero thought tokens without thought parts; stable model version",
        minimal_is_not_assumed_to_disable_thinking=True, human_annotation_produced=False,
    )


def _usage_totals(rows):
    totals = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "thoughts_tokens"):
        values = [row.get("usage", {}).get(field) for row in rows]
        totals[field] = sum(value for value in values if type(value) is int and value >= 0)
    totals["responses_without_verified_zero_thoughts"] = sum(
        not row["google_thinking_evidence"].get("zero_thoughts_verified") for row in rows)
    totals["responses_with_explicit_zero_thought_count"] = sum(
        row["google_thinking_evidence"].get("thoughts_evidence") == "explicit_zero" for row in rows)
    totals["responses_with_zero_inferred_from_usage_balance"] = sum(
        row["google_thinking_evidence"].get("thoughts_evidence") == "usage_balance_zero" for row in rows)
    totals["responses_with_missing_usage_counts"] = sum(
        any(row.get("usage", {}).get(field) is None for field in
            ("prompt_tokens", "completion_tokens", "total_tokens")) for row in rows)
    return totals


def run_google_panel_plan(plan, output, *, execute=False, stage="pilot", client=None,
                          max_requests=4000, max_retries=4, min_request_interval=8.0,
                          sleep=time.sleep):
    """Mantiene il piano intero e limita il primo avvio ai pannelli del pilota."""
    if (stage not in {"pilot", "full"} or type(max_requests) is not int or max_requests < 1
            or type(max_retries) is not int or max_retries < 0
            or isinstance(min_request_interval, bool)
            or not isinstance(min_request_interval, (int, float))
            or not math.isfinite(min_request_interval) or not 0 <= min_request_interval <= 60):
        raise ValueError("Fase, budget, numero di retry o intervallo Google non validi")
    started = time.monotonic()
    by_request = validate_google_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "judge_plan.json"
    if plan_path.exists():
        if _digest(json.loads(plan_path.read_text())) != _digest(plan):
            raise ValueError("Il piano Google è diverso: usare un'altra cartella")
    else:
        _write_json(plan_path, plan)
    results_path, attempts_path, errors_path = (output/name for name in
                                               ("judgments.jsonl", "api_attempts.jsonl", "api_errors.jsonl"))
    rows, attempts, errors = (_read_rows(path) for path in (results_path, attempts_path, errors_path))
    done, pinned_version = _validate_stored(plan, rows, attempts, errors, by_request)
    old_path = output / "judge_run.json"
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    model_check = previous.get("model_check")
    pilot = _pilot_check(plan, rows, attempts, errors)
    pilot_sources = set(plan["pilot_source_ids"])
    targets = [item for item in plan["requests"] if stage == "full" or item["source_id"] in pilot_sources]
    pending = [item for item in targets if item["request_id"] not in done]
    status, blocked_reason, provider_error = "planned", None, None
    problem = _stored_problem(rows)
    if problem:
        status, blocked_reason = "pilot_failed" if stage == "pilot" else "partial", problem
    elif execute and stage == "full" and not pilot["passed"]:
        status, blocked_reason = "partial", "pilot_not_passed"
    elif not pending:
        if stage == "pilot":
            status = "pilot_complete" if pilot["passed"] else "pilot_failed"
            blocked_reason = None if pilot["passed"] else "pilot_acceptance_criteria_failed"
        else:
            status = "completed" if all(row["status"] == "ok" for row in rows) else "partial"
            blocked_reason = None if status == "completed" else "stored_invalid_or_incomplete_responses"
    elif execute and len(attempts) >= max_requests:
        status, blocked_reason = "partial", "request_budget_exhausted"
    elif execute:
        client = client if client is not None else GoogleGemmaClient()
        model_check = None
        try:
            model_check = client.check_model(plan["model"])
        except APIError as exc:
            status, blocked_reason, provider_error = "partial", f"model_check_http_{exc.status}", _safe_error(exc, client)
        except ValueError as exc:
            status, blocked_reason, provider_error = "partial", "requested_model_unavailable", _safe_error(exc, client)
        else:
            status, stop, last_post_started = "partial", False, None
            for item in pending:
                for retry in range(max_retries+1):
                    if len(attempts) >= max_requests:
                        blocked_reason, stop = "request_budget_exhausted", True
                        break
                    if last_post_started is not None and min_request_interval:
                        delay = min_request_interval-(time.monotonic()-last_post_started)
                        if delay > 0:
                            sleep(delay)
                    attempt = dict(request_id=item["request_id"], attempt_number=len(attempts)+1,
                                   created_at=datetime.now(timezone.utc).isoformat(), provider="google",
                                   requested_model=plan["model"], thinking_level="minimal", stage=stage)
                    _append_jsonl(attempts_path, attempt)
                    attempts.append(attempt)
                    sent_at = last_post_started = time.monotonic()
                    try:
                        raw = client.request("POST", "/chat/completions", item["native_payload"])
                    except APIError as exc:
                        try:
                            wait = _retry_seconds(exc.retry_after)
                        except (TypeError, ValueError, OverflowError):
                            wait = 1.0
                        error = dict(request_id=item["request_id"], attempt_number=attempt["attempt_number"],
                                     created_at=datetime.now(timezone.utc).isoformat(), status_code=exc.status,
                                     error=_safe_error(exc, client),
                                     retry_after_seconds=wait if math.isfinite(wait) else None)
                        _append_jsonl(errors_path, error)
                        errors.append(error)
                        if (exc.status in RETRYABLE_HTTP and retry < max_retries and len(attempts) < max_requests
                                and math.isfinite(wait) and wait <= 60):
                            sleep(max(min(20 * (2**retry), 60), wait))
                            continue
                        blocked_reason, provider_error, stop = f"provider_http_{exc.status}", _safe_error(exc, client), True
                        break
                    raw = raw if isinstance(raw, dict) else {"invalid_response_value": raw}
                    evidence = inspect_google_thinking(raw)
                    row = {name: item[name] for name in ITEM_METADATA}
                    row.update(provider="google", label_source="LLM", thinking_level="minimal",
                               requested_model=plan["model"], response_model=None, status="invalid_response",
                               provider_model_version=raw.get("modelVersion"), raw_response=raw,
                               usage_metadata=raw.get("usageMetadata", {}),
                               usage=_google_usage(raw, evidence), google_thinking_evidence=evidence,
                               elapsed_seconds=time.monotonic()-sent_at,
                               created_at=datetime.now(timezone.utc).isoformat())
                    validation_problem = _thinking_problem(evidence)
                    try:
                        raw_version = raw.get("modelVersion")
                        if not _version_matches(raw_version, plan["model"]):
                            validation_problem = "provider_model_mismatch"
                            row["validation_error_kind"] = "provider_model_version"
                            raise ValueError("La risposta Google non documenta la versione del modello richiesto")
                        if validation_problem:
                            raise ValueError("Il controllo Google non verifica zero token di pensiero")
                        if pinned_version is not None and raw.get("modelVersion") != pinned_version:
                            validation_problem = "provider_model_version_changed"
                            row["validation_error_kind"] = "provider_model_version"
                            raise ValueError("modelVersion differisce da quella già registrata")
                        if pinned_version is None:
                            pinned_version = raw_version
                        normalized = normalize_google_response(raw, plan["model"])
                        version = normalized.get("provider_model_version")
                        if not isinstance(version, str) or not version:
                            raise ValueError("La risposta Google non documenta modelVersion")
                        ids = [f"T{i+1}" for i in range(len(item["system_order"]))]
                        row["judgment"] = parse_panel_judgment(normalized, plan["model"], ids)
                        row.update(status="ok", response_model=plan["model"], provider_model_version=version,
                                   provider_model_match_rule=normalized.get("provider_model_match_rule"))
                        pinned_version = version
                    except (ValueError, TypeError, KeyError) as exc:
                        row.update(error=_safe_error(exc, client),
                                   validation_reason=validation_problem or "invalid_model_response")
                        blocked_reason, stop = row["validation_reason"], True
                    _append_jsonl(results_path, row)
                    rows.append(row)
                    done.add(item["request_id"])
                    print(f"Giudizi Google validi: {sum(row['status']=='ok' for row in rows)}/{len(plan['requests'])}; "
                          f"POST: {len(attempts)}/{max_requests}; {row['status']}", flush=True)
                    break
                if stop:
                    break
            pilot = _pilot_check(plan, rows, attempts, errors)
            if not stop:
                if stage == "pilot":
                    status = "pilot_complete" if pilot["passed"] else "pilot_failed"
                    blocked_reason = None if pilot["passed"] else "pilot_acceptance_criteria_failed"
                else:
                    status = "completed" if all(row["status"] == "ok" for row in rows) else "partial"
                    blocked_reason = None if status == "completed" else "stored_invalid_or_incomplete_responses"
    elif attempts or rows:
        status = "partial"
        blocked_reason = previous.get("blocked_reason")
        provider_error = previous.get("last_provider_error")
    pilot = _pilot_check(plan, rows, attempts, errors)
    _write_json(output/"pilot_check.json", pilot)
    manifest = dict(
        protocol=plan["protocol"], provider="google", requested_model=plan["model"], stage=stage,
        thinking_level="minimal", minimal_is_not_assumed_to_disable_thinking=True,
        plan_sha256=_digest(plan), status=status, blocked_reason=blocked_reason,
        last_provider_error=provider_error, api_execution_requested=bool(execute),
        generation_api_executed=bool(attempts), expected_completion_requests=len(plan["requests"]),
        stage_expected_requests=len(targets), completed_valid_requests=sum(row["status"] == "ok" for row in rows),
        invalid_response_requests=sum(row["status"] != "ok" for row in rows), stored_responses=len(rows),
        generation_post_attempts=len(attempts), max_requests=max_requests,
        remaining_request_budget=max(0, max_requests-len(attempts)), max_retries=max_retries,
        min_request_interval_seconds=min_request_interval, model_check=model_check,
        provider_model_version_pinned=pinned_version, pilot_passed=pilot["passed"],
        thinking_evidence_counts=dict(Counter(row["google_thinking_evidence"].get("thoughts_evidence", "unknown")
                                             for row in rows)),
        token_usage=_usage_totals(rows),
        http_error_counts=dict(Counter(str(error["status_code"]) for error in errors)),
        elapsed_seconds_this_run=time.monotonic()-started,
        api_response_elapsed_seconds_total=sum(row.get("elapsed_seconds", 0) for row in rows),
        updated_at=datetime.now(timezone.utc).isoformat(),
        client_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        http_transport="requests; fixed Google host; redirects disabled; API key in header only",
        budget_note="Tutti i POST consumano il budget, inclusi HTTP 429/503, retry e riprese",
        primary_scope="Confronto sui soli originali con tutti i giudizi validi per entrambi i modelli",
        no_implicit_model_or_reasoning_fallback=True, human_annotation_produced=False, label_source="LLM",
        **_source_coverage(plan, rows),
    )
    _write_json(output/"judge_run.json", manifest)
    return manifest
