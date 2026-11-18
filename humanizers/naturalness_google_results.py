"""Controlla i metadati di pensiero e normalizza le sole risposte Google valide."""
from __future__ import annotations

import re


_COUNTERS = (
    "promptTokenCount", "candidatesTokenCount", "totalTokenCount",
    "toolUsePromptTokenCount", "thoughtsTokenCount",
)
_NON_TEXT_PART_FIELDS = (
    "functionCall", "functionResponse", "inlineData", "fileData",
    "executableCode", "codeExecutionResult",
)


def _token_count(value):
    """Accetta contatori interi non negativi, senza convertire stringhe o booleani."""
    return value if type(value) is int and value >= 0 else None


def _empty_thought_part(part):
    """Riconosce soltanto una parte di pensiero testuale priva di contenuto."""
    return (isinstance(part, dict) and part.get("thought") is True
            and isinstance(part.get("text"), str) and not part["text"].strip()
            and set(part) <= {"thought", "text", "thoughtSignature", "thought_signature"})


def _thought_part_counts(raw):
    """Conserva i marcatori vuoti e tratta quelli malformati come non sicuri."""
    empty, nonempty = 0, 0
    if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
        return empty, nonempty
    for candidate in raw["candidates"]:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            continue
        for part in content["parts"]:
            if isinstance(part, dict) and part.get("thought") is True:
                if _empty_thought_part(part):
                    empty += 1
                else:
                    nonempty += 1
    return empty, nonempty


def _normalize_json_code_fence(text):
    """Rimuove solo un unico blocco Markdown che contiene l'intero oggetto JSON."""
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text.strip(), re.IGNORECASE)
    if match:
        body = match.group(1)
        if "```" not in body and body.lstrip().startswith("{") and body.rstrip().endswith("}"):
            return body, "single_json_code_fence"
    return text, "none"


def inspect_google_thinking(raw):
    """Distingue lo zero esplicito da quello ricavato dal bilancio dei token.

    L'assenza dell'intero usage non viene mai interpretata come assenza di
    pensiero. Le parti di pensiero vuote rimangono contate separatamente;
    parti non vuote, malformate o contatori contraddittori impediscono la verifica.
    """
    empty_parts, nonempty_parts = _thought_part_counts(raw)
    result = dict(thoughts_token_count=None, thoughts_field_present=False,
                  thoughts_evidence="unknown", zero_thoughts_verified=False,
                  has_thought_parts=bool(empty_parts or nonempty_parts),
                  has_nonempty_thought_parts=bool(nonempty_parts),
                  empty_thought_part_count=empty_parts, nonempty_thought_part_count=nonempty_parts)
    usage = raw.get("usageMetadata") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return result
    present = "thoughtsTokenCount" in usage
    result["thoughts_field_present"] = present
    counts = {key: _token_count(usage.get(key)) for key in _COUNTERS}
    malformed = any(key in usage and counts[key] is None for key in _COUNTERS)
    required = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
    known_balance = all(counts[key] is not None for key in required)
    tool = counts["toolUsePromptTokenCount"] if "toolUsePromptTokenCount" in usage else 0
    known_balance = known_balance and tool is not None
    unaccounted = None
    if known_balance:
        unaccounted = (counts["totalTokenCount"]-counts["promptTokenCount"]
                       -counts["candidatesTokenCount"]-tool)
    if present:
        value = counts["thoughtsTokenCount"]
        if value is None:
            return result
        result["thoughts_token_count"] = value
        result["thoughts_evidence"] = "explicit_zero" if value == 0 else "explicit_positive"
        result["zero_thoughts_verified"] = bool(
            value == 0 and not malformed and not nonempty_parts
            and (unaccounted is None or unaccounted == 0)
        )
    elif not malformed and known_balance and unaccounted == 0:
        result.update(thoughts_token_count=0, thoughts_evidence="usage_balance_zero",
                      zero_thoughts_verified=not nonempty_parts)
    return result


def _reported_model(raw, requested):
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("Il modello Google richiesto deve essere esplicito")
    version = raw.get("modelVersion")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("La risposta Google non contiene modelVersion")
    if version == requested:
        return version, "exact"
    if version.startswith(requested + "-") and len(version) > len(requested) + 1:
        return version, "configured_prefix_suffix"
    raise ValueError("Google ha restituito una versione estranea al modello richiesto")


def normalize_google_response(raw, model):
    """Conserva il modello effettivo e prepara il formato del parser JSON comune.

    La normalizzazione ignora i marcatori di pensiero vuoti e può rimuovere
    un solo involucro Markdown. Non corregge né completa voti o JSON: il parser
    del pannello continua ad accettare soltanto i quattro interi previsti.
    """
    if not isinstance(raw, dict):
        raise ValueError("La risposta Google deve essere un oggetto JSON")
    version, match_rule = _reported_model(raw, model)
    feedback = raw.get("promptFeedback")
    if feedback is not None and not isinstance(feedback, dict):
        raise ValueError("I metadati di sicurezza Google non hanno il formato previsto")
    if isinstance(feedback, dict) and feedback.get("blockReason") not in {
            None, "", "BLOCK_REASON_UNSPECIFIED"}:
        raise ValueError("Google ha bloccato il prompt")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("Il giudice Google deve restituire un solo candidato completo")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise ValueError("La risposta Google è bloccata, incompleta o troncata")
    for container in (feedback or {}, candidate):
        safety = container.get("safetyRatings", [])
        if not isinstance(safety, list) or any(
                isinstance(item, dict) and item.get("blocked") is True for item in safety):
            raise ValueError("Google ha segnalato un blocco di sicurezza")
    if candidate.get("refusal") or raw.get("refusal"):
        raise ValueError("Google ha restituito un rifiuto invece dei voti")
    evidence = inspect_google_thinking(raw)
    if not evidence["zero_thoughts_verified"]:
        raise ValueError("I metadati Google non verificano zero token di pensiero")
    content = candidate.get("content")
    if not isinstance(content, dict) or content.get("role", "model") != "model":
        raise ValueError("Il candidato Google non contiene un messaggio del modello")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Il candidato Google non contiene parti di testo")
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Una parte Google non ha il formato previsto")
        if "thought" in part and type(part["thought"]) is not bool:
            raise ValueError("L'indicatore di pensiero Google non è booleano")
        if any(key in part for key in _NON_TEXT_PART_FIELDS):
            raise ValueError("Sono ammessi soltanto voti testuali, senza strumenti o dati multimediali")
        if part.get("thought") is True:
            if _empty_thought_part(part):
                continue
            raise ValueError("Il candidato Google contiene una parte di pensiero non vuota o malformata")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Una parte del candidato Google non contiene testo")
        texts.append(text)
    text = "".join(texts)
    if not text.strip():
        raise ValueError("Il candidato Google contiene soltanto testo vuoto")
    text, text_normalization = _normalize_json_code_fence(text)
    usage = raw["usageMetadata"]
    counts = {key: _token_count(usage.get(key)) for key in _COUNTERS}
    if any(counts[key] is None for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")):
        raise ValueError("Mancano contatori Google interi per prompt, candidato o totale")
    return dict(
        model=model, provider_model_version=version, provider_model_match_rule=match_rule,
        response_text_normalization=text_normalization,
        choices=[dict(finish_reason="stop", message=dict(content=text))],
        usage=dict(prompt_tokens=counts["promptTokenCount"],
                   completion_tokens=counts["candidatesTokenCount"],
                   total_tokens=counts["totalTokenCount"], thoughts_tokens=evidence["thoughts_token_count"],
                   tool_use_prompt_tokens=counts["toolUsePromptTokenCount"]
                   if "toolUsePromptTokenCount" in usage else 0),
        google_thinking_evidence=evidence,
    )
