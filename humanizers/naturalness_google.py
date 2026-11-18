"""Accede al solo endpoint Google dichiarato, senza sostituire il giudice."""
from __future__ import annotations

import json
import os
import re

from .naturalness import APIError


GOOGLE_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GOOGLE_MODEL = "gemma-4-31b-it"


def _redact(value, key):
    """Rimuove la credenziale anche da messaggi restituiti dal provider."""
    text = str(value).replace(key, "[REDACTED]")
    text = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
    return " ".join(text.split())[:800]


def _provider_detail(response, key):
    """Legge un dettaglio limitato, senza riportare intestazioni o chiavi."""
    body = response.text[:16384]
    try:
        data = json.loads(body)
        error = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(error, dict):
            detail = "; ".join(f"{name}: {error[name]}"
                               for name in ("status", "code", "message") if error.get(name))
        else:
            detail = str(error)
    except (TypeError, ValueError):
        detail = re.sub(r"<[^>]*>", " ", body)
    return _redact(detail, key) or "Nessun dettaglio restituito dal provider"


def _retry_after(response):
    """Conserva l'attesa HTTP o quella comunicata nei dettagli Google."""
    if response.headers.get("Retry-After") is not None:
        return response.headers["Retry-After"]
    try:
        data = json.loads(response.text[:16384])
        error = data.get("error", {}) if isinstance(data, dict) else {}
        details = error.get("details", []) if isinstance(error, dict) else []
        if not isinstance(details, list):
            return None
        for entry in details:
            if not isinstance(entry, dict):
                continue
            delay = entry.get("retryDelay")
            if isinstance(delay, str) and re.fullmatch(r"\d+(?:\.\d+)?s", delay):
                return delay[:-1]
    except (TypeError, ValueError):
        pass
    return None


class GoogleGemmaClient:
    """Invia richieste native Google usando la chiave soltanto nell'header."""

    provider = "google"

    def __init__(self, api_key=None, *, model=GOOGLE_MODEL, timeout=120, session=None):
        import requests

        if model != GOOGLE_MODEL:
            raise ValueError(f"Il client richiede esattamente {GOOGLE_MODEL}; nessun fallback")
        key = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Impostare GOOGLE_API_KEY nei Secrets, senza incollarla nei risultati")
        self._key = key.strip()
        self.model = model
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()

    def _request_json(self, method, path, *, payload=None, params=None):
        import requests

        try:
            response = self.session.request(
                method, GOOGLE_API_ROOT + path, json=payload, params=params,
                headers={"x-goog-api-key": self._key,
                         "Content-Type": "application/json", "Accept": "application/json"},
                timeout=self.timeout, allow_redirects=False,
            )
        except requests.RequestException:
            raise APIError(0, "Richiesta Google non riuscita; nessun cambio implicito di provider o modello") from None
        if not 200 <= response.status_code < 300:
            phase = "catalogo" if method == "GET" else "generazione"
            detail = _provider_detail(response, self._key)
            raise APIError(response.status_code, f"Google HTTP {response.status_code} durante {phase}. {detail}",
                           _retry_after(response))
        try:
            data = response.json()
        except ValueError:
            raise APIError(response.status_code, "Google ha restituito JSON non valido") from None
        if not isinstance(data, dict):
            raise APIError(response.status_code, "Google non ha restituito un oggetto JSON")
        return data

    def _catalog(self):
        """Segue solo i token di pagina, mantenendo fisso l'host del catalogo."""
        models, page_token, seen_tokens, seen_names = [], None, set(), set()
        for _ in range(100):
            params = {"pageSize": 1000}
            if page_token is not None:
                params["pageToken"] = page_token
            page = self._request_json("GET", "/models", params=params)
            entries = page.get("models", [])
            if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
                raise ValueError("Il catalogo Google non contiene una lista di modelli valida")
            for item in entries:
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Un modello Google non contiene il nome")
                if name in seen_names:
                    raise ValueError("Il catalogo Google ripete un nome di modello")
                seen_names.add(name)
                models.append(item)
            next_token = page.get("nextPageToken")
            if next_token is None or next_token == "":
                return {"models": models}
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise ValueError("La paginazione del catalogo Google non avanza")
            seen_tokens.add(next_token)
            page_token = next_token
        raise ValueError("Il catalogo Google supera il numero massimo di pagine previsto")

    def request(self, method, endpoint, payload=None):
        """Mantiene il payload nativo intatto e restituisce la risposta grezza."""
        if method == "GET" and endpoint == "/models" and payload is None:
            return self._catalog()
        if method == "POST" and endpoint in {"/chat/completions", "/generateContent"}:
            if not isinstance(payload, dict):
                raise ValueError("La richiesta Google richiede un payload nativo JSON")
            return self._request_json("POST", f"/models/{self.model}:generateContent", payload=payload)
        raise ValueError("Endpoint Google non previsto")

    def check_model(self, model):
        """Controlla nome e metodo; il catalogo non certifica l'assenza di pensiero."""
        if model != self.model:
            raise ValueError(f"Il confronto richiede esattamente {self.model}; nessun fallback")
        entries = self.request("GET", "/models")["models"]
        selected = [item for item in entries if item["name"] in {model, "models/" + model}]
        if len(selected) != 1:
            available = sorted(item["name"] for item in entries)
            raise ValueError(f"Il modello {model!r} non è presente una sola volta nel catalogo Google; "
                             f"nessun fallback. Nomi disponibili: {available}")
        methods = selected[0].get("supportedGenerationMethods", [])
        if not isinstance(methods, list) or "generateContent" not in methods:
            raise ValueError("Il modello Google selezionato non dichiara generateContent")
        return dict(
            provider="google", requested_model=model, present_in_authenticated_catalog=True,
            supports_generate_content=True, thinking_level="minimal",
            no_thinking_option=("thinkingConfig.thinkingLevel=minimal non certifica il pensiero spento; "
                                "i token di pensiero devono essere verificati in ogni risposta"),
        )
