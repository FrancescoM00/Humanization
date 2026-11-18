"""Controlla il trasporto Google con risposte locali e senza credenziali reali."""
from copy import deepcopy
import json

import pytest
import requests

from humanizers.naturalness import APIError
from humanizers.naturalness_google import GOOGLE_API_ROOT, GOOGLE_MODEL, GoogleGemmaClient


KEY = "AIza" + "x" * 30


class Response:
    def __init__(self, value=None, *, status=200, text=None, headers=None, invalid_json=False):
        self.value = {} if value is None else value
        self.status_code = status
        self.text = json.dumps(self.value) if text is None else text
        self.headers = {} if headers is None else headers
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("Risposta locale non JSON")
        return deepcopy(self.value)


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, deepcopy(kwargs)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(responses):
    return GoogleGemmaClient(api_key=KEY, session=Session(responses))


def model(name=GOOGLE_MODEL, methods=None):
    return dict(name="models/" + name,
                supportedGenerationMethods=["generateContent"] if methods is None else methods)


def payload():
    return dict(contents=[dict(parts=[dict(text="Rate these four passages.")])],
                generationConfig=dict(temperature=0, maxOutputTokens=256,
                                      thinkingConfig=dict(thinkingLevel="minimal")))


def test_generation_uses_fixed_google_host_header_and_unchanged_native_payload():
    raw = dict(candidates=[dict(content=dict(parts=[dict(text='{"ratings": {}}')]), finishReason="STOP")],
               usageMetadata=dict(promptTokenCount=50, candidatesTokenCount=9, thoughtsTokenCount=0),
               modelVersion=GOOGLE_MODEL)
    api = client([Response(raw)])
    value = payload()
    before = deepcopy(value)
    assert api.request("POST", "/chat/completions", value) == raw
    assert value == before
    method, url, options = api.session.calls[0]
    assert method == "POST" and url == GOOGLE_API_ROOT + "/models/" + GOOGLE_MODEL + ":generateContent"
    assert KEY not in url and options["params"] is None
    assert options["headers"]["x-goog-api-key"] == KEY
    assert "Authorization" not in options["headers"]
    assert options["allow_redirects"] is False and options["timeout"] == 120
    assert options["json"] == before


def test_catalog_follows_page_tokens_and_check_model_never_posts():
    api = client([
        Response(dict(models=[model("another-model")], nextPageToken="next-page")),
        Response(dict(models=[model()])),
    ])
    result = api.check_model(GOOGLE_MODEL)
    assert result["provider"] == "google" and result["thinking_level"] == "minimal"
    assert result["supports_generate_content"] is True
    assert "non certifica" in result["no_thinking_option"]
    assert len(api.session.calls) == 2
    for method, url, options in api.session.calls:
        assert method == "GET" and url == GOOGLE_API_ROOT + "/models"
        assert options["allow_redirects"] is False
        assert KEY not in url and "key" not in options["params"]
    assert api.session.calls[0][2]["params"] == {"pageSize": 1000}
    assert api.session.calls[1][2]["params"] == {"pageSize": 1000, "pageToken": "next-page"}


def test_unavailable_model_stops_without_inference_or_fallback():
    api = client([Response(dict(models=[model("another-model")]))])
    with pytest.raises(ValueError, match="nessun fallback"):
        api.check_model(GOOGLE_MODEL)
    assert len(api.session.calls) == 1 and api.session.calls[0][0] == "GET"


def test_model_without_generate_content_is_not_an_acceptable_judge():
    api = client([Response(dict(models=[model(methods=["countTokens"])]))])
    with pytest.raises(ValueError, match="generateContent"):
        api.check_model(GOOGLE_MODEL)
    assert len(api.session.calls) == 1


def test_catalog_rejects_repeated_page_tokens_before_an_unbounded_loop():
    api = client([
        Response(dict(models=[model("first")], nextPageToken="repeated")),
        Response(dict(models=[model("second")], nextPageToken="repeated")),
    ])
    with pytest.raises(ValueError, match="non avanza"):
        api.check_model(GOOGLE_MODEL)
    assert len(api.session.calls) == 2


@pytest.mark.parametrize("entries", [[model(), model()], [dict(description="missing-name")], "not-a-list"])
def test_catalog_rejects_ambiguous_or_malformed_model_records(entries):
    api = client([Response(dict(models=entries))])
    with pytest.raises(ValueError):
        api.check_model(GOOGLE_MODEL)


@pytest.mark.parametrize("method,endpoint", [
    ("GET", "https://example.com/models"), ("POST", "/models"),
    ("GET", "/generateContent"), ("DELETE", "/models"),
])
def test_unexpected_endpoints_do_not_receive_credentials(method, endpoint):
    api = client([])
    with pytest.raises(ValueError, match="Endpoint"):
        api.request(method, endpoint, payload())
    assert api.session.calls == []


def test_wrong_model_is_rejected_before_network():
    with pytest.raises(ValueError, match="esattamente"):
        GoogleGemmaClient(api_key=KEY, model="another-model")
    api = client([])
    with pytest.raises(ValueError, match="esattamente"):
        api.check_model("another-model")
    assert api.session.calls == []


def test_empty_secret_is_not_accepted_and_environment_secret_is_used(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GoogleGemmaClient()
    with pytest.raises(ValueError):
        GoogleGemmaClient(api_key="  ")
    monkeypatch.setenv("GOOGLE_API_KEY", " " + KEY + " ")
    api = GoogleGemmaClient(session=Session([Response({})]))
    api.request("POST", "/generateContent", payload())
    assert api.session.calls[0][2]["headers"]["x-goog-api-key"] == KEY


def test_http_error_preserves_status_and_wait_but_redacts_credential():
    api = client([Response(dict(error=dict(status="UNAVAILABLE", code=503, message="Overloaded " + KEY)),
                           status=503, headers={"Retry-After": "20"})])
    with pytest.raises(APIError) as raised:
        api.request("POST", "/generateContent", payload())
    assert raised.value.status == 503 and raised.value.retry_after == "20"
    assert KEY not in str(raised.value) and "[REDACTED]" in str(raised.value)


def test_google_retry_info_is_used_when_header_is_absent():
    api = client([Response(dict(error=dict(code=429, message="Quota exhausted", details=[
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42.5s"}])), status=429)])
    with pytest.raises(APIError) as raised:
        api.request("POST", "/generateContent", payload())
    assert raised.value.status == 429 and raised.value.retry_after == "42.5"


def test_redirect_is_not_followed_or_treated_as_a_model_response():
    api = client([Response({}, status=302, text="<p>Redirect " + KEY + "</p>",
                           headers={"Location": "https://example.com"})])
    with pytest.raises(APIError) as raised:
        api.request("POST", "/generateContent", payload())
    assert raised.value.status == 302 and KEY not in str(raised.value)
    assert len(api.session.calls) == 1 and api.session.calls[0][2]["allow_redirects"] is False


def test_network_exception_does_not_expose_credentials():
    api = client([requests.ConnectionError("Network request with " + KEY)])
    with pytest.raises(APIError) as raised:
        api.request("POST", "/generateContent", payload())
    assert raised.value.status == 0 and KEY not in str(raised.value)


@pytest.mark.parametrize("response", [Response(invalid_json=True), Response([1, 2])])
def test_non_object_or_invalid_json_is_rejected(response):
    api = client([response])
    with pytest.raises(APIError, match="JSON"):
        api.request("POST", "/generateContent", payload())


def test_empty_candidate_data_are_preserved_for_the_judgment_parser():
    raw = dict(candidates=[], promptFeedback=dict(blockReason="SAFETY"), usageMetadata={})
    api = client([Response(raw)])
    assert api.request("POST", "/generateContent", payload()) == raw
