"""Credential-bearing redirects are exercised through urllib's real pipeline.

Only the protocol transport is replaced; response/error processors and redirect
handlers remain real. No socket, external callback or actual token is used.
"""

from email.message import Message
import io
import json
import urllib.error
import urllib.request
import urllib.response

import pytest

from vla_agent.main import Client


ORIGIN = "https://controller.invalid"
TOKEN = "synthetic-transport-proof-nonsecret"


def response(request, status, payload=None, location=None):
    headers = Message()
    if location:
        headers["Location"] = location
    value = urllib.response.addinfourl(io.BytesIO(json.dumps(payload or {}).encode()),
                                      headers, request.full_url, status)
    value.msg = "Found" if status != 200 else "OK"
    return value


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("destination", [
    "https://untrusted.invalid/collect",
    "http://untrusted.invalid/collect",
    "http://127.0.0.1:65534/collect",
    ORIGIN + "/another-endpoint",
])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_agent_rejects_every_redirect_without_forwarding_token(monkeypatch, status, destination, method):
    observed = []

    def send(_transport, request):
        observed.append(request)
        return response(request, status, location=destination) if len(observed) == 1 else response(request, 200)

    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", send)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", send)
    client = Client({"control_plane_url": ORIGIN, "http_timeout_seconds": 2})
    body = {"node_id": "synthetic"} if method == "POST" else None
    with pytest.raises(urllib.error.HTTPError) as rejected:
        client.request(method, "/agent/v1/heartbeat" if body else "/agent/v1/commands", TOKEN, body)
    assert rejected.value.code == status
    assert len(observed) == 1, "Authorization must never reach a redirect destination"
    assert observed[0].get_header("Authorization") == "Bearer " + TOKEN
    assert observed[0].full_url.startswith(ORIGIN + "/agent/v1/")


@pytest.mark.parametrize("origin", [ORIGIN, "http://127.0.0.1:18001"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_stable_agent_endpoints_keep_json_auth_body_timeout_and_idempotency(monkeypatch, origin, method):
    observed = []

    def send(_transport, request):
        observed.append(request)
        return response(request, 200, {"ok": True})

    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", send)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", send)
    client = Client({"control_plane_url": origin, "http_timeout_seconds": 2})
    body = {"name": "synthetic-machine"} if method == "POST" else None
    assert client.request(method, "/agent/v1/register" if body else "/agent/v1/commands", TOKEN, body,
                          {"Idempotency-Key": "synthetic-stable-key"}) == {"ok": True}
    assert len(observed) == 1
    request = observed[0]
    assert request.get_method() == method
    assert request.get_header("Authorization") == "Bearer " + TOKEN
    assert request.get_header("Idempotency-key") == "synthetic-stable-key"
    assert request.timeout == 2
    assert json.loads(request.data) == body if body is not None else request.data is None


def test_agent_remote_plaintext_is_still_rejected_before_transport():
    with pytest.raises(ValueError, match="require HTTPS"):
        Client({"control_plane_url": "http://controller.invalid", "http_timeout_seconds": 2})
