import pytest

from .conftest import ADMIN, ENROLL
from .test_onboarding import request_body


@pytest.fixture(autouse=True)
def clear_configured_url(monkeypatch):
    monkeypatch.delenv("VLA_AGENT_CONTROL_PLANE_URL", raising=False)


def without_url():
    body = request_body()
    del body["control_plane_url"]
    return body


def test_defaults_require_admin_and_do_not_return_credentials(platform, monkeypatch):
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "https://training.example.com")
    node = platform.register()
    for headers in ({}, ENROLL, node["headers"], {"Authorization": "Bearer wrong"}):
        response = platform.client.get("/api/v1/agent-config-defaults", headers=headers)
        assert response.status_code == 401
    response = platform.client.get("/api/v1/agent-config-defaults", headers=ADMIN)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert platform.settings.admin_token not in response.text
    assert platform.settings.enrollment_token not in response.text


def test_configured_url_wins_over_public_request_origin_and_is_used_for_omitted_input(platform, monkeypatch):
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "https://agent-entry.example.com/control/")
    defaults = platform.client.get("https://browser.example.com/api/v1/agent-config-defaults", headers=ADMIN).json()
    assert defaults["control_plane_url"] == "https://agent-entry.example.com/control"
    assert defaults["source"] == "configured"
    response = platform.client.post("https://browser.example.com/api/v1/agent-configs", headers=ADMIN, json=without_url())
    assert response.status_code == 201, response.text
    assert response.json()["config"]["control_plane_url"] == defaults["control_plane_url"]


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000/", "http://localhost:18000", "http://[::1]:18000"])
def test_configured_tunnel_is_explicit_and_its_prerequisite_is_explained(platform, monkeypatch, url):
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", url)
    response = platform.client.get("/api/v1/agent-config-defaults", headers=ADMIN)
    assert response.status_code == 200
    assert response.json()["source"] == "configured"
    assert response.json()["control_plane_url"] == url.rstrip("/")
    assert "隧道" in response.json()["message"]
    assert "目标机器" in response.json()["message"]


def test_public_https_origin_is_inferred_and_applied_when_url_is_omitted(platform):
    defaults = platform.client.get("https://training.example.com:8443/api/v1/agent-config-defaults", headers=ADMIN).json()
    assert defaults["source"] == "request"
    assert defaults["control_plane_url"] == "https://training.example.com:8443"
    response = platform.client.post("https://training.example.com:8443/api/v1/agent-configs", headers=ADMIN, json=without_url())
    assert response.status_code == 201, response.text
    assert response.json()["config"]["control_plane_url"] == defaults["control_plane_url"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8000", "http://training.example.com:8000", "https://localhost",
                                    "https://agent.localhost", "https://127.0.0.1", "https://127.0.0.2",
                                    "https://[::1]", "https://0.0.0.0", "https://[::]", "https://10.1.2.3"])
def test_non_public_https_origin_does_not_guess_agent_address(platform, origin):
    response = platform.client.get(origin + "/api/v1/agent-config-defaults", headers=ADMIN)
    assert response.status_code == 200
    assert response.json()["source"] == "unavailable"
    assert response.json()["control_plane_url"] is None
    response = platform.client.post(origin + "/api/v1/agent-configs", headers=ADMIN, json=without_url())
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "AGENT_CONTROL_PLANE_URL_REQUIRED"
    assert platform.settings.enrollment_token not in response.text


def test_explicit_advanced_override_is_still_used(platform, monkeypatch):
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "https://standard.example.com")
    body = request_body()
    body["control_plane_url"] = "http://127.0.0.1:18000"
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    assert response.json()["config"]["control_plane_url"] == body["control_plane_url"]


@pytest.mark.parametrize("invalid", ["https://user:secret-value@example.com", "http://public.example.com", "   ", "ftp://example.com"])
def test_invalid_platform_configuration_reports_safe_error(platform, monkeypatch, invalid):
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", invalid)
    for response in (
        platform.client.get("/api/v1/agent-config-defaults", headers=ADMIN),
        platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=without_url()),
    ):
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "AGENT_DEFAULT_URL_INVALID"
        assert "VLA_AGENT_CONTROL_PLANE_URL" in response.json()["error"]["message"]
        assert invalid not in response.text
        assert platform.settings.enrollment_token not in response.text
        assert "secret-value" not in response.text


def test_forwarded_headers_do_not_create_an_automatic_public_origin(platform):
    headers = {**ADMIN, "X-Forwarded-Host": "training.example.com", "X-Forwarded-Proto": "https",
               "Forwarded": "proto=https;host=training.example.com"}
    response = platform.client.get("http://127.0.0.1:8000/api/v1/agent-config-defaults", headers=headers)
    assert response.status_code == 200
    assert response.json()["source"] == "unavailable"
    assert response.json()["control_plane_url"] is None


def test_public_origin_ignores_conflicting_forwarded_headers(platform):
    response = platform.client.get("https://training.example.com/api/v1/agent-config-defaults",
                                   headers={**ADMIN, "X-Forwarded-Host": "attacker.example.com", "X-Forwarded-Proto": "http"})
    assert response.status_code == 200
    assert response.json()["source"] == "request"
    assert response.json()["control_plane_url"] == "https://training.example.com"
