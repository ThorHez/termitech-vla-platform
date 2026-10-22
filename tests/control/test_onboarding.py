from copy import deepcopy
import json
import uuid

import pytest
from sqlalchemy import func, select

from vla_control.models import Node

from .conftest import ADMIN, ENROLL


def request_body():
    return {"name": "GPU worker", "control_plane_url": "https://training.example.com/",
            "roots": ["/data2"], "output_root": "/data2/vla/outputs", "state_dir": "/data2/vla/state",
            "profiles": [{"id": "openpi-sft", "name": "OpenPI environment", "code_dir": "/data2/openpi-umi",
                          "python": "/data2/openpi-umi/.venv/bin/python", "adapter": "openpi-sft",
                          "recipe": "umi_sft_recipe", "execution_mode": "native"}]}


def test_generated_config_has_explicit_training_python_and_no_node_side_effects(platform):
    # This path deliberately does not exist on the control server: paths are for
    # the target machine and this endpoint must neither probe nor fabricate it.
    body = request_body()
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    assert response.headers["Cache-Control"] == "no-store"
    generated = response.json()
    assert str(uuid.UUID(generated["id"])) == generated["id"]
    assert generated["filename"] == f"vla-agent-{generated['id']}.json"
    config = generated["config"]
    assert config["control_plane_url"] == "https://training.example.com"
    assert config["enrollment_token"] == platform.settings.enrollment_token
    assert config["poll_interval_seconds"] == 2 and config["heartbeat_interval_seconds"] == 10
    assert config["profiles"][0]["python"] == body["profiles"][0]["python"]
    assert config["profiles"][0]["metadata"] == {"validated": False, "onboarding_id": generated["id"]}
    assert platform.settings.admin_token not in json.dumps(generated)
    with platform.app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(Node)) == 0

    # A real Agent can register the generated profile using the existing API.
    register_body = {key: config[key] for key in ("name", "roots", "output_root", "profiles")}
    register_body["hostname"] = "actual-worker"
    registration = platform.client.post("/agent/v1/register", headers={**ENROLL, "Idempotency-Key": "real-worker"}, json=register_body)
    assert registration.status_code == 201, registration.text
    profiles = platform.client.get("/api/v1/profiles", headers=ADMIN).json()
    assert profiles[0]["metadata"]["onboarding_id"] == generated["id"]
    assert profiles[0]["metadata"]["validated"] is False


def test_config_requires_admin_and_never_leaks_tokens_on_errors(platform):
    node = platform.register()
    for headers in ({}, ENROLL, node["headers"], {"Authorization": "Bearer wrong"}):
        response = platform.client.post("/api/v1/agent-configs", headers=headers, json=request_body())
        assert response.status_code == 401
        assert platform.settings.enrollment_token not in response.text
        assert platform.settings.admin_token not in response.text


@pytest.mark.parametrize("field", ["enrollment_token", "admin_token", "ssh_password", "metadata"])
def test_browser_cannot_supply_secrets_or_trusted_fields(platform, field):
    body = request_body()
    body[field] = "browser-value"
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "browser-value" not in response.text


@pytest.mark.parametrize("field,value", [("metadata", {"validated": True}), ("docker_image", "image:latest"),
                                        ("execution_mode", "docker"), ("adapter", "arbitrary-executable"),
                                        ("id", "../../escape"), ("python", ""),
                                        ("recipe", "umi.sft.recipe")])
def test_profile_has_only_explicit_supported_inputs(platform, field, value):
    body = request_body()
    body["profiles"][0][field] = value
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("value", [None, "", "  ", "omitted"])
def test_recipe_is_not_required_to_onboard_and_register_runtime(platform, value):
    body = request_body()
    if value == "omitted":
        del body["profiles"][0]["recipe"]
    else:
        body["profiles"][0]["recipe"] = value
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    generated = response.json()
    config = generated["config"]
    assert config["profiles"][0]["recipe"] is None
    assert config["profiles"][0]["metadata"]["validated"] is False
    registration = {key: config[key] for key in ("name", "roots", "output_root", "profiles")}
    registration["hostname"] = "worker-without-recipe"
    result = platform.client.post("/agent/v1/register", headers=ENROLL, json=registration)
    assert result.status_code == 201, result.text
    profiles = platform.client.get("/api/v1/profiles", headers=ADMIN).json()
    assert profiles[0]["recipe"] is None


def test_legacy_named_recipe_is_preserved_when_supplied(platform):
    body = request_body()
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201
    assert response.json()["config"]["profiles"][0]["recipe"] == "umi_sft_recipe"


@pytest.mark.parametrize("path", ["relative/python", "~/venv/bin/python", "/data2/../usr/python", "/data2/py\nthon", "/data2/py\x00thon"])
def test_training_python_must_be_absolute_and_without_traversal_or_controls(platform, path):
    body = request_body()
    body["profiles"][0]["python"] = path
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_AGENT_PATH"


@pytest.mark.parametrize("url", ["http://public.example.com:8000", "https://user:secret@example.com", "https://example.com?token=x",
                                "https://example.com#fragment", "ftp://example.com", "https://example.com:99999",
                                "https://example.com:0", "https://example.com/../prefix", "https://exa mple.com", "https://example.com\n"])
def test_invalid_urls_cannot_leak_enrollment_credentials(platform, url):
    body = request_body()
    body["control_plane_url"] = url
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422, response.text
    assert platform.settings.enrollment_token not in response.text


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:18000/", "http://[::1]:8000", "https://vla.example.com/prefix/"])
def test_loopback_tunnels_and_https_are_accepted(platform, url):
    body = request_body()
    body["control_plane_url"] = url
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    assert response.json()["config"]["control_plane_url"] == url.rstrip("/")


@pytest.mark.parametrize("field,value,code", [("roots", ["data2"], "INVALID_AGENT_PATH"),
                                             ("output_root", "/outside", "PATH_OUTSIDE_ROOTS"),
                                             ("state_dir", "/data2/openpi-umi/state", "AGENT_PATH_OVERLAP"),
                                             ("state_dir", "/data2/vla/outputs/state", "AGENT_PATH_OVERLAP"),
                                             ("output_root", "/data2/openpi-umi/checkpoints", "AGENT_PATH_OVERLAP")])
def test_directories_are_managed_and_separated(platform, field, value, code):
    body = request_body()
    body[field] = value
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_profiles_must_be_unique_nonempty_and_within_roots(platform):
    body = request_body()
    body["profiles"] = []
    assert platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body).status_code == 422
    body = request_body()
    body["profiles"].append(deepcopy(body["profiles"][0]))
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.json()["error"]["code"] == "DUPLICATE_PROFILE"
    body = request_body()
    body["profiles"][0]["code_dir"] = "/data20/openpi"
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.json()["error"]["code"] == "PATH_OUTSIDE_ROOTS"


def test_normalization_and_multiple_runtime_profiles(platform):
    body = request_body()
    body["roots"] = ["/data2/", "/data2/."]
    body["profiles"][0]["code_dir"] = "//data2//openpi-umi/."
    smoke = {**body["profiles"][0], "id": "smoke", "adapter": "smoke", "recipe": "smoke"}
    body["profiles"].append(smoke)
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    generated = response.json()
    assert generated["config"]["roots"] == ["/data2"]
    assert generated["config"]["profiles"][0]["code_dir"] == "/data2/openpi-umi"
    assert generated["config"]["profiles"][1]["metadata"] == {"validated": False, "simulation": True, "onboarding_id": generated["id"]}


def test_agent_package_requires_admin_and_missing_package_is_clear(platform, monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_AGENT_BUNDLE", str(tmp_path / "absent.tar.gz"))
    for headers in ({}, ENROLL):
        assert platform.client.get("/api/v1/agent-package", headers=headers).status_code == 401
    response = platform.client.get("/api/v1/agent-package", headers=ADMIN)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AGENT_PACKAGE_UNAVAILABLE"
    assert "build" in response.json()["error"]["message"]
    assert platform.settings.enrollment_token not in response.text


def test_agent_package_serves_fixed_administrator_artifact_without_caching(platform, monkeypatch, tmp_path):
    package = tmp_path / "agent-bundle.tar.gz"
    package.write_bytes(b"administratively-built-bundle")
    monkeypatch.setenv("VLA_AGENT_BUNDLE", str(package))
    # User-controlled paths are not endpoint inputs and cannot select a file.
    response = platform.client.get("/api/v1/agent-package?path=/etc/passwd", headers=ADMIN)
    assert response.status_code == 200
    assert response.content == package.read_bytes()
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Content-Type"] == "application/gzip"
    assert 'filename="vla-agent-bundle.tar.gz"' in response.headers["Content-Disposition"]
    assert platform.settings.enrollment_token.encode() not in response.content
