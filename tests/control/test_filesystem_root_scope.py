"""Whole-machine browsing does not make Agent-owned directories filesystem-wide."""

import json

import pytest

from vla_control.common import APIError
from vla_control.remote_deployment import DeploymentManager
from vla_control.ssh_remote import RemoteError, SSHRemote

from .conftest import ADMIN, ENROLL
from .test_onboarding import request_body
from .test_path_suggestions import ack as path_ack, create as path_create, get as path_get, result as path_result
from .test_ssh_remote import uninstall_script_fixture


def root_scope_config(platform):
    body = request_body()
    body["roots"] = ["/", "/.", "//"]
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    config = response.json()["config"]
    assert config["roots"] == ["/"]
    assert config["state_dir"] == "/data2/vla/state" and config["output_root"] == "/data2/vla/outputs"
    registration = {key: config[key] for key in ("name", "roots", "output_root", "profiles")}
    response = platform.client.post("/agent/v1/register", headers=ENROLL, json={**registration, "hostname": "root-scoped"})
    assert response.status_code == 201, response.text
    node = response.json()
    node["headers"] = {"Authorization": "Bearer " + node["token"]}
    platform.heartbeat(node, gpus=[])
    return node


def test_explicit_filesystem_root_survives_onboarding_registration_assets_and_path_browsing(platform):
    node = root_scope_config(platform)
    for path in ("/data1/demo", "/mnt/datasets/demo", "/home/research/demo"):
        response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
            "node_id": node["node_id"], "name": path, "kind": "DATASET", "path": path})
        assert response.status_code == 201 and response.json()["path"] == path
    for prefix, parent, entries in (("/", "/", [{"name": "mnt", "path": "/mnt/", "kind": "directory"}]),
                                   ("/mnt/", "/mnt", [{"name": "datasets", "path": "/mnt/datasets/", "kind": "directory"}])):
        query = path_create(platform, node, prefix=prefix)
        assert query.status_code == 202, query.text
        query_id = query.json()["id"]
        assert path_ack(platform, node, query_id, path_result(prefix=prefix, parent=parent, entries=entries)).status_code == 200
        checked = path_get(platform, node, query_id).json()
        assert checked["status"] == "READY" and checked["entries"] == entries


@pytest.mark.parametrize("role", ["state_dir", "output_root", "profiles.code_dir", "profiles.python"])
@pytest.mark.parametrize("path", ["/", "/.", "//"])
def test_only_browsing_roots_may_use_filesystem_root(platform, role, path):
    body = request_body()
    body["roots"] = ["/"]
    if role.startswith("profiles."):
        body["profiles"][0][role.split(".")[1]] = path
    else:
        body[role] = path
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_AGENT_PATH"


@pytest.mark.parametrize("path", ["/opt", "/usr", "/root", "/var", "/home", "/etc", "/tmp"])
def test_state_system_directories_are_rejected_before_deployment(platform, path):
    body = request_body()
    body["roots"] = ["/"]
    body["state_dir"] = path
    response = platform.client.post("/api/v1/agent-configs", headers=ADMIN, json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_AGENT_PATH"


@pytest.mark.parametrize("path", ["/", "/.", "//"])
def test_install_prefix_stays_dedicated_when_browsing_root_is_slash(tmp_path, path):
    with pytest.raises(APIError):
        DeploymentManager._install_path(path)
    remote = SSHRemote(host="host", port=22, username="root", password="test-only", state_dir=tmp_path)
    with pytest.raises(RemoteError, match="独立|简单|目录"):
        remote.probe(path)


def test_uninstall_scopes_private_components_even_with_whole_machine_browsing(tmp_path, monkeypatch):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    config["roots"] = ["/"]
    config_path.write_text(json.dumps(config))
    unrelated = tmp_path / "training-data"
    unrelated.mkdir()
    data = unrelated / "keep.txt"
    data.write_text("preserve training data")
    result = namespace["uninstall_agent"](str(prefix), unit_dir)
    assert "/" not in result["removed_paths"]
    assert data.read_text() == "preserve training data"
    assert (prefix / "runs/result.txt").is_file()
    assert str(unrelated) not in result["removed_paths"]
    assert not system["active"]
