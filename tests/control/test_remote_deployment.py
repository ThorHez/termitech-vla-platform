"""Remote jobs are accepted only after SSH proof and authenticated Agent proof."""

import json
import stat
import time

import pytest
from sqlalchemy import select

from vla_control.common import now, uid
from vla_control.models import Asset, Command, Node, Profile, Run
from vla_control.remote_deployment import DeploymentManager
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.ssh_remote import RemoteError

from .conftest import ADMIN


PASSWORD = "unit-password-never-public"


class FakeRemote:
    probe_result = {"hostname": "gpu-host", "checks": [{"name": "systemd", "ok": True, "message": "ready"}], "can_install": True}
    error = None
    installer = None
    uninstall_result = {"backup_dir": "/data2/vla-agent-backup/20261015", "removed_paths": ["/data2/vla-agent/.venv"], "service_removed": True}
    calls = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.fingerprint = "SHA256:unit-host-key"
        self.closed = False
        self.calls.append(self)

    def connect(self):
        if self.error:
            raise self.error

    def probe(self, prefix):
        return json.loads(json.dumps(self.probe_result))

    def install(self, bundle, config, prefix, emit):
        assert config["enrollment_token"] == "test-enrollment"
        emit("installing", "Install " + PASSWORD + " test-enrollment")
        if self.installer:
            type(self).installer(config)

    def close(self):
        self.closed = True

    def uninstall(self, prefix, emit):
        emit("UNINSTALLING", "正在卸载 Agent")
        return dict(self.uninstall_result)


class FakeTunnel:
    calls = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.actual_port = kwargs["remote_port"] or 21001 + len(self.calls)
        self.healthy = False
        self.closed = False
        self.calls.append(self)

    def start(self):
        self.healthy = True
        return self.actual_port

    def close(self):
        self.healthy = False
        self.closed = True


@pytest.fixture
def remote(platform, monkeypatch, tmp_path):
    FakeRemote.calls = []
    FakeRemote.error = None
    FakeRemote.installer = None
    FakeRemote.uninstall_result = {"backup_dir": "/data2/vla-agent-backup/20261015", "removed_paths": ["/data2/vla-agent/.venv"], "service_removed": True}
    FakeRemote.probe_result = {"hostname": "gpu-host", "checks": [{"name": "systemd", "ok": True, "message": "ready"}], "can_install": True}
    FakeTunnel.calls = []
    manager = platform.app.state.remote_manager
    manager._remote_factory, manager._tunnel_factory = FakeRemote, FakeTunnel
    manager._launch = lambda job_id: None
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "https://control.example.com")
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"unit-test-package")
    monkeypatch.setenv("VLA_AGENT_BUNDLE", str(bundle))
    return platform, manager


def connection(platform, **changes):
    body = {"host": "192.0.2.169", "port": 22, "username": "root", "password": PASSWORD,
            "install_dir": "/data2/vla-agent", **changes}
    return platform.client.post("/api/v1/ssh-connections/test", headers=ADMIN, json=body)


def payload(connection_id, **agent_changes):
    return {"connection_id": connection_id, "install_dir": "/data2/vla-agent", "agent": {
        "name": "remote-test", "roots": ["/data2"], "state_dir": "/data2/vla-agent/state",
        "output_root": "/data2/vla-agent/outputs", "profiles": [{"id": "smoke", "name": "test runtime",
        "code_dir": "/data2/code", "python": "/data2/code/.venv/bin/python", "adapter": "smoke", "recipe": "smoke"}], **agent_changes}}


def create(platform, body, key="unit-deploy"):
    return platform.client.post("/api/v1/remote-deployments", headers={**ADMIN, "Idempotency-Key": key}, json=body)


def agent_online(platform, config, status="READY", node_id=None, stale=False, wrong_provenance=False):
    node_id = node_id or uid()
    with platform.app.state.db.write() as session:
        node = session.get(Node, node_id)
        timestamp = "2000-01-01T00:00:00.000000+00:00" if stale else now()
        checks = [{"profile_id": p["id"], "status": status} for p in config["profiles"]]
        if node is None:
            node = Node(id=node_id, token_hash=uid(), name="old-name", hostname="gpu-host", status="ONLINE",
                last_seen=timestamp, created_at=now(), roots=["/old"], output_root="/old/output", gpus=[], metrics={"runtime_checks": checks})
            session.add(node)
            session.flush()
        else:
            node.last_seen, node.metrics = timestamp, {"runtime_checks": checks}
            node.status = "ONLINE"
        for p in config["profiles"]:
            profile = session.get(Profile, (node_id, p["id"]))
            metadata = {**p["metadata"], **({"onboarding_id": "wrong"} if wrong_provenance else {})}
            if profile is None:
                session.add(Profile(node_id=node_id, id=p["id"], name=p["name"], code_dir=p["code_dir"], python=p["python"],
                    adapter=p["adapter"], recipe=p["recipe"], execution_mode="native", metadata_json=metadata))
            else:
                profile.metadata_json = metadata
                profile.active = True
    return node_id


def test_ssh_connection_credentials_are_encrypted_and_private(remote):
    platform, manager = remote
    result = connection(platform)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["hostname"] == "gpu-host" and body["can_install"]
    assert body["fingerprint"] == "SHA256:unit-host-key"
    assert PASSWORD not in result.text and "encrypted_password" not in result.text
    with platform.app.state.db.read() as session:
        stored = session.get(RemoteConnection, body["connection_id"])
        assert PASSWORD not in stored.encrypted_password
        assert manager.vault.decrypt(stored.encrypted_password) == PASSWORD
    assert stat.S_IMODE((manager.state_dir / "credentials.key").stat().st_mode) == 0o600
    assert result.headers["cache-control"] == "no-store"


def test_all_remote_routes_require_administrator(remote):
    platform, _ = remote
    assert platform.client.get("/api/v1/remote-deployments").status_code == 401
    assert platform.client.post("/api/v1/ssh-connections/test", json={"host": "192.0.2.1", "username": "root", "password": PASSWORD,
        "install_dir": "/data2/agent"}).status_code == 401
    assert not FakeRemote.calls


def test_ssh_failure_does_not_leak_raw_exception_or_store_credentials(remote):
    platform, _ = remote
    FakeRemote.error = RuntimeError("ssh://root:" + PASSWORD + "@host")
    response = connection(platform)
    assert response.status_code == 422
    assert PASSWORD not in response.text
    assert response.json()["error"]["code"] == "SSH_OPERATION_FAILED"
    with platform.app.state.db.read() as session:
        assert session.scalar(select(RemoteConnection)) is None


@pytest.mark.parametrize("host", ["ssh://host", "host;echo hi", "host/path", "user@host", "-option"])
def test_non_host_input_rejected_before_ssh(remote, host):
    platform, _ = remote
    assert connection(platform, host=host).status_code == 422
    assert not FakeRemote.calls


def test_generation_queued_and_idempotent_without_plaintext_enrollment(remote, monkeypatch):
    platform, _ = remote
    tested = connection(platform).json()
    body = payload(tested["connection_id"])
    response = create(platform, body)
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["state"] == "QUEUED" and job["connection_mode"] == "direct"
    assert job["history"][0]["state"] == "QUEUED"
    assert "test-enrollment" not in response.text and PASSWORD not in response.text
    monkeypatch.delenv("VLA_AGENT_CONTROL_PLANE_URL")
    assert create(platform, body).json()["id"] == job["id"]
    body["agent"]["name"] = "changed"
    assert create(platform, body).status_code == 409
    with platform.app.state.db.read() as session:
        stored = session.get(RemoteDeployment, job["id"])
        assert "enrollment_token" not in stored.agent_config


def test_connection_path_change_and_prerequisite_failure_prevent_install(remote):
    platform, _ = remote
    tested = connection(platform).json()
    body = payload(tested["connection_id"])
    body["install_dir"] = "/data2/other-agent"
    assert create(platform, body).json()["error"]["code"] == "SSH_TARGET_CHANGED"
    FakeRemote.probe_result["can_install"] = False
    failed = connection(platform).json()
    assert not failed["can_install"]
    assert create(platform, payload(failed["connection_id"])).json()["error"]["code"] == "SSH_PREREQUISITES_FAILED"


def test_pending_command_blocks_reinstall_even_without_running_training(remote):
    platform, _ = remote
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=node["node_id"], type="PREFLIGHT", status="PENDING", payload={}, created_at=now()))
    FakeRemote.probe_result["existing_agent"] = {"node_id": node["node_id"], "profiles": []}
    tested = connection(platform).json()
    response = create(platform, payload(tested["connection_id"]))
    assert response.status_code == 409 and response.json()["error"]["code"] == "NODE_BUSY"


def test_duplicate_target_has_one_active_job(remote):
    platform, _ = remote
    tested = connection(platform).json()
    first = create(platform, payload(tested["connection_id"]))
    assert first.status_code == 202
    conflict = create(platform, payload(tested["connection_id"]), key="second")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["details"]["deployment_id"] == first.json()["id"]


def test_host_alias_cannot_start_second_deployment_on_same_registered_node(remote):
    platform, _ = remote
    node = platform.register()
    FakeRemote.probe_result["existing_agent"] = {"node_id": node["node_id"], "profiles": []}
    tested_ip = connection(platform).json()
    tested_alias = connection(platform, host="gpu-worker.example.com").json()
    first = create(platform, payload(tested_ip["connection_id"]))
    assert first.status_code == 202, first.text
    conflict = create(platform, payload(tested_alias["connection_id"]), key="host-alias")
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["error"]["code"] == "REMOTE_NODE_MANAGED"
    assert conflict.json()["error"]["details"]["deployment_id"] == first.json()["id"]


@pytest.mark.parametrize("with_recipe", [True, False])
def test_direct_install_requires_matching_fresh_agent_and_runtime_ready(remote, with_recipe):
    platform, manager = remote
    tested = connection(platform).json()
    body = payload(tested["connection_id"])
    if not with_recipe:
        del body["agent"]["profiles"][0]["recipe"]
    job = create(platform, body).json()
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(job["id"])
    completed = manager.get(job["id"])
    assert completed["state"] == "SUCCEEDED", completed
    assert completed["node_id"]
    assert completed["agent"]["profiles"][0]["recipe"] == ("smoke" if with_recipe else None)
    assert not FakeTunnel.calls
    assert PASSWORD not in json.dumps(completed) and "test-enrollment" not in json.dumps(completed)
    with platform.app.state.db.read() as session:
        node = session.get(Node, completed["node_id"])
        assert node.name == "remote-test" and node.roots == ["/data2"]
        assert node.output_root == "/data2/vla-agent/outputs"


@pytest.mark.parametrize("stale,wrong", [(True, False), (False, True)])
def test_old_heartbeat_or_wrong_onboarding_cannot_claim_success(remote, stale, wrong):
    platform, manager = remote
    manager._wait_seconds = 0.01
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config, stale=stale, wrong_provenance=wrong)
    manager._execute(job["id"])
    completed = manager.get(job["id"])
    assert completed["state"] == "FAILED"
    assert completed["error"]["code"] == "AGENT_ONLINE_TIMEOUT"


def test_runtime_failure_keeps_node_and_tunnel_for_diagnosis(remote, monkeypatch):
    platform, manager = remote
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:18000")
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config, status="FAILED")
    manager._execute(job["id"])
    failed = manager.get(job["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "RUNTIME_CHECK_FAILED"
    assert failed["node_id"]
    assert FakeTunnel.calls[0].healthy and not FakeTunnel.calls[0].closed
    assert failed["agent"]["control_plane_url"].endswith(str(FakeTunnel.calls[0].actual_port))


def test_tunnel_reconnect_reuses_assigned_port_and_pinned_host(remote, monkeypatch):
    platform, manager = remote
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:18000")
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(job["id"])
    first = FakeTunnel.calls[0]
    assert first.options["remote_port"] == 0
    assert first.options["expected_fingerprint"] == "SHA256:unit-host-key"
    first.healthy = False
    manager._maintain_tunnels()
    assert first.closed
    assert FakeTunnel.calls[-1].options["remote_port"] == first.actual_port
    assert manager.get(job["id"])["tunnel_state"] == "HEALTHY"


def test_successful_update_preserves_old_tunnel_until_new_success(remote, monkeypatch):
    platform, manager = remote
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:18000")
    tested = connection(platform).json()
    first = create(platform, payload(tested["connection_id"])).json()
    identity = []
    def install(config):
        identity.append(agent_online(platform, config, node_id=identity[0] if identity else None))
    FakeRemote.installer = install
    manager._execute(first["id"])
    old_tunnel = FakeTunnel.calls[0]
    second = create(platform, payload(tested["connection_id"], name="renamed"), key="updated")
    assert second.status_code == 202, second.text
    assert not old_tunnel.closed
    manager._execute(second.json()["id"])
    assert manager.get(second.json()["id"])["state"] == "SUCCEEDED"
    assert old_tunnel.closed and manager.get(first["id"])["tunnel_state"] == "SUPERSEDED"
    assert identity[0] == identity[1]


def test_restart_marks_queued_job_interrupted_and_restores_only_managed_tunnel(remote):
    platform, manager = remote
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    manager.close()
    restored = DeploymentManager(platform.app.state.db, platform.settings, remote_factory=FakeRemote, tunnel_factory=FakeTunnel)
    try:
        restored.start()
        current = restored.get(job["id"])
        assert current["state"] == "INTERRUPTED"
        assert current["error"]["code"] == "DEPLOYMENT_INTERRUPTED"
        assert not FakeTunnel.calls
        with platform.app.state.db.read() as session:
            saved = session.get(RemoteConnection, tested["connection_id"])
            assert restored.vault.decrypt(saved.encrypted_password) == PASSWORD
    finally:
        restored.close()


@pytest.mark.parametrize("with_recipe", [True, False])
def test_failed_job_retry_uses_persisted_connection_without_exposing_password(remote, with_recipe):
    platform, manager = remote
    tested = connection(platform).json()
    body = payload(tested["connection_id"])
    if not with_recipe:
        del body["agent"]["profiles"][0]["recipe"]
    job = create(platform, body).json()
    FakeRemote.error = RemoteError("SSH_AUTH_FAILED", "SSH 认证失败")
    manager._execute(job["id"])
    response = platform.client.post(f"/api/v1/remote-deployments/{job['id']}/retry", headers=ADMIN)
    assert response.status_code == 202, response.text
    assert response.json()["id"] != job["id"]
    assert response.json()["agent"]["profiles"][0]["recipe"] == ("smoke" if with_recipe else None)
    assert PASSWORD not in response.text and "test-enrollment" not in response.text


def test_ssh_flow_manages_tunnel_without_global_url_while_manual_defaults_stay_unavailable(remote, monkeypatch):
    platform, _ = remote
    monkeypatch.delenv("VLA_AGENT_CONTROL_PLANE_URL")
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"]))
    assert job.status_code == 202, job.text
    assert job.json()["connection_mode"] == "ssh-tunnel"
    defaults = platform.client.get("/api/v1/agent-config-defaults", headers=ADMIN).json()
    assert defaults["control_plane_url"] is None


@pytest.mark.parametrize("interrupted", [False, True])
def test_restart_restores_successful_or_interrupted_tunnel_to_same_port(remote, monkeypatch, interrupted):
    platform, manager = remote
    monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:18000")
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(job["id"])
    port = FakeTunnel.calls[-1].actual_port
    if interrupted:
        with platform.app.state.db.write() as session:
            stored = session.get(RemoteDeployment, job["id"])
            stored.state = "WAITING_AGENT"
    manager.close()
    previous_tunnels = len(FakeTunnel.calls)
    restored = DeploymentManager(platform.app.state.db, platform.settings, remote_factory=FakeRemote, tunnel_factory=FakeTunnel)
    try:
        restored.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(FakeTunnel.calls) <= previous_tunnels:
            time.sleep(0.01)
        assert FakeTunnel.calls[-1].options["remote_port"] == port
        assert FakeTunnel.calls[-1].healthy
        assert restored.get(job["id"])["state"] == ("INTERRUPTED" if interrupted else "SUCCEEDED")
    finally:
        restored.close()


def managed_node(platform, manager, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setenv("VLA_AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:18000")
    tested = connection(platform).json()
    job = create(platform, payload(tested["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(job["id"])
    completed = manager.get(job["id"])
    assert completed["state"] == "SUCCEEDED", completed
    return completed["node_id"], job, tested


def remove_node(platform, node_id, operation, key="unit-remove"):
    return platform.client.post(f"/api/v1/nodes/{node_id}/{operation}", headers={**ADMIN, "Idempotency-Key": key})


def test_disconnect_closes_tunnels_revokes_alias_credentials_and_preserves_history(remote, monkeypatch):
    platform, manager = remote
    node_id, job, tested = managed_node(platform, manager, monkeypatch)
    with platform.app.state.db.write() as session:
        node = session.get(Node, node_id)
        token_hash = node.token_hash
        session.add(Asset(id=uid(), node_id=node_id, name="dataset", kind="DATASET", path="/data2/dataset", version="1", created_at=now()))
    FakeRemote.probe_result["existing_agent"] = {"node_id": node_id, "profiles": []}
    alias = connection(platform, host="worker.example.com").json()
    FakeRemote.probe_result.pop("existing_agent")
    other_path = connection(platform, install_dir="/data2/new-agent").json()
    response = remove_node(platform, node_id, "disconnect")
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["state"] == "DISCONNECTED" and record["operation"] == "disconnect"
    assert "_operation" not in response.text and PASSWORD not in response.text
    assert FakeTunnel.calls[0].closed
    assert remove_node(platform, node_id, "disconnect").json()["id"] == record["id"]
    manager._maintain_tunnels()
    assert len(FakeTunnel.calls) == 1
    with platform.app.state.db.read() as session:
        node = session.get(Node, node_id)
        assert node.status == "DISCONNECTED" and node.token_hash == token_hash
        assert session.scalar(select(Asset).where(Asset.node_id == node_id))
        assert not session.scalar(select(Profile).where(Profile.node_id == node_id)).active
        for item in (tested, alias, other_path):
            assert session.get(RemoteConnection, item["connection_id"]).encrypted_password == ""
        assert session.get(RemoteDeployment, job["id"]).state == "SUCCEEDED"
    assert not manager.management(node_id)["can_uninstall"]
    assert create(platform, payload(tested["connection_id"]), key="revoked").json()["error"]["code"] == "SSH_CONNECTION_REVOKED"


def test_unmanaged_node_can_disconnect_without_ssh_but_cannot_uninstall(remote):
    platform, manager = remote
    node = platform.register()
    info = manager.management(node["node_id"])
    assert not info["managed"] and info["can_disconnect"] and not info["can_uninstall"]
    assert remove_node(platform, node["node_id"], "uninstall").status_code == 409
    response = remove_node(platform, node["node_id"], "disconnect", key="local-disconnect")
    assert response.status_code == 200 and response.json()["agent"] == {}
    assert not FakeRemote.calls


def test_uninstall_is_async_and_only_confirmed_remote_success_retires_node(remote, monkeypatch):
    platform, manager = remote
    node_id, old, tested = managed_node(platform, manager, monkeypatch)
    response = remove_node(platform, node_id, "uninstall")
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["state"] == "QUEUED" and queued["operation"] == "uninstall"
    with platform.app.state.db.read() as session:
        assert session.get(Node, node_id).status == "ONLINE"
        assert session.get(RemoteConnection, tested["connection_id"]).encrypted_password
    manager._execute(queued["id"])
    done = manager.get(queued["id"])
    assert done["state"] == "UNINSTALLED" and done["operation_result"]["service_removed"]
    assert done["operation_result"]["backup_dir"].startswith("/data2/")
    assert FakeTunnel.calls[0].closed
    assert remove_node(platform, node_id, "uninstall").json()["id"] == queued["id"]
    with platform.app.state.db.read() as session:
        assert session.get(Node, node_id).status == "REMOVED"
        assert not session.get(RemoteConnection, tested["connection_id"]).encrypted_password
        assert not session.scalar(select(Profile).where(Profile.node_id == node_id)).active
    manager._maintain_tunnels()
    assert len(FakeTunnel.calls) == 1


def test_uninstall_failure_keeps_connection_and_can_retry_without_false_success(remote):
    platform, manager = remote
    node_id, _, tested = managed_node(platform, manager)
    queued = remove_node(platform, node_id, "uninstall").json()
    FakeRemote.uninstall_result = {"service_removed": False}
    manager._execute(queued["id"])
    failed = manager.get(queued["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "AGENT_UNINSTALL_NOT_CONFIRMED"
    with platform.app.state.db.read() as session:
        assert session.get(Node, node_id).status == "ONLINE"
        assert session.get(RemoteConnection, tested["connection_id"]).encrypted_password
        assert session.scalar(select(Profile).where(Profile.node_id == node_id)).active
    retry = platform.client.post(f"/api/v1/remote-deployments/{queued['id']}/retry", headers=ADMIN)
    assert retry.status_code == 202, retry.text
    assert retry.json()["operation"] == "uninstall"


@pytest.mark.parametrize("operation", ["disconnect", "uninstall"])
def test_queued_training_prevents_removal(remote, operation):
    platform, _ = remote
    node = platform.register()
    platform.run(node)
    result = remove_node(platform, node["node_id"], operation)
    assert result.status_code == 409 and result.json()["error"]["code"] == "NODE_BUSY"


@pytest.mark.parametrize("operation", ["disconnect", "uninstall"])
def test_leftover_gpu_allocation_prevents_removal_after_run_terminal(remote, operation):
    platform, _ = remote
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    with platform.app.state.db.write() as session:
        session.get(Run, run["id"]).state = "FAILED"
    result = remove_node(platform, node["node_id"], operation)
    assert result.status_code == 409 and result.json()["error"]["code"] == "NODE_BUSY"


def test_uninstall_maintenance_blocks_alias_reinstall_and_disconnect(remote):
    platform, manager = remote
    node_id, _, _ = managed_node(platform, manager)
    queued = remove_node(platform, node_id, "uninstall")
    assert queued.status_code == 202
    assert remove_node(platform, node_id, "disconnect", key="parallel-disconnect").status_code == 409
    FakeRemote.probe_result["existing_agent"] = {"node_id": node_id, "profiles": []}
    alias = connection(platform, host="alias.example.com").json()
    result = create(platform, payload(alias["connection_id"]), key="parallel-install")
    assert result.status_code == 409 and result.json()["error"]["code"] == "REMOTE_NODE_MANAGED"


def test_readding_disconnected_node_requires_new_password_and_restores_original_identity(remote):
    platform, manager = remote
    node_id, _, old_connection = managed_node(platform, manager)
    remove_node(platform, node_id, "disconnect")
    FakeRemote.probe_result["existing_agent"] = {"node_id": node_id, "profiles": []}
    fresh = connection(platform).json()
    response = create(platform, payload(fresh["connection_id"]), key="reconnect")
    assert response.status_code == 202, response.text
    with platform.app.state.db.read() as session:
        assert session.get(Node, node_id).status == "REGISTERED"
    FakeRemote.installer = lambda config: agent_online(platform, config, node_id=node_id)
    manager._execute(response.json()["id"])
    assert manager.get(response.json()["id"])["state"] == "SUCCEEDED"
    assert manager.get(response.json()["id"])["node_id"] == node_id


def test_readding_uninstalled_prefix_creates_fresh_node_without_old_identity_fallback(remote):
    platform, manager = remote
    node_id, _, _ = managed_node(platform, manager)
    queued = remove_node(platform, node_id, "uninstall").json()
    manager._execute(queued["id"])
    fresh = connection(platform).json()
    result = create(platform, payload(fresh["connection_id"]), key="fresh-after-uninstall")
    assert result.status_code == 202, result.text
    assert result.json()["node_id"] is None
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(result.json()["id"])
    done = manager.get(result.json()["id"])
    assert done["state"] == "SUCCEEDED" and done["node_id"] != node_id


def test_disconnect_during_tunnel_reconnect_cannot_resurrect_connection(remote, monkeypatch):
    platform, manager = remote
    node_id, _, _ = managed_node(platform, manager, monkeypatch)
    FakeTunnel.calls[0].healthy = False
    class RacingTunnel(FakeTunnel):
        def start(self):
            self.healthy = True
            manager.remove(node_id, "disconnect", "disconnect-during-reconnect")
            return self.actual_port
    manager._tunnel_factory = RacingTunnel
    manager._maintain_tunnels()
    assert FakeTunnel.calls[-1].closed
    assert not manager._tunnels
    assert manager.management(node_id)["node_status"] == "DISCONNECTED"


def test_install_discovering_node_after_probe_still_respects_existing_uninstall(remote):
    platform, manager = remote
    node_id, _, _ = managed_node(platform, manager)
    alias = connection(platform, host="late-alias.example.com").json()
    install = create(platform, payload(alias["connection_id"]), key="late-install").json()
    assert install["node_id"] is None
    uninstall = remove_node(platform, node_id, "uninstall")
    assert uninstall.status_code == 202
    FakeRemote.probe_result["existing_agent"] = {"node_id": node_id, "profiles": []}
    manager._execute(install["id"])
    failed = manager.get(install["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "REMOTE_NODE_MANAGED"


def test_uninstall_stops_when_remote_agent_identity_changed(remote):
    platform, manager = remote
    node_id, _, tested = managed_node(platform, manager)
    queued = remove_node(platform, node_id, "uninstall").json()
    FakeRemote.probe_result["existing_agent"] = {"node_id": uid(), "profiles": []}
    manager._execute(queued["id"])
    failed = manager.get(queued["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "REMOTE_NODE_CHANGED"
    with platform.app.state.db.read() as session:
        assert session.get(Node, node_id).status == "ONLINE"
        assert session.get(RemoteConnection, tested["connection_id"]).encrypted_password
