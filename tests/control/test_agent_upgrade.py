"""Connected Agents upgrade through saved SSH with fresh target-build proof."""

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import zipfile

import pytest

from vla_control.common import APIError, now, offset_time, token_hash, uid
from vla_control.models import Allocation, Attempt, Command, GPUReservation, Node, Profile, Run
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.ssh_remote import _UPGRADE_MERGE_SCRIPT, upgrade_public_config

from .conftest import ADMIN
from .test_remote_deployment import FakeRemote, FakeTunnel, PASSWORD, agent_online, managed_node, remote


RELEASE = {"version": "0.1.0", "build_id": "a" * 64}


@pytest.fixture
def connected(remote, monkeypatch):
    platform, manager = remote
    node_id, source, connection = managed_node(platform, manager)
    source = manager.get(source["id"])
    current = deepcopy(source["agent"])
    current.update(node_id=node_id, active_attempts=0)
    FakeRemote.probe_result["existing_agent"] = current
    write_bundle(Path(os.environ["VLA_AGENT_BUNDLE"]))

    def upgraded(config):
        agent_online(platform, config, node_id=node_id)
        with platform.app.state.db.write() as session:
            node = session.get(Node, node_id)
            node.metrics = {**node.metrics, "agent": dict(RELEASE)}
            for profile in config["profiles"]:
                row = session.get(Profile, (node_id, profile["id"]))
                for field in ("name", "python", "code_dir", "adapter", "recipe", "execution_mode"):
                    setattr(row, field, profile.get(field))

    FakeRemote.installer = upgraded
    return platform, manager, node_id, current, source, connection


def write_bundle(path, release=None):
    identity = {"schema_version": 1, **(RELEASE if release is None else release)}
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vla_agent/agent-release.json", json.dumps(identity))
    files = {name: b"fixture" for name in ("requirements.txt", "install_agent.py", "install-agent.sh",
                                          "agent-bootstrap.sh", "agent_ownership.py")}
    files.update({"release.json": json.dumps(identity).encode(), "agent.whl": wheel.getvalue()})
    files["manifest.json"] = json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in files.items()}).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def upgrade(platform, node_id, key="unit-upgrade"):
    return platform.client.post(f"/api/v1/nodes/{node_id}/upgrade", headers={**ADMIN, "Idempotency-Key": key})


def test_upgrade_requires_management_authority_and_idempotency_key(connected):
    platform, _, node_id, *_ = connected
    assert platform.client.post(f"/api/v1/nodes/{node_id}/upgrade").status_code == 401
    assert platform.client.post(f"/api/v1/nodes/{node_id}/upgrade", headers=ADMIN).status_code == 422


def test_upgrade_is_durable_idempotent_and_reuses_saved_expired_ssh(connected):
    platform, manager, node_id, _, source, connection = connected
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, connection["connection_id"]).expires_at = offset_time(-3600)
    queued = upgrade(platform, node_id)
    assert queued.status_code == 202, queued.text
    job = queued.json()
    assert job["operation"] == "upgrade" and job["state"] == "QUEUED"
    assert job["node_id"] == node_id and job["install_dir"] == source["install_dir"]
    assert job["connection_id"] == source["connection_id"]
    assert upgrade(platform, node_id).json()["id"] == job["id"]
    assert upgrade(platform, node_id, key="parallel-upgrade").status_code == 409
    assert PASSWORD not in queued.text and "test-enrollment" not in queued.text
    info = manager.management(node_id)
    assert not info["can_upgrade"] and info["latest_upgrade"]["id"] == job["id"]
    with platform.app.state.db.read() as session:
        assert "enrollment_token" not in session.get(RemoteDeployment, job["id"]).agent_config


def test_reviewed_build_cannot_drift_before_upgrade_submission(connected):
    platform, manager, node_id, *_ = connected
    reviewed = manager.management(node_id)["target_build_id"]
    write_bundle(Path(os.environ["VLA_AGENT_BUNDLE"]), {**RELEASE, "build_id": "b" * 64})
    response = platform.client.post(f"/api/v1/nodes/{node_id}/upgrade",
        headers={**ADMIN, "Idempotency-Key": "reviewed-build"}, json={"expected_build_id": reviewed})
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_PACKAGE_CHANGED"
    assert manager.management(node_id)["latest_upgrade"] is None


def test_reviewed_target_is_idempotent_and_different_reviewed_target_conflicts(connected):
    platform, _, node_id, *_ = connected
    path = f"/api/v1/nodes/{node_id}/upgrade"
    headers = {**ADMIN, "Idempotency-Key": "reviewed-build"}
    body = {"expected_build_id": RELEASE["build_id"]}
    first = platform.client.post(path, headers=headers, json=body)
    assert first.status_code == 202, first.text
    write_bundle(Path(os.environ["VLA_AGENT_BUNDLE"]), {**RELEASE, "build_id": "b" * 64})
    assert platform.client.post(path, headers=headers, json=body).json()["id"] == first.json()["id"]
    conflict = platform.client.post(path, headers=headers, json={"expected_build_id": "b" * 64})
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("build_id", ["latest", "a" * 63, "A" * 64, 123])
def test_reviewed_build_requires_explicit_hex_identity(connected, build_id):
    platform, _, node_id, *_ = connected
    response = platform.client.post(f"/api/v1/nodes/{node_id}/upgrade",
        headers={**ADMIN, "Idempotency-Key": "reviewed-build"}, json={"expected_build_id": build_id})
    assert response.status_code == 422


def test_upgrade_preserves_current_paths_profiles_identity_and_confirms_build(connected):
    platform, manager, node_id, current, source, _ = connected
    current.update(output_root="/data2/new-output", cache_root="/data2/cache", http_timeout_seconds=45)
    current["profiles"][0].update(code_dir="/data2/rebound-code", python="/data2/env/bin/python", recipe="new_recipe")
    job = upgrade(platform, node_id).json()
    manager._execute(job["id"])
    done = manager.get(job["id"])
    assert done["state"] == "SUCCEEDED", done
    assert done["operation"] == "upgrade" and done["node_id"] == node_id
    for field in ("state_dir", "roots", "output_root", "cache_root", "http_timeout_seconds"):
        assert done["agent"][field] == current[field]
    assert done["agent"]["profiles"][0]["python"] == "/data2/env/bin/python"
    assert done["agent"]["profiles"][0]["metadata"]["onboarding_id"] != source["onboarding_id"]
    with platform.app.state.db.read() as session:
        node = session.get(Node, node_id)
        assert node.output_root == current["output_root"] and node.cache_root == current["cache_root"]
    info = manager.management(node_id)
    assert info["deployment_id"] == job["id"] and info["current_version"] == RELEASE["version"]
    assert info["target_build_id"] == RELEASE["build_id"]
    assert info["latest_upgrade"]["state"] == "SUCCEEDED"


@pytest.mark.parametrize("change,code", [({"node_id": "different-node"}, "REMOTE_NODE_CHANGED"),
                                        ({"state_dir": "/data2/other-state"}, "AGENT_STATE_DIR_CONFLICT")])
def test_upgrade_stops_before_install_if_remote_identity_or_state_changes(connected, change, code):
    platform, manager, node_id, current, *_ = connected
    installed = []
    FakeRemote.installer = lambda config: installed.append(config)
    job = upgrade(platform, node_id).json()
    current.update(change)
    manager._execute(job["id"])
    failed = manager.get(job["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == code
    assert not installed


@pytest.mark.parametrize("proof", [None, {**RELEASE, "build_id": "b" * 64}, {**RELEASE, "version": "old"}])
def test_old_process_or_wrong_build_cannot_confirm_upgrade(connected, proof):
    platform, manager, node_id, *_ = connected
    job = upgrade(platform, node_id).json()
    with platform.app.state.db.write() as session:
        row = session.get(RemoteDeployment, job["id"])
        row.started_at = offset_time(-1)
        row.state = "WAITING_AGENT"
        config = deepcopy(row.agent_config)
    agent_online(platform, config, node_id=node_id)
    if proof is not None:
        with platform.app.state.db.write() as session:
            node = session.get(Node, node_id)
            node.metrics = {**node.metrics, "agent": proof}
    assert not manager._confirm_agent(job["id"])
    assert manager.get(job["id"])["state"] == "WAITING_AGENT"


def test_replaced_bundle_fails_safely_and_retry_remains_upgrade(connected, monkeypatch):
    platform, manager, node_id, *_ = connected
    job = upgrade(platform, node_id).json()
    write_bundle(Path(os.environ["VLA_AGENT_BUNDLE"]), {**RELEASE, "build_id": "b" * 64})
    manager._execute(job["id"])
    assert manager.get(job["id"])["error"]["code"] == "AGENT_PACKAGE_CHANGED"
    retry = platform.client.post(f"/api/v1/remote-deployments/{job['id']}/retry", headers={**ADMIN, "Idempotency-Key": "retry-upgrade"})
    assert retry.status_code == 202 and retry.json()["operation"] == "upgrade"
    with platform.app.state.db.read() as session:
        assert session.get(RemoteDeployment, retry.json()["id"]).agent_config["_target_release"]["build_id"] == "b" * 64


def test_install_uses_verified_private_snapshot_if_administrator_rebuilds_bundle_during_upload(connected):
    from vla_protocol.agent_release import read_agent_bundle

    platform, manager, node_id, *_ = connected
    configured_bundle = Path(os.environ["VLA_AGENT_BUNDLE"])
    snapshots = []

    class RebuildingRemote(FakeRemote):
        def install(self, bundle, config, prefix, emit):
            snapshots.append(bundle)
            write_bundle(configured_bundle, {**RELEASE, "build_id": "b" * 64})
            assert bundle != configured_bundle and read_agent_bundle(bundle) == RELEASE
            super().install(bundle, config, prefix, emit)

    manager._remote_factory = RebuildingRemote
    queued = upgrade(platform, node_id).json()
    manager._execute(queued["id"])
    assert manager.get(queued["id"])["state"] == "SUCCEEDED"
    assert snapshots and not snapshots[0].exists()


def test_gpu_reservation_allows_upgrade_but_still_blocks_disconnect_and_uninstall(connected):
    from vla_control.auth_models import User

    platform, manager, node_id, *_ = connected
    with platform.app.state.db.write() as session:
        user = User(id=uid(), username="reservation-owner", display_name="Reservation owner", password_hash="unused", role="operator", enabled=True,
                    created_at=now(), updated_at=now())
        session.add(user)
        session.flush()
        session.add(GPUReservation(node_id=node_id, gpu_uuid="GPU-reserved", user_id=user.id, created_at=now()))
    info = manager.management(node_id)
    assert info["can_upgrade"] and not info["can_disconnect"] and not info["can_uninstall"]
    assert manager.upgrade(node_id, "reserved-upgrade")["state"] == "QUEUED"


def isolated_training(connected):
    platform, _, node_id, current, *_ = connected
    run_id, attempt_id = uid(), uid()
    with platform.app.state.db.write() as session:
        session.add(Run(id=run_id, name="active", node_id=node_id, profile_id=current["profiles"][0]["id"],
            template_id="smoke", state="RUNNING", created_at=now(), updated_at=now(), inputs={}, parameters={},
            gpu_count=1, code_expectation={}))
        session.flush()
        session.add(Attempt(id=attempt_id, run_id=run_id, node_id=node_id, number=1, execution_token=uid(),
            state="RUNNING", created_at=now(), started_at=now(), gpu_uuids=["GPU-1"], latest_metrics={}, resources={}))
        session.flush()
        session.add(Allocation(node_id=node_id, gpu_uuid="GPU-1", attempt_id=attempt_id, created_at=now()))
        node = session.get(Node, node_id)
        node.metrics = {**node.metrics, "training_supervisor": {"protocol_version": 1, "upgrade_safe": True,
            "active_attempts": 1, "active_attempt_ids": [attempt_id]}}
    return run_id, attempt_id


def test_upgrade_during_isolated_training_preserves_run_and_gpu_allocation(connected):
    platform, manager, node_id, *_ = connected
    run_id, attempt_id = isolated_training(connected)
    info = manager.management(node_id)
    assert info["can_upgrade"] and not info["can_uninstall"] and not info["can_disconnect"]
    job = upgrade(platform, node_id).json()
    manager._execute(job["id"])
    assert manager.get(job["id"])["state"] == "SUCCEEDED"
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).state == "RUNNING"
        assert session.get(Attempt, attempt_id).state == "RUNNING"
        assert session.get(Allocation, (node_id, "GPU-1")).attempt_id == attempt_id


@pytest.mark.parametrize("change", [None, "recipe", "metadata"])
def test_live_upgrade_confirms_authenticated_heartbeat_without_mutating_running_profile(connected, change):
    platform, manager, node_id, current, *_ = connected
    run_id, attempt_id = isolated_training(connected)
    profile_id = current["profiles"][0]["id"]
    with platform.app.state.db.write() as session:
        node = session.get(Node, node_id)
        node.token_hash = token_hash("isolated-agent-token")
        metadata = dict(session.get(Profile, (node_id, profile_id)).metadata_json)
        old_recipe = session.get(Profile, (node_id, profile_id)).recipe
    responses = []
    def heartbeat(config):
        # SSH merge retains the target's non-provenance metadata. Unlike the
        # old fake installer, only this authenticated API writes profiles.
        profiles = [{**profile, "metadata": {**metadata, **profile["metadata"]}} for profile in config["profiles"]]
        if change == "recipe":
            profiles[0]["recipe"] = "unexpected_recipe"
        elif change == "metadata":
            profiles[0]["metadata"]["validated"] = not metadata.get("validated", False)
        result = platform.client.post("/agent/v1/heartbeat", headers={"Authorization": "Bearer isolated-agent-token"},
            json={"node_id": node_id, "gpus": [], "active_attempts": [{"attempt_id": attempt_id, "state": "RUNNING"}],
                "profiles": profiles, "metrics": {"agent": RELEASE,
                    "runtime_checks": [{"profile_id": profile_id, "status": "READY"}],
                    "training_supervisor": {"protocol_version": 1, "upgrade_safe": True, "active_attempts": 1,
                        "active_attempt_ids": [attempt_id]}}})
        assert result.status_code == 200, result.text
        responses.append(result.json())
    FakeRemote.installer = heartbeat
    manager._wait_seconds = 0
    job = upgrade(platform, node_id).json()
    manager._execute(job["id"])
    done = manager.get(job["id"])
    assert done["state"] == ("SUCCEEDED" if change is None else "FAILED"), done
    assert responses[0]["profile_sync"]["status"] == ("APPLIED" if change is None else "BLOCKED")
    with platform.app.state.db.read() as session:
        profile = session.get(Profile, (node_id, profile_id))
        assert profile.recipe == old_recipe and profile.metadata_json["validated"] == metadata["validated"]
        assert profile.metadata_json["onboarding_id"] == (done["onboarding_id"] if change is None else metadata["onboarding_id"])
        assert session.get(Run, run_id).state == session.get(Attempt, attempt_id).state == "RUNNING"
        assert session.get(Allocation, (node_id, "GPU-1")).attempt_id == attempt_id


@pytest.mark.parametrize("change", ["legacy", "unsafe", "missing", "stale", "unknown"])
def test_live_upgrade_requires_fresh_complete_supervisor_proof(connected, change):
    platform, manager, node_id, *_ = connected
    _, attempt_id = isolated_training(connected)
    with platform.app.state.db.write() as session:
        node = session.get(Node, node_id)
        proof = dict(node.metrics["training_supervisor"])
        if change == "legacy":
            proof["protocol_version"] = 0
        elif change == "unsafe":
            proof["upgrade_safe"] = False
        elif change == "missing":
            proof["active_attempt_ids"] = []
        elif change == "stale":
            node.last_seen = offset_time(-1000)
        else:
            session.get(Attempt, attempt_id).state = "UNKNOWN"
        node.metrics = {**node.metrics, "training_supervisor": proof}
    assert not manager.management(node_id)["can_upgrade"]
    assert upgrade(platform, node_id).status_code == 409


def test_pending_command_blocks_upgrade_and_unmanaged_agent_cannot_upgrade(remote, monkeypatch):
    platform, manager = remote
    node = platform.register()
    monkeypatch.setattr(manager, "_upgrade_release", lambda: dict(RELEASE))
    assert not manager.management(node["node_id"])["can_upgrade"]
    assert upgrade(platform, node["node_id"]).status_code == 409
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=node["node_id"], type="PROFILE_REBIND", status="PENDING", payload={}, created_at=now()))
    response = upgrade(platform, node["node_id"], key="pending-upgrade")
    assert response.status_code == 409 and response.json()["error"]["code"] == "NODE_BUSY"


def test_legacy_bundle_is_not_advertised_as_upgrade_target(remote):
    platform, manager = remote
    node_id, *_ = managed_node(platform, manager)
    info = manager.management(node_id)
    assert not info["can_upgrade"] and info["target_version"] is None
    assert "重新构建" in info["upgrade_reason"]
    assert upgrade(platform, node_id).status_code == 409


def test_upgrade_tunnel_is_published_monitored_and_supersedes_previous_tunnel(connected, monkeypatch):
    platform, manager, node_id, _, source, _ = connected
    with platform.app.state.db.write() as session:
        old = session.get(RemoteDeployment, source["id"])
        old.connection_mode, old.tunnel_state, old.tunnel_remote_port = "ssh-tunnel", "HEALTHY", 23001
        old.control_plane_url = "http://127.0.0.1:23001"
        old.agent_config = {**old.agent_config, "control_plane_url": old.control_plane_url}
    old_tunnel = FakeTunnel(remote_port=23001)
    old_tunnel.start()
    manager._tunnels[source["id"]] = old_tunnel
    queued = upgrade(platform, node_id).json()
    manager._execute(queued["id"])
    done = manager.get(queued["id"])
    assert done["state"] == "SUCCEEDED" and done["tunnel_state"] == "HEALTHY"
    assert old_tunnel.closed and queued["id"] in manager._tunnels
    manager._tunnels[queued["id"]].healthy = False
    manager._maintain_tunnels()
    assert manager._tunnels[queued["id"]].healthy


def test_upgrade_transport_merges_privately_and_aborts_if_target_changed(tmp_path):
    from unittest.mock import Mock
    from vla_control.ssh_remote import RemoteError
    from .test_ssh_remote import install_fixture

    ssh, bundle, _ = install_fixture(tmp_path)
    ssh._is_root = False
    ssh._execute = Mock(side_effect=[(0, ""), (0, ""), (0, "")])
    config = {"_operation": "upgrade", "_upgrade_node_id": "node", "_upgrade_expected": {}, "profiles": [],
              "enrollment_token": "private-token"}
    ssh.install(bundle, config, "/data2/agent", Mock())
    calls = ssh._execute.call_args_list
    assert _UPGRADE_MERGE_SCRIPT in __import__("shlex").split(calls[0].args[0])
    assert calls[0].kwargs["sudo"] is True
    assert "install_agent.py" in calls[1].args[0]
    ssh, bundle, _ = install_fixture(tmp_path)
    ssh._execute = Mock(return_value=(1, "private-output-never-public"))
    with pytest.raises(RemoteError, match="远端 Agent 身份或配置已变化"):
        ssh.install(bundle, config, "/data2/agent", Mock())
    assert len(ssh._execute.call_args_list) == 1


def test_upgrade_merge_preserves_unknown_private_settings_and_profile_metadata_on_target(tmp_path):
    prefix = tmp_path / "agent"
    (prefix / "config").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    old = {"name": "worker", "roots": [str(tmp_path)], "output_root": str(tmp_path / "output"),
        "state_dir": str(state_dir), "cache_root": str(tmp_path / "cache"),
        "_output_root_history": [str(tmp_path / "old-output")], "private_extension": {"token": "keep-private"},
        "profiles": [{"id": "smoke", "python": sys.executable, "private_field": "keep-profile-private",
                      "metadata": {"custom": {"unknown": 42}, "onboarding_id": "previous"}}]}
    original = prefix / "config/agent.json"
    original.write_text(json.dumps(old))
    with sqlite3.connect(state_dir / "agent.sqlite3") as state:
        state.execute("CREATE TABLE credentials(id INTEGER,value TEXT)")
        state.execute("INSERT INTO credentials VALUES(1,?)", (json.dumps({"node_id": "worker-id", "token": "node-secret"}),))
    incoming = {**upgrade_public_config(old), "control_plane_url": "https://control.example.com", "enrollment_token": "new-enrollment",
        "_upgrade_expected": upgrade_public_config(old), "_upgrade_node_id": "worker-id",
        "profiles": [{"id": "smoke", "metadata": {"onboarding_id": "new-onboarding"}}]}
    upload = tmp_path / "upload.json"
    upload.write_text(json.dumps(incoming))
    result = subprocess.run([sys.executable, "-I", "-c", _UPGRADE_MERGE_SCRIPT, str(upload), str(prefix)], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
    merged = json.loads(upload.read_text())
    assert merged["private_extension"] == old["private_extension"]
    assert merged["_output_root_history"] == old["_output_root_history"]
    assert merged["profiles"][0]["private_field"] == "keep-profile-private"
    assert merged["profiles"][0]["metadata"] == {"custom": {"unknown": 42}, "onboarding_id": "new-onboarding"}
    assert json.loads(original.read_text()) == old
    # A changed target config cannot be silently replaced by an earlier snapshot.
    upload.write_text(json.dumps(incoming))
    original.write_text(json.dumps({**old, "output_root": str(tmp_path / "moved-output")}))
    refused = subprocess.run([sys.executable, "-I", "-c", _UPGRADE_MERGE_SCRIPT, str(upload), str(prefix)], capture_output=True, text=True)
    assert refused.returncode != 0 and "AGENT_UPGRADE_CONFIG_CHANGED" in refused.stderr
    assert "keep-private" not in refused.stderr


def test_sudo_upgrade_reads_identity_as_database_owner_without_root_sqlite_sidecars(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pwd

    prefix = tmp_path / "agent"
    (prefix / "config").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    database = state_dir / "agent.sqlite3"
    database.touch()
    old = {"name": "worker", "roots": [str(tmp_path)], "output_root": str(tmp_path / "output"),
           "state_dir": str(state_dir), "profiles": [{"id": "smoke"}]}
    (prefix / "config/agent.json").write_text(json.dumps(old))
    upload = tmp_path / "upload.json"
    upload.write_text(json.dumps({"_upgrade_expected": upgrade_public_config(old), "_upgrade_node_id": "node-id",
        "control_plane_url": "https://control.example.com", "enrollment_token": "new-enrollment",
        "profiles": [{"id": "smoke", "metadata": {"onboarding_id": "new-onboarding"}}]}))
    calls = []

    def read_as_owner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout='{"node_id":"node-id"}', stderr="")

    # Simulate the merge's sudo context while the state file belongs to the
    # ordinary service user. The mocked child is the only database reader.
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    real_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda path, *a, **kw: SimpleNamespace(st_uid=12345)
                        if path == database else real_stat(path, *a, **kw))
    monkeypatch.setattr(pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="agent-owner"))
    monkeypatch.setattr(subprocess, "run", read_as_owner)
    monkeypatch.setattr(sys, "argv", ["merge", str(upload), str(prefix)])
    exec(compile(_UPGRADE_MERGE_SCRIPT, "upgrade-merge", "exec"), {})
    assert len(calls) == 1
    args, options = calls[0]
    assert args[:4] == ["runuser", "-u", "agent-owner", "--"]
    assert args[-1] == str(database) and options == {"capture_output": True, "text": True, "timeout": 10}
    assert "'token'" not in args[-2] and "'node_id'" in args[-2]
    assert not (state_dir / "agent.sqlite3-wal").exists() and not (state_dir / "agent.sqlite3-shm").exists()
