"""Transfers use pinned machine access and never deploy unverified checkpoints."""

from copy import deepcopy
import hashlib
import json
import threading

import pytest
from sqlalchemy import inspect, select
from vla_protocol import canonical_digest

from vla_control.checkpoint_transfer_models import CheckpointTransfer
from vla_control.checkpoint_transfers import active_transfers
from vla_control.common import APIError, now, offset_time, uid
from vla_control.models import Asset, Attempt, Command, Node, Profile, Run
from vla_control.policy_deployment_models import PolicyDeployment
from vla_control.profile_rebindings import _in_use
from vla_control.recipe_writeback import require_checkout_idle
from vla_control.remote_deployment import DeploymentManager
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN
from .test_policy_deployments import CHECKPOINT_SOURCE_WARNING, add_run, running
from .test_recipe_details import register


BASE = "/api/v1/checkpoint-transfers"
SECRET = "transfer-ssh-password-not-public"
PORTABLE = {"schema_version": 1, "model": {"kind": "test-model"}, "code_files": {"src/model.py": "a" * 64}}


def bind_ssh(platform, node, host):
    manager = platform.app.state.remote_manager
    stamp = now()
    connection = RemoteConnection(id=uid(), host=host, port=22, username="fixture-user",
        encrypted_password=manager.vault.encrypt(SECRET), fingerprint="SHA256:verified-fixture-" + host,
        probe_result={}, install_dir="/machine/agent", created_at=stamp, expires_at=offset_time(-60))
    with platform.app.state.db.write() as session:
        session.add(connection)
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection.id, idempotency_key=uid(), request_digest="d" * 64,
            target_key=uid(), onboarding_id=uid(), node_id=node["node_id"], state="SUCCEEDED", stage="SUCCEEDED", logs=[], history=[],
            agent_config={"_operation": "install"}, install_dir="/machine/agent", control_plane_url="https://controller.example",
            connection_mode="direct", tunnel_state="NOT_REQUIRED", created_at=stamp, updated_at=stamp, finished_at=stamp))
    return connection


@pytest.fixture
def transfer(platform):
    source, target = register(platform, name="source"), register(platform, name="target")
    source_connection = bind_ssh(platform, source, "192.0.2.10")
    target_connection = bind_ssh(platform, target, "192.0.2.20")
    with platform.app.state.db.write() as session:
        for node in (source, target):
            session.get(Node, node["node_id"]).metrics = {"capabilities": ["POLICY_DEPLOY_PORTABLE_V1", "POLICY_EXPORT", "POLICY_EXPORT_CANCEL"]}
        run = add_run(session, source)
        attempt = Attempt(id=uid(), run_id=run.id, node_id=source["node_id"], number=1, execution_token=uid(),
            state="SUCCEEDED", created_at=now(), output_dir="/machine/output/owned", gpu_uuids=[], latest_metrics={}, resources={})
        session.add(attempt)
    manager = platform.app.state.transfer_manager
    manager._launch = lambda _id: None
    return platform, manager, source, target, run, source_connection, target_connection


def create(fixture, **changes):
    platform, _, _, target, run, *_ = fixture
    return platform.client.post(BASE, headers=ADMIN, json={"source_run_id": run.id,
        "source_checkpoint_path": "/machine/output/owned/10000", "target_node_id": target["node_id"],
        "target_profile_id": "default", "target_directory": "/machine/models", **changes})


def get(fixture, transfer_id):
    return fixture[0].client.get(BASE + "/" + transfer_id, headers=ADMIN).json()


def export_command(fixture, kind="POLICY_EXPORT"):
    return next(command for command in fixture[0].commands(fixture[2]) if command["type"] == kind)


def ack_export(fixture, command=None, result=None, node=None, status="APPLIED"):
    platform, _, source, *_ = fixture
    node = node or source
    command = command or export_command(fixture)
    result = result or {"state": "READY", "portable_config": PORTABLE, "portable_digest": canonical_digest(PORTABLE)}
    return platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": status, "result": result})


def test_create_queues_export_with_pinned_source_and_never_exposes_credentials(transfer):
    platform, _, source, target, run, *_ = transfer
    response = create(transfer)
    assert response.status_code == 202, response.text
    row = response.json()
    assert row["state"] == "QUEUED" and row["source_node_id"] == source["node_id"]
    assert row["target_node_id"] == target["node_id"] and row["target_path"] is None
    command = export_command(transfer)
    assert command["payload"]["checkpoint_path"] == row["source_checkpoint_path"]
    assert command["payload"]["source_training"]["run_id"] == run.id
    assert command["payload"]["export_id"] == row["id"] + "-1"
    assert now() < command["payload"]["expires_at"] < offset_time(121)
    assert not any(item["type"] == "POLICY_DEPLOY_START" for item in platform.commands(target))
    duplicate = create(transfer)
    assert duplicate.json()["id"] == row["id"]
    listing = platform.client.get(BASE, headers=ADMIN)
    assert listing.json() == [row]
    assert listing.headers["Cache-Control"] == "no-store"
    assert SECRET not in listing.text and "encrypted_password" not in listing.text and "portable_config" not in listing.text
    with platform.app.state.db.read() as session:
        stored = session.get(CheckpointTransfer, row["id"])
        assert stored.binding_snapshot["source_connection_id"] == transfer[5].id
        assert stored.binding_snapshot["target_connection_id"] == transfer[6].id


@pytest.mark.parametrize("side,index", [("source", 5), ("target", 6)])
def test_invalid_saved_credential_fails_before_export_or_worker_start(transfer, monkeypatch, side, index):
    platform, manager, source, target, *_ = transfer
    launched = []
    monkeypatch.setattr(manager, "_launch", launched.append)
    monkeypatch.setattr("vla_control.ssh_remote.SSHRemote.connect", lambda *_: pytest.fail("Credential validation must not connect"))
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, transfer[index].id).encrypted_password = "invalid-encrypted-credential"
    response = create(transfer)
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "TRANSFER_" + side.upper() + "_SSH_CREDENTIAL_UNAVAILABLE"
    assert response.json()["error"]["details"] == {"side": side}
    assert ("源机器" if side == "source" else "目标机器") in response.json()["error"]["message"]
    assert SECRET not in response.text and "invalid-encrypted-credential" not in response.text
    assert launched == [] and platform.commands(source) == platform.commands(target) == []
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(CheckpointTransfer))) == []


def test_valid_saved_credentials_construct_and_close_original_pinned_identities_without_connecting(transfer, monkeypatch):
    platform, manager, *_ = transfer
    constructed = []
    class ValidationRemote:
        def __init__(self, **kwargs):
            assert kwargs["password"] == SECRET
            self.identity = {key: kwargs[key] for key in ("host", "port", "username", "expected_fingerprint")}
            self.closed = False
            constructed.append(self)
        def connect(self):
            pytest.fail("Credential validation must not connect")
        def close(self):
            self.closed = True
    monkeypatch.setattr(platform.app.state.remote_manager, "_remote_factory", ValidationRemote)
    response = create(transfer)
    assert response.status_code == 202, response.text
    assert len(constructed) == 2 and all(remote.closed for remote in constructed)
    for remote, connection in zip(constructed, transfer[5:7]):
        assert remote.identity == {"host": connection.host, "port": connection.port, "username": connection.username,
            "expected_fingerprint": connection.fingerprint}
    with platform.app.state.db.read() as session:
        snapshot = session.get(CheckpointTransfer, response.json()["id"]).binding_snapshot
        assert snapshot["source_connection_id"] == transfer[5].id and snapshot["target_connection_id"] == transfer[6].id


@pytest.mark.parametrize("side,index", [("source", 5), ("target", 6)])
def test_saved_connection_constructor_error_is_directional_and_does_not_leak_raw_text(transfer, monkeypatch, side, index):
    platform, manager, *_ = transfer
    original = platform.app.state.remote_manager._remote
    launched = []
    monkeypatch.setattr(manager, "_launch", launched.append)
    def construct(connection):
        if connection.id == transfer[index].id:
            raise ValueError("private constructor error " + SECRET)
        return original(connection)
    monkeypatch.setattr(platform.app.state.remote_manager, "_remote", construct)
    response = create(transfer)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "TRANSFER_" + side.upper() + "_SSH_CONFIGURATION_INVALID"
    assert SECRET not in response.text and "private constructor error" not in response.text
    assert launched == [] and platform.commands(transfer[2]) == []


def test_retry_checks_saved_credentials_before_changing_attempt_or_queuing_export(transfer, monkeypatch):
    platform, manager, source, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer, result={"state": "FAILED", "error": {"code": "EXPORT_TEST", "message": "fixture"}})
    launched = []
    monkeypatch.setattr(manager, "_launch", launched.append)
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, transfer[5].id).encrypted_password = "invalid-encrypted-credential"
    response = platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN)
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "TRANSFER_SOURCE_SSH_CREDENTIAL_UNAVAILABLE"
    assert launched == [] and platform.commands(source) == []
    unchanged = get(transfer, row["id"])
    assert unchanged["attempt"] == 1 and unchanged["state"] == "FAILED"


def test_model_import_remains_available_with_busy_or_reserved_gpus(transfer):
    """Copying weights and exporting CPU configuration do not allocate a GPU."""
    platform, _, source, target, *_ = transfer
    with platform.app.state.db.write() as session:
        for registered in (source, target):
            session.get(Node, registered['node_id']).gpus = [{
                'uuid': 'GPU-1', 'name': 'busy GPU', 'memory_total_mb': 24000,
                'memory_used_mb': 23000, 'memory_free_mb': 100, 'utilization_pct': 100,
                'processes': [{'pid': 123, 'kind': 'compute', 'ownership': 'external'}]}]
        session.add(PolicyDeployment(id=uid(), name='existing target inference', node_id=target['node_id'],
            profile_id='default', checkpoint_path='/machine/existing-model', config_name='pi0_base', port=8000,
            gpu_uuid='GPU-1', state='RUNNING', logs='', checks=[], created_at=now(), updated_at=now()))
    response = create(transfer)
    assert response.status_code == 202, response.text
    imported = response.json()
    assert imported['state'] == 'QUEUED'
    assert export_command(transfer)['type'] == 'POLICY_EXPORT'
    assert not any(item['type'] == 'POLICY_DEPLOY_START' for item in platform.commands(target))


@pytest.mark.parametrize("changes,code", [
    ({"source_checkpoint_path": "/outside/model"}, "PATH_OUTSIDE_ROOTS"),
    ({"source_checkpoint_path": "/machine/../outside"}, "PATH_OUTSIDE_ROOTS"),
    ({"target_directory": "/outside/models"}, "PATH_OUTSIDE_ROOTS"),
    ({"target_directory": "/machine/models\n;touch bad"}, "TRANSFER_PATH_INVALID"),
    ({"target_profile_id": "missing"}, "TRANSFER_PROFILE_UNSUPPORTED"),
])
def test_transfer_rejects_invalid_source_target_and_shell_paths(transfer, changes, code):
    response = create(transfer, **changes)
    assert response.status_code in {409, 422}, response.text
    assert response.json()["error"]["code"] == code
    assert transfer[0].commands(transfer[2]) == []


@pytest.mark.parametrize("has_attempt", [True, False])
def test_unrecorded_checkpoint_transfer_warns_and_preserves_frozen_provenance(transfer, has_attempt):
    platform, _, _, target, run, *_ = transfer
    with platform.app.state.db.write() as session:
        if not has_attempt:
            for attempt in session.scalars(select(Attempt).where(Attempt.run_id == run.id)):
                session.delete(attempt)
        asset_count = len(list(session.scalars(select(Asset))))
        expected = deepcopy(session.get(Run, run.id).code_expectation)
    response = create(transfer, source_checkpoint_path="/machine/external/59999")
    assert response.status_code == 202, response.text
    row = response.json()
    assert row["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    assert get(transfer, row["id"])["warnings"] == row["warnings"]
    frozen = export_command(transfer)["payload"]["source_training"]
    assert frozen["checkpoint_roots"] == (["/machine/output/owned"] if has_attempt else [])
    assert frozen["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    with platform.app.state.db.read() as session:
        assert session.get(Run, run.id).code_expectation == expected
        assert len(list(session.scalars(select(Asset)))) == asset_count
        assert session.get(CheckpointTransfer, row["id"]).binding_snapshot["source_training"] == frozen
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "跨机警告保留",
        "node_id": target["node_id"], "profile_id": "default", "checkpoint_path": finished["target_path"],
        "source_run_id": run.id, "transfer_id": row["id"]})
    assert response.status_code == 202, response.text
    assert response.json()["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    detail = platform.client.get("/api/v1/policy-deployments/" + response.json()["id"], headers=ADMIN).json()
    assert detail["warnings"] == [CHECKPOINT_SOURCE_WARNING]


@pytest.mark.parametrize("kind", ["POLICY_EXPORT", "POLICY_EXPORT_STATUS"])
def test_accepted_export_warning_is_canonical_and_reaches_transfer_and_portable_deployment(transfer, kind):
    platform, _, source, target, run, *_ = transfer
    row = create(transfer).json()
    initial = export_command(transfer)
    frozen = deepcopy(initial["payload"]["source_training"])
    receipt = initial
    if kind == "POLICY_EXPORT_STATUS":
        ack_export(transfer, initial, {"state": "RUNNING"})
        with platform.app.state.db.write() as session:
            receipt_id = uid()
            payload = {key: initial["payload"][key] for key in ("export_id", "transfer_id", "transfer_attempt")}
            session.add(Command(id=receipt_id, node_id=source["node_id"], type=kind,
                status="PENDING", payload=payload, created_at=now()))
        receipt = {"id": receipt_id}
    response = ack_export(transfer, receipt, {"state": "READY", "portable_config": PORTABLE,
        "portable_digest": canonical_digest(PORTABLE), "warnings": [
            {"code": CHECKPOINT_SOURCE_WARNING["code"], "message": "untrusted text", "private": "discard"},
            CHECKPOINT_SOURCE_WARNING, {"code": "OTHER", "message": "discard"}]})
    assert response.status_code == 200, response.text
    assert get(transfer, row["id"])["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    with platform.app.state.db.read() as session:
        assert session.get(Command, receipt["id"]).result["warnings"] == [CHECKPOINT_SOURCE_WARNING]
        assert session.get(CheckpointTransfer, row["id"]).binding_snapshot["source_training"] == frozen
    complete(transfer, row)
    finished = get(transfer, row["id"])
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "真实路径警告",
        "node_id": target["node_id"], "profile_id": "default", "checkpoint_path": finished["target_path"],
        "source_run_id": run.id, "transfer_id": row["id"]})
    assert response.status_code == 202, response.text
    assert response.json()["warnings"] == [CHECKPOINT_SOURCE_WARNING]


@pytest.mark.parametrize("case", ["pending", "rejected", "other-transfer", "other-node", "other-type", "other-attempt", "other-export", "unknown-code"])
def test_unaccepted_or_unrelated_export_warning_never_pollutes_transfer(transfer, case):
    platform, _, source, target, *_ = transfer
    row = create(transfer).json()
    payload = {"transfer_id": uid() if case == "other-transfer" else row["id"],
        "transfer_attempt": 2 if case == "other-attempt" else 1,
        "export_id": uid() if case == "other-export" else row["id"] + "-1"}
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=target["node_id"] if case == "other-node" else source["node_id"],
            type="POLICY_EXPORT_CANCEL" if case == "other-type" else "POLICY_EXPORT_STATUS",
            status={"pending": "PENDING", "rejected": "REJECTED"}.get(case, "APPLIED"), payload=payload,
            result={"warnings": [{"code": "OTHER"}] if case == "unknown-code" else [CHECKPOINT_SOURCE_WARNING]}, created_at=now()))
    assert get(transfer, row["id"])["warnings"] == []


def test_requires_distinct_online_nodes_and_bound_ssh_access(transfer):
    platform, _, source, target, *_ = transfer
    assert create(transfer, target_node_id=source["node_id"]).json()["error"]["code"] == "TRANSFER_SAME_NODE"
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).status = "OFFLINE"
    assert create(transfer).json()["error"]["code"] == "NODE_OFFLINE"
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).status = "ONLINE"
        for job in session.scalars(select(RemoteDeployment).where(RemoteDeployment.node_id == target["node_id"])):
            job.state = "DISCONNECTED"
    assert create(transfer).json()["error"]["code"] == "TRANSFER_SSH_UNAVAILABLE"


def test_export_receipt_requires_correct_agent_digest_and_terminal_first_result(transfer):
    row = create(transfer).json()
    command = export_command(transfer)
    assert ack_export(transfer, command, node=transfer[3]).status_code == 403
    assert ack_export(transfer, command, result={"state": "READY", "portable_config": PORTABLE, "portable_digest": "bad"}).status_code == 200
    assert get(transfer, row["id"])["error"]["code"] == "TRANSFER_EXPORT_INVALID"
    assert ack_export(transfer, command).status_code == 200
    assert get(transfer, row["id"])["state"] == "FAILED"


def test_old_agent_maps_export_error_to_actionable_result(transfer):
    row = create(transfer).json()
    ack_export(transfer, result={"error": {"code": "COMMAND_UNSUPPORTED", "message": "unknown command"}}, status="REJECTED")
    assert get(transfer, row["id"])["error"]["code"] == "TRANSFER_AGENT_UNSUPPORTED"


def test_export_failure_redacts_machine_and_platform_credentials(transfer):
    row = create(transfer).json()
    ack_export(transfer, result={"state": "FAILED", "error": {"code": "EXPORT_TEST", "message": SECRET + " test-enrollment"}})
    failed = get(transfer, row["id"])
    assert SECRET not in str(failed) and "test-enrollment" not in str(failed)


class FakeRemote:
    def __init__(self):
        self.closed = False
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True


def complete(transfer, row, *, transport=None):
    platform, manager, *_ = transfer
    remotes = []
    def remote(_connection):
        result = FakeRemote()
        remotes.append(result)
        return result
    platform.app.state.remote_manager._remote = remote
    def fake(source, target, source_path, target_directory, transfer_id, source_roots, target_roots, *, progress, cancelled, extra_files):
        assert source.connected and target.connected
        assert source_path == row["source_checkpoint_path"]
        assert source_roots == target_roots == ["/machine"]
        package = __import__("json").loads(extra_files["termitech-policy.json"])
        assert package == {"config": PORTABLE, "digest": canonical_digest(PORTABLE)}
        progress({"stage": "TRANSFERRING", "bytes_done": 50, "bytes_total": 100, "files_done": 1, "files_total": 2,
            "speed_bps": 250, "percent": 50})
        assert not cancelled()
        assert get(transfer, row["id"])["progress_pct"] == 50
        progress({"stage": "VERIFYING", "bytes_done": 100, "bytes_total": 100, "percent": 100})
        package_bytes = extra_files["termitech-policy.json"]
        entries = [{"path": "params/weights", "size": 100, "sha256": "a" * 64},
            {"path": "assets/norm_stats.json", "size": 2, "sha256": "b" * 64},
            {"path": "termitech-policy.json", "size": len(package_bytes), "sha256": hashlib.sha256(package_bytes).hexdigest()}]
        manifest = {"schema_version": 1, "files": entries,
            "bytes_total": sum(item["size"] for item in entries), "files_total": len(entries)}
        return {"target_path": target_directory + "/" + transfer_id, "manifest": manifest,
            "manifest_sha256": canonical_digest(manifest), "bytes_total": manifest["bytes_total"], "files_total": manifest["files_total"]}
    manager.transfer_function = transport or fake
    manager._run(row["id"])
    return remotes


def test_verified_transfer_registers_target_asset_and_requires_separate_deploy(transfer):
    platform, _, _, target, run, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    remotes = complete(transfer, row)
    finished = get(transfer, row["id"])
    assert finished["state"] == "SUCCEEDED" and finished["progress_pct"] == 100
    assert finished["target_path"] == "/machine/models/" + row["id"]
    assert all(remote.closed for remote in remotes)
    assert not any(item["type"] == "POLICY_DEPLOY_START" for item in platform.commands(target))
    with platform.app.state.db.read() as session:
        assets = list(session.scalars(select(Asset).where(Asset.node_id == target["node_id"])))
        assert len(assets) == 1 and assets[0].path == finished["target_path"]
        assert assets[0].origin_attempt_id is None
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "跨机部署", "node_id": target["node_id"],
        "profile_id": "default", "checkpoint_path": finished["target_path"], "config_name": "pi0_base",
        "source_run_id": run.id, "transfer_id": row["id"]})
    assert response.status_code == 202, response.text
    assert response.json()["transfer_id"] == row["id"]
    command = next(item for item in platform.commands(target) if item["type"] == "POLICY_DEPLOY_START")
    assert "source_training" not in command["payload"]
    assert command["payload"]["portable_policy"]["config"] == PORTABLE
    assert command["payload"]["portable_policy"]["checkpoint_manifest_digest"] == finished["manifest_sha256"]


def test_unsuccessful_or_mismatched_transfer_cannot_deploy(transfer):
    platform, _, source, target, run, *_ = transfer
    row = create(transfer).json()
    body = {"name": "跨机部署", "node_id": target["node_id"], "profile_id": "default", "checkpoint_path": "/machine/models/" + row["id"],
        "config_name": "pi0_base", "source_run_id": run.id, "transfer_id": row["id"]}
    assert platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json=body).json()["error"]["code"] == "TRANSFER_NOT_READY"
    ack_export(transfer)
    complete(transfer, row)
    for changes in ({"source_run_id": uid()}, {"checkpoint_path": "/machine/models/someone-else"}, {"node_id": source["node_id"]}):
        response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={**body, **changes})
        assert response.json()["error"]["code"] == "TRANSFER_DEPLOYMENT_MISMATCH"


@pytest.mark.parametrize("supplied", [{}, {"config_name": None}, {"config_name": "target_recipe_override"}])
def test_transferred_deployment_uses_exported_training_recipe_not_target_registry(transfer, supplied):
    platform, _, _, target, run, *_ = transfer
    with platform.app.state.db.write() as session:
        source = session.get(Run, run.id)
        source.code_expectation = {**source.code_expectation, "recipe": "source_training_only"}
    row = create(transfer).json()
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    # Deployment uses the frozen export snapshot, even if mutable source metadata changes later.
    with platform.app.state.db.write() as session:
        source = session.get(Run, run.id)
        source.code_expectation = {**source.code_expectation, "recipe": "later_training_metadata"}
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "跨机自动配置",
        "node_id": target["node_id"], "profile_id": "default", "checkpoint_path": finished["target_path"],
        "source_run_id": run.id, "transfer_id": row["id"], **supplied})
    assert response.status_code == 202, response.text
    assert response.json()["config_name"] == response.json()["source_training_recipe"] == "source_training_only"
    assert response.json()["config_source"] == "training"
    command = next(item for item in platform.commands(target) if item["type"] == "POLICY_DEPLOY_START")
    assert command["payload"]["config_name"] == "source_training_only"
    assert "source_training" not in command["payload"]
    assert command["payload"]["portable_policy"]["digest"] == finished["portable_digest"]
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, response.json()["id"]).config_name = "legacy_client_label"
    historical = platform.client.get("/api/v1/policy-deployments/" + response.json()["id"], headers=ADMIN).json()
    assert historical["config_name"] == "legacy_client_label"
    assert historical["source_training_recipe"] == "source_training_only"


def test_transferred_training_snapshot_cannot_fall_back_to_client_config(transfer):
    platform, _, _, target, run, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    with platform.app.state.db.write() as session:
        stored = session.get(CheckpointTransfer, row["id"])
        snapshot = deepcopy(stored.binding_snapshot)
        snapshot["source_training"]["code_expectation"]["automatic_preflight"]["plan"].pop("generated_config")
        stored.binding_snapshot = snapshot
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "拒绝缺失配置",
        "node_id": target["node_id"], "profile_id": "default", "checkpoint_path": finished["target_path"],
        "source_run_id": run.id, "transfer_id": row["id"], "config_name": "pi0_base"})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "POLICY_DEPLOY_SOURCE_CONFIG_MISSING"
    assert not any(item["type"] == "POLICY_DEPLOY_START" for item in platform.commands(target))


def test_retry_has_fresh_export_id_reuses_transfer_and_rejects_changed_binding(transfer):
    platform, _, _, target, *_ = transfer
    row = create(transfer).json()
    first = export_command(transfer)
    ack_export(transfer, result={"state": "FAILED", "error": {"code": "EXPORT_TEST", "message": "fixture failure"}})
    response = platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN)
    assert response.status_code == 202, response.text
    assert response.json()["id"] == row["id"] and response.json()["attempt"] == 2
    second = export_command(transfer)
    assert second["payload"]["export_id"] != first["payload"]["export_id"]
    ack_export(transfer, second, result={"state": "FAILED", "error": {"code": "EXPORT_TEST", "message": "fixture failure"}})
    with platform.app.state.db.write() as session:
        session.get(Profile, (target["node_id"], "default")).python = "/changed/python"
    retry = platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN)
    assert retry.status_code == 409 and retry.json()["error"]["code"] == "TRANSFER_BINDING_CHANGED"


def test_retry_clears_old_inventory_progress_until_the_partial_is_recounted(transfer):
    platform, manager, *_ = transfer
    row = create(transfer).json()
    manager._progress(row["id"], 1, {"stage": "TRANSFERRING", "bytes_done": 1168000000,
        "bytes_total": 44744756700, "files_done": 5, "files_total": 86, "speed_bps": 1000, "percent": 2.6})
    platform.client.post(BASE + "/" + row["id"] + "/cancel", headers=ADMIN)
    response = platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN)
    assert response.status_code == 202, response.text
    renewed = response.json()
    assert renewed["state"] == "QUEUED" and renewed["attempt"] == 2
    assert {key: renewed[key] for key in ("bytes_total", "bytes_transferred", "files_total", "files_transferred",
        "speed_bytes_per_second", "progress_pct")} == dict.fromkeys(("bytes_total", "bytes_transferred",
        "files_total", "files_transferred", "speed_bytes_per_second", "progress_pct"), 0)
    manager._progress(row["id"], 1, {"bytes_done": 9000000000, "bytes_total": 44744756700, "percent": 50})
    assert get(transfer, row["id"])["bytes_transferred"] == 0


def test_cancel_retires_commands_late_ack_never_resurrects_and_retry_can_resume(transfer):
    platform, *_ = transfer
    row = create(transfer).json()
    command = export_command(transfer)
    cancelled = platform.client.post(BASE + "/" + row["id"] + "/cancel", headers=ADMIN)
    assert cancelled.json()["state"] == "CANCELLED"
    assert ack_export(transfer, command).status_code == 200
    assert get(transfer, row["id"])["state"] == "CANCELLED"
    assert [item["type"] for item in platform.commands(transfer[2])] == ["POLICY_EXPORT_CANCEL"]
    cancellation = export_command(transfer, "POLICY_EXPORT_CANCEL")
    assert ack_export(transfer, cancellation, result={"state": "FAILED", "error": {
        "code": "POLICY_EXPORT_CANCELLED", "message": "owned source export stopped"}}).status_code == 200
    assert get(transfer, row["id"])["state"] == "CANCELLED"
    with platform.app.state.db.read() as session:
        assert session.get(Command, cancellation["id"]).status == "APPLIED"
    retry = platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN)
    assert retry.status_code == 202 and retry.json()["attempt"] == 2


def test_controller_restart_marks_interrupted_and_preserves_partial_progress(transfer):
    _, manager, *_ = transfer
    row = create(transfer).json()
    manager._progress(row["id"], 1, {"stage": "TRANSFERRING", "bytes_done": 50, "bytes_total": 100, "percent": 50})
    manager.start()
    state = get(transfer, row["id"])
    assert state["state"] == "INTERRUPTED" and state["bytes_transferred"] == 50
    assert state["error"]["code"] == "TRANSFER_INTERRUPTED"
    manager._progress(row["id"], 1, {"stage": "TRANSFERRING", "bytes_done": 90})
    assert get(transfer, row["id"])["state"] == "INTERRUPTED"


def test_failed_transport_closes_connections_keeps_progress_and_redacts_credentials(transfer):
    row = create(transfer).json()
    ack_export(transfer)
    def failure(*args, progress, **kwargs):
        progress({"stage": "TRANSFERRING", "bytes_done": 50, "bytes_total": 100, "percent": 50})
        raise APIError(502, "TRANSFER_CHECKSUM_MISMATCH", "checksum mismatch: test-enrollment " + SECRET)
    remotes = complete(transfer, row, transport=failure)
    finished = get(transfer, row["id"])
    assert finished["state"] == "FAILED" and finished["bytes_transferred"] == 50
    assert finished["error"]["code"] == "TRANSFER_CHECKSUM_MISMATCH"
    assert "test-enrollment" not in str(finished)
    assert SECRET not in str(finished)
    assert all(remote.closed for remote in remotes)


def test_active_transfer_blocks_both_agent_management_source_edits_and_rebinding(transfer):
    platform, _, source, target, *_ = transfer
    row = create(transfer).json()
    with platform.app.state.db.write() as session:
        for node in (source, target):
            assert active_transfers(session, node["node_id"])
            assert _in_use(session, node["node_id"], "default")
            with pytest.raises(APIError, match="源码"):
                require_checkout_idle(session, node["node_id"], "/machine/code")
            with pytest.raises(APIError) as caught:
                DeploymentManager._assert_idle(session, node["node_id"])
            assert caught.value.code == "NODE_BUSY"
    platform.client.post(BASE + "/" + row["id"] + "/cancel", headers=ADMIN)
    with platform.app.state.db.read() as session:
        assert not active_transfers(session, target["node_id"])


def test_export_timeout_fails_without_opening_ssh_and_can_retry(transfer):
    platform, manager, *_ = transfer
    row = create(transfer).json()
    manager._export_timeout = 0
    original = platform.app.state.remote_manager._remote
    platform.app.state.remote_manager._remote = lambda *_: pytest.fail("Timeout must not open SSH")
    manager._run(row["id"])
    failed = get(transfer, row["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "TRANSFER_EXPORT_TIMEOUT"
    assert [item["type"] for item in platform.commands(transfer[2])] == ["POLICY_EXPORT_CANCEL"]
    platform.app.state.remote_manager._remote = original
    assert platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN).status_code == 202


def test_cancel_running_copy_closes_both_connections_and_never_publishes_asset(transfer):
    platform, manager, _, target, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    entered, released = threading.Event(), threading.Event()
    remotes = []
    def remote(_connection):
        result = FakeRemote()
        remotes.append(result)
        return result
    def copying(*args, cancelled, **kwargs):
        entered.set()
        assert released.wait(5)
        assert cancelled()
        raise APIError(409, "TRANSFER_CANCELLED", "fixture cancellation")
    platform.app.state.remote_manager._remote = remote
    manager.transfer_function = copying
    worker = threading.Thread(target=manager._run, args=(row["id"],), daemon=True)
    manager._workers[row["id"]] = worker
    worker.start()
    try:
        assert entered.wait(5)
        cancelled = platform.client.post(BASE + "/" + row["id"] + "/cancel", headers=ADMIN)
        assert cancelled.status_code == 202 and cancelled.json()["state"] == "CANCELLED"
        assert all(remote.closed for remote in remotes)
        assert platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN).json()["error"]["code"] == "TRANSFER_CANCELLING"
    finally:
        released.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert get(transfer, row["id"])["state"] == "CANCELLED"
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Asset).where(Asset.node_id == target["node_id"]))) == []
    assert platform.client.post(BASE + "/" + row["id"] + "/retry", headers=ADMIN).status_code == 202


@pytest.mark.parametrize("change", ["missing_sidecar", "wrong_sidecar", "wrong_total", "unsafe_path", "duplicate_path"])
def test_incomplete_or_forged_manifest_never_becomes_success(transfer, change):
    platform, manager, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    platform.app.state.remote_manager._remote = lambda _: FakeRemote()
    def invalid(*args, **kwargs):
        package = kwargs["extra_files"]["termitech-policy.json"]
        entries = [{"path": "params/weights", "size": 100, "sha256": "a" * 64},
            {"path": "assets/norm_stats.json", "size": 2, "sha256": "b" * 64},
            {"path": "termitech-policy.json", "size": len(package), "sha256": hashlib.sha256(package).hexdigest()}]
        if change == "missing_sidecar": entries.pop()
        if change == "wrong_sidecar": entries[-1]["sha256"] = "0" * 64
        if change == "unsafe_path": entries[0]["path"] = "../weights"
        if change == "duplicate_path": entries.append(deepcopy(entries[0]))
        manifest = {"schema_version": 1, "files": entries,
            "bytes_total": sum(item["size"] for item in entries), "files_total": len(entries)}
        if change == "wrong_total": manifest["bytes_total"] += 1
        return {"target_path": row["target_directory"] + "/" + row["id"], "manifest": manifest,
            "manifest_sha256": canonical_digest(manifest), "bytes_total": manifest["bytes_total"], "files_total": manifest["files_total"]}
    manager.transfer_function = invalid
    manager._run(row["id"])
    failed = get(transfer, row["id"])
    assert failed["state"] == "FAILED" and failed["error"]["code"] == "TRANSFER_INVALID_RESULT"


def test_completed_transfer_deploy_needs_only_target_online_and_target_contract_unchanged(transfer):
    platform, _, source, target, run, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    with platform.app.state.db.write() as session:
        session.get(Node, source["node_id"]).status = "OFFLINE"
    body = {"name": "已同步模型离线源机部署", "node_id": target["node_id"], "profile_id": "default",
        "checkpoint_path": finished["target_path"], "config_name": "pi0_base", "source_run_id": run.id,
        "transfer_id": row["id"]}
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json=body)
    assert response.status_code == 202, response.text
    with platform.app.state.db.write() as session:
        session.get(Profile, (target["node_id"], "default")).code_dir = "/machine/changed"
    mismatch = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={**body, "port": 8001})
    assert mismatch.status_code == 409 and mismatch.json()["error"]["code"] == "TRANSFER_BINDING_CHANGED"


def test_unknown_progress_file_count_stays_unknown_until_verified_receipt(transfer):
    platform, manager, *_ = transfer
    row = create(transfer).json()
    manager._progress(row["id"], 1, {"stage": "TRANSFERRING", "files_done": None, "bytes_done": 50, "bytes_total": 100})
    assert get(transfer, row["id"])["files_transferred"] is None
    # Existing previews already have this column as NOT NULL. Unknown progress
    # must remain representable without dropping/rebuilding their job table.
    columns = inspect(platform.app.state.db.engine).get_columns("checkpoint_transfers")
    assert next(column for column in columns if column["name"] == "files_transferred")["nullable"] is False
    with platform.app.state.db.read() as session:
        assert session.get(CheckpointTransfer, row["id"]).files_transferred == -1


@pytest.mark.parametrize("capabilities", [None, [], ["POLICY_DEPLOY_START"], "POLICY_DEPLOY_PORTABLE_V1", {"POLICY_DEPLOY_PORTABLE_V1": True}])
def test_old_target_agent_is_blocked_before_sync_or_source_export(transfer, capabilities):
    platform, _, _, target, *_ = transfer
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).metrics = {"capabilities": capabilities}
    response = create(transfer)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "POLICY_DEPLOY_AGENT_UNSUPPORTED"
    assert platform.commands(transfer[2]) == []


def _portable_deployment(transfer):
    platform, _, _, target, run, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    body = {"name": "可移植策略", "node_id": target["node_id"], "profile_id": "default",
        "checkpoint_path": finished["target_path"], "config_name": "pi0_base", "source_run_id": run.id,
        "transfer_id": row["id"]}
    result = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json=body)
    assert result.status_code == 202, result.text
    command = next(item for item in platform.commands(target) if item["type"] == "POLICY_DEPLOY_START")
    return result.json(), command, finished, body


def test_target_agent_capability_is_rechecked_after_sync(transfer):
    platform, _, _, target, run, *_ = transfer
    row = create(transfer).json()
    ack_export(transfer)
    complete(transfer, row)
    finished = get(transfer, row["id"])
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).metrics = {"capabilities": ["POLICY_DEPLOY_START"]}
    response = platform.client.post("/api/v1/policy-deployments", headers=ADMIN, json={"name": "目标旧 Agent", "node_id": target["node_id"],
        "profile_id": "default", "checkpoint_path": finished["target_path"], "config_name": "pi0_base", "source_run_id": run.id,
        "transfer_id": row["id"]})
    assert response.status_code == 409 and response.json()["error"]["code"] == "POLICY_DEPLOY_AGENT_UNSUPPORTED"
    assert not any(item["type"] == "POLICY_DEPLOY_START" for item in platform.commands(target))


@pytest.mark.parametrize("bad", [{}, {"portable_digest": "0" * 64}, {"checkpoint_manifest_digest": "0" * 64},
    {"portable_digest": None, "checkpoint_manifest_digest": None}])
def test_portable_running_requires_exact_config_and_checkpoint_receipts_and_keeps_gpu(transfer, bad):
    platform, _, _, target, *_ = transfer
    deployment, command, finished, _ = _portable_deployment(transfer)
    receipt = running()
    if bad:
        receipt.update(portable_digest=finished["portable_digest"], checkpoint_manifest_digest=finished["manifest_sha256"])
        receipt.update(bad)
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=target["headers"],
        json={"node_id": target["node_id"], "status": "APPLIED", "result": receipt})
    assert response.status_code == 200
    detail = platform.client.get("/api/v1/policy-deployments/" + deployment["id"], headers=ADMIN).json()
    assert detail["state"] == "UNKNOWN" and detail["error"]["code"] == "POLICY_DEPLOY_PORTABLE_MISMATCH"
    node = platform.client.get("/api/v1/nodes/" + target["node_id"], headers=ADMIN).json()
    assert node["gpus"][0]["reserved_by"] == "deployment:" + deployment["id"]
    # A process-stop receipt remains sufficient to recover a rejected legacy
    # launch; it does not need to claim the model was ever loaded correctly.
    platform.client.post("/api/v1/policy-deployments/" + deployment["id"] + "/stop", headers=ADMIN)
    stop = next(item for item in platform.commands(target) if item["type"] == "POLICY_DEPLOY_STOP")
    platform.client.post(f"/agent/v1/commands/{stop['id']}/ack", headers=target["headers"], json={"node_id": target["node_id"],
        "status": "APPLIED", "result": {"state": "STOPPED", "pid": None, "checks": [], "logs": ""}})
    detail = platform.client.get("/api/v1/policy-deployments/" + deployment["id"], headers=ADMIN).json()
    assert detail["state"] == "STOPPED"


def test_portable_ready_receipt_accepts_only_the_exact_transferred_artifact(transfer):
    platform, _, _, target, *_ = transfer
    deployment, command, finished, _ = _portable_deployment(transfer)
    receipt = running(portable_digest=finished["portable_digest"], checkpoint_manifest_digest=finished["manifest_sha256"])
    result = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=target["headers"],
        json={"node_id": target["node_id"], "status": "APPLIED", "result": receipt})
    assert result.status_code == 200
    detail = platform.client.get("/api/v1/policy-deployments/" + deployment["id"], headers=ADMIN).json()
    assert detail["state"] == "RUNNING"
    with platform.app.state.db.read() as session:
        assert session.get(Command, command["id"]).result["portable_digest"] == finished["portable_digest"]
