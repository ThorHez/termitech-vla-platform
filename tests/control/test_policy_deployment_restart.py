"""Restart one business record while fencing its old Agent execution."""

from copy import deepcopy

import pytest
from sqlalchemy import select

from vla_control.checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer
from vla_control.common import now, offset_time, uid
from vla_control.models import Command, Node, Profile, Run
from vla_control.policy_deployment_models import PolicyDeployment
from vla_control.policy_deployments import queue_status_checks

from .conftest import ADMIN
from .test_checkpoint_transfers import _portable_deployment, transfer
from .test_policy_deployments import BASE, ack, action, add_run, command, create, get, ready, running
from .test_recipe_details import register


def enable_restart(platform, node):
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        metrics = deepcopy(row.metrics or {})
        metrics["capabilities"] = [*(metrics.get("capabilities") or []), "POLICY_DEPLOY_RESTART_V1"]
        row.metrics = metrics


def terminal(platform, node, state="STOPPED", **changes):
    deployment = create(platform, node, **changes).json()
    initial = command(platform, node, "POLICY_DEPLOY_START")
    if state == "FAILED":
        assert ack(platform, node, initial, {"state": "FAILED", "error": {"code": "LOAD_FAILED", "message": "fixture"}}).status_code == 200
    else:
        assert ack(platform, node, initial, running()).status_code == 200
        assert action(platform, deployment, "stop").status_code == 202
        assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_STOP"), {"state": "STOPPED"}).status_code == 200
    return deployment, initial


@pytest.mark.parametrize("state", ["STOPPED", "FAILED"])
def test_restart_reuses_business_record_and_is_idempotent_while_start_pending(platform, state):
    node = register(platform)
    enable_restart(platform, node)
    deployment, initial = terminal(platform, node, state)
    with platform.app.state.db.write() as session:
        row = session.get(PolicyDeployment, deployment["id"])
        row.test_result = {"status": "FAILED", "error": {"code": "OLD_TEST", "message": "old"}}
        row.error = {"code": "OLD_ERROR", "message": "old"}
    response = action(platform, deployment, "start", checkpoint_path="/different", config_name="tamper")
    assert response.status_code == 202, response.text
    current = response.json()
    restarted = command(platform, node, "POLICY_DEPLOY_START")
    execution = restarted["payload"]["execution_id"]
    assert current["id"] == deployment["id"] and current["created_at"] == deployment["created_at"]
    assert current["state"] == "PENDING" and current["execution_id"] == execution
    assert current["pending_command"] == "POLICY_DEPLOY_START"
    assert current["pid"] is current["error"] is current["endpoint"] is current["test_result"] is None
    assert "execution_id" not in initial["payload"]
    assert execution != deployment["id"] and now() < restarted["payload"]["expires_at"] <= offset_time(121)
    for key in ("checkpoint_path", "config_name", "gpu_uuid", "gpu_mode", "gpu_memory_mb", "profile_id", "port"):
        assert restarted["payload"][key] == initial["payload"][key]
    assert action(platform, deployment, "start").json()["execution_id"] == execution
    assert len([item for item in platform.commands(node) if item["type"] == "POLICY_DEPLOY_START"]) == 1
    with platform.app.state.db.read() as session:
        assert len(list(session.scalars(select(PolicyDeployment)))) == 1
        starts = list(session.scalars(select(Command).where(Command.type == "POLICY_DEPLOY_START")))
        assert len(starts) == 2 and session.get(Command, initial["id"]).result is not None


def test_new_execution_token_follows_refresh_test_stop_scheduler_and_next_restart(platform):
    node = register(platform)
    enable_restart(platform, node)
    deployment, _ = terminal(platform, node)
    action(platform, deployment, "start")
    started = command(platform, node, "POLICY_DEPLOY_START")
    token = started["payload"]["execution_id"]
    assert ack(platform, node, started, running(execution_id=token)).status_code == 200
    action(platform, deployment, "refresh")
    status = command(platform, node, "POLICY_DEPLOY_STATUS")
    assert status["payload"]["execution_id"] == token
    ack(platform, node, status, running(execution_id=token))
    action(platform, deployment, "test", observation_path="/machine/observation.json")
    tested = command(platform, node, "POLICY_DEPLOY_TEST")
    assert tested["payload"]["execution_id"] == token
    ack(platform, node, tested, running(execution_id=token, test_result={"status": "FAILED"}))
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).updated_at = offset_time(-30)
        queue_status_checks(session, session.get(Node, node["node_id"]), platform.settings)
    scheduled = command(platform, node, "POLICY_DEPLOY_STATUS")
    assert scheduled["payload"]["execution_id"] == token
    ack(platform, node, scheduled, running(execution_id=token))
    action(platform, deployment, "stop")
    stopped = command(platform, node, "POLICY_DEPLOY_STOP")
    assert stopped["payload"]["execution_id"] == token
    ack(platform, node, stopped, {"state": "STOPPED", "execution_id": token})
    next_start = action(platform, deployment, "start")
    assert next_start.status_code == 202, next_start.text
    assert next_start.json()["execution_id"] != token


def test_restart_retires_confirmed_old_pending_and_ignores_old_token_or_created_at_receipts(platform):
    node = register(platform)
    enable_restart(platform, node)
    deployment = ready(platform, node)
    action(platform, deployment, "refresh")
    older = command(platform, node, "POLICY_DEPLOY_STATUS")
    action(platform, deployment, "stop")
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_STOP"), {"state": "STOPPED"})
    response = action(platform, deployment, "start")
    assert response.status_code == 202, response.text
    token = response.json()["execution_id"]
    with platform.app.state.db.read() as session:
        retired = session.get(Command, older["id"])
        assert retired.status == "REJECTED" and retired.result["error"]["code"] == "POLICY_DEPLOY_EXECUTION_SUPERSEDED"
    ack(platform, node, older, running())
    for by_timestamp in (False, True):
        with platform.app.state.db.write() as session:
            stale = Command(id=uid(), node_id=node["node_id"], type="POLICY_DEPLOY_STOP", status="PENDING",
                payload={"deployment_id": deployment["id"], **({"execution_id": token} if by_timestamp else {})},
                created_at=older["payload"].get("created_at", offset_time(-10)) if by_timestamp else now())
            session.add(stale)
            stale_id = stale.id
        ack(platform, node, {"id": stale_id}, {"state": "STOPPED", **({"execution_id": token} if by_timestamp else {})})
        assert get(platform, deployment)["state"] == "PENDING"
        with platform.app.state.db.read() as session:
            assert session.get(Command, stale_id).result["error"]["code"] == "POLICY_DEPLOY_EXECUTION_SUPERSEDED"


@pytest.mark.parametrize("receipt_id", [None, "wrong-execution"])
def test_agent_must_echo_the_new_execution_id_before_any_ready_state_is_accepted(platform, receipt_id):
    node = register(platform)
    enable_restart(platform, node)
    deployment, _ = terminal(platform, node)
    action(platform, deployment, "start")
    result = running(**({"execution_id": receipt_id} if receipt_id is not None else {}))
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), result)
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN" and current["error"]["code"] == "POLICY_DEPLOY_EXECUTION_MISMATCH"
    assert action(platform, deployment, "start").status_code == 409


@pytest.mark.parametrize("state", ["RUNNING", "STARTING", "PENDING", "UNKNOWN", "STOPPING"])
def test_restart_rejects_active_uncertain_or_stopping_records(platform, state):
    node = register(platform)
    enable_restart(platform, node)
    deployment = create(platform, node).json()
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).state = state
    response = action(platform, deployment, "start")
    assert response.status_code == 409 and response.json()["error"]["code"] == "POLICY_DEPLOY_NOT_RESTARTABLE"


@pytest.mark.parametrize("state", ["STOPPED", "FAILED"])
def test_terminal_label_without_stop_or_failure_receipt_is_not_restart_permission(platform, state):
    node = register(platform)
    enable_restart(platform, node)
    deployment = create(platform, node).json()
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).state = state
    response = action(platform, deployment, "start")
    assert response.status_code == 409 and response.json()["error"]["code"] == "POLICY_DEPLOY_STOP_UNCONFIRMED"


@pytest.mark.parametrize("blocker,code", [
    ("gpu", "POLICY_DEPLOY_GPU_UNAVAILABLE"), ("port", "POLICY_DEPLOY_PORT_BUSY"),
    ("capability", "POLICY_DEPLOY_RESTART_UNSUPPORTED"), ("offline", None),
    ("roots", "PATH_OUTSIDE_ROOTS"), ("profile", "PROFILE_NODE_MISMATCH"),
    ("rebind", "PROFILE_REBIND_PENDING"), ("pending", "POLICY_DEPLOY_BUSY"),
])
def test_restart_rechecks_resources_environment_and_agent_support(platform, blocker, code):
    node = register(platform)
    enable_restart(platform, node)
    deployment, _ = terminal(platform, node)
    if blocker == "pending":
        action(platform, deployment, "refresh")
    with platform.app.state.db.write() as session:
        target = session.get(Node, node["node_id"])
        if blocker == "gpu":
            gpu = deepcopy(target.gpus)
            gpu[0]["processes"] = [{"pid": 123, "kind": "compute", "ownership": "external"}]
            target.gpus = gpu
        elif blocker == "port":
            session.add(PolicyDeployment(id=uid(), name="another service", node_id=target.id, profile_id="default",
                checkpoint_path="/machine/model", config_name="pi0_base", port=8000, gpu_uuid="GPU-OTHER",
                state="RUNNING", logs="", checks=[], created_at=now(), updated_at=now()))
        elif blocker == "capability":
            target.metrics = {"capabilities": []}
        elif blocker == "offline":
            target.status = "OFFLINE"
        elif blocker == "roots":
            target.roots = ["/other"]
        elif blocker == "profile":
            session.get(Profile, (target.id, "default")).active = False
        elif blocker == "rebind":
            session.add(Command(id=uid(), node_id=target.id, type="PROFILE_REBIND", status="PENDING", created_at=now(),
                payload={"profile_id": "default", "expected_code_dir": "/machine/code", "code_dir": "/machine/new"}))
    response = action(platform, deployment, "start")
    assert response.status_code in {409, 422}, response.text
    if code:
        assert response.json()["error"]["code"] == code
    with platform.app.state.db.read() as session:
        starts = list(session.scalars(select(Command).where(Command.type == "POLICY_DEPLOY_START")))
        assert len(starts) == 1


def test_same_machine_restart_uses_original_start_snapshot_even_if_source_run_metadata_changes(platform):
    node = register(platform)
    enable_restart(platform, node)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        run_id = run.id
    deployment, initial = terminal(platform, node, source_run_id=run_id)
    with platform.app.state.db.write() as session:
        session.get(Run, run_id).code_expectation = {"recipe": "later-broken-recipe"}
    response = action(platform, deployment, "start")
    assert response.status_code == 202, response.text
    restarted = command(platform, node, "POLICY_DEPLOY_START")
    assert restarted["payload"]["source_training"] == initial["payload"]["source_training"]
    assert restarted["payload"]["config_name"] == "pi0_base"


@pytest.mark.parametrize("change,code", [(None, None), ("profile", "TRANSFER_BINDING_CHANGED"),
    ("digest", "TRANSFER_RECEIPT_INVALID"), ("state", "TRANSFER_NOT_READY")])
def test_portable_restart_preserves_and_revalidates_the_exact_transfer(transfer, change, code):
    platform, _, _, target, *_ = transfer
    deployment, initial, finished, _ = _portable_deployment(transfer)
    enable_restart(platform, target)
    ack(platform, target, initial, running(portable_digest=finished["portable_digest"],
        checkpoint_manifest_digest=finished["manifest_sha256"]))
    action(platform, deployment, "stop")
    ack(platform, target, command(platform, target, "POLICY_DEPLOY_STOP"), {"state": "STOPPED"})
    with platform.app.state.db.write() as session:
        source = session.get(CheckpointTransfer, finished["id"])
        if change == "profile":
            session.get(Profile, (target["node_id"], "default")).python = "/machine/another/python"
        elif change == "digest":
            source.portable_digest = "0" * 64
        elif change == "state":
            source.state = "FAILED"
    response = action(platform, deployment, "start")
    if code:
        assert response.status_code in {409, 422} and response.json()["error"]["code"] == code
    else:
        assert response.status_code == 202, response.text
        restarted = command(platform, target, "POLICY_DEPLOY_START")
        assert restarted["payload"]["portable_policy"] == initial["payload"]["portable_policy"]
        assert response.json()["transfer_id"] == finished["id"]
        with platform.app.state.db.read() as session:
            assert len(list(session.scalars(select(PolicyDeployment)))) == 1
            assert session.get(PolicyDeploymentTransfer, deployment["id"]).transfer_id == finished["id"]


@pytest.mark.parametrize("role,expected", [("operator", 202), ("viewer", 403)])
def test_restart_permission_is_training_write(platform, monkeypatch, role, expected):
    from vla_control.auth import ROLE_PERMISSIONS
    node = register(platform)
    enable_restart(platform, node)
    deployment, _ = terminal(platform, node)
    monkeypatch.setitem(ROLE_PERMISSIONS, "operator", frozenset({"platform.read", "training.write"}))
    platform.app.state.auth.create_user("restart-user", "test-user-password", role)
    platform.client.base_url = "https://testserver"
    login = platform.client.post("/api/v1/auth/login", json={"username": "restart-user", "password": "test-user-password"})
    response = platform.client.post(BASE + "/" + deployment["id"] + "/start",
        headers={"X-CSRF-Token": login.json()["csrf_token"]})
    assert response.status_code == expected, response.text
