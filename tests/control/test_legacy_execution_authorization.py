"""Turning off legacy admission cannot start unclaimed accountless execution."""

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.checkpoint_transfer_models import CheckpointTransfer
from vla_control.common import now, uid
from vla_control.main import create_app
from vla_control.models import Allocation, Command, Preview, Run
from vla_control.settings import Settings
from vla_protocol import canonical_digest

from .conftest import ADMIN, Platform, snapshot
from .test_asset_transfers import create as copy_asset, fixture, result_for, run_copy
from .test_checkpoint_transfers import (PORTABLE, ack_export, complete,
    create as copy_checkpoint, transfer)
from .test_policy_deployments import create as create_policy, running
from .test_recipe_details import register


TASKS = ("START", "PREFLIGHT", "POLICY_DEPLOY_START", "POLICY_EXPORT")


@pytest.fixture(params=TASKS)
def pending_task(platform, request):
    kind = request.param
    if kind == "POLICY_EXPORT":
        selected = request.getfixturevalue("transfer")
        node = selected[2]
        response = copy_checkpoint(selected)
    elif kind == "POLICY_DEPLOY_START":
        node = register(platform)
        response = create_policy(platform, node)
    else:
        node = platform.register()
        if kind == "START":
            platform.run(node)
            assert platform.schedule() == 1
            response = None
        else:
            response = platform.client.post("/api/v1/run-previews", headers=ADMIN,
                json={"name": "legacy preflight", "node_id": node["node_id"],
                      "profile_id": "default", "template_id": "smoke", "inputs": {},
                      "parameters": {"steps": 3}, "gpu_count": 1})
    if response is not None:
        assert response.status_code == 202, response.text
    with platform.app.state.db.read() as session:
        command = session.scalar(select(Command).where(Command.type == kind))
        assert command is not None and command.delivered_at is None
        return platform, node, command.id, kind, dict(command.payload)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("users_exist", [False, True])
def test_first_delivery_requires_explicit_legacy_and_empty_user_table(pending_task, legacy, users_exist):
    platform, node, identifier, _, _ = pending_task
    if users_exist:
        platform.app.state.auth.create_user("existing-admin", "legacy-boundary-test-password", "admin")
    # Use a client without lifespan so the matrix isolates first delivery,
    # rather than the separate transfer-startup interruption policy.
    app = create_app(replace(platform.settings, legacy_admin_enabled=legacy))
    client = TestClient(app)
    try:
        response = client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]})
        assert response.status_code == 200, response.text
        delivered = identifier in {item["id"] for item in response.json()}
    finally:
        client.close()
        app.state.db.close()
    assert delivered is (legacy and not users_exist)
    with platform.app.state.db.read() as session:
        command = session.get(Command, identifier)
        assert command.status == "PENDING"
        assert (command.delivered_at is not None) is delivered


def test_previously_delivered_task_keeps_retries_and_late_ack_when_legacy_is_disabled(pending_task):
    platform, node, identifier, kind, payload = pending_task
    assert identifier in {item["id"] for item in platform.commands(node)}
    if kind == "START":
        result = {"attempt_id": payload["attempt_id"], "accepted": True}
    elif kind == "PREFLIGHT":
        result = {"checks": [{"name": "fixture", "ok": True, "message": "checked"}],
                  "code_snapshot": snapshot(recipe=payload["recipe"])}
    elif kind == "POLICY_DEPLOY_START":
        result = running()
    else:
        result = {"state": "READY", "portable_config": PORTABLE,
                  "portable_digest": canonical_digest(PORTABLE)}
    app = create_app(replace(platform.settings, legacy_admin_enabled=False))
    client = TestClient(app)
    try:
        retry = client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]})
        assert retry.status_code == 200 and identifier in {item["id"] for item in retry.json()}
        ack = client.post(f"/agent/v1/commands/{identifier}/ack", headers=node["headers"],
            json={"node_id": node["node_id"], "status": "APPLIED", "result": result})
        assert ack.status_code == 200, ack.text
        with platform.app.state.db.read() as session:
            assert session.get(Command, identifier).status == "APPLIED"
    finally:
        client.close()
        app.state.db.close()


@pytest.mark.parametrize("delivered_before_restart", [False, True])
def test_real_restart_with_legacy_disabled_freezes_only_never_delivered_start(tmp_path, delivered_before_restart):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "restart.sqlite"),
        admin_token="test-admin", enrollment_token="test-enrollment",
        scheduler_enabled=False, legacy_admin_enabled=True)
    original = create_app(settings)
    with TestClient(original) as client:
        platform = Platform(client, original, settings)
        node = platform.register()
        active = platform.run(node)
        queued = platform.run(node)
        assert platform.schedule() == 1
        with original.state.db.read() as session:
            command = session.scalar(select(Command).where(Command.type == "START"))
            identifier, payload = command.id, dict(command.payload)
            assert command.delivered_at is None
        if delivered_before_restart:
            assert identifier in {item["id"] for item in platform.commands(node)}

    current = replace(settings, legacy_admin_enabled=False)
    restarted = create_app(current)
    with TestClient(restarted) as client:
        platform = Platform(client, restarted, current)
        assert client.get("/api/v1/nodes", headers=ADMIN).status_code == 401
        assert platform.schedule() == 0
        assert (identifier in {item["id"] for item in platform.commands(node)}) is delivered_before_restart
        with restarted.state.db.read() as session:
            assert session.get(Run, queued["id"]).state == "QUEUED"
            assert session.get(Run, queued["id"]).queue_reason == "ACCOUNT_EXECUTION_REVOKED"
            assert session.get(Command, identifier).status == "PENDING"
            assert (session.get(Command, identifier).delivered_at is not None) is delivered_before_restart
            assert len(list(session.scalars(select(Allocation)))) == 1
        if delivered_before_restart:
            ack = client.post(f"/agent/v1/commands/{identifier}/ack", headers=node["headers"],
                json={"node_id": node["node_id"], "status": "APPLIED",
                      "result": {"attempt_id": payload["attempt_id"], "accepted": True}})
            assert ack.status_code == 200, ack.text
            event = platform.emit(node, payload["attempt_id"], 1, "STARTED",
                {"pid": 1234, "code_snapshot": active["code_expectation"],
                 "output_dir": "/machine/output/" + payload["attempt_id"]})
            assert event.status_code == 200, event.text
            with restarted.state.db.read() as session:
                assert session.get(Run, active["id"]).state == "RUNNING"
                assert len(list(session.scalars(select(Allocation)))) == 1


def test_legacy_disabled_retains_safety_command_delivery(platform):
    node = platform.register()
    kinds = ("CANCEL", "POLICY_DEPLOY_STOP", "POLICY_DEPLOY_STATUS", "POLICY_EXPORT_CANCEL", "POLICY_EXPORT_STATUS")
    with platform.app.state.db.write() as session:
        identifiers = []
        for kind in kinds:
            identifier = uid()
            identifiers.append(identifier)
            session.add(Command(id=identifier, node_id=node["node_id"], type=kind,
                payload={}, status="PENDING", created_at=now()))
    app = create_app(replace(platform.settings, legacy_admin_enabled=False))
    client = TestClient(app)
    try:
        response = client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]})
        assert response.status_code == 200, response.text
        assert {item["id"] for item in response.json()} == set(identifiers)
    finally:
        client.close()
        app.state.db.close()


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
def test_worker_with_legacy_disabled_rejects_unclaimed_work_without_ssh_and_releases_slot(request, monkeypatch, kind):
    selected = request.getfixturevalue("fixture" if kind == "asset" else "transfer")
    platform, manager, *_ = selected
    response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    manager.settings = replace(manager.settings, legacy_admin_enabled=False)
    monkeypatch.setattr(manager.remote_manager, "_remote", lambda *_: pytest.fail("An unauthorized new claim must not construct SSH"))
    manager._run(identifier)
    model = AssetTransfer if kind == "asset" else CheckpointTransfer
    with platform.app.state.db.read() as session:
        row = session.get(model, identifier)
        assert row.state == "INTERRUPTED" and row.error["code"] == "ACCOUNT_EXECUTION_REVOKED"
        assert row.owner_user_id is None and row.execution_user_id is None
        if kind == "checkpoint":
            export = session.scalar(select(Command).where(Command.type == "POLICY_EXPORT"))
            assert export.status == "REJECTED" and export.delivered_at is None
            assert session.scalar(select(Command.id).where(Command.type == "POLICY_EXPORT_CANCEL"))
    assert manager._semaphore.acquire(blocking=False)
    assert manager._semaphore.acquire(blocking=False)
    assert not manager._semaphore.acquire(blocking=False)
    manager._semaphore.release()
    manager._semaphore.release()


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
def test_explicit_empty_user_legacy_workers_can_complete(request, monkeypatch, kind):
    selected = request.getfixturevalue("fixture" if kind == "asset" else "transfer")
    platform, manager, *_ = selected
    response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
    assert response.status_code == 202, response.text
    item = response.json()
    assert manager.settings.legacy_admin_enabled
    if kind == "asset":
        assert run_copy(selected, item["id"], monkeypatch)["state"] == "SUCCEEDED"
    else:
        assert ack_export(selected).status_code == 200
        remotes = complete(selected, item)
        assert remotes and all(remote.connected and remote.closed for remote in remotes)
        with platform.app.state.db.read() as session:
            assert session.get(CheckpointTransfer, item["id"]).state == "SUCCEEDED"


def test_already_delivered_export_can_finish_after_legacy_is_disabled(transfer):
    platform, manager, source, *_ = transfer
    response = copy_checkpoint(transfer)
    assert response.status_code == 202, response.text
    item = response.json()
    command = next(row for row in platform.commands(source) if row["type"] == "POLICY_EXPORT")
    manager.settings = replace(manager.settings, legacy_admin_enabled=False)
    assert ack_export(transfer, command).status_code == 200
    remotes = complete(transfer, item)
    assert remotes and all(remote.connected and remote.closed for remote in remotes)
    with platform.app.state.db.read() as session:
        assert session.get(CheckpointTransfer, item["id"]).state == "SUCCEEDED"
        assert session.scalar(select(Command.id).where(Command.type == "POLICY_EXPORT_CANCEL")) is None


def test_asset_copy_already_claimed_can_finish_when_legacy_is_disabled(fixture, monkeypatch):
    _, manager, *_ = fixture
    response = copy_asset(fixture)
    assert response.status_code == 202, response.text

    def disable_during_active_copy(*args, **kwargs):
        assert args[0].connected and args[1].connected
        manager.settings = replace(manager.settings, legacy_admin_enabled=False)
        return result_for(args, kwargs)

    finished = run_copy(fixture, response.json()["id"], monkeypatch, transfer=disable_during_active_copy)
    assert finished["state"] == "SUCCEEDED"


def test_real_operator_run_is_created_scheduled_and_delivered_with_legacy_disabled(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "account-run.sqlite"),
        enrollment_token="test-enrollment", scheduler_enabled=False, legacy_admin_enabled=False)
    original = create_app(settings)
    with TestClient(original) as client:
        platform = Platform(client, original, settings)
        node = platform.register()
        password = "production-account-regression-password"
        original.state.auth.create_user("account-admin", password, "admin")
        operator = original.state.auth.create_user("account-operator", password, "operator")
        actor_client = TestClient(original, base_url="https://testserver")
        try:
            login = actor_client.post("/api/v1/auth/login",
                json={"username": "account-operator", "password": password})
            assert login.status_code == 200, login.text
            actor_client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
            authenticated = actor_client.get("/api/v1/auth/me").json()
            assert authenticated["user"]["id"] == operator["id"]
            assert authenticated["user"]["role"] == "operator"
            # Normal routed preflight and run creation, with a real account
            # cookie; no ownership columns are patched after creation.
            actor = Platform(actor_client, original, settings)
            run = actor.run(node)
            assert run["owner_user_id"] == operator["id"]
            assert actor.schedule() == 1
            with original.state.db.read() as session:
                command = session.scalar(select(Command).where(Command.type == "START"))
                identifier = command.id
                assert command.payload["run_id"] == run["id"] and command.delivered_at is None
                assert session.scalar(select(Preview).where(Preview.run_id == run["id"])).owner_user_id == operator["id"]
        finally:
            actor_client.close()

    restarted = create_app(settings)
    with TestClient(restarted) as client:
        platform = Platform(client, restarted, settings)
        assert identifier in {item["id"] for item in platform.commands(node)}
        with restarted.state.db.read() as session:
            assert session.get(Command, identifier).delivered_at is not None
            assert session.get(Run, run["id"]).owner_user_id == operator["id"]
            assert session.scalar(select(Preview).where(Preview.run_id == run["id"])).owner_user_id == operator["id"]
