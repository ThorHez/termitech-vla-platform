"""Deployment record removal waits for stopped services and finished transfers."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.checkpoint_transfer_models import CheckpointTransfer
from vla_control.common import now, uid
from vla_control.deployment_record_models import DeploymentRecord
from vla_control.policy_deployment_models import PolicyDeployment

from .conftest import ADMIN, ENROLL
from .test_checkpoint_transfers import create as create_transfer, transfer
from .test_deployment_records import cancel, deploy, snapshot, succeed
from .test_policy_deployment_restart import terminal
from .test_policy_deployments import ack, action, command, running
from .test_recipe_details import register


BASE = "/api/v1/deployment-records"


def remove(platform, record_id):
    return platform.client.delete(f"{BASE}/{record_id}", headers=ADMIN)


def draft(platform):
    identifier = uid()
    with platform.app.state.db.write() as session:
        session.add(DeploymentRecord(id=identifier, name="empty draft", created_at=now()))
    return identifier


def test_finished_preparation_can_be_removed_without_deleting_receipts_or_training(transfer):
    platform, _, _, _, run, *_ = transfer
    child = create_transfer(transfer, record_name="finished preparation").json()
    cancel(transfer, child)
    record_id = child["record_id"]
    before = snapshot(platform)
    response = remove(platform, record_id)
    assert response.status_code == 204 and response.content == b""
    assert snapshot(platform) == before
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    for alias in (record_id, child["id"]):
        assert platform.client.get(f"{BASE}/{alias}", headers=ADMIN).status_code == 404
    assert platform.client.get(f"/api/v1/runs/{run.id}", headers=ADMIN).status_code == 200
    assert platform.client.get("/api/v1/checkpoint-transfers", headers=ADMIN).json() == []
    child_path = f"/api/v1/checkpoint-transfers/{child['id']}"
    assert platform.client.get(child_path, headers=ADMIN).status_code == 404
    for operation in ("retry", "cancel"):
        assert platform.client.post(child_path + "/" + operation, headers=ADMIN).status_code == 404
    with platform.app.state.db.read() as session:
        deleted_at = session.get(DeploymentRecord, record_id).deleted_at
        assert deleted_at and session.get(CheckpointTransfer, child["id"]) is not None
    assert remove(platform, record_id).status_code == 204
    with platform.app.state.db.read() as session:
        assert session.get(DeploymentRecord, record_id).deleted_at == deleted_at
    platform.app.state.db.initialize()
    assert platform.client.get(BASE, headers=ADMIN).json() == []


@pytest.mark.parametrize("state", ["PENDING", "STARTING", "RUNNING", "STOPPING", "UNKNOWN"])
def test_active_runtime_prevents_record_removal_without_mutating_execution(platform, state):
    node = register(platform)
    runtime, _ = terminal(platform, node)
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, runtime["id"]).state = state
    before = snapshot(platform, include_records=True)
    response = remove(platform, runtime["record_id"])
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "DEPLOYMENT_RECORD_ACTIVE"
    assert snapshot(platform, include_records=True) == before
    assert platform.client.get(f"{BASE}/{runtime['record_id']}", headers=ADMIN).status_code == 200


@pytest.mark.parametrize("witness", [{"pid": 1234}, {"endpoint": "ws://127.0.0.1:8000"}])
def test_terminal_runtime_with_uncertain_process_cannot_be_removed(platform, witness):
    node = register(platform)
    runtime, _ = terminal(platform, node)
    with platform.app.state.db.write() as session:
        row = session.get(PolicyDeployment, runtime["id"])
        for key, value in witness.items():
            setattr(row, key, value)
    before = snapshot(platform, include_records=True)
    response = remove(platform, runtime["record_id"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "DEPLOYMENT_RECORD_ACTIVE"
    assert snapshot(platform, include_records=True) == before


@pytest.mark.parametrize("state", ["QUEUED", "PREPARING", "EXPORTING", "TRANSFERRING", "VERIFYING"])
def test_unfinished_transfer_prevents_record_removal(transfer, state):
    platform, *_ = transfer
    child = create_transfer(transfer, record_name="active transfer").json()
    with platform.app.state.db.write() as session:
        session.get(CheckpointTransfer, child["id"]).state = state
    before = snapshot(platform, include_records=True)
    response = remove(platform, child["record_id"])
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "DEPLOYMENT_RECORD_ACTIVE"
    assert snapshot(platform, include_records=True) == before


@pytest.mark.parametrize("state", ["STOPPED", "FAILED"])
def test_confirmed_terminal_runtime_can_be_removed_and_hidden_from_old_actions(platform, state):
    node = register(platform)
    runtime, _ = terminal(platform, node, state)
    before = snapshot(platform)
    response = remove(platform, runtime["record_id"])
    assert response.status_code == 204, response.text
    assert snapshot(platform) == before
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    assert platform.client.get("/api/v1/policy-deployments", headers=ADMIN).json() == []
    for alias in (runtime["record_id"], runtime["id"]):
        assert platform.client.get(f"{BASE}/{alias}", headers=ADMIN).status_code == 404
    runtime_path = f"/api/v1/policy-deployments/{runtime['id']}"
    assert platform.client.get(runtime_path, headers=ADMIN).status_code == 404
    for action in ("refresh", "start", "stop", "test"):
        response = platform.client.post(runtime_path + "/" + action, headers=ADMIN,
                                        json={"observation_path": "/machine/observation.json"} if action == "test" else {})
        assert response.status_code == 404, response.text
    with platform.app.state.db.read() as session:
        assert session.get(PolicyDeployment, runtime["id"]) is not None
    platform.app.state.db.initialize()
    assert platform.client.get(BASE, headers=ADMIN).json() == []


@pytest.mark.parametrize("role,expected", [("admin", 204), ("operator", 403), ("viewer", 403)])
def test_record_delete_requires_training_permission_csrf_and_same_origin(platform, role, expected):
    record_id = draft(platform)
    username, password = "record-delete-" + role, "record-deletion-test-password"
    platform.app.state.auth.create_user(username, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert login.status_code == 200, login.text
        path = f"{BASE}/{record_id}"
        assert client.delete(path).status_code == 403
        csrf = {"X-CSRF-Token": login.json()["csrf_token"]}
        if expected == 204:
            foreign = client.delete(path, headers={**csrf, "Origin": "https://foreign.invalid"})
            assert foreign.status_code == 403 and foreign.json()["error"]["code"] == "ORIGIN_REJECTED"
        response = client.delete(path, headers=csrf)
        assert response.status_code == expected, response.text
        assert client.get(path).status_code == (404 if expected == 204 else 200)


def test_record_delete_rejects_anonymous_and_machine_credentials_and_missing_ids(platform):
    node = register(platform)
    record_id = draft(platform)
    for headers in ({}, ENROLL, node["headers"]):
        assert platform.client.delete(f"{BASE}/{record_id}", headers=headers).status_code == 401
    assert platform.client.get(f"{BASE}/{record_id}", headers=ADMIN).status_code == 200
    assert remove(platform, uid()).status_code == 404


def test_record_deletion_marker_migrates_existing_database_without_recreating_removed_records(platform):
    record_id = draft(platform)
    database = platform.app.state.db
    before = platform.client.get(f"{BASE}/{record_id}", headers=ADMIN).json()
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE deployment_records DROP COLUMN deleted_at")
    assert "deleted_at" not in {column["name"] for column in inspect(database.engine).get_columns("deployment_records")}
    database.initialize()
    database.initialize()
    assert platform.client.get(f"{BASE}/{record_id}", headers=ADMIN).json() == before
    with database.read() as session:
        assert session.get(DeploymentRecord, record_id).deleted_at is None
    assert remove(platform, record_id).status_code == 204
    database.initialize()
    assert platform.client.get(f"{BASE}/{record_id}", headers=ADMIN).status_code == 404
    with database.read() as session:
        assert list(session.scalars(select(DeploymentRecord.id))) == [record_id]


def test_shared_transfer_owner_waits_for_referencing_record_to_be_deleted(transfer):
    platform, _, _, target, *_ = transfer
    child = succeed(transfer, create_transfer(transfer, record_name="shared checkpoint owner").json())
    witnesses = {"portable_digest": child["portable_digest"], "checkpoint_manifest_digest": child["manifest_sha256"]}

    def stopped_runtime(**changes):
        response = deploy(transfer, child, **changes)
        assert response.status_code == 202, response.text
        runtime = response.json()
        assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_START"), running(**witnesses)).status_code == 200
        assert action(platform, runtime, "stop").status_code == 202
        assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_STOP"), {"state": "STOPPED", **witnesses}).status_code == 200
        return runtime

    owner = stopped_runtime(record_id=child["record_id"])
    dependent = stopped_runtime(port=8001)
    assert owner["record_id"] != dependent["record_id"]
    before = snapshot(platform, include_records=True)
    response = remove(platform, owner["record_id"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "DEPLOYMENT_RECORD_IN_USE"
    assert snapshot(platform, include_records=True) == before
    # A reference alone must not count as ownership and create a mutual block.
    assert remove(platform, dependent["record_id"]).status_code == 204
    assert remove(platform, owner["record_id"]).status_code == 204
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    assert snapshot(platform) == {key: value for key, value in before.items()
                                  if key not in {"deployment_records", "deployment_record_runtimes", "deployment_record_transfers"}}


def test_delete_through_merged_alias_removes_canonical_record_and_all_alias_reads(platform):
    record_id = draft(platform)
    alias_id = uid()
    with platform.app.state.db.write() as session:
        session.add(DeploymentRecord(id=alias_id, name="historical alias", created_at=now(), merged_into_id=record_id))
    response = remove(platform, alias_id)
    assert response.status_code == 204, response.text
    for identifier in (record_id, alias_id):
        assert platform.client.get(f"{BASE}/{identifier}", headers=ADMIN).status_code == 404
        assert remove(platform, identifier).status_code == 204
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    with platform.app.state.db.read() as session:
        assert session.get(DeploymentRecord, record_id).deleted_at
        assert session.get(DeploymentRecord, alias_id).merged_into_id == record_id
