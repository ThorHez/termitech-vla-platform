"""Deleting finished training records preserves execution and asset lineage."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.common import now, offset_time, uid
from vla_control.models import Allocation, Asset, Attempt, Command, Event, Idempotency, Node, Preview, Run, RunInput

from .conftest import ADMIN, ENROLL
from .test_checkpoint_transfers import create as create_transfer, transfer
from .test_deployment_records import cancel as cancel_preparation, deploy, snapshot as deployment_snapshot, succeed
from .test_policy_deployments import add_run
from .test_policy_deployments import ack, action, command, running


def delete(platform, run):
    return platform.client.delete(f"/api/v1/runs/{run['id']}", headers=ADMIN)


def finished(platform, state="SUCCEEDED"):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    data = {"exit_code": 0 if state == "SUCCEEDED" else -15 if state == "CANCELLED" else 1,
            "outputs": []}
    if state == "FAILED":
        data["error"] = {"code": "TRAINING_FAILED", "message": "fixture failure"}
    response = platform.emit(node, attempt["id"], 2, state, data)
    assert response.status_code == 200, response.text
    return node, run, attempt


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED"])
def test_finished_record_disappears_from_all_training_views(platform, state):
    node, run, _ = finished(platform, state)
    path = f"/api/v1/runs/{run['id']}"
    assert run["id"] in {item["id"] for item in platform.client.get("/api/v1/runs", headers=ADMIN).json()}
    response = delete(platform, run)
    assert response.status_code == 204 and response.content == b""
    assert platform.client.get("/api/v1/runs", headers=ADMIN).json() == []
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["runs"] == []
    for suffix in ("", "/events", "/logs", "/metrics"):
        assert platform.client.get(path + suffix, headers=ADMIN).status_code == 404
    assert platform.client.post(path + "/cancel", headers=ADMIN).status_code == 404
    overview = platform.client.get("/api/v1/overview", headers=ADMIN).json()
    assert overview["runs_active"] == overview["runs_failed"] == 0
    assert all(item.get("run_id") != run["id"] for item in
               platform.client.get("/api/v1/alerts", headers=ADMIN).json())
    with platform.app.state.db.read() as session:
        deleted_at = session.get(Run, run["id"]).deleted_at
        assert deleted_at
    assert delete(platform, run).status_code == 204
    with platform.app.state.db.read() as session:
        assert session.get(Run, run["id"]).deleted_at == deleted_at


@pytest.mark.parametrize("state", ["QUEUED", "STARTING", "RUNNING", "CANCELLING", "UNKNOWN"])
def test_unfinished_record_cannot_be_deleted_or_release_gpu(platform, state):
    node = platform.register()
    run = platform.run(node)
    if state != "QUEUED":
        platform.schedule()
    if state in {"RUNNING", "CANCELLING", "UNKNOWN"}:
        platform.start(node, run)
    if state == "CANCELLING":
        assert platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN).status_code == 202
    elif state == "UNKNOWN":
        with platform.app.state.db.write() as session:
            session.get(Node, node["node_id"]).last_seen = offset_time(-100)
        platform.schedule()
    before = platform.get_run(run)
    assert before["state"] == state
    with platform.app.state.db.read() as session:
        allocation_ids = [(row.node_id, row.gpu_uuid, row.attempt_id) for row in session.scalars(select(Allocation))]
        command_ids = list(session.scalars(select(Command.id)))
    response = delete(platform, run)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "RUN_NOT_TERMINAL"
    assert platform.get_run(run) == before
    with platform.app.state.db.read() as session:
        assert session.get(Run, run["id"]).deleted_at is None
        assert [(row.node_id, row.gpu_uuid, row.attempt_id) for row in session.scalars(select(Allocation))] == allocation_ids
        assert list(session.scalars(select(Command.id))) == command_ids


@pytest.mark.parametrize("remaining", ["active_attempt", "gpu_allocation"])
def test_terminal_state_with_unfinished_execution_cannot_be_deleted(platform, remaining):
    node, run, attempt = finished(platform)
    with platform.app.state.db.write() as session:
        if remaining == "active_attempt":
            session.get(Attempt, attempt["id"]).state = "RUNNING"
        else:
            session.add(Allocation(node_id=node["node_id"], gpu_uuid="GPU-1", attempt_id=attempt["id"], created_at=now()))
    response = delete(platform, run)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RUN_NOT_TERMINAL"
    assert platform.client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Run, run["id"]).deleted_at is None
        if remaining == "active_attempt":
            assert session.get(Attempt, attempt["id"]).state == "RUNNING"
        else:
            assert session.get(Allocation, (node["node_id"], "GPU-1")).attempt_id == attempt["id"]


def test_deletion_preserves_checkpoints_inputs_and_agent_event_delivery(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    output = {"kind": "MODEL", "name": "checkpoint", "version": "100",
              "path": f"/machine/output/{attempt['id']}/checkpoints/100", "metadata": {"step": 100}}
    assert platform.emit(node, attempt["id"], 2, "LOG", {"text": "training log"}).status_code == 200
    assert platform.emit(node, attempt["id"], 3, "METRIC", {"step": 100, "loss": 0.1}).status_code == 200
    terminal_time = now()
    terminal_data = {"exit_code": 0, "outputs": [output]}
    assert platform.emit(node, attempt["id"], 4, "SUCCEEDED", terminal_data, terminal_time).status_code == 200
    asset = platform.get_run(run)["outputs"][0]
    consumer = platform.run(node)
    with platform.app.state.db.write() as session:
        session.get(Run, consumer["id"]).inputs = {"model": asset["id"]}
        session.add(RunInput(run_id=consumer["id"], node_id=node["node_id"], role="model", asset_id=asset["id"]))
    assets_before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    with platform.app.state.db.read() as session:
        commands_before = [(row.id, row.status, row.payload) for row in session.scalars(select(Command))]
        events_before = [(row.seq, row.type, row.digest, row.data) for row in
                         session.scalars(select(Event).where(Event.attempt_id == attempt["id"]).order_by(Event.seq))]
    assert delete(platform, run).status_code == 204
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == assets_before
    assert platform.get_run(consumer)["input_assets"][0]["id"] == asset["id"]
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).origin_attempt_id == attempt["id"]
        assert session.get(Attempt, attempt["id"]).run_id == run["id"]
        assert [(row.seq, row.type, row.digest, row.data) for row in
                session.scalars(select(Event).where(Event.attempt_id == attempt["id"]).order_by(Event.seq))] == events_before
        assert [(row.id, row.status, row.payload) for row in session.scalars(select(Command))] == commands_before
    # An Agent may retry the final batch after the user removes its record.
    # Its acknowledgement must remain valid, without restoring the record.
    response = platform.emit(node, attempt["id"], 4, "SUCCEEDED", terminal_data, terminal_time)
    assert response.status_code == 200, response.text
    assert platform.client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN).status_code == 404
    assert [item["id"] for item in platform.client.get("/api/v1/runs", headers=ADMIN).json()] == [consumer["id"]]


def test_deleted_run_cannot_be_recreated_by_submission_retries(platform):
    node = platform.register()
    preview = platform.preview(node)
    key = "delete-preserves-idempotency"
    body = {"preflight_id": preview["id"]}
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": key}, json=body)
    assert response.status_code == 201, response.text
    run = response.json()
    assert platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN).status_code == 202
    assert delete(platform, run).status_code == 204
    replay = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": key}, json=body)
    assert replay.status_code == 409 and replay.json()["error"]["code"] == "RUN_DELETED"
    duplicate = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "new-submit-key"}, json=body)
    assert duplicate.status_code == 409 and duplicate.json()["error"]["code"] == "PREFLIGHT_ALREADY_USED"
    with platform.app.state.db.read() as session:
        assert session.get(Idempotency, key).run_id == run["id"]
        assert session.get(Preview, preview["id"]).run_id == run["id"]
        assert list(session.scalars(select(Run.id))) == [run["id"]]
    assert platform.client.get("/api/v1/runs", headers=ADMIN).json() == []


@pytest.mark.parametrize("role,expected", [("admin", 204), ("operator", 204), ("viewer", 403)])
def test_delete_requires_training_permission_and_csrf(platform, role, expected):
    _, run, _ = finished(platform)
    username = "delete-" + role
    password = "delete-training-record-password"
    platform.app.state.auth.create_user(username, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        logged = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert logged.status_code == 200
        path = f"/api/v1/runs/{run['id']}"
        assert client.delete(path).status_code == 403
        csrf = {"X-CSRF-Token": logged.json()["csrf_token"]}
        if expected == 204:
            foreign = client.delete(path, headers={**csrf, "Origin": "https://foreign.invalid"})
            assert foreign.status_code == 403 and foreign.json()["error"]["code"] == "ORIGIN_REJECTED"
        response = client.delete(path, headers=csrf)
        assert response.status_code == expected, response.text
        assert client.get(path).status_code == (404 if expected == 204 else 200)


def test_missing_and_nonhuman_credentials_do_not_delete_records(platform):
    node, run, _ = finished(platform)
    path = f"/api/v1/runs/{run['id']}"
    for headers in ({}, ENROLL, node["headers"]):
        assert platform.client.delete(path, headers=headers).status_code == 401
    assert platform.client.get(path, headers=ADMIN).status_code == 200
    assert platform.client.delete("/api/v1/runs/missing", headers=ADMIN).status_code == 404


def test_existing_database_adds_deletion_marker_without_changing_training(platform):
    _, run, attempt = finished(platform)
    database = platform.app.state.db
    with database.read() as session:
        events_before = [(row.seq, row.digest) for row in session.scalars(
            select(Event).where(Event.attempt_id == attempt["id"]).order_by(Event.seq))]
    before = platform.get_run(run)
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE runs DROP COLUMN deleted_at")
    assert "deleted_at" not in {column["name"] for column in inspect(database.engine).get_columns("runs")}
    database.initialize()
    database.initialize()
    assert "deleted_at" in {column["name"] for column in inspect(database.engine).get_columns("runs")}
    assert platform.get_run(run) == before
    with database.read() as session:
        assert session.get(Run, run["id"]).deleted_at is None
        assert [(row.seq, row.digest) for row in session.scalars(
            select(Event).where(Event.attempt_id == attempt["id"]).order_by(Event.seq))] == events_before
    assert delete(platform, run).status_code == 204
    database.initialize()
    assert platform.client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN).status_code == 404


@pytest.mark.parametrize("record_state", ["preparing", "cancelled", "runtime"])
def test_linked_deployment_records_prevent_training_deletion(transfer, record_state):
    platform, _, _, _, run, *_ = transfer
    response = create_transfer(transfer, record_name="training remains linked")
    assert response.status_code == 202, response.text
    child = response.json()
    record_path = f"/api/v1/deployment-records/{child['record_id']}"
    if record_state == "cancelled":
        cancel_preparation(transfer, child)
    elif record_state == "runtime":
        child = succeed(transfer, child)
        response = deploy(transfer, child, record_id=child["record_id"])
        assert response.status_code == 202, response.text
    training_path = f"/api/v1/runs/{run.id}"
    training_before = platform.client.get(training_path, headers=ADMIN).json()
    record_before = platform.client.get(record_path, headers=ADMIN).json()
    preserved = deployment_snapshot(platform, include_records=True)
    response = delete(platform, {"id": run.id})
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "RUN_HAS_DEPLOYMENTS"
    assert platform.client.get(training_path, headers=ADMIN).json() == training_before
    assert platform.client.get(record_path, headers=ADMIN).json() == record_before
    assert deployment_snapshot(platform, include_records=True) == preserved
    assert [item["id"] for item in platform.client.get("/api/v1/runs", headers=ADMIN).json()] == [run.id]
    with platform.app.state.db.read() as session:
        assert session.get(Run, run.id).deleted_at is None


def test_other_run_deployment_on_same_node_and_checkpoint_does_not_block_deletion(transfer):
    platform, _, source, _, run, *_ = transfer
    with platform.app.state.db.write() as session:
        unrelated = add_run(session, source)
        session.add(Attempt(id=uid(), run_id=unrelated.id, node_id=source["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(),
            output_dir="/machine/output/owned", gpu_uuids=[], latest_metrics={}, resources={}))
    response = create_transfer(transfer, record_name="only the exact source run is linked")
    assert response.status_code == 202, response.text
    child = response.json()
    before = deployment_snapshot(platform, include_records=True)
    assert delete(platform, {"id": unrelated.id}).status_code == 204
    assert platform.client.get(f"/api/v1/runs/{unrelated.id}", headers=ADMIN).status_code == 404
    assert platform.client.get(f"/api/v1/runs/{run.id}", headers=ADMIN).status_code == 200
    record = platform.client.get(f"/api/v1/deployment-records/{child['record_id']}", headers=ADMIN).json()
    assert record["source_run_id"] == run.id
    assert deployment_snapshot(platform, include_records=True) == before


@pytest.mark.parametrize("with_runtime", [False, True])
def test_training_can_be_deleted_after_its_deployment_record_is_deleted(transfer, with_runtime):
    platform, _, _, target, run, *_ = transfer
    child = create_transfer(transfer, record_name="remove deployment before training").json()
    if with_runtime:
        child = succeed(transfer, child)
        response = deploy(transfer, child, record_id=child["record_id"])
        assert response.status_code == 202, response.text
        runtime = response.json()
        witnesses = {"portable_digest": child["portable_digest"], "checkpoint_manifest_digest": child["manifest_sha256"]}
        assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_START"), running(**witnesses)).status_code == 200
        assert action(platform, runtime, "stop").status_code == 202
        assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_STOP"), {"state": "STOPPED", **witnesses}).status_code == 200
    else:
        cancel_preparation(transfer, child)
    assert delete(platform, {"id": run.id}).json()["error"]["code"] == "RUN_HAS_DEPLOYMENTS"
    record_path = f"/api/v1/deployment-records/{child['record_id']}"
    response = platform.client.delete(record_path, headers=ADMIN)
    assert response.status_code == 204, response.text
    assert platform.client.get(record_path, headers=ADMIN).status_code == 404
    assert delete(platform, {"id": run.id}).status_code == 204
    assert platform.client.get(f"/api/v1/runs/{run.id}", headers=ADMIN).status_code == 404


def test_all_linked_deployment_records_must_be_deleted_before_training(transfer):
    platform, _, _, _, run, *_ = transfer
    first = create_transfer(transfer, record_name="first deployment intent").json()
    second = create_transfer(transfer, record_name="second deployment intent").json()
    assert first["record_id"] != second["record_id"]
    for child in (first, second):
        cancel_preparation(transfer, child)
    assert platform.client.delete(f"/api/v1/deployment-records/{first['record_id']}", headers=ADMIN).status_code == 204
    response = delete(platform, {"id": run.id})
    assert response.status_code == 409 and response.json()["error"]["code"] == "RUN_HAS_DEPLOYMENTS"
    assert platform.client.get(f"/api/v1/runs/{run.id}", headers=ADMIN).status_code == 200
    assert platform.client.delete(f"/api/v1/deployment-records/{second['record_id']}", headers=ADMIN).status_code == 204
    assert delete(platform, {"id": run.id}).status_code == 204
