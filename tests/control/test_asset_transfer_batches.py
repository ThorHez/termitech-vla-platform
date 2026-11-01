"""Atomic groups reuse existing durable copies and their individual lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from vla_control import asset_transfer_batches as batches
from vla_control import asset_transfers
from vla_control.asset_transfer_batch_models import AssetTransferBatch
from vla_control.asset_transfer_models import AssetTransfer
from vla_control.checkpoint_transfer_transport import TransferError
from vla_control.common import now, offset_time, uid
from vla_control.models import Asset, Command, Node
from vla_control.remote_models import RemoteConnection

from .conftest import ADMIN
from .test_asset_transfers import BASE, create, fixture, receipt, result_for, run_copy


BATCH = "/api/v1/asset-transfer-batches"


@pytest.fixture
def group(fixture):
    platform, manager, source, target, dataset, _ = fixture
    values = []
    for kind, path in (("MODEL", "/machine/models/base"), ("NORM", "/machine/stats/norm_stats.json")):
        response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
            "node_id": source["node_id"], "kind": kind, "name": kind.lower(),
            "path": path, "version": "v1", "metadata": {}})
        assert response.status_code == 201, response.text
        values.append(response.json())
    with platform.app.state.db.write() as session:
        session.get(Asset, dataset["id"]).norm_asset_id = values[1]["id"]
    platform.app.state.auth.create_user("batch-copy-functional", "isolated-functional-password", "operator")
    client = TestClient(platform.app, base_url="https://testserver")
    login = client.post("/api/v1/auth/login", json={
        "username": "batch-copy-functional", "password": "isolated-functional-password"}).json()
    client.headers.update({"X-CSRF-Token": login["csrf_token"], "X-Asset-Transfer-Batch-User": login["user"]["id"]})
    platform.client = client  # Existing lifecycle helpers use this authenticated session too.
    launched = []
    manager._launch = launched.append
    yield fixture, client, [dataset, *values], launched, login["user"]["id"]
    client.close()


def body(group, **changes):
    return {"idempotency_key": uid(), "target_node_id": group[0][3]["node_id"],
            "target_directory": "/machine/copied", "entries": [
                {"source_asset_id": asset["id"], "target_name": f"copy-{index}",
                 "include_norm_stats": index == 0,
                 **({"expected_norm_asset_id": group[2][2]["id"]} if index == 0 else {})}
                for index, asset in enumerate(group[2])], **changes}


def counts(group):
    with group[0][0].app.state.db.read() as session:
        return tuple(session.scalar(select(func.count()).select_from(model))
                     for model in (AssetTransfer, AssetTransferBatch, Asset, Command))


def test_mixed_assets_and_optional_norm_queue_atomically_before_first_worker_launch(group):
    observed = []
    def launch(transfer_id):
        observed.append((transfer_id, counts(group)))
    group[0][1]._launch = launch
    before = counts(group)
    request = body(group)
    response = group[1].post(BATCH, json=request)
    assert response.status_code == 202, response.text
    result = response.json()
    rows = result["transfers"]
    assert result["idempotency_key"] == request["idempotency_key"]
    assert [row["source_kind"] for row in rows] == ["DATASET", "MODEL", "NORM"]
    assert all(row["state"] == "QUEUED" and row["attempt"] == 1 for row in rows)
    assert [row["include_norm_stats"] for row in rows] == [True, False, False]
    assert rows[0]["norm_sync"]["source_asset_id"] == group[2][2]["id"]
    assert rows[0]["norm_sync"]["target_name"] == "norm-stats-" + rows[0]["id"]
    assert rows[1]["norm_sync"] is None and rows[2]["norm_sync"] is None
    assert all(row["target_node_id"] == request["target_node_id"] for row in rows)
    assert observed == [(row["id"], (3, 1, before[2], before[3])) for row in rows]
    with group[0][0].app.state.db.read() as session:
        for row in rows:
            stored = session.get(AssetTransfer, row["id"])
            assert stored.binding_snapshot["source_asset"]["id"] == row["source_asset_id"]
            assert stored.request_key.startswith("asset-batch:")
    # The private frozen receipt is not a replacement for the existing global
    # transfer listing, detail or individual controls.
    listed = group[0][0].client.get(BASE, headers=ADMIN).json()
    assert {row["id"] for row in listed} == {row["id"] for row in rows}


def test_committed_group_replays_without_revalidation_or_relaunch(group):
    request = body(group)
    first = group[1].post(BATCH, json=request)
    assert first.status_code == 202, first.text
    result, launched = first.json(), list(group[3])
    with group[0][0].app.state.db.write() as session:
        session.get(AssetTransfer, launched[0]).state = "FAILED"
        session.get(Asset, group[2][0]["id"]).state = "ARCHIVED"
        session.get(Node, request["target_node_id"]).last_seen = offset_time(-1000)
    replay = group[1].post(BATCH, json=request)
    assert replay.status_code == 202 and replay.json() == result
    assert group[3] == launched and counts(group)[:2] == (3, 1)
    request["entries"][0]["target_name"] = "changed"
    changed = group[1].post(BATCH, json=request)
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "ASSET_TRANSFER_BATCH_IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("case", ["duplicate-id", "duplicate-path", "duplicate-target", "active-target", "norm-binding"])
def test_group_contract_failures_never_queue_partial_transfers(group, case):
    request = body(group)
    if case == "duplicate-id":
        request["entries"][2]["source_asset_id"] = request["entries"][0]["source_asset_id"]
    elif case == "duplicate-path":
        with group[0][0].app.state.db.write() as session:
            session.get(Asset, group[2][1]["id"]).path = group[2][0]["path"] + "/./"
    elif case == "duplicate-target":
        request["entries"][2]["target_name"] = request["entries"][0]["target_name"]
    elif case == "norm-binding":
        request["entries"][0]["expected_norm_asset_id"] = uid()
    else:
        assert create(group[0], target_name=request["entries"][2]["target_name"]).status_code == 202
    before, launched = counts(group), list(group[3])
    response = group[1].post(BATCH, json=request)
    assert response.status_code in {409, 422}, response.text
    assert counts(group) == before and group[3] == launched


def test_companion_target_collision_is_rejected_before_any_row_is_committed(group, monkeypatch):
    original = batches._norm_sync
    def conflicting(transfer_id, source):
        return {**original(transfer_id, source), "target_name": "copy-1"}
    monkeypatch.setattr(batches, "_norm_sync", conflicting)
    before = counts(group)
    response = group[1].post(BATCH, json=body(group))
    assert response.status_code == 422 and response.json()["error"]["code"] == "ASSET_TRANSFER_BATCH_TARGET_CONFLICT"
    assert counts(group) == before and group[3] == []


def test_same_physical_host_checks_other_entries_sources_against_all_destinations(group):
    platform, _, _, _, _, connections = group[0]
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, connections[1].id).fingerprint = connections[0].fingerprint
        session.get(Asset, group[2][1]["id"]).path = "/machine/copied/copy-0"
    # Each single pair is disjoint: model goes to copy-1. The dataset's copy-0
    # would nevertheless overwrite that model source through a node alias.
    response = group[1].post(BATCH, json=body(group))
    assert response.status_code == 422 and response.json()["error"]["code"] == "ASSET_TRANSFER_PATH_OVERLAP"
    assert counts(group)[:2] == (0, 0) and group[3] == []


def test_batch_jobs_can_fail_complete_cancel_and_retry_independently(group, monkeypatch):
    response = group[1].post(BATCH, json=body(group))
    assert response.status_code == 202, response.text
    tasks = response.json()["transfers"]
    def fail(*args, **kwargs):
        raise TransferError("FIXTURE_COPY_FAILED", "Synthetic data copy failure")
    assert run_copy(group[0], tasks[0]["id"], monkeypatch, fail)["state"] == "FAILED"
    success = run_copy(group[0], tasks[1]["id"], monkeypatch)
    assert success["state"] == "SUCCEEDED" and success["target_asset_id"]
    cancelled = group[1].post(BASE + "/" + tasks[2]["id"] + "/cancel")
    assert cancelled.status_code == 202 and cancelled.json()["state"] == "CANCELLED"
    retry = group[1].post(BASE + "/" + tasks[0]["id"] + "/retry")
    assert retry.status_code == 202 and retry.json()["attempt"] == 2
    def copied(*args, **kwargs):
        return result_for(args, kwargs, file_name="norm_stats.json" if args[2].endswith("norm_stats.json") else None)
    completed = run_copy(group[0], tasks[0]["id"], monkeypatch, copied)
    assert completed["state"] == "SUCCEEDED" and completed["norm_sync"]["state"] == "SUCCEEDED"
    assert receipt(group[0], tasks[1]["id"])["state"] == "SUCCEEDED"
    assert receipt(group[0], tasks[2]["id"])["state"] == "CANCELLED"
    with group[0][0].app.state.db.read() as session:
        target = session.get(Asset, completed["target_asset_id"])
        assert target.norm_asset_id == completed["norm_sync"]["target_asset_id"]


def test_batch_capacity_is_shared_with_single_creates_retries_and_cancelling_jobs(group):
    initial = create(group[0], target_name="occupied-0").json()
    with group[0][0].app.state.db.write() as session:
        seed = session.get(AssetTransfer, initial["id"])
        for index in range(1, 19):
            session.add(AssetTransfer(id=uid(), request_key=uid(), request_digest="a" * 64,
                name="capacity fixture", source_asset_id=seed.source_asset_id,
                source_node_id=seed.source_node_id, source_path=seed.source_path, source_kind=seed.source_kind,
                target_node_id=seed.target_node_id, target_directory="/machine/copied",
                target_name=f"occupied-{index}", state="QUEUED", attempt=1,
                logs="", binding_snapshot=deepcopy(seed.binding_snapshot), created_at=now(), updated_at=now()))
        seed.state = "CANCELLING"
    request = body(group)
    response = group[1].post(BATCH, json=request)
    assert response.status_code == 429 and counts(group)[:2] == (19, 0)
    request["entries"] = request["entries"][:1]
    accepted = group[1].post(BATCH, json=request)
    assert accepted.status_code == 202 and counts(group)[:2] == (20, 1)
    assert create(group[0], target_name="over-capacity").status_code == 429
    with group[0][0].app.state.db.write() as session:
        seed = session.get(AssetTransfer, initial["id"])
        seed.state = "FAILED"
    # Retry itself occupies the last slot. A further retry at full capacity
    # cannot bypass the common guard, even though its previous row is terminal.
    retried = group[1].post(BASE + "/" + initial["id"] + "/retry")
    assert retried.status_code == 202, retried.text
    with group[0][0].app.state.db.write() as session:
        session.get(AssetTransfer, initial["id"]).state = "FAILED"
    another = create(group[0], target_name="last-slot")
    assert another.status_code == 202
    denied = group[1].post(BASE + "/" + initial["id"] + "/retry")
    assert denied.status_code == 429


def test_simultaneous_same_batch_request_creates_one_receipt_and_launches_once(group):
    request = batches.AssetTransferBatchInput.model_validate(body(group))
    barrier = threading.Barrier(2)
    def queue():
        barrier.wait()
        return batches.create_batch(group[0][1], group[4], request)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: queue(), range(2)))
    assert results[0] == results[1]
    assert group[3] == [row["id"] for row in results[0]["transfers"]]
    assert counts(group)[:2] == (3, 1)


def test_batch_prelocks_all_asset_inputs_nowait_before_any_node_and_shared_admission(group, monkeypatch):
    actions = []
    original_inputs, original_online, original_admission = batches._lock_inputs, batches._online, batches._admission_lock
    def admission(session):
        actions.append("admission")
        return original_admission(session)
    def inputs(session, request):
        result = original_inputs(session, request)
        actions.append("all-assets")
        return result
    def online(session, node_id, settings):
        actions.append(("node", node_id))
        return original_online(session, node_id, settings)
    monkeypatch.setattr(batches, "_admission_lock", admission)
    monkeypatch.setattr(batches, "_lock_inputs", inputs)
    monkeypatch.setattr(batches, "_online", online)
    response = group[1].post(BATCH, json=body(group))
    assert response.status_code == 202, response.text
    assert actions[:2] == ["admission", "all-assets"]
    assert actions[2:] == [("node", value) for value in sorted({group[0][2]["node_id"], group[0][3]["node_id"]})]
    # Compile the very SELECTs issued by input prelocking for the PostgreSQL
    # dialect: SQLite cannot exercise actual row lock contention.
    with group[0][0].app.state.db.read() as session:
        source = session.get(Asset, group[2][0]["id"])
        norm = session.get(Asset, group[2][2]["id"])
    statements, values = [], iter((source, norm))
    class LockSession:
        def scalar(self, statement):
            statements.append(statement)
            return next(values)
    batches._lock_inputs(LockSession(), batches.AssetTransferBatchInput.model_validate(body(group, entries=[
        {"source_asset_id": source.id, "target_name": "data", "include_norm_stats": True}])))
    assert len(statements) == 2
    assert all("FOR UPDATE NOWAIT" in str(value.compile(dialect=postgresql.dialect())) for value in statements)
    assert all(value.get_execution_options()["populate_existing"] for value in statements)


def test_postgres_asset_contention_fails_fast_without_partial_queue_or_raw_error(group, monkeypatch):
    class Locked(Exception):
        sqlstate = "55P03"
    def busy(*args):
        raise OperationalError("private SQL and fields", {}, Locked("private raw lock error"))
    monkeypatch.setattr(batches, "_lock_inputs", busy)
    response = group[1].post(BATCH, json=body(group))
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_TRANSFER_SOURCE_BUSY"
    assert "private" not in response.text and counts(group)[:2] == (0, 0) and group[3] == []
