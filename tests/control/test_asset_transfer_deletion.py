"""Removing sync history preserves copied assets, norm bindings and receipts."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.models import Asset, Command

from .conftest import ADMIN, ENROLL
from .test_asset_transfer_norms import copied_norm, paired
from .test_asset_transfers import BASE, create, fixture, receipt, run_copy


def remove(platform, transfer_id):
    return platform.client.delete(f"{BASE}/{transfer_id}", headers=ADMIN)


def rows(database, model, *, omit=()):
    with database.read() as session:
        table = model.__table__
        columns = [column for column in table.columns if column.name not in omit]
        return [dict(row) for row in session.execute(select(*columns).order_by(*table.primary_key.columns)).mappings()]


def fail_external_action(*args, **kwargs):
    pytest.fail("Deleting sync history must not launch a worker or access a machine")


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"])
def test_ended_sync_records_can_be_deleted_and_are_hidden_from_old_actions(fixture, monkeypatch, state):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    if state == "SUCCEEDED":
        run_copy(fixture, task["id"], monkeypatch)
    else:
        with platform.app.state.db.write() as session:
            row = session.get(AssetTransfer, task["id"])
            row.state, row.finished_at = state, now()
    database = platform.app.state.db
    transfers_before = rows(database, AssetTransfer, omit=("deleted_at",))
    assets_before, commands_before = rows(database, Asset), rows(database, Command)
    monkeypatch.setattr(manager, "_launch", fail_external_action)
    monkeypatch.setattr(manager.remote_manager, "_remote", fail_external_action)
    monkeypatch.setattr(manager, "transfer_function", fail_external_action)
    response = remove(platform, task["id"])
    assert response.status_code == 204 and response.content == b""
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    path = f"{BASE}/{task['id']}"
    assert platform.client.get(path, headers=ADMIN).status_code == 404
    for operation in ("retry", "cancel"):
        assert platform.client.post(path + "/" + operation, headers=ADMIN).status_code == 404
    assert rows(database, AssetTransfer, omit=("deleted_at",)) == transfers_before
    assert rows(database, Asset) == assets_before and rows(database, Command) == commands_before
    with database.read() as session:
        deleted_at = session.get(AssetTransfer, task["id"]).deleted_at
        assert deleted_at
    assert remove(platform, task["id"]).status_code == 204
    with database.read() as session:
        assert session.get(AssetTransfer, task["id"]).deleted_at == deleted_at


@pytest.mark.parametrize("state", ["QUEUED", "PREPARING", "TRANSFERRING", "VERIFYING", "CANCELLING"])
def test_active_sync_cannot_be_deleted_or_lose_progress(fixture, monkeypatch, state):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    with platform.app.state.db.write() as session:
        row = session.get(AssetTransfer, task["id"])
        row.state, row.bytes_total, row.bytes_transferred = state, 1000, 375
        row.files_total, row.files_transferred, row.progress_pct = 10, 3, 37.5
        row.logs = "partial transfer progress\n"
    before = receipt(fixture, task["id"])
    stored_before = rows(platform.app.state.db, AssetTransfer)
    monkeypatch.setattr(manager, "_launch", fail_external_action)
    monkeypatch.setattr(manager.remote_manager, "_remote", fail_external_action)
    response = remove(platform, task["id"])
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "ASSET_TRANSFER_NOT_TERMINAL"
    assert receipt(fixture, task["id"]) == before
    assert rows(platform.app.state.db, AssetTransfer) == stored_before


def test_deleting_paired_copy_preserves_both_source_and_target_norm_links_and_receipts(paired, monkeypatch):
    fixture, source_norm = paired
    platform, manager, _, target, source_dataset, _ = fixture
    task = create(fixture, include_norm_stats=True).json()
    final = run_copy(fixture, task["id"], monkeypatch, copied_norm)
    assert final["state"] == final["norm_sync"]["state"] == "SUCCEEDED"
    target_dataset_id, target_norm_id = final["target_asset_id"], final["norm_sync"]["target_asset_id"]
    database = platform.app.state.db
    assets_before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    transfers_before = rows(database, AssetTransfer, omit=("deleted_at",))
    monkeypatch.setattr(manager, "_launch", fail_external_action)
    monkeypatch.setattr(manager.remote_manager, "_remote", fail_external_action)
    monkeypatch.setattr(manager, "transfer_function", fail_external_action)
    assert remove(platform, task["id"]).status_code == 204
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == assets_before
    assert platform.client.get(BASE, headers=ADMIN, params={"source_asset_id": source_dataset["id"],
                                                         "target_node_id": target["node_id"]}).json() == []
    assert rows(database, AssetTransfer, omit=("deleted_at",)) == transfers_before
    with database.read() as session:
        assert session.get(Asset, source_dataset["id"]).norm_asset_id == source_norm["id"]
        assert session.get(Asset, target_dataset_id).norm_asset_id == target_norm_id
        assert session.get(Asset, target_norm_id).metadata_json["asset_sync"]["transfer_id"] == task["id"]
        row = session.get(AssetTransfer, task["id"])
        assert row.manifest and row.manifest_sha256
        assert row.norm_sync["manifest_sha256"] and row.norm_sync["bytes_total"] == 7
        assert row.norm_sync["target_asset_id"] == target_norm_id
    database.initialize()
    assert platform.client.get(BASE, headers=ADMIN).json() == []
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == assets_before


def test_deleted_sync_request_key_cannot_recreate_or_restart_a_transfer(fixture, monkeypatch):
    platform, manager, *_ = fixture
    key = "deleted-sync-stays-consumed"
    task = create(fixture, idempotency_key=key).json()
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, task["id"]).state = "FAILED"
    assert remove(platform, task["id"]).status_code == 204
    before = rows(platform.app.state.db, AssetTransfer)
    monkeypatch.setattr(manager, "_launch", fail_external_action)
    monkeypatch.setattr(manager.remote_manager, "_remote", fail_external_action)
    response = create(fixture, idempotency_key=key)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "ASSET_TRANSFER_DELETED"
    assert rows(platform.app.state.db, AssetTransfer) == before
    assert platform.client.get(BASE, headers=ADMIN).json() == []


@pytest.mark.parametrize("role,expected", [("admin", 204), ("operator", 204), ("viewer", 403)])
def test_sync_deletion_requires_asset_write_permission_csrf_and_same_origin(fixture, role, expected):
    platform, *_ = fixture
    task = create(fixture).json()
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, task["id"]).state = "CANCELLED"
    username, password = "sync-delete-" + role, "asset-sync-deletion-password"
    platform.app.state.auth.create_user(username, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert login.status_code == 200, login.text
        path = f"{BASE}/{task['id']}"
        assert client.delete(path).status_code == 403
        csrf = {"X-CSRF-Token": login.json()["csrf_token"]}
        if expected == 204:
            response = client.delete(path, headers={**csrf, "Origin": "https://foreign.invalid"})
            assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_REJECTED"
        response = client.delete(path, headers=csrf)
        assert response.status_code == expected, response.text
        assert client.get(path).status_code == (404 if expected == 204 else 200)


def test_sync_delete_requires_human_credentials_and_missing_records_return_404(fixture):
    platform, _, source, *_ = fixture
    task = create(fixture).json()
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, task["id"]).state = "INTERRUPTED"
    for headers in ({}, ENROLL, source["headers"]):
        assert platform.client.delete(f"{BASE}/{task['id']}", headers=headers).status_code == 401
    assert receipt(fixture, task["id"])["state"] == "INTERRUPTED"
    assert remove(platform, uid()).status_code == 404


def test_existing_sync_database_gets_additive_marker_without_losing_progress_or_assets(paired):
    fixture, source_norm = paired
    platform, *_ = fixture
    task = create(fixture, include_norm_stats=True).json()
    database = platform.app.state.db
    with database.write() as session:
        row = session.get(AssetTransfer, task["id"])
        row.state, row.bytes_total, row.bytes_transferred = "TRANSFERRING", 1000, 425
        row.files_total, row.files_transferred, row.progress_pct = 10, 4, 42.5
        row.logs = "keep partial sync history\n"
    transfer_before = rows(database, AssetTransfer, omit=("deleted_at",))
    assets_before = rows(database, Asset)
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE asset_transfers DROP COLUMN deleted_at")
    assert "deleted_at" not in {column["name"] for column in inspect(database.engine).get_columns("asset_transfers")}
    database.initialize()
    database.initialize()
    assert rows(database, AssetTransfer, omit=("deleted_at",)) == transfer_before
    assert rows(database, Asset) == assets_before
    with database.read() as session:
        assert session.get(AssetTransfer, task["id"]).deleted_at is None
        assert session.get(Asset, fixture[4]["id"]).norm_asset_id == source_norm["id"]
    assert receipt(fixture, task["id"])["bytes_transferred"] == 425
