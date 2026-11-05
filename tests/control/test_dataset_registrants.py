"""Only a verified manual registration establishes an asset's registrant."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from vla_control.common import uid
from vla_control.db import Database
from vla_control.models import Asset

from .conftest import ADMIN
from .test_asset_transfers import create as create_copy, fixture, result_for, run_copy
from .test_dataset_archive_permissions import registered
from .test_dataset_batches import PASSWORD, assets, batch, operator_batch, scan, submit
from .test_dataset_norm_migration import _legacy_table, _snapshot
from .test_multiuser_task_permissions import team  # noqa: F401
from .test_resource_ownership import accounts, login


@pytest.mark.parametrize("kind", ["DATASET", "NORM", "MODEL"])
def test_single_manual_registration_persists_actual_registrar_for_every_supported_kind(team, kind):
    platform, _, users, clients = team
    asset = registered(team, "alice", kind=kind)
    database = Database(platform.settings.database_url)
    try:
        database.initialize()
        with database.read() as session:
            row = session.get(Asset, asset["id"])
            assert row.kind == kind and row.registered_by_user_id == users["alice"]["id"]
    finally:
        database.close()
    listed = next(row for row in clients["bob"].get("/api/v1/assets").json() if row["id"] == asset["id"])
    assert listed["registered_by_user_id"] == users["alice"]["id"]


def test_additive_registrant_migration_preserves_history_and_never_infers_authorship(platform):
    node = platform.register()
    historical = []
    for kind in ("DATASET", "MODEL", "NORM"):
        response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
            "node_id": node["node_id"], "kind": kind, "name": "historical " + kind,
            "path": "/machine/" + kind.lower(), "metadata": {"registered_by_user_id": "unproven"}})
        assert response.status_code == 201, response.text
        historical.append(response.json())
    user = platform.app.state.auth.create_user("registrant-admin", PASSWORD, "admin")
    platform.client.base_url = "https://testserver"
    authenticated = platform.client.post("/api/v1/auth/login", json={"username": "registrant-admin", "password": PASSWORD})
    assert authenticated.status_code == 200, authenticated.text
    database = platform.app.state.db
    queries = {name: f"SELECT * FROM {name} ORDER BY id" for name in ("nodes", "users", "user_sessions", "commands")}
    raw = database.engine.raw_connection()
    try:
        raw.commit()
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        retained = _legacy_table(cursor, "assets", {"registered_by_user_id"})
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    queries["assets"] = "SELECT " + ",".join('"' + column + '"' for column in retained) + " FROM assets ORDER BY id"
    preserved = _snapshot(database, queries)
    barrier = Barrier(2)

    def migrate(_):
        separate = Database(platform.settings.database_url)
        try:
            barrier.wait(timeout=5)
            separate.initialize()
        finally:
            separate.close()

    with ThreadPoolExecutor(max_workers=2) as threads:
        list(threads.map(migrate, range(2)))
    database.initialize()
    assert _snapshot(database, queries) == preserved
    with database.read() as session:
        assert all(session.get(Asset, asset["id"]).registered_by_user_id is None for asset in historical)
    assert any(foreign["constrained_columns"] == ["registered_by_user_id"]
               and foreign["referred_table"] == "users" for foreign in inspect(database.engine).get_foreign_keys("assets"))
    assert platform.client.get("/api/v1/auth/me").status_code == 200
    with pytest.raises(IntegrityError):
        with database.write() as session:
            session.get(Asset, historical[0]["id"]).registered_by_user_id = uid()
            session.flush()
    with database.write() as session:
        session.get(Asset, historical[0]["id"]).registered_by_user_id = user["id"]
    database.initialize()
    with database.read() as session:
        assert session.get(Asset, historical[0]["id"]).registered_by_user_id == user["id"]
    assert _snapshot(database, {key: value for key, value in queries.items() if key != "assets"}) == {
        key: value for key, value in preserved.items() if key != "assets"}


def test_batch_new_dataset_registrant_persists_but_companion_norm_is_unclaimed(operator_batch):
    value = scan(operator_batch)
    response = submit(operator_batch, value, entries=[{"path": "/machine/datasets/first", "name": "own dataset",
        "norm_path": "/machine/norms/automatic.json"}])
    assert response.status_code == 200, response.text
    created = response.json()["created"][0]
    actual_user = operator_batch[5]["user"]["id"]
    assert created["registered_by_user_id"] == actual_user
    database = Database(operator_batch[0].state.settings.database_url)
    try:
        database.initialize()
        with database.read() as session:
            assert session.get(Asset, created["id"]).registered_by_user_id == actual_user
            assert session.get(Asset, created["norm_asset_id"]).registered_by_user_id is None
    finally:
        database.close()
    listed = {row["id"]: row for row in assets(operator_batch)}
    assert listed[created["id"]]["registered_by_user_id"] == actual_user
    assert listed[created["norm_asset_id"]]["registered_by_user_id"] is None


@pytest.mark.parametrize("field", ["registered_by_user_id", "owner_user_id", "user_id"])
def test_client_cannot_supply_a_batch_registrant(operator_batch, field):
    value = scan(operator_batch)
    response = submit(operator_batch, value, entries=[{"path": "/machine/datasets/first", "name": "forged",
        field: uid()}])
    assert response.status_code == 422, response.text
    assert assets(operator_batch) == []


@pytest.mark.parametrize("historical_null", [False, True])
def test_other_accounts_batch_skip_and_original_replay_never_transfer_registrant(batch, historical_null):
    value = scan(batch)
    key = uid()
    first = submit(batch, value, key=key)
    assert first.status_code == 200, first.text
    identifiers = [asset["id"] for asset in first.json()["created"]]
    if historical_null:
        with batch[0].state.db.write() as session:
            for identifier in identifiers:
                session.get(Asset, identifier).registered_by_user_id = None
    before = assets(batch)
    app, client, manager, node_id, connection_id, _, remotes = batch
    app.state.auth.create_user("other-registrant", PASSWORD, "operator")
    client.cookies.clear()
    current = client.post("/api/v1/auth/login", json={"username": "other-registrant", "password": PASSWORD}).json()
    client.headers.update({"X-CSRF-Token": current["csrf_token"], "X-Dataset-Scan-User": current["user"]["id"]})
    other = (app, client, manager, node_id, connection_id, current, remotes)
    other_scan = scan(other)
    skipped = submit(other, other_scan)
    assert skipped.status_code == 200, skipped.text
    assert skipped.json()["created_count"] == 0 and skipped.json()["skipped_count"] == 2
    assert assets(other) == before
    client.cookies.clear()
    original = client.post("/api/v1/auth/login", json={"username": "batch-admin", "password": PASSWORD}).json()
    client.headers.update({"X-CSRF-Token": original["csrf_token"], "X-Dataset-Scan-User": original["user"]["id"]})
    replay = submit(batch, value, key=key)
    assert replay.status_code == 200 and replay.json() == first.json()
    assert assets(batch) == before


@pytest.mark.parametrize("kind", ["DATASET", "NORM", "MODEL"])
def test_copy_output_does_not_inherit_asset_registrant_or_transfer_task_owner(fixture, monkeypatch, kind):
    platform, _, _, _, source, _ = fixture
    users = accounts(platform)
    with platform.app.state.db.write() as session:
        row = session.get(Asset, source["id"])
        row.kind, row.registered_by_user_id = kind, users["alice"]
        if kind == "NORM":
            row.path = "/machine/norms/norm_stats.json"
    login(platform, "alice")
    response = create_copy(fixture)
    assert response.status_code == 202, response.text
    finished = run_copy(fixture, response.json()["id"], monkeypatch,
        transfer=lambda *args, **kwargs: result_for(args, kwargs, file_name="norm_stats.json" if kind == "NORM" else None))
    assert finished["state"] == "SUCCEEDED" and finished["owner_user_id"] == users["alice"]
    listed = {row["id"]: row for row in platform.client.get("/api/v1/assets").json()}
    assert listed[source["id"]]["registered_by_user_id"] == users["alice"]
    assert listed[finished["target_asset_id"]]["registered_by_user_id"] is None
