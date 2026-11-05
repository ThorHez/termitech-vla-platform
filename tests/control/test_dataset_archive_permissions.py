"""Archiving a dataset requires its persisted registrar, without deleting files."""

from copy import deepcopy

import pytest
from sqlalchemy import delete, func, select

from vla_control.auth_models import User, UserSession
from vla_control.common import now, offset_time, uid
from vla_control.db import Database
from vla_control.models import Asset, Command, Run, RunInput

from .conftest import ENROLL
from .test_asset_transfers import create as copy_asset, fixture
from .test_dataset_profiles import profile
from .test_dataset_registration_permissions import body
from .test_multiuser_task_permissions import team  # noqa: F401
from .test_resource_ownership import accounts, login


def registered(team, creator="alice", **changes):
    _, node, users, clients = team
    response = clients[creator].post("/api/v1/assets", json=body(node, **changes))
    assert response.status_code == 201, response.text
    value = response.json()
    if value["kind"] == "DATASET":
        assert value["registered_by_user_id"] == users[creator]["id"]
    return value


def archive_path(asset):
    return "/api/v1/assets/" + asset["id"] + "/archive"


def snapshot(platform, identifier):
    with platform.app.state.db.read() as session:
        row = session.get(Asset, identifier)
        return {"state": row.state, "registered_by_user_id": row.registered_by_user_id,
                "metadata": deepcopy(row.metadata_json), "norm_asset_id": row.norm_asset_id,
                "assets": session.scalar(select(func.count()).select_from(Asset)),
                "commands": session.scalar(select(func.count()).select_from(Command))}


@pytest.mark.parametrize("owner", ["alice", "bob", None])
def test_persisted_registrar_allows_only_own_dataset_and_admin_can_handle_history(team, owner):
    platform, _, users, clients = team
    asset = registered(team, owner or "team-admin")
    if owner is None:
        # A historical row has no proof of its registrar, even if somebody's
        # identity happens to appear in old imported JSON metadata.
        with platform.app.state.db.write() as session:
            row = session.get(Asset, asset["id"])
            row.registered_by_user_id = None
            row.metadata_json = {**row.metadata_json, "registered_by_user_id": users["alice"]["id"]}
    independent = Database(platform.settings.database_url)
    try:
        independent.initialize()
        with independent.read() as session:
            expected = users[owner]["id"] if owner else None
            assert session.get(Asset, asset["id"]).registered_by_user_id == expected
    finally:
        independent.close()
    before = snapshot(platform, asset["id"])
    response = clients["alice"].post(archive_path(asset))
    assert response.status_code == (200 if owner == "alice" else 403), response.text
    if owner != "alice":
        assert response.json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"
        assert snapshot(platform, asset["id"]) == before
    else:
        assert response.json()["registered_by_user_id"] == users["alice"]["id"]
        assert snapshot(platform, asset["id"])["state"] == "ARCHIVED"
    assert clients["team-admin"].post(archive_path(asset)).status_code == 200
    assert snapshot(platform, asset["id"])["registered_by_user_id"] == (users[owner]["id"] if owner else None)


@pytest.mark.parametrize("creator", ["alice", "bob"])
def test_json_metadata_cannot_replace_the_server_recorded_registrar(team, creator):
    platform, _, users, clients = team
    forged = users["bob" if creator == "alice" else "alice"]["id"]
    asset = registered(team, creator, metadata={"registered_by_user_id": forged,
        "owner_user_id": forged, "registration": {"actor_user_id": forged}})
    assert asset["metadata"]["registered_by_user_id"] == forged
    assert snapshot(platform, asset["id"])["registered_by_user_id"] == users[creator]["id"]
    response = clients["alice"].post(archive_path(asset), json={"registered_by_user_id": users["alice"]["id"]})
    assert response.status_code == (200 if creator == "alice" else 403), response.text
    assert snapshot(platform, asset["id"])["state"] == ("ARCHIVED" if creator == "alice" else "REGISTERED")


def test_client_cannot_select_a_registrar_in_the_creation_request(team):
    platform, node, users, clients = team
    with platform.app.state.db.read() as session:
        before = session.scalar(select(func.count()).select_from(Asset))
    response = clients["bob"].post("/api/v1/assets",
        json=body(node, registered_by_user_id=users["alice"]["id"]))
    assert response.status_code == 422, response.text
    with platform.app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(Asset)) == before


@pytest.mark.parametrize("kind", ["MODEL", "NORM"])
def test_even_a_matching_registrar_does_not_grant_operator_non_dataset_archive_rights(team, kind):
    platform, _, users, clients = team
    asset = registered(team, "team-admin", kind=kind)
    # Cover kind independently from the registrar equality check. Legacy or
    # internal records must not convert a dataset-only right into catalogue edit.
    with platform.app.state.db.write() as session:
        session.get(Asset, asset["id"]).registered_by_user_id = users["alice"]["id"]
    before = snapshot(platform, asset["id"])
    denied = clients["alice"].post(archive_path(asset))
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"
    assert snapshot(platform, asset["id"]) == before
    assert clients["team-admin"].post(archive_path(asset)).status_code == 200


def test_own_dataset_referenced_by_an_active_training_run_cannot_be_archived(team):
    platform, node, users, clients = team
    asset = registered(team)
    with platform.app.state.db.write() as session:
        run = Run(id=uid(), name="active reference", owner_user_id=users["alice"]["id"],
            node_id=node["node_id"], profile_id="default", template_id="smoke", state="QUEUED",
            inputs={"dataset": asset["id"]}, parameters={}, gpu_count=1, code_expectation={},
            created_at=now(), updated_at=now())
        session.add(run)
        session.flush()
        session.add(RunInput(run_id=run.id, node_id=node["node_id"], role="dataset", asset_id=asset["id"]))
        identifier = run.id
    before = snapshot(platform, asset["id"])
    response = clients["alice"].post(archive_path(asset))
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_IN_USE"
    assert snapshot(platform, asset["id"]) == before
    with platform.app.state.db.write() as session:
        session.get(Run, identifier).state = "SUCCEEDED"
    assert clients["alice"].post(archive_path(asset)).status_code == 200


def test_own_dataset_being_synchronized_cannot_be_archived_until_copy_is_cancelled(fixture):
    platform, _, source, *_ = fixture
    users = accounts(platform)
    login(platform, "alice")
    response = platform.client.post("/api/v1/assets", json=body(source,
        path="/machine/data/alice-owned", metadata={"dataset_profile": profile()}))
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["registered_by_user_id"] == users["alice"]
    copied = copy_asset(fixture, source_asset_id=asset["id"], target_name="alice-owned-copy")
    assert copied.status_code == 202, copied.text
    before = snapshot(platform, asset["id"])
    response = platform.client.post(archive_path(asset))
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_IN_USE"
    assert snapshot(platform, asset["id"]) == before
    assert platform.client.post("/api/v1/asset-transfers/" + copied.json()["id"] + "/cancel").status_code == 202
    assert platform.client.post(archive_path(asset)).status_code == 200


@pytest.mark.parametrize("change", ["disabled", "viewer", "logout", "idle", "expired", "password-change"])
def test_archive_rechecks_original_session_after_dependency_without_mutating_asset(team, monkeypatch, change):
    platform, _, users, clients = team
    asset = registered(team)
    path = archive_path(asset)
    before = snapshot(platform, asset["id"])
    identifier = users["alice"]["id"]
    original = platform.app.state.auth.require
    changed = []

    def revoke_after_admission(request, permission=None):
        principal = original(request, permission)
        if request.method == "POST" and request.url.path == path and not changed:
            changed.append(True)
            if change in {"disabled", "viewer"}:
                platform.app.state.auth.update_user(None, identifier,
                    {"enabled": False} if change == "disabled" else {"role": "viewer"})
            else:
                with platform.app.state.db.write() as session:
                    if change == "password-change":
                        session.get(User, identifier).must_change_password = True
                    elif change == "logout":
                        session.execute(delete(UserSession).where(UserSession.user_id == identifier))
                    else:
                        active = session.scalar(select(UserSession).where(UserSession.user_id == identifier))
                        if change == "idle":
                            active.last_activity_at = offset_time(-3601)
                        else:
                            active.expires_at = offset_time(-1)
        return principal

    monkeypatch.setattr(platform.app.state.auth, "require", revoke_after_admission)
    response = clients["alice"].post(path)
    assert changed and response.status_code in {401, 403}, response.text
    assert snapshot(platform, asset["id"]) == before


@pytest.mark.parametrize("failure", ["csrf", "account-switched", "viewer"])
def test_archive_retains_csrf_and_current_account_assertion(team, failure):
    platform, _, users, clients = team
    asset = registered(team, "bob")
    before = snapshot(platform, asset["id"])
    client, headers = clients["bob"], {}
    if failure == "csrf":
        client.headers.pop("X-CSRF-Token")
    elif failure == "account-switched":
        headers["X-Operation-User"] = users["alice"]["id"]
    else:
        client = clients["reader"]
    response = client.post(archive_path(asset), headers=headers)
    assert response.status_code == 403, response.text
    assert snapshot(platform, asset["id"]) == before


def test_archiving_own_dataset_preserves_physical_files(team, tmp_path, monkeypatch):
    platform, _, users, clients = team
    data = tmp_path / "registered-dataset"
    data.mkdir()
    sample = data / "sample.bin"
    sample.write_bytes(b"physical dataset must survive catalogue archive")
    response = platform.client.post("/agent/v1/register", headers=ENROLL, json={
        "name": "local-files-fixture", "hostname": "local-files-fixture", "roots": [str(tmp_path)],
        "output_root": str(tmp_path / "output"), "profiles": [{"id": "default", "name": "fixture",
            "code_dir": str(tmp_path / "code"), "python": "/usr/bin/python3", "adapter": "smoke", "recipe": "smoke"}]})
    assert response.status_code == 201, response.text
    node = response.json()
    created = clients["alice"].post("/api/v1/assets", json=body(node, path=str(data)))
    assert created.status_code == 201, created.text
    asset = created.json()
    monkeypatch.setattr(platform.app.state.remote_manager, "_remote", lambda *_: pytest.fail("Catalogue archive must not open SSH"))
    response = clients["alice"].post(archive_path(asset))
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "ARCHIVED"
    assert snapshot(platform, asset["id"])["registered_by_user_id"] == users["alice"]["id"]
    assert data.is_dir() and sample.read_bytes() == b"physical dataset must survive catalogue archive"
