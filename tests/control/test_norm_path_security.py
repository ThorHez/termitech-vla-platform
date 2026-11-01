"""Independent routed checks for atomic norm path registration and binding."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.models import Asset, Command

from .conftest import ADMIN


def register(platform, node, kind="DATASET", **changes):
    return platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "node_id": node["node_id"], "name": "fixture " + kind, "kind": kind,
        "path": "/machine/" + kind.lower(), **changes})


def setup_assets(platform):
    node = platform.register()
    dataset = register(platform, node).json()
    norm = register(platform, node, "NORM", path="/machine/known").json()
    return node, dataset, norm


def rows(platform):
    with platform.app.state.db.read() as session:
        return [dict(row) for row in session.execute(select(Asset.__table__).order_by(Asset.id)).mappings()]


def bind_path(platform, dataset, path, **extra):
    return platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=ADMIN,
                               json={"norm_path": path, **extra})


@pytest.mark.parametrize("path", [
    "relative/norm", "/outside/norm", "/machine-other/norm", "/machine/../norm",
    "/machine/a/../norm", "/machine\\norm", "/machine/norm\nsecret", "/machine/norm\x1b",
    "/machine/norm\x7f", "/machine/norm\u0085", "/", "", None,
])
def test_invalid_paths_never_register_norm_or_change_dataset(platform, path):
    node, dataset, _ = setup_assets(platform)
    before = rows(platform)
    update = bind_path(platform, dataset, path)
    assert update.status_code == 422, update.text
    create = register(platform, node, norm_path=path)
    assert create.status_code == 422, create.text
    assert rows(platform) == before


@pytest.mark.parametrize("body", [
    {}, {"norm_path": "/machine/norm", "norm_asset_id": None},
    {"norm_path": "/machine/norm", "norm_asset_id": "fake"},
    {"norm_path": "/machine/norm", "node_id": "another"},
    {"norm_path": "/machine/norm", "kind": "NORM"},
])
def test_binding_contract_rejects_ambiguous_or_identity_changing_fields(platform, body):
    _, dataset, _ = setup_assets(platform)
    before = rows(platform)
    response = platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=ADMIN, json=body)
    assert response.status_code == 422, response.text
    assert rows(platform) == before


def test_paths_do_not_claim_remote_file_verification_or_issue_agent_commands(platform, monkeypatch):
    _, dataset, _ = setup_assets(platform)
    original = Path.stat

    def no_machine_stat(path, *args, **kwargs):
        assert not str(path).startswith("/machine"), "Registration must not inspect remote paths on the controller"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", no_machine_stat)
    response = bind_path(platform, dataset, "/machine/does-not-exist-on-controller/norm_stats.json")
    assert response.status_code == 200, response.text
    with platform.app.state.db.read() as session:
        norm = session.get(Asset, response.json()["norm_asset_id"])
        assert norm.kind == "NORM" and norm.state == "REGISTERED"
        assert not norm.metadata_json.get("byte_verified")
        assert not norm.metadata_json.get("verified")
        assert list(session.scalars(select(Command))) == []


def test_same_path_reuses_only_the_exact_node_and_norm_kind(platform):
    source, dataset, norm = setup_assets(platform)
    other = platform.register(name="other")
    foreign_norm = register(platform, other, "NORM", path="/machine/new-norm").json()
    wrong_kind = register(platform, source, "MODEL", path="/machine/new-norm").json()
    response = bind_path(platform, dataset, "/machine/new-norm")
    assert response.status_code == 200, response.text
    new_id = response.json()["norm_asset_id"]
    assert new_id not in {foreign_norm["id"], wrong_kind["id"]}
    second = bind_path(platform, dataset, "/machine/new-norm/")
    assert second.status_code == 200 and second.json()["norm_asset_id"] == new_id
    assert bind_path(platform, dataset, norm["path"]).json()["norm_asset_id"] == norm["id"]
    norms = [row for row in rows(platform) if row["kind"] == "NORM" and row["node_id"] == source["node_id"]]
    assert len(norms) == 2


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_unavailable_same_path_norm_is_not_silently_reactivated_or_duplicated(platform, state):
    _, dataset, norm = setup_assets(platform)
    with platform.app.state.db.write() as session:
        session.get(Asset, norm["id"]).state = state
    before = rows(platform)
    response = bind_path(platform, dataset, norm["path"])
    assert response.status_code == 409, response.text
    assert rows(platform) == before


def test_transfer_blocker_rolls_back_automatic_norm_registration(platform):
    source, dataset, norm = setup_assets(platform)
    target = platform.register(name="target")
    with platform.app.state.db.write() as session:
        session.get(Asset, dataset["id"]).norm_asset_id = norm["id"]
        session.add(AssetTransfer(id=uid(), request_key=uid(), request_digest="a" * 64,
            name="synthetic pending transfer", source_asset_id=dataset["id"], source_node_id=source["node_id"],
            source_path=dataset["path"], source_kind="DATASET", target_node_id=target["node_id"],
            target_directory="/machine/copied", target_name="dataset", state="QUEUED",
            binding_snapshot={}, logs="", created_at=now(), updated_at=now()))
    before = rows(platform)
    response = bind_path(platform, dataset, "/machine/should-not-be-created")
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_IN_USE"
    assert rows(platform) == before
    identical = bind_path(platform, dataset, norm["path"])
    assert identical.status_code == 200 and identical.json()["norm_asset_id"] == norm["id"]
    assert rows(platform) == before


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_dataset_state_blocker_precedes_automatic_norm_registration(platform, state):
    _, dataset, _ = setup_assets(platform)
    with platform.app.state.db.write() as session:
        session.get(Asset, dataset["id"]).state = state
    before = rows(platform)
    response = bind_path(platform, dataset, "/machine/no-orphan")
    assert response.status_code == 409
    assert rows(platform) == before


def test_missing_or_wrong_kind_dataset_and_bad_create_path_leave_no_orphan_norm(platform):
    node, _, norm = setup_assets(platform)
    before = rows(platform)
    for target, status in (({"id": "missing"}, 404), (norm, 409)):
        response = bind_path(platform, target, "/machine/no-orphan")
        assert response.status_code == status
        assert rows(platform) == before
    for response in (
        register(platform, node, path="/outside/data", norm_path="/machine/no-orphan"),
        register(platform, node, "MODEL", norm_path="/machine/no-orphan"),
    ):
        assert response.status_code in {409, 422}
        assert rows(platform) == before


@pytest.mark.parametrize("role,status", [("admin", 200), ("operator", 200), ("viewer", 403)])
def test_direct_path_binding_preserves_role_csrf_and_origin_checks(platform, role, status):
    _, dataset, _ = setup_assets(platform)
    password = "norm-path-policy-fixture-123"
    platform.app.state.auth.create_user("path-" + role, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "path-" + role, "password": password})
        assert login.status_code == 200
        csrf = login.json()["csrf_token"]
        url = f"/api/v1/assets/{dataset['id']}/norm-binding"
        before = rows(platform)
        response = client.put(url, json={"norm_path": "/machine/role-norm"}, headers={"X-CSRF-Token": csrf})
        assert response.status_code == status, response.text
        if role == "viewer":
            assert rows(platform) == before
        else:
            before = rows(platform)
            for headers in ({}, {"X-CSRF-Token": csrf, "Origin": "https://foreign.example"}):
                blocked = client.put(url, json={"norm_path": "/machine/forbidden-norm"}, headers=headers)
                assert blocked.status_code == 403
                assert rows(platform) == before
