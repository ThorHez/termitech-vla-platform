"""Dataset normalization defaults are explicit, local and independent of frozen jobs."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.models import Asset, Command, Run

from .conftest import ADMIN, ENROLL


def register_asset(platform, node, kind="DATASET", **changes):
    return platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "node_id": node["node_id"], "kind": kind, "name": "asset " + kind,
        "path": "/machine/" + kind.lower(), "metadata": {"custom": {"keep": True}}, **changes})


def bind(platform, dataset, norm_asset_id, **changes):
    return platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=ADMIN,
                               json={"norm_asset_id": norm_asset_id, **changes})


def listed(platform, asset_id):
    return next(item for item in platform.client.get("/api/v1/assets", headers=ADMIN).json()
                if item["id"] == asset_id)


def test_dataset_can_bind_replace_and_clear_optional_registered_norm(platform):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    norms = [register_asset(platform, node, "NORM", name=name).json() for name in ("raw14", "full32")]
    assert dataset["norm_asset_id"] is None
    for norm in norms:
        response = bind(platform, dataset, norm["id"])
        assert response.status_code == 200, response.text
        assert response.json() == {**dataset, "norm_asset_id": norm["id"]}
        assert listed(platform, dataset["id"])["norm_asset_id"] == norm["id"]
        assert listed(platform, norm["id"])["norm_asset_id"] is None
    assert bind(platform, dataset, None).json() == dataset
    with platform.app.state.db.read() as session:
        assert session.get(Asset, dataset["id"]).norm_asset_id is None


def test_dataset_registration_can_explicitly_bind_a_norm_without_name_inference(platform):
    node = platform.register()
    norm = register_asset(platform, node, "NORM", name="same name").json()
    bound = register_asset(platform, node, name="dataset", norm_asset_id=norm["id"])
    assert bound.status_code == 201, bound.text
    assert bound.json()["norm_asset_id"] == norm["id"]
    plain = register_asset(platform, node, name="same name").json()
    assert plain["norm_asset_id"] is None


@pytest.mark.parametrize("kind", ["MODEL", "NORM"])
def test_non_dataset_cannot_set_a_norm_during_create_or_update(platform, kind):
    node = platform.register()
    norm = register_asset(platform, node, "NORM").json()
    response = register_asset(platform, node, kind, norm_asset_id=norm["id"])
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ASSET_KIND_MISMATCH"
    plain = register_asset(platform, node, kind).json()
    assert plain["norm_asset_id"] is None
    for value in (norm["id"], None):
        response = bind(platform, plain, value)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ASSET_KIND_MISMATCH"


@pytest.mark.parametrize("case,code", [
    ("missing", "NORM_ASSET_NOT_FOUND"),
    ("wrong_node", "NORM_ASSET_NODE_MISMATCH"),
    ("wrong_kind", "NORM_ASSET_KIND_MISMATCH"),
    ("ARCHIVED", "NORM_ASSET_UNAVAILABLE"),
    ("UNAVAILABLE", "NORM_ASSET_UNAVAILABLE"),
])
def test_invalid_norm_never_changes_existing_binding_or_creates_a_dataset(platform, case, code):
    node = platform.register()
    valid = register_asset(platform, node, "NORM").json()
    dataset = register_asset(platform, node, norm_asset_id=valid["id"]).json()
    candidate = register_asset(platform, platform.register(name="other") if case == "wrong_node" else node,
                               "MODEL" if case == "wrong_kind" else "NORM").json()
    if case in {"ARCHIVED", "UNAVAILABLE"}:
        with platform.app.state.db.write() as session:
            session.get(Asset, candidate["id"]).state = case
    value = "missing" if case == "missing" else candidate["id"]
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    for response in (bind(platform, dataset, value), register_asset(platform, node, norm_asset_id=value)):
        assert response.status_code in {404, 409}, response.text
        assert response.json()["error"]["code"] == code
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_unavailable_dataset_cannot_change_or_remove_its_binding(platform, state):
    node = platform.register()
    norm = register_asset(platform, node, "NORM").json()
    dataset = register_asset(platform, node, norm_asset_id=norm["id"]).json()
    with platform.app.state.db.write() as session:
        session.get(Asset, dataset["id"]).state = state
    for value in (None, norm["id"]):
        response = bind(platform, dataset, value)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "DATASET_ASSET_UNAVAILABLE"
    assert listed(platform, dataset["id"])["norm_asset_id"] == norm["id"]


def test_binding_update_is_scoped_and_never_rewrites_frozen_training_or_commands(platform):
    node = platform.register()
    first, second = [register_asset(platform, node, "NORM", name=name).json() for name in ("old", "new")]
    dataset = register_asset(platform, node, norm_asset_id=first["id"]).json()
    run = platform.run(node)
    with platform.app.state.db.write() as session:
        row = session.get(Run, run["id"])
        row.inputs = {"dataset": dataset["id"], "norm": first["id"]}
        row.code_expectation = {**row.code_expectation,
            "dataset_bindings": [{"dataset_asset_id": dataset["id"], "norm_asset_id": first["id"]}]}
    with platform.app.state.db.read() as session:
        before = dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one())
        commands = deepcopy([(row.id, row.payload) for row in session.scalars(select(Command))])
    assert bind(platform, dataset, second["id"]).status_code == 200
    with platform.app.state.db.read() as session:
        after = dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one())
        assert after == before
        assert [(row.id, row.payload) for row in session.scalars(select(Command))] == commands
    assert listed(platform, dataset["id"]) == {**dataset, "norm_asset_id": second["id"]}


def test_active_transfer_prevents_changed_binding_but_replayed_put_is_idempotent(platform):
    source, target = platform.register(), platform.register(name="target")
    first, second = [register_asset(platform, source, "NORM", name=name).json() for name in ("old", "new")]
    dataset = register_asset(platform, source, norm_asset_id=first["id"]).json()
    transfer_id = uid()
    with platform.app.state.db.write() as session:
        session.add(AssetTransfer(id=transfer_id, request_key=uid(), request_digest="a" * 64,
            name="isolated pending copy", source_asset_id=dataset["id"], source_node_id=source["node_id"],
            source_path=dataset["path"], source_kind="DATASET", target_node_id=target["node_id"],
            target_directory="/machine/copied", target_name="dataset", state="QUEUED",
            binding_snapshot={}, logs="", created_at=now(), updated_at=now()))
    assert bind(platform, dataset, first["id"]).status_code == 200
    for value in (None, second["id"]):
        response = bind(platform, dataset, value)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ASSET_IN_USE"
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, transfer_id).state = "FAILED"
    assert bind(platform, dataset, second["id"]).status_code == 200


@pytest.mark.parametrize("body", [{}, {"norm_asset_id": ""}, {"norm_asset_id": "x" * 37},
    {"norm_asset_id": 1}, {"norm_asset_id": None, "path": "/another/path"}])
def test_binding_request_rejects_missing_invalid_and_unrelated_fields(platform, body):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    response = platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=ADMIN, json=body)
    assert response.status_code == 422
    assert listed(platform, dataset["id"]) == dataset


@pytest.mark.parametrize("role,status", [("admin", 200), ("operator", 200), ("viewer", 403)])
def test_norm_binding_requires_asset_write_permission_and_valid_csrf(platform, role, status):
    node = platform.register()
    norm = register_asset(platform, node, "NORM").json()
    dataset = register_asset(platform, node).json()
    password = "norm-binding-isolated-password"
    platform.app.state.auth.create_user("norm-" + role, password, role)
    client = TestClient(platform.app, base_url="https://testserver")
    try:
        login = client.post("/api/v1/auth/login", json={"username": "norm-" + role, "password": password})
        assert login.status_code == 200
        path = f"/api/v1/assets/{dataset['id']}/norm-binding"
        csrf = login.json()["csrf_token"]
        response = client.put(path, json={"norm_asset_id": norm["id"]}, headers={"X-CSRF-Token": csrf})
        assert response.status_code == status, response.text
        if status == 200:
            response = client.put(path, json={"norm_asset_id": None})
            assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"
            response = client.put(path, json={"norm_asset_id": None},
                headers={"X-CSRF-Token": csrf, "Origin": "https://foreign.invalid"})
            assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_REJECTED"
    finally:
        client.close()


def test_missing_dataset_and_nonhuman_credentials_cannot_mutate_a_binding(platform):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    assert bind(platform, {"id": "missing"}, None).status_code == 404
    for headers in ({}, ENROLL, node["headers"]):
        response = platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=headers,
                                       json={"norm_asset_id": None})
        assert response.status_code == 401
    assert listed(platform, dataset["id"]) == dataset
