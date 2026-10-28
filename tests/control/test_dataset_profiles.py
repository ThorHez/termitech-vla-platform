"""Dataset semantics are bounded metadata; editing never rewrites training state."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.dataset_profiles import normalize_dataset_profile, validate_asset_metadata
from vla_control.models import Asset, Command, Run

from .conftest import ADMIN, ENROLL


def profile(**changes):
    return {"schema_version": 1, "format": "lerobot", "robot_type": "ARX5",
            "arm_mode": "bimanual", "action_representation": "eef",
            "cameras": [{"name": "left wrist", "key": "left_wrist_0_rgb_0", "kind": "rgb"},
                        {"name": "right wrist", "key": "right_wrist_0_rgb_0", "kind": "rgb"}],
            "task_description": "Fold the towel", **changes}


def register_asset(platform, node, metadata=None, kind="DATASET"):
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "node_id": node["node_id"], "name": "training data", "kind": kind,
        "path": "/machine/data", "version": "v1", "metadata": metadata or {}})
    assert response.status_code == 201, response.text
    return response.json()


def patch_profile(platform, asset, value=None, **extra):
    return platform.client.patch(f"/api/v1/assets/{asset['id']}/dataset-profile", headers=ADMIN,
                                 json={"dataset_profile": profile() if value is None else value, **extra})


def asset_row(platform, asset_id):
    with platform.app.state.db.read() as session:
        return dict(session.execute(select(Asset.__table__).where(Asset.id == asset_id)).mappings().one())


def test_registration_normalizes_v1_semantics_without_losing_existing_metadata(platform):
    node = platform.register()
    declared = profile(robot_type=" ARX5 ", task_description="  Fold towel\nThen place it away.  ",
                       cameras=[{"name": " wrist ", "key": " observation.images.wrist ", "kind": "rgb"}],
                       action_dim=20, state_dim=44, fps=30, action_mode="absolute",
                       rotation_representation="rotation_6d", feature_keys=[" state ", "action"])
    asset = register_asset(platform, node, {"dataset_profile": declared, "availability": "available", "custom": {"value": 4}})
    clean = asset["metadata"]["dataset_profile"]
    assert clean == normalize_dataset_profile(declared)
    assert clean["robot_type"] == "ARX5" and clean["task_description"] == "Fold towel\nThen place it away."
    assert clean["cameras"] == [{"name": "wrist", "key": "observation.images.wrist", "kind": "rgb"}]
    assert clean["feature_keys"] == ["state", "action"]
    assert asset["metadata"]["custom"] == {"value": 4} and asset["metadata"]["availability"] == "available"
    assert asset["state"] == "REGISTERED"
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json()[0]["metadata"] == asset["metadata"]


def test_legacy_registration_without_profile_remains_supported(platform):
    node = platform.register()
    for kind in ("DATASET", "MODEL", "NORM"):
        assert register_asset(platform, node, kind=kind)["metadata"] == {}
    assert validate_asset_metadata("DATASET", {"custom": [1, 2]}) == {"custom": [1, 2]}


@pytest.mark.parametrize("kind", ["MODEL", "NORM"])
def test_only_dataset_assets_accept_dataset_semantics_on_create(platform, kind):
    node = platform.register()
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "node_id": node["node_id"], "name": "asset", "kind": kind, "path": "/machine/data",
        "metadata": {"dataset_profile": profile()}})
    assert response.status_code == 422
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Asset))) == []


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"schema_version": "1"},
    {"format": "arbitrary"}, {"robot_type": " "}, {"robot_type": "x" * 101}, {"robot_type": 1},
    {"robot_type": "robot\x00name"}, {"arm_mode": "triple"}, {"action_representation": "torque"},
    {"cameras": []}, {"cameras": [{"name": "a", "key": "k", "kind": "rgb"}] * 17},
    {"cameras": [{"name": "same", "key": "a", "kind": "rgb"}, {"name": "same", "key": "b", "kind": "rgb"}]},
    {"cameras": [{"name": "a", "key": "same", "kind": "rgb"}, {"name": "b", "key": " same ", "kind": "depth"}]},
    {"cameras": [{"name": "a\nname", "key": "key", "kind": "rgb"}]},
    {"cameras": [{"name": "a", "key": "key", "kind": "thermal"}]},
    {"cameras": [{"name": "a", "key": "key", "kind": "rgb", "python": "unsafe()"}]},
    {"task_description": " "}, {"task_description": "x" * 2001}, {"task_description": "task\x1bcommand"},
    {"action_dim": True}, {"action_dim": "20"}, {"action_dim": 0}, {"action_dim": 513},
    {"state_dim": 0}, {"fps": 0}, {"fps": True}, {"fps": 1001},
    {"action_mode": "velocity"}, {"rotation_representation": "euler"},
    {"feature_keys": ["same", " same "]}, {"feature_keys": [""]},
    {"feature_keys": ["key\x85name"]}, {"feature_keys": ["k" + str(i) for i in range(257)]},
    {"processor": "arbitrary_python"},
])
def test_invalid_profiles_cannot_enter_registration_or_modify_existing_metadata(platform, change):
    node = platform.register()
    asset = register_asset(platform, node, {"existing": True})
    before = asset_row(platform, asset["id"])
    invalid = profile(**change)
    response = patch_profile(platform, asset, invalid)
    assert response.status_code == 422, response.text
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "node_id": node["node_id"], "name": "invalid", "kind": "DATASET", "path": "/machine/data",
        "metadata": {"dataset_profile": invalid}})
    assert response.status_code == 422, response.text
    assert asset_row(platform, asset["id"]) == before


def test_profile_update_only_changes_its_namespace_preserving_asset_and_frozen_run(platform):
    node = platform.register()
    asset = register_asset(platform, node, {"dataset_profile": profile(), "availability": "available", "custom": {"x": 2}})
    run = platform.run(node)
    with platform.app.state.db.write() as session:
        session.get(Asset, asset["id"]).state = "READY"
        original = deepcopy(session.get(Run, run["id"]).code_expectation)
        session.get(Run, run["id"]).code_expectation = {**original, "dataset_profiles": {asset["id"]: asset["metadata"]["dataset_profile"]}}
    before = asset_row(platform, asset["id"])
    frozen_before = platform.get_run(run)
    commands = platform.commands(node)
    changed = profile(task_description="Different task", arm_mode="single")
    response = patch_profile(platform, asset, changed)
    assert response.status_code == 200, response.text
    after = asset_row(platform, asset["id"])
    assert after == {**before, "metadata": {**before["metadata"], "dataset_profile": normalize_dataset_profile(changed)}}
    assert response.json()["state"] == "READY"
    assert platform.get_run(run)["code_expectation"] == frozen_before["code_expectation"]
    assert platform.get_run(run)["parameters"] == frozen_before["parameters"]
    assert platform.commands(node) == commands


@pytest.mark.parametrize("kind", ["MODEL", "NORM"])
def test_patch_rejects_non_dataset_assets(platform, kind):
    node = platform.register()
    asset = register_asset(platform, node, {"custom": 1}, kind=kind)
    before = asset_row(platform, asset["id"])
    response = patch_profile(platform, asset)
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_KIND_MISMATCH"
    assert asset_row(platform, asset["id"]) == before


def test_patch_rejects_archived_missing_and_extra_mutation_fields(platform):
    node = platform.register()
    asset = register_asset(platform, node)
    response = patch_profile(platform, asset, path="/different", version="v2", metadata={"custom": 2})
    assert response.status_code == 422
    assert patch_profile(platform, {"id": "missing"}).status_code == 404
    assert platform.client.post(f"/api/v1/assets/{asset['id']}/archive", headers=ADMIN).status_code == 200
    before = asset_row(platform, asset["id"])
    response = patch_profile(platform, asset)
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_ARCHIVED"
    assert asset_row(platform, asset["id"]) == before


@pytest.mark.parametrize("role,status", [("admin", 200), ("operator", 200), ("viewer", 403)])
def test_dataset_profile_patch_respects_asset_write_permission_and_csrf(platform, role, status):
    node = platform.register()
    asset = register_asset(platform, node)
    password = "dataset-profile-isolated-password"
    platform.app.state.auth.create_user("dataset-" + role, password, role)
    client = TestClient(platform.app, base_url="https://testserver")
    try:
        login = client.post("/api/v1/auth/login", json={"username": "dataset-" + role, "password": password})
        assert login.status_code == 200
        path = f"/api/v1/assets/{asset['id']}/dataset-profile"
        csrf = login.json()["csrf_token"]
        response = client.patch(path, json={"dataset_profile": profile()}, headers={"X-CSRF-Token": csrf})
        assert response.status_code == status, response.text
        if status == 200:
            response = client.patch(path, json={"dataset_profile": profile()})
            assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"
            response = client.patch(path, json={"dataset_profile": profile()},
                                    headers={"X-CSRF-Token": csrf, "Origin": "https://foreign.invalid"})
            assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_REJECTED"
    finally:
        client.close()


def test_anonymous_enrollment_and_agent_credentials_cannot_patch_profile(platform):
    node = platform.register()
    asset = register_asset(platform, node)
    path = f"/api/v1/assets/{asset['id']}/dataset-profile"
    for headers in ({}, ENROLL, node["headers"]):
        response = platform.client.patch(path, headers=headers, json={"dataset_profile": profile()})
        assert response.status_code == 401
    assert asset["metadata"] == {}


def test_normalization_is_independent_of_original_mutable_metadata():
    value = {"dataset_profile": profile(), "custom": {"x": [1]}}
    clean = validate_asset_metadata("DATASET", value)
    clean["custom"]["x"].append(2)
    clean["dataset_profile"]["cameras"][0]["name"] = "other"
    assert value == {"dataset_profile": profile(), "custom": {"x": [1]}}


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), float("-inf")])
def test_dataset_profile_rejects_nonfinite_measurements(fps):
    with pytest.raises(ValueError):
        normalize_dataset_profile(profile(fps=fps))
