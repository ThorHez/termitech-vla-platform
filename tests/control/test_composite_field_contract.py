"""Composite labels survive registration and reuse without becoming training rules."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_control.models import Asset, Command, Run

from .conftest import ADMIN, profile_inspection


def profile():
    return {
        "schema_version": 1,
        "format": "lerobot",
        "robot_type": "ARX5",
        "arm_mode": "single",
        "action_representation": "unknown",
        "task_description": "Pick the electrical component.",
        "cameras": [{"name": "wrist", "key": "observation.images.wrist", "kind": "rgb"}],
        "feature_keys": ["observation.images.wrist", "observation.state", "actions"],
        "field_annotations": [
            {"key": "observation.state", "role": "state", "representation": "composite",
             "value_mode": "absolute", "component": "other", "arm": "left", "shape": [2, 16]},
            {"key": "actions", "role": "action", "representation": "composite",
             "value_mode": "delta", "component": "other", "arm": "left", "shape": [16, 10], "horizon": 16},
        ],
    }


def register(platform, node, declaration=None):
    declared = profile() if declaration is None else declaration
    return platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "name": "composite source", "kind": "DATASET", "node_id": node["node_id"],
        "path": "/machine/composite", "version": "v1",
        "metadata": {"dataset_profile": declared,
                     "custom": {"preserve": True}},
        "dataset_inspection_id": profile_inspection(platform, node["node_id"], "/machine/composite", declared),
    })


def assets(platform):
    with platform.app.state.db.read() as session:
        return [dict(row) for row in session.execute(select(Asset.__table__).order_by(Asset.id)).mappings()]


def test_composite_registration_preserves_source_shape_horizon_and_explicit_unknown_global_representation(platform):
    node = platform.register()
    response = register(platform, node)
    assert response.status_code == 201, response.text
    saved = response.json()
    declared = saved["metadata"]["dataset_profile"]
    assert declared["action_representation"] == "unknown"
    assert declared["field_annotations"] == profile()["field_annotations"]
    assert declared["field_annotations"][1]["horizon"] == 16
    assert "processing" not in declared
    assert saved["metadata"]["custom"] == {"preserve": True}
    listed = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    assert listed == [saved]
    assert platform.commands(node) == []


@pytest.mark.parametrize("role", ["state", "action"])
@pytest.mark.parametrize("component", ["position", "rotation", "joint", "gripper"])
def test_composite_is_not_a_single_semantic_component_and_rejection_is_atomic(platform, role, component):
    node = platform.register()
    baseline = register(platform, node)
    assert baseline.status_code == 201, baseline.text
    saved = baseline.json()
    before = assets(platform)
    invalid = profile()
    next(row for row in invalid["field_annotations"] if row["role"] == role)["component"] = component
    assert register(platform, node, invalid).status_code == 422
    update = platform.client.patch(f"/api/v1/assets/{saved['id']}/dataset-profile", headers=ADMIN,
                                  json={"dataset_profile": invalid})
    assert update.status_code == 422, update.text
    assert assets(platform) == before
    assert platform.commands(node) == []


@pytest.mark.parametrize("change", ["wrong_horizon", "state_horizon", "global_composite"])
def test_composite_does_not_relax_source_horizon_or_global_model_representation_contract(platform, change):
    node = platform.register()
    invalid = profile()
    if change == "wrong_horizon":
        invalid["field_annotations"][1]["horizon"] = 50
    elif change == "state_horizon":
        invalid["field_annotations"][0]["horizon"] = 2
    else:
        invalid["action_representation"] = "composite"
    assert register(platform, node, invalid).status_code == 422
    assert assets(platform) == []
    assert platform.commands(node) == []


def test_changing_source_to_composite_keeps_frozen_training_and_other_metadata(platform):
    node = platform.register()
    initial = profile()
    for row in initial["field_annotations"]:
        row["representation"] = "unknown"
    response = register(platform, node, initial)
    assert response.status_code == 201, response.text
    asset = response.json()
    run = platform.run(node)
    with platform.app.state.db.write() as session:
        row = session.get(Run, run["id"])
        row.code_expectation = {**row.code_expectation, "dataset_profiles": {asset["id"]: initial}}
    with platform.app.state.db.read() as session:
        frozen = deepcopy(dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one()))
        commands = deepcopy([(row.id, row.payload) for row in session.scalars(select(Command))])
    update = platform.client.patch(f"/api/v1/assets/{asset['id']}/dataset-profile", headers=ADMIN,
                                  json={"dataset_profile": profile()})
    assert update.status_code == 200, update.text
    assert update.json()["metadata"]["custom"] == {"preserve": True}
    with platform.app.state.db.read() as session:
        assert dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one()) == frozen
        assert [(row.id, row.payload) for row in session.scalars(select(Command))] == commands


def test_private_format_template_roundtrip_keeps_composite_source_facts_without_mutating_assets(platform):
    node = platform.register()
    asset = register(platform, node)
    assert asset.status_code == 201, asset.text
    before = assets(platform)
    password = "composite-template-password-123"
    platform.app.state.auth.create_user("composite-owner", password, "operator")
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "composite-owner", "password": password})
        assert login.status_code == 200, login.text
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        client.headers["X-Dataset-Template-User"] = login.json()["user"]["id"]
        created = client.post("/api/v1/dataset-format-templates", json={"name": "Mixed 16-step format", "profile": profile()})
        assert created.status_code == 201, created.text
        saved = created.json()
        assert saved["profile"]["field_annotations"] == profile()["field_annotations"]
        restored = client.get("/api/v1/dataset-format-templates/" + saved["id"])
        assert restored.status_code == 200 and restored.json() == saved
        assert "processing" not in restored.json()["profile"]
        assert assets(platform) == before
        deleted = client.request("DELETE", "/api/v1/dataset-format-templates/" + saved["id"],
                                 json={"expected_revision": saved["revision"]})
        assert deleted.status_code == 200


def info():
    return {"fps": 30, "features": {
        "observation.images.wrist": {"dtype": "video", "shape": [224, 224, 3]},
        "observation.state": {"dtype": "float32", "shape": [2, 16]},
        "actions": {"dtype": "float32", "shape": [16, 10]},
    }}


def test_composite_preflight_still_checks_actual_field_metadata_and_source_horizon():
    declared, metadata = profile(), info()
    before = deepcopy((declared, metadata))
    validated = verify_registered_dataset_profile(declared, metadata)
    assert validated["field_annotations"] == profile()["field_annotations"]
    assert (declared, metadata) == before
    for key, bad in (("actions", [50, 10]), ("observation.state", [2, 15])):
        changed = info()
        changed["features"][key]["shape"] = bad
        with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH"):
            verify_registered_dataset_profile(declared, changed)
    changed = info()
    changed["features"]["actions"]["dtype"] = "video"
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH"):
        verify_registered_dataset_profile(declared, changed)
