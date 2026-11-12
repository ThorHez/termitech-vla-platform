"""Link manually annotated data to the exact completed Value training inputs."""

import pytest

from vla_control.models import Asset, Attempt, Node

from . import test_recap_stages as recap_stages
from .conftest import ADMIN
from .test_automatic_preflight import acknowledge, submit
from .test_dataset_validations import ack, result
from .test_recap_stages import checks, pending
from .test_resource_ownership import accounts, login


def completed_value(platform, checkpoint_format="openpi-pytorch"):
    node, preview_id, command, code = pending(platform)
    assert acknowledge(platform, node, preview_id, command, code, checks())["status"] == "READY"
    response = submit(platform, preview_id)
    assert response.status_code == 201, response.text
    run = response.json()
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    step = run["parameters"]["steps"] - 1
    checkpoint = {"kind": "MODEL", "name": "value head", "version": str(step),
                  "path": f"/machine/output/{attempt['id']}/checkpoints/mixed/platform/{step}",
                  "metadata": {"format": "openpi-pytorch", "backend": "pytorch",
                               "model_role": "value_estimator", "step": step}}
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": [checkpoint]})
    assert response.status_code == 200, response.text
    run = platform.get_run(run)
    assert run["state"] == "SUCCEEDED", run["error"]
    checkpoint = run["outputs"][0]
    if checkpoint_format == "openpi-orbax":
        with platform.app.state.db.write() as session:
            session.get(Asset, checkpoint["id"]).metadata_json = {"format": checkpoint_format, "step": step}
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).metrics = {"capabilities": ["DATASET_VALIDATE"]}
    return node, run, attempt, checkpoint


def register_target(platform, node, *, path="/machine/annotated", headers=ADMIN, **changes):
    response = platform.client.post("/api/v1/assets", headers=headers, json={
        "kind": "DATASET", "name": "annotated output", "path": path, "node_id": node["node_id"], **changes})
    assert response.status_code == 201, response.text
    return response.json()


def link(platform, run, checkpoint, target, *, headers=ADMIN, **changes):
    return platform.client.post(f"/api/v1/runs/{run['id']}/recap-annotations", headers=headers, json={
        "checkpoint_asset_id": checkpoint["id"], "source_dataset_asset_id": run["inputs"]["dataset_0"],
        "dataset_asset_id": target["id"], **changes})


@pytest.mark.parametrize("checkpoint_format", ["openpi-pytorch", "openpi-orbax"])
def test_link_saves_frozen_provenance_and_is_idempotent_without_staling_validation(platform, checkpoint_format):
    node, run, attempt, checkpoint = completed_value(platform, checkpoint_format)
    target = register_target(platform, node, norm_asset_id=run["inputs"]["norm_1"], metadata={"note": "keep"})
    assert target["validation"]["status"] == "PENDING"
    first = link(platform, run, checkpoint, target)
    assert first.status_code == 200, first.text
    linked = first.json()
    assert linked["metadata"] == {"note": "keep", "recap_annotation_source": {
        "run_id": run["id"], "attempt_id": attempt["id"], "checkpoint_asset_id": checkpoint["id"],
        "source_dataset_asset_id": run["inputs"]["dataset_0"], "source_norm_asset_id": run["inputs"]["norm_0"],
        "source_dataset_index": 0}}
    assert linked["norm_asset_id"] == run["inputs"]["norm_1"]
    assert linked["validation"] == target["validation"]
    assert link(platform, run, checkpoint, target).json() == linked
    assert link(platform, run, checkpoint, target, source_dataset_index=0).json() == linked
    conflict = link(platform, run, checkpoint, target, source_dataset_asset_id=run["inputs"]["dataset_1"])
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "RECAP_ANNOTATION_CONFLICT"
    assert ack(platform, node, target["validation"], result(target)).status_code == 200
    repeated = link(platform, run, checkpoint, target)
    assert repeated.json()["validation"]["status"] == "SUCCEEDED"


@pytest.mark.parametrize("same_norm", [False, True])
def test_index_selects_exact_repeated_dataset_binding(platform, monkeypatch, same_norm):
    original = recap_stages.configured

    def configured(*args, **kwargs):
        node, inputs, rows = original(*args, **kwargs)
        rows[1]["dataset_asset_id"] = rows[0]["dataset_asset_id"]
        if same_norm:
            rows[1]["norm_asset_id"] = rows[0]["norm_asset_id"]
        return node, inputs, rows

    monkeypatch.setattr(recap_stages, "configured", configured)
    node, run, _, checkpoint = completed_value(platform)
    target = register_target(platform, node)
    legacy = link(platform, run, checkpoint, target)
    if same_norm:
        assert legacy.status_code == 200 and "source_dataset_index" not in legacy.json()["metadata"]["recap_annotation_source"]
    else:
        assert legacy.status_code == 409 and legacy.json()["error"]["code"] == "RECAP_ANNOTATION_SOURCE_INVALID"
    wrong_row = link(platform, run, checkpoint, target, source_dataset_index=2)
    assert wrong_row.status_code == 409 and wrong_row.json()["error"]["code"] == "RECAP_ANNOTATION_SOURCE_INVALID"
    response = link(platform, run, checkpoint, target, source_dataset_index=1)
    assert response.status_code == 200, response.text
    provenance = response.json()["metadata"]["recap_annotation_source"]
    assert provenance["source_norm_asset_id"] == run["inputs"]["norm_1"]
    assert provenance["source_dataset_index"] == 1
    assert link(platform, run, checkpoint, target, source_dataset_index=1).json() == response.json()
    conflict = link(platform, run, checkpoint, target, source_dataset_index=0)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "RECAP_ANNOTATION_CONFLICT"
    if same_norm:
        assert link(platform, run, checkpoint, target).json() == response.json()


@pytest.mark.parametrize("index", [-1, 64, True, "0"])
def test_source_index_requires_a_bounded_integer(platform, index):
    response = link(platform, {"id": "run", "inputs": {"dataset_0": "source"}}, {"id": "model"}, {"id": "target"},
                    source_dataset_index=index)
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/machine/dataset-0", "/machine//dataset-0/.", "/machine/dataset-0/labels", "/machine"])
def test_reject_in_place_and_nested_annotation_paths(platform, path):
    node, run, _, checkpoint = completed_value(platform)
    target = register_target(platform, node, path=path)
    response = link(platform, run, checkpoint, target)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECAP_ANNOTATION_IN_PLACE"


def test_reject_unbound_source_wrong_attempt_and_archived_target(platform):
    node, run, attempt, checkpoint = completed_value(platform)
    target = register_target(platform, node)
    response = link(platform, run, checkpoint, target, source_dataset_asset_id=target["id"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECAP_ANNOTATION_SOURCE_INVALID"
    with platform.app.state.db.write() as session:
        session.get(Attempt, attempt["id"]).state = "FAILED"
    response = link(platform, run, checkpoint, target)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECAP_ANNOTATION_CHECKPOINT_INVALID"
    with platform.app.state.db.write() as session:
        session.get(Attempt, attempt["id"]).state = "SUCCEEDED"
        session.get(Asset, target["id"]).state = "ARCHIVED"
    response = link(platform, run, checkpoint, target)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECAP_ANNOTATION_DATASET_INVALID"


def test_link_requires_target_ownership_or_admin_and_registration_permission(platform):
    node, run, _, checkpoint = completed_value(platform)
    accounts(platform)
    login(platform, "alice")
    first = register_target(platform, node, headers={})
    second = register_target(platform, node, path="/machine/another-annotation", headers={})
    login(platform, "bob")
    response = link(platform, run, checkpoint, first, headers={})
    assert response.status_code == 403 and response.json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"
    login(platform, "reader")
    response = link(platform, run, checkpoint, first, headers={})
    assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN"
    login(platform, "alice")
    assert link(platform, run, checkpoint, first, headers={}).status_code == 200
    login(platform, "owner-admin")
    assert link(platform, run, checkpoint, second, headers={}).status_code == 200


def test_asset_registration_cannot_claim_recap_lineage(platform):
    node = platform.register()
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "kind": "DATASET", "name": "forged output", "path": "/machine/annotated", "node_id": node["node_id"],
        "metadata": {"recap_annotation_source": {"run_id": "forged"}}})
    assert response.status_code == 422
