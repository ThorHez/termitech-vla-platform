"""Explicit OpenPI model formats remain editable manual asset metadata."""

import pytest
from sqlalchemy import select

from vla_control.auth_models import AuditEvent
from vla_control.models import Asset

from .conftest import ADMIN
from .test_asset_transfers import create as create_transfer, fixture as transfer_fixture
from .test_multi_dataset_control import configured
from .test_recap_annotations import completed_value
from .test_resource_ownership import accounts, login
from .test_task_recipe import request_preview


def register(platform, node, metadata, *, name="checkpoint", kind="MODEL", headers=ADMIN):
    response = platform.client.post("/api/v1/assets", headers=headers, json={
        "kind": kind, "name": name, "path": f"/machine/{name}", "node_id": node["node_id"],
        "metadata": metadata,
    })
    assert response.status_code == 201, response.text
    return response.json()


def update(platform, asset, model_format, *, headers=ADMIN):
    return platform.client.patch(f"/api/v1/assets/{asset['id']}/model-format", headers=headers,
                                 json={"format": model_format})


def test_register_and_update_explicit_format_without_changing_asset_identity(platform):
    node = platform.register()
    original = register(platform, node, {"format": "openpi-orbax", "notes": "keep"})
    changed = update(platform, original, "openpi-pytorch")
    assert changed.status_code == 200, changed.text
    expected = {**original, "metadata": {"format": "openpi-pytorch", "notes": "keep"}}
    assert changed.json() == expected
    assert update(platform, original, "openpi-pytorch").json() == expected
    assert update(platform, original, None).json() == {**original, "metadata": {"notes": "keep"}}
    legacy = register(platform, node, {"format": "OpenPI-JAX"}, name="legacy")
    assert update(platform, legacy, None).json()["metadata"] == {}
    unrelated = register(platform, node, {"format": "gguf", "notes": "keep"}, name="generic")
    assert update(platform, unrelated, None).json() == unrelated


def test_manual_model_format_edits_require_owner_or_admin_and_are_audited(platform):
    node = platform.register()
    unowned = register(platform, node, {}, name="unowned")
    users = accounts(platform)
    login(platform, "alice")
    owned = register(platform, node, {}, headers={})
    assert owned["registered_by_user_id"] == users["alice"]
    assert update(platform, owned, "openpi-pytorch", headers={}).status_code == 200
    assert update(platform, unowned, "openpi-pytorch", headers={}).json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"
    login(platform, "bob")
    assert update(platform, owned, "openpi-orbax", headers={}).json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"
    login(platform, "reader")
    assert update(platform, owned, "openpi-orbax", headers={}).status_code == 403
    login(platform, "owner-admin")
    assert update(platform, unowned, "openpi-orbax", headers={}).status_code == 200
    assert update(platform, owned, "openpi-orbax", headers={}).status_code == 200
    with platform.app.state.db.read() as session:
        updates = list(session.scalars(select(AuditEvent).where(
            AuditEvent.action == "assets.update_model_format", AuditEvent.status == 200)))
        assert any(event.actor_id == users["alice"] and event.details["target_id"] == owned["id"]
                   for event in updates)


@pytest.mark.parametrize("body", [{}, {"format": "pytorch"}, {"format": "gguf"},
                                    {"format": False}, {"format": None, "path": "/machine/other"}])
def test_format_patch_requires_an_explicit_supported_value(platform, body):
    response = platform.client.patch("/api/v1/assets/absent/model-format", headers=ADMIN, json=body)
    assert response.status_code == 422, response.text


def test_model_format_patch_cannot_relabel_training_or_sync_outputs_or_unavailable_assets(platform):
    node, _, _, checkpoint = completed_value(platform)
    assert update(platform, checkpoint, "openpi-orbax").json()["error"]["code"] == "MODEL_FORMAT_READONLY"
    for name, metadata in (("asset-sync", {"format": "openpi-pytorch", "asset_sync": {"byte_verified": True}}),
                           ("checkpoint-sync", {"format": "openpi-transferred", "transfer_id": "sync"})):
        synced = register(platform, node, metadata, name=name)
        assert update(platform, synced, "openpi-orbax").json()["error"]["code"] == "MODEL_FORMAT_READONLY"
    manual = register(platform, node, {})
    archive = platform.client.post(f"/api/v1/assets/{manual['id']}/archive", headers=ADMIN)
    assert archive.status_code == 200, archive.text
    assert update(platform, manual, "openpi-pytorch").json()["error"]["code"] == "ASSET_UNAVAILABLE"
    norm = register(platform, node, {}, name="norm", kind="NORM")
    assert update(platform, norm, "openpi-pytorch").json()["error"]["code"] == "ASSET_KIND_MISMATCH"


def test_format_patch_cannot_change_a_frozen_sync_source(transfer_fixture):
    platform, _, source, *_ = transfer_fixture
    asset = register(platform, source, {"format": "openpi-pytorch"})
    response = create_transfer(transfer_fixture, source_asset_id=asset["id"])
    assert response.status_code == 202, response.text
    rejected = update(platform, asset, "openpi-orbax")
    assert rejected.status_code == 409 and rejected.json()["error"]["code"] == "ASSET_IN_USE"
    transfer_id = response.json()["id"]
    ended = platform.client.post(f"/api/v1/asset-transfers/{transfer_id}/cancel", headers=ADMIN)
    assert ended.json()["state"] == "CANCELLED"
    assert update(platform, asset, "openpi-orbax").status_code == 200


@pytest.mark.parametrize("model_format", ["openpi-orbax", "orbax", "jax", " OpenPI-JAX ",
                                          "openpi-pytorch", "torch", "openpi-transferred", "other", None])
def test_stage_advantage_rejects_known_orbax_and_retains_file_preflight_for_other_formats(platform, model_format):
    node, inputs, rows = configured(platform)
    with platform.app.state.db.write() as session:
        asset = session.get(Asset, inputs["model"])
        asset.metadata_json = {} if model_format is None else {"format": model_format}
    response = request_preview(platform, node, template_id="recap", training_stage="stage_progress",
                               inputs=inputs, dataset_bindings=rows[:1])
    if model_format in {"openpi-orbax", "orbax", "jax", " OpenPI-JAX "}:
        assert response.status_code == 422 and response.json()["error"]["code"] == "MODEL_FORMAT"
        assert platform.commands(node) == []
    else:
        assert response.status_code == 202, response.text
        assert any(command["type"] == "PREFLIGHT" for command in platform.commands(node))
