"""Image decisions are explicit and bound to trusted source metadata."""

from copy import deepcopy

import pytest
from sqlalchemy import select

from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_adapters.image_policy import image_dimensions, normalize_image_processing, validate_profile_image_processing
from vla_control.models import Asset

from .conftest import ADMIN
from .test_dataset_batches import batch, scan, submit  # noqa: F401
from .test_dataset_inspection_queries import ack, create


def profile(policy=None):
    camera = {"name": "head", "key": "observation.images.head", "kind": "rgb"}
    if policy is not None:
        camera["image_processing"] = policy
    return {"schema_version": 1, "format": "lerobot", "robot_type": "AR5", "arm_mode": "single",
            "action_representation": "unknown", "task_description": "Pick an item", "cameras": [camera]}


POLICY = {"resize_mode": "pad", "augmentation": "crop_rotate_color"}


def inspection(platform, node, *, shape=(3, 360, 640), path="/machine/data"):
    query = create(platform, node, path).json()
    value = {"schema_version": 1, "path": path, "source": {"info_sha256": "f" * 64, "tasks_sha256": None},
             "features": [{"key": "observation.images.head", "dtype": "video", "shape": list(shape), "names": []}],
             "suggestions": {}, "processing": {}, "evidence": [], "warnings": []}
    assert ack(platform, node, query, value).status_code == 200
    return query["id"]


def register(platform, node, declared=None, query_id=None, **extra):
    return platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "name": "camera dataset", "kind": "DATASET", "node_id": node["node_id"], "path": "/machine/data",
        "metadata": {"dataset_profile": declared} if declared is not None else {},
        **({"dataset_inspection_id": query_id} if query_id is not None else {}), **extra})


def test_profiled_registration_requires_real_inspection_even_with_declared_policy(platform):
    node = platform.register()
    response = register(platform, node, profile(POLICY))
    assert response.status_code == 409 and response.json()["error"]["code"] == "DATASET_IMAGE_INSPECTION_REQUIRED"
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == []


def test_non224_registration_requires_choice_before_norm_or_asset_creation(platform):
    node = platform.register()
    query_id = inspection(platform, node)
    response = register(platform, node, profile(), query_id, norm_path="/machine/new-norm")
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMAGE_PROCESSING_REQUIRED"
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == []
    response = register(platform, node, profile(POLICY), query_id)
    assert response.status_code == 201, response.text
    saved = response.json()["metadata"]
    assert saved["dataset_profile"]["cameras"][0]["image_processing"] == POLICY
    evidence = saved["_image_registration"]
    assert evidence["inspection_id"] == query_id and evidence["source"]["info_sha256"] == "f" * 64
    assert evidence["features"][0]["shape"] == [3, 360, 640]


def test_224_sources_need_no_resize_choice_and_legacy_path_only_is_not_assumed_224(platform):
    node = platform.register()
    query_id = inspection(platform, node, shape=(224, 224, 3))
    assert register(platform, node, profile(), query_id).status_code == 201
    legacy = register(platform, node)
    assert legacy.status_code == 201 and "_image_registration" not in legacy.json()["metadata"]
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_REQUIRED"):
        validate_profile_image_processing(None, {"observation.images.head": {"dtype": "video", "shape": [3, 360, 640]}})


def test_observed_non224_cannot_bypass_choice_by_omitting_profile_or_camera(platform):
    node = platform.register()
    query_id = inspection(platform, node)
    response = register(platform, node, query_id=query_id)
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMAGE_PROCESSING_REQUIRED"
    wrong = profile(POLICY)
    wrong["cameras"][0]["key"] = "other.image"
    response = register(platform, node, wrong, query_id)
    assert response.status_code == 422 and response.json()["error"]["code"] == "DATASET_IMAGE_METADATA_INVALID"


@pytest.mark.parametrize("case", ["wrong_path", "wrong_node", "pending", "forged_metadata"])
def test_registration_evidence_cannot_be_forged_or_crossed(platform, case):
    node = platform.register()
    source = platform.register(name="other") if case == "wrong_node" else node
    path = "/machine/other" if case == "wrong_path" else "/machine/data"
    query_id = (create(platform, source, path).json()["id"] if case == "pending"
                else inspection(platform, source, path=path))
    changes = {}
    if case == "forged_metadata":
        changes["metadata"] = {"dataset_profile": profile(POLICY), "_image_registration": {"features": []}}
    response = register(platform, node, profile(POLICY), query_id, **changes)
    assert response.status_code in {409, 422}, response.text
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == []


def test_existing_asset_can_add_policy_and_cannot_remove_it_against_saved_source(platform):
    node = platform.register()
    old = register(platform, node).json()
    query_id = inspection(platform, node)
    path = f"/api/v1/assets/{old['id']}/dataset-profile"
    response = platform.client.patch(path, headers=ADMIN,
        json={"dataset_profile": profile(POLICY), "dataset_inspection_id": query_id})
    assert response.status_code == 200, response.text
    before = response.json()
    response = platform.client.patch(path, headers=ADMIN, json={"dataset_profile": profile()})
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMAGE_PROCESSING_REQUIRED"
    current = platform.client.get("/api/v1/assets", headers=ADMIN).json()[0]
    assert current == before


def test_new_geometry_only_registration_and_legacy_asset_edit(platform):
    node = platform.register()
    query_id = inspection(platform, node)
    geometry = {"resize_mode": "stretch"}
    fresh = register(platform, node, profile(geometry), query_id)
    assert fresh.status_code == 201, fresh.text
    assert fresh.json()["metadata"]["dataset_profile"]["cameras"][0]["image_processing"] == geometry
    legacy = register(platform, node, profile(POLICY), query_id)
    assert legacy.status_code == 201, legacy.text
    path = f"/api/v1/assets/{legacy.json()['id']}/dataset-profile"
    changed = platform.client.patch(path, headers=ADMIN, json={"dataset_profile": profile(geometry)})
    assert changed.status_code == 200, changed.text
    assert changed.json()["metadata"]["dataset_profile"]["cameras"][0]["image_processing"] == {
        **geometry, "augmentation": POLICY["augmentation"]}


def test_batch_registration_requires_per_source_image_choice_and_stays_atomic(batch):
    original = batch[2].scanner
    def scanner(*args, **kwargs):
        value = original(*args, **kwargs)
        value["entries"][0]["features"][0]["shape"] = [3, 360, 640]
        return value
    batch[2].scanner = scanner
    value = scan(batch)
    assert value["entries"][0]["features"][0]["shape"] == [3, 360, 640]
    response = submit(batch, value)
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMAGE_PROCESSING_REQUIRED"
    with batch[0].state.db.read() as session:
        assert list(session.scalars(select(Asset))) == []
    declared = deepcopy(value["entries"][0])
    source_profile = profile(POLICY)
    source_profile["cameras"][0]["key"] = "observation.images.top"
    response = submit(batch, value, entries=[{"path": declared["path"], "name": "chosen", "profile": source_profile}])
    assert response.status_code == 200, response.text
    assert response.json()["created"][0]["metadata"]["_image_registration"]["scan_id"] == value["id"]


def test_norm_stats_ignores_visual_policy_but_sft_checks_real_source():
    data = {"features": {"observation.images.head": {"dtype": "video", "shape": [3, 360, 640]}}}
    assert verify_registered_dataset_profile(profile(), data, norm_stats=True)
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_REQUIRED"):
        verify_registered_dataset_profile(profile(), data)
    assert verify_registered_dataset_profile(profile(POLICY), data)["cameras"][0]["image_processing"] == POLICY


@pytest.mark.parametrize("shape,kind,expected", [
    ([3, 360, 640], "rgb", (360, 640)), ([360, 640, 3], "rgb", (360, 640)),
    ([4, 3, 224, 224], "rgb", (224, 224)), ([360, 640], "depth", (360, 640)),
    ([3, 224, 3], "rgb", None), ([224, 224], "rgb", None), ([True, 224, 224], "rgb", None),
])
def test_only_unambiguous_image_shapes_are_reported_224(shape, kind, expected):
    assert image_dimensions({"shape": shape}, kind) == expected


@pytest.mark.parametrize("value", [None, {}, {**POLICY, "resize_mode": "execute"},
    {**POLICY, "augmentation": ["none"]}, {**POLICY, "python": "run()"}])
def test_image_processing_is_a_strict_bounded_declaration(value):
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_INVALID"):
        normalize_image_processing(value)
