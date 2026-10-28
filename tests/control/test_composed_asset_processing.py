"""A selected training composition stays frozen independently of asset history."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from vla_protocol import canonical_digest
from vla_adapters.composition_spec import composition_choice, composition_digest
from vla_control.data_processors import _entry

from .conftest import ADMIN, automatic_preflight, snapshot
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview
from .test_automatic_preflight import acknowledge, submit


def description():
    spec = json.loads((Path(__file__).resolve().parents[1] / "fixtures/composed-processing-v1.json").read_text())["valid"][0]
    return {"schema_version": 1, "format": "lerobot", "robot_type": "ARX5", "arm_mode": "single",
            "action_representation": "eef", "cameras": [{"name": "wrist", "key": spec["cameras"][0]["key"], "kind": "rgb"}],
            "task_description": "Pick up cup", "action_dim": 10, "state_dim": 7, "processing": spec}


def setup_composed(platform, *, source_only=False):
    node, inputs, rows = configured(platform)
    rows = rows[:1]
    declared = description()
    selected = composition_choice(declared)
    if source_only:
        declared.pop("processing")
        declared["field_annotations"] = [
            {"key": "observation.joints", "role": "state", "representation": "joint", "value_mode": "absolute",
             "component": "joint", "arm": "left", "shape": [6]},
            {"key": "observation.gripper", "role": "state", "representation": "unknown", "value_mode": "absolute",
             "component": "gripper", "arm": "left", "shape": [1]},
            {"key": "actions", "role": "action", "representation": "eef", "value_mode": "absolute",
             "component": "other", "arm": "unknown", "shape": [16, 10]},
        ]
    response = platform.client.patch(f"/api/v1/assets/{rows[0]['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                     json={"dataset_profile": declared})
    assert response.status_code == 200, response.text
    profile = response.json()["metadata"]["dataset_profile"]
    rows[0].pop("recipe_index")
    rows[0]["processor"] = selected
    return node, inputs, rows, profile


def pending(platform, *, source_only=False, customized=False):
    node, inputs, rows, profile = setup_composed(platform, source_only=source_only)
    selected = rows[0]["processor"]["parameters"]["composition"]
    if customized:
        selected["state_fields"].reverse()
        selected["normalization"] = "zscore"
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    evidence = automatic_preflight(command["payload"])
    data = evidence["plan"]["datasets"][0]
    data["registered_profile"] = deepcopy(profile)
    data["composition"] = {"schema_version": 1, "spec": deepcopy(selected),
        "digest": composition_digest(selected), "training_stage": "sft", "model_type": "Pi0Config",
        "action_horizon": 16, "action_dim": 32,
        "pi05": True, "raw_state_dim": 7, "raw_action_dim": 10, "state_history": 1, "normalization": selected["normalization"],
        "image_keys": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], "depth_keys": [],
        "action_metadata_sha256": data["metadata_sha256"], "action_sequence_keys": [], "action_loss_mask": [1.] * 10 + [0.] * 22,
        "source": {"path": "src/openpi/training/composed_processing.py", "sha256": "c" * 64}}
    data["actions_shape"] = [16, 10]
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    code = snapshot(recipe="mixed", dataset_bindings=rows, automatic_preflight=evidence)
    return node, identifier, command, code, profile


def test_composition_is_frozen_with_asset_even_after_edits_and_reaches_start(platform):
    node, identifier, command, code, profile = pending(platform)
    assert command["payload"]["inputs"]["dataset_0"]["dataset_profile"] == profile
    changed = deepcopy(profile); changed["processing"]["coordinate_frame"] = "another_frame"
    asset_id = command["payload"]["inputs"]["dataset_0"]["id"]
    response = platform.client.patch(f"/api/v1/assets/{asset_id}/dataset-profile", headers=ADMIN,
                                     json={"dataset_profile": changed})
    assert response.status_code == 200, response.text
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    run = submit(platform, identifier)
    assert run.status_code == 201, run.text
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["inputs"]["dataset_0"]["dataset_profile"] == profile
    assert start["payload"]["dataset_bindings"][0]["processor"] == composition_choice(profile)


@pytest.mark.parametrize("change", ["legacy", "different", "omit"])
def test_asset_history_does_not_choose_the_current_training_processor(platform, change):
    node, inputs, rows, profile = setup_composed(platform)
    if change == "legacy": rows[0].pop("processor"); rows[0]["recipe_index"] = 0
    if change == "different": rows[0]["processor"]["parameters"]["composition"]["coordinate_frame"] = "other"
    if change == "omit": rows[0]["processor"] = {"name": "OtherData", "parameters": {}}
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 202, response.text
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    assert command["payload"]["dataset_bindings"] == rows


@pytest.mark.parametrize("source_only", [False, True])
def test_training_owns_customized_composition_with_or_without_asset_history(platform, source_only):
    node, identifier, command, code, declared = pending(platform, source_only=source_only, customized=True)
    selected = command["payload"]["dataset_bindings"][0]["processor"]
    assert selected["parameters"]["composition"]["normalization"] == "zscore"
    assert selected["parameters"]["composition"]["state_fields"][0]["key"] == "observation.gripper"
    if source_only:
        assert "processing" not in declared
    else:
        assert declared["processing"]["normalization"] == "quantile"
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    run = submit(platform, identifier)
    assert run.status_code == 201, run.text
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["dataset_bindings"][0]["processor"] == selected
    assert start["payload"]["inputs"]["dataset_0"]["dataset_profile"] == declared


@pytest.mark.parametrize("field,value", [("composition", None), ("digest", "0" * 64), ("source", {"path": "../evil.py", "sha256": "a" * 64}),
    ("model_type", "Pi0MemConfig"), ("training_stage", "value"), ("action_horizon", True),
    ("action_loss_mask", [1.] * 32), ("action_sequence_keys", ["actions"]), ("action_metadata_sha256", "0" * 64),
    ("raw_state_dim", 8), ("pi05", None), ("image_keys", ["base_0_rgb"]),
    ("image_keys", ["left_wrist_0_rgb", "left_wrist_0_rgb"]), ("depth_keys", ["left_wrist_0_rgb"])])
def test_agent_must_prove_composition_model_and_source(platform, field, value):
    node, identifier, command, code, _ = pending(platform)
    evidence = code["automatic_preflight"]
    data = evidence["plan"]["datasets"][0]
    if field == "composition": data.pop("composition")
    else: data["composition"][field] = value
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] == "AGENT_COMPOSITION_UNSUPPORTED"


def test_catalog_preserves_only_traceable_v1_composition_capability():
    value = {"name": "ComposableDataConfig", "description": "组合数据处理", "parameters": [], "unsupported_parameters": [],
             "available": True, "source": {"path": "src/openpi/training/config.py", "sha256": "b" * 64}, "composition_schema_version": 1}
    assert _entry(value)["composition_schema_version"] == 1
    for field, update in [("name", "OtherData"), ("available", False), ("composition_schema_version", True), ("composition_schema_version", 2)]:
        with pytest.raises(ValueError): _entry({**value, field: update})


@pytest.mark.parametrize("format", ["rlds", "other"])
def test_non_lerobot_asset_can_keep_history_but_cannot_select_composable_training(platform, format):
    node, inputs, rows, _ = setup_composed(platform)
    changed = description(); changed["format"] = format
    response = platform.client.patch(f"/api/v1/assets/{rows[0]['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                     json={"dataset_profile": changed})
    assert response.status_code == 200
    preview = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert preview.status_code == 422, preview.text


def test_profile_and_composition_share_javascript_unicode_trimming(platform):
    from vla_control.dataset_profiles import normalize_dataset_profile
    value = description()
    value["robot_type"] = "\ufeffARX5\ufeff"
    value["processing"]["robot_model"] = "\ufeffARX5\ufeff"
    value["cameras"][0]["key"] = "\ufeffobservation.images.wrist\ufeff"
    value["processing"]["cameras"][0]["key"] = "\ufeffobservation.images.wrist\ufeff"
    normalized = normalize_dataset_profile(value)
    assert normalized["robot_type"] == "ARX5"
    assert normalized["cameras"][0]["key"] == "observation.images.wrist"
    assert normalized["processing"]["cameras"][0]["key"] == "observation.images.wrist"
    for control in ("\u001c", "\u0085"):
        invalid = description(); invalid["robot_type"] = control + "ARX5" + control
        with pytest.raises(ValueError): normalize_dataset_profile(invalid)
