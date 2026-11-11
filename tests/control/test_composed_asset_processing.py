"""A selected training composition stays frozen independently of asset history."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from vla_protocol import canonical_digest
from vla_adapters.composition_spec import composition_choice, composition_digest
from vla_control.data_processors import _entry

from .conftest import ADMIN, automatic_preflight, snapshot, inspected_profile_payload
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview
from .test_automatic_preflight import acknowledge, submit


def description(*, action_layout=None, arm_mode="single", action_representation="eef"):
    spec = json.loads((Path(__file__).resolve().parents[1] / "fixtures/composed-processing-v1.json").read_text())["valid"][0]
    spec["arm_mode"] = arm_mode
    spec["action_representation"] = action_representation
    if action_representation == "joint":
        spec["rotation_representation"] = "none"
    if action_layout is not None:
        spec["action_layout"] = deepcopy(action_layout)
    return {"schema_version": 1, "format": "lerobot", "robot_type": "ARX5", "arm_mode": arm_mode,
            "action_representation": action_representation,
            "cameras": [{"name": "wrist", "key": spec["cameras"][0]["key"], "kind": "rgb"}],
            "task_description": "Pick up cup", "action_dim": sum(row["width"] for row in spec["action_layout"]),
            "state_dim": 7, "processing": spec}


def setup_composed(platform, *, source_only=False, **composition_options):
    node, inputs, rows = configured(platform)
    rows = rows[:1]
    declared = description(**composition_options)
    selected = composition_choice(declared)
    if source_only:
        declared.pop("processing")
        declared["field_annotations"] = [
            {"key": "observation.joints", "role": "state", "representation": "joint", "value_mode": "absolute",
             "component": "joint", "arm": "left", "shape": [6]},
            {"key": "observation.gripper", "role": "state", "representation": "unknown", "value_mode": "absolute",
             "component": "gripper", "arm": "left", "shape": [1]},
            {"key": "actions", "role": "action", "representation": "composite", "value_mode": "absolute",
             "component": "other", "arm": "shared", "shape": [16, declared["action_dim"]]},
        ]
    response = platform.client.patch(f"/api/v1/assets/{rows[0]['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                     json=inspected_profile_payload(platform, rows[0]['dataset_asset_id'], declared))
    assert response.status_code == 200, response.text
    profile = response.json()["metadata"]["dataset_profile"]
    rows[0].pop("recipe_index")
    rows[0]["processor"] = selected
    return node, inputs, rows, profile


def pending(platform, *, source_only=False, customized=False, augmentation=None, automatic=False,
            state_history=1, **composition_options):
    node, inputs, rows, profile = setup_composed(platform, source_only=source_only, **composition_options)
    selected = rows[0]["processor"]["parameters"]["composition"]
    selected["state_history"] = state_history
    if augmentation is not None:
        selected["cameras"][0]["augmentation"] = augmentation
    if customized:
        selected["state_fields"].reverse()
        selected["normalization"] = "zscore"
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows,
                               **({"auto_config": {"model_type": "pi05", "action_horizon": 16}} if automatic else {}))
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    rows = command["payload"]["dataset_bindings"]
    selected = rows[0]["processor"]["parameters"]["composition"]
    evidence = automatic_preflight(command["payload"])
    data = evidence["plan"]["datasets"][0]
    data["registered_profile"] = deepcopy(profile)
    raw_action = sum(field["width"] for field in selected["action_layout"])
    data["composition"] = {"schema_version": 1, "spec": deepcopy(selected),
        "digest": composition_digest(selected), "training_stage": "sft", "model_type": "Pi0Config",
        "action_horizon": 16, "action_dim": 32,
        "pi05": True, "raw_state_dim": 7, "raw_action_dim": raw_action, "state_history": state_history,
        "normalization": selected["normalization"],
        "image_keys": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], "depth_keys": [],
        "action_metadata_sha256": data["metadata_sha256"], "action_sequence_keys": [],
        "action_loss_mask": [1.] * raw_action + [0.] * (32 - raw_action),
        "source": {"path": "src/openpi/training/composed_processing.py", "sha256": "c" * 64}}
    data["actions_shape"] = [16, raw_action]
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    code = snapshot(recipe=command["payload"]["recipe"], dataset_bindings=rows, automatic_preflight=evidence,
                    **({"auto_config": command["payload"]["auto_config"]} if automatic else {}))
    return node, identifier, command, code, profile


@pytest.mark.parametrize(("automatic", "state_history"), [(False, 1), (True, 1), (True, 2)])
def test_composition_is_frozen_with_asset_even_after_edits_and_reaches_start(platform, automatic, state_history):
    node, identifier, command, code, profile = pending(platform, automatic=automatic, state_history=state_history)
    assert command["payload"]["inputs"]["dataset_0"]["dataset_profile"] == profile
    assert profile["processing"]["model_transform"] == "standard"
    processor = command["payload"]["dataset_bindings"][0]["processor"]
    assert processor["parameters"]["composition"]["model_transform"] == ("flat_state" if automatic else "standard")
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
    assert start["payload"]["dataset_bindings"][0]["processor"] == processor


@pytest.mark.parametrize("tampered", [False, True])
def test_training_freezes_the_selected_shared_action_mask(platform, tampered):
    mask = [True] * 9 + [False]
    layout = [{"arm": "shared", "kind": "shared", "width": 10, "normalize": True, "normalize_mask": mask}]
    node, identifier, command, code, profile = pending(platform, source_only=True, action_layout=layout)
    processor = command["payload"]["dataset_bindings"][0]["processor"]
    assert processor["parameters"]["composition"]["action_layout"] == layout
    assert "processing" not in profile
    if tampered:
        evidence = code["automatic_preflight"]
        composition = evidence["plan"]["datasets"][0]["composition"]
        composition["spec"]["action_layout"][0]["normalize_mask"] = list(reversed(mask))
        composition["digest"] = composition_digest(composition["spec"])
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    if tampered:
        assert checked["status"] == "FAILED", checked
        assert checked["error"]["code"] == "AGENT_COMPOSITION_UNSUPPORTED"
        return
    assert checked["status"] == "READY", checked
    run = submit(platform, identifier)
    assert run.status_code == 201, run.text
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["dataset_bindings"][0]["processor"] == processor


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


@pytest.mark.parametrize("route", [None, 1, 0])
def test_explicit_default_is_frozen_and_requires_agent_augmentation_evidence(platform, route):
    node, identifier, command, code, _ = pending(platform, source_only=True, augmentation="default")
    selected = command["payload"]["dataset_bindings"][0]["processor"]["parameters"]["composition"]["cameras"][0]
    assert selected["augmentation"] == "default"
    evidence = code["automatic_preflight"]
    if route is not None:
        evidence["plan"]["datasets"][0]["image_processing"] = {"schema_version": 1,
            "augmentation_codes": {selected["slot"]: route}, "uniform_augmentation_code": None}
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    if route != 0:
        assert checked["status"] == "FAILED", checked
        assert checked["error"]["code"] == "AGENT_IMAGE_AUGMENTATION_MISMATCH"
        return
    assert checked["status"] == "READY", checked
    run = submit(platform, identifier)
    assert run.status_code == 201, run.text
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    actual = start["payload"]["dataset_bindings"][0]["processor"]["parameters"]["composition"]["cameras"][0]
    assert actual == selected


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


def test_training_still_requires_visual_configuration_when_norm_stats_does_not(platform):
    node, inputs, rows, _ = setup_composed(platform)
    rows[0]["processor"]["parameters"]["composition"]["cameras"] = []
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


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
