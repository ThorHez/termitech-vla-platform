"""Registration validates source labels without requiring a processing plan."""
from copy import deepcopy

import pytest

from vla_control.dataset_profiles import normalize_dataset_profile


def profile():
    return {"schema_version": 1, "format": "lerobot", "robot_type": "umi", "arm_mode": "single",
            "action_representation": "unknown", "cameras": [{"name": "wrist", "key": "image", "kind": "rgb"}],
            "task_description": "pick", "feature_keys": ["state", "actions", "image"],
            "field_annotations": [
                {"key": "state", "role": "state", "representation": "eef", "value_mode": "absolute",
                 "component": "other", "arm": "unknown", "shape": [6], "dtype": "float32", "names": []},
                {"key": "actions", "role": "action", "representation": "joint", "value_mode": "delta",
                 "component": "joint", "arm": "left", "shape": [7]}]}


def test_registration_accepts_independent_labels_and_unknowns_without_a_processing_plan():
    result = normalize_dataset_profile(profile())
    assert result["action_representation"] == "unknown" and "processing" not in result
    assert result["field_annotations"][0]["representation"] == "eef"
    assert result["field_annotations"][1]["representation"] == "joint"
    unknown = profile();unknown["field_annotations"][0]["value_mode"] = "unknown"
    assert normalize_dataset_profile(unknown)["field_annotations"][0]["value_mode"] == "unknown"


@pytest.mark.parametrize("role", ["state", "action"])
def test_standalone_gripper_representation_roundtrips_without_changing_global_arm_representation(role):
    candidate = profile()
    candidate["field_annotations"][0].update(
        role=role, representation="gripper", component="gripper", value_mode="unknown", shape=[1],
    )
    if role == "action":
        candidate["field_annotations"][0]["horizon"] = 1
    clean = normalize_dataset_profile(candidate)
    assert clean["field_annotations"][0]["representation"] == "gripper"
    assert clean["field_annotations"][0]["value_mode"] == "unknown"
    assert clean["action_representation"] == "unknown"


@pytest.mark.parametrize("component", ["position", "rotation", "joint", "other"])
def test_gripper_representation_rejects_non_gripper_components(component):
    candidate = profile()
    candidate["field_annotations"][0].update(representation="gripper", component=component)
    with pytest.raises(ValueError, match="夹爪"):
        normalize_dataset_profile(candidate)


@pytest.mark.parametrize("representation", ["eef", "joint", "unknown"])
def test_legacy_gripper_component_labels_remain_accepted(representation):
    candidate = profile()
    candidate["field_annotations"][0].update(representation=representation, component="gripper")
    assert normalize_dataset_profile(candidate)["field_annotations"][0]["representation"] == representation


@pytest.mark.parametrize("change", [
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(key="missing"),
    lambda rows: rows[0].update(value_mode="relative"),
    lambda rows: rows[0].update(normalize=True),
    lambda rows: rows[0].update(shape=[]),
    lambda rows: rows[0].update(shape=[True]),
    lambda rows: rows[0].update(shape=[0]),
    lambda rows: rows[0].update(names=["bad\nname"]),
])
def test_registration_rejects_invalid_or_executable_field_annotations(change):
    candidate = profile();change(candidate["field_annotations"])
    with pytest.raises(ValueError):
        normalize_dataset_profile(candidate)
