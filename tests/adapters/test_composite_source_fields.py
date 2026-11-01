"""Packed source vectors retain facts without pretending to describe one component."""

from copy import deepcopy

import pytest

from vla_adapters.composition_spec import (normalize_field_annotations, validate_composition_profile_input,
                                         verify_composition_metadata)
from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_control.dataset_profiles import normalize_dataset_profile

from test_composed_processing_contract import profile
from test_source_field_annotations import source


def packed_source():
    declared, info = source()
    declared.update(state_dim=7, action_representation="unknown")
    declared["field_annotations"] = [
        {"key": "observation.packed", "role": "state", "representation": "composite",
         "value_mode": "absolute", "component": "other", "arm": "left", "shape": [7]},
        {**declared["field_annotations"][2], "representation": "composite", "horizon": 16},
    ]
    info["features"]["observation.packed"] = {"shape": [7], "dtype": "float32"}
    selected = deepcopy(profile()["processing"])
    selected["state_fields"] = [{"key": "observation.packed", "width": 7, "arm": "left",
                                  "kind": "other", "normalize": True}]
    selected["action_mode"] = "delta"
    return declared, info, selected


def test_composite_round_trip_and_explicit_training_plan_preserve_source_facts():
    declared, info, selected = packed_source()
    before = deepcopy((declared, info, selected))
    clean = normalize_dataset_profile(declared)
    assert clean["action_representation"] == "unknown"
    assert all(row["representation"] == "composite" for row in clean["field_annotations"])
    assert normalize_field_annotations(declared["field_annotations"]) == declared["field_annotations"]
    assert verify_registered_dataset_profile(declared, info) == declared
    assert validate_composition_profile_input(declared, selected) == selected
    assert verify_composition_metadata(selected, info, action_horizon=16) == selected
    assert (declared, info, selected) == before


@pytest.mark.parametrize("role", ["state", "action"])
@pytest.mark.parametrize("component", ["position", "rotation", "joint", "gripper"])
def test_composite_requires_generic_component_in_both_registration_and_adapter(role, component):
    declared, _, _ = packed_source()
    row = next(row for row in declared["field_annotations"] if row["role"] == role)
    row["component"] = component
    with pytest.raises(ValueError, match="组合向量"):
        normalize_dataset_profile(declared)
    with pytest.raises(ValueError, match="COMPOSITION_INVALID.*组合向量"):
        normalize_field_annotations(declared["field_annotations"])


@pytest.mark.parametrize("kind", ["position", "rotation", "joint", "gripper"])
def test_packed_state_cannot_be_relabelled_as_a_single_component_by_training(kind):
    declared, _, selected = packed_source()
    selected["state_fields"][0]["kind"] = kind
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*整字段"):
        validate_composition_profile_input(declared, selected)


def test_packed_state_cannot_be_silently_truncated_or_assigned_to_another_arm():
    declared, _, selected = packed_source()
    selected["state_fields"][0]["width"] = 6
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*完整宽度"):
        validate_composition_profile_input(declared, selected)
    selected["state_fields"][0].update(width=7, arm="shared")
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*状态拼接"):
        validate_composition_profile_input(declared, selected)


def test_state_value_mode_must_still_be_confirmed():
    declared, _, selected = packed_source()
    declared["field_annotations"][0]["value_mode"] = "unknown"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_UNCONFIRMED"):
        validate_composition_profile_input(declared, selected)


def test_composite_action_allows_an_explicit_joint_layout_without_changing_values_or_width():
    declared, info, selected = packed_source()
    selected.update(action_representation="joint", rotation_representation="none")
    selected["action_layout"] = [
        {"arm": "left", "kind": "joint", "width": 9, "normalize": True},
        {"arm": "left", "kind": "gripper", "width": 1, "normalize": False},
    ]
    original = deepcopy((declared, info))
    assert validate_composition_profile_input(declared, selected) == selected
    assert verify_composition_metadata(selected, info, action_horizon=16) == selected
    assert (declared, info) == original


def test_composite_action_width_and_known_encoding_are_not_inferred_or_overridden():
    declared, _, selected = packed_source()
    selected["action_layout"][-1]["width"] = 2
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*原始宽度"):
        validate_composition_profile_input(declared, selected)
    selected["action_layout"][-1]["width"] = 1
    selected["action_mode"] = "absolute"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*绝对/增量"):
        validate_composition_profile_input(declared, selected)


def test_packed_label_does_not_skip_real_metadata_or_horizon_validation():
    declared, info, selected = packed_source()
    info["features"]["actions"]["shape"] = [16, 9]
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH"):
        verify_registered_dataset_profile(declared, info)
    with pytest.raises(ValueError, match="COMPOSITION_INVALID.*动作字段"):
        verify_composition_metadata(selected, info, action_horizon=16)
    info["features"]["actions"]["shape"] = [16, 10]
    with pytest.raises(ValueError, match="COMPOSITION_INVALID.*动作字段"):
        verify_composition_metadata(selected, info, action_horizon=50)


@pytest.mark.parametrize("field", ["state_representation", "action_representation"])
def test_composite_is_a_source_tag_not_a_new_training_target_representation(field):
    declared, _, selected = packed_source()
    selected[field] = "composite"
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        validate_composition_profile_input(declared, selected)
