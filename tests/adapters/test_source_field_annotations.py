"""Source labels describe data independently of the chosen training layout."""

from copy import deepcopy

import pytest

from vla_adapters.composition_spec import (normalize_field_annotations, validate_composition_profile_input,
                                           verify_composition_metadata)
from vla_adapters.dataset_contracts import verify_registered_dataset_profile

from test_composed_processing_contract import metadata, profile


def source():
    declared, info = profile(), metadata()
    declared.pop("processing")
    declared["state_dim"] = 123  # Legacy summary is not an ordered model input.
    for key in ("observation.joints", "observation.gripper", "actions"):
        info["features"][key]["dtype"] = "float32"
    info["features"]["actions"]["names"] = ["coordinate_" + str(i) for i in range(10)]
    declared["field_annotations"] = [
        {"key": "observation.joints", "role": "state", "representation": "joint", "value_mode": "absolute",
         "component": "joint", "arm": "left", "shape": [6], "dtype": "float32"},
        {"key": "observation.gripper", "role": "state", "representation": "unknown", "value_mode": "absolute",
         "component": "gripper", "arm": "left", "shape": [1]},
        {"key": "actions", "role": "action", "representation": "eef", "value_mode": "delta",
         "component": "other", "arm": "unknown", "shape": [16, 10], "dtype": "float32",
         "names": list(info["features"]["actions"]["names"])},
    ]
    return declared, info


def test_raw_labels_allow_independent_state_action_semantics_and_training_order():
    declared, info = source()
    before = deepcopy(declared)
    assert verify_registered_dataset_profile(declared, info) == declared
    selected = deepcopy(profile()["processing"])
    selected["state_fields"].reverse()
    selected["action_mode"] = "delta"
    selected["normalization"] = "zscore"
    assert validate_composition_profile_input(declared, selected) == selected
    assert declared == before and "processing" not in declared


@pytest.mark.parametrize("field,value", [("shape", [8]), ("dtype", "float64"), ("key", "missing"),
                                         ("names", ["incorrect"]), ("role", "image")])
def test_labels_cannot_forge_original_feature_facts(field, value):
    declared, info = source()
    declared["field_annotations"][2][field] = value
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH"):
        verify_registered_dataset_profile(declared, info)


@pytest.mark.parametrize("index,field,value", [(0, "representation", "unknown"), (0, "value_mode", "unknown"),
    (2, "representation", "unknown"), (0, "role", "action")])
def test_unknown_or_wrong_role_labels_do_not_authorize_training(index, field, value):
    declared, _ = source()
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    declared["field_annotations"][index][field] = value
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_UNCONFIRMED"):
        validate_composition_profile_input(declared, selected)


@pytest.mark.parametrize("mode", ["", "absolute", "delta"])
def test_unknown_action_mode_does_not_block_an_explicit_or_unspecified_declaration(mode):
    declared, _ = source()
    declared["field_annotations"][2]["value_mode"] = "unknown"
    selected = deepcopy(profile()["processing"]); selected["action_mode"] = mode
    assert validate_composition_profile_input(declared, selected)["action_mode"] == mode


@pytest.mark.parametrize("change", ["state_representation", "action_representation", "action_mode", "camera", "arm", "kind"])
def test_repacking_cannot_implicitly_convert_source_semantics(change):
    declared, _ = source()
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    if change == "state_representation": selected["state_representation"] = "eef"
    elif change == "action_representation": declared["field_annotations"][2]["representation"] = "joint"
    elif change == "action_mode": selected["action_mode"] = "absolute"
    elif change == "camera": declared["cameras"] = []
    elif change == "arm": declared["field_annotations"][0]["arm"] = "right"
    elif change == "kind": declared["field_annotations"][0]["component"] = "position"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH"):
        validate_composition_profile_input(declared, selected)


def test_unselected_source_fields_do_not_force_the_training_concatenation():
    declared, info = source()
    declared["field_annotations"].append({"key": "observation.extra", "role": "state", "representation": "unknown",
        "value_mode": "unknown", "component": "other", "arm": "unknown", "shape": [3]})
    info["features"]["observation.extra"] = {"shape": [3], "dtype": "float32"}
    assert verify_registered_dataset_profile(declared, info)["field_annotations"][-1]["key"] == "observation.extra"
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    validate_composition_profile_input(declared, selected)


def test_generic_state_inputs_can_keep_unknown_representation_but_actions_cannot():
    declared, _ = source()
    declared["field_annotations"][0].update(component="other", representation="unknown")
    selected = deepcopy(profile()["processing"])
    selected["state_fields"][0]["kind"] = "other"
    selected["action_mode"] = "delta"
    validate_composition_profile_input(declared, selected)
    declared["field_annotations"][2]["representation"] = "unknown"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_UNCONFIRMED"):
        validate_composition_profile_input(declared, selected)


@pytest.mark.parametrize("shape,horizon,model_horizon", [([10], 1, 32), ([16, 10], 16, 16)])
def test_source_action_horizon_is_verified_without_changing_model_sampling(shape, horizon, model_horizon):
    declared, info = source()
    action = declared["field_annotations"][2]
    action.update(shape=shape, horizon=horizon)
    info["features"]["actions"]["shape"] = shape
    before = deepcopy(declared)
    assert verify_registered_dataset_profile(declared, info) == declared
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    config = {"action_horizon": model_horizon}
    verify_composition_metadata(selected, info, action_horizon=config["action_horizon"])
    assert config == {"action_horizon": model_horizon}
    assert declared == before and declared["field_annotations"][2]["horizon"] == horizon
    if len(shape) == 2:
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            verify_composition_metadata(selected, info, action_horizon=model_horizon + 1)


@pytest.mark.parametrize("value", [0, -1, True, "16", 16.5, None, 2**31])
def test_source_action_horizon_rejects_invalid_values(value):
    declared, _ = source()
    declared["field_annotations"][2]["horizon"] = value
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        normalize_field_annotations(declared["field_annotations"])


def test_state_fields_cannot_acquire_action_horizon():
    declared, _ = source()
    declared["field_annotations"][0]["horizon"] = 1
    with pytest.raises(ValueError, match="只有动作字段"):
        normalize_field_annotations(declared["field_annotations"])


@pytest.mark.parametrize("shape,horizon", [([10], 16), ([16, 10], 1), ([2, 16, 10], 16)])
def test_declared_action_horizon_must_agree_with_supported_source_shape(shape, horizon):
    declared, _ = source()
    declared["field_annotations"][2].update(shape=shape, horizon=horizon)
    with pytest.raises(ValueError, match="horizon"):
        normalize_field_annotations(declared["field_annotations"])


@pytest.mark.parametrize("shape,horizon", [([10], 16), ([16, 10], 1), ([2, 16, 10], 16)])
def test_omitting_saved_shape_cannot_bypass_real_action_horizon_verification(shape, horizon):
    declared, info = source()
    action = declared["field_annotations"][2]
    action.pop("shape")
    action["horizon"] = horizon
    info["features"]["actions"]["shape"] = shape
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH.*horizon"):
        verify_registered_dataset_profile(declared, info)


def test_old_labels_without_action_horizon_keep_their_original_contract():
    declared, info = source()
    before = deepcopy(declared)
    assert verify_registered_dataset_profile(declared, info) == before
    assert "horizon" not in declared["field_annotations"][2]
    # Higher rank source facts remain representable without guessing a time axis.
    declared["field_annotations"][2]["shape"] = [2, 16, 10]
    info["features"]["actions"]["shape"] = [2, 16, 10]
    assert "horizon" not in verify_registered_dataset_profile(declared, info)["field_annotations"][2]


@pytest.mark.parametrize("representation", ["gripper", "unknown", "eef", "joint"])
def test_gripper_state_is_independent_of_arm_representation_and_legacy_tags_remain_valid(representation):
    declared, info = source()
    declared["field_annotations"][1]["representation"] = representation
    before = deepcopy(declared)
    assert verify_registered_dataset_profile(declared, info) == declared
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    assert validate_composition_profile_input(declared, selected) == selected
    assert declared == before
    selected["state_fields"][1]["kind"] = "other"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH"):
        validate_composition_profile_input(declared, selected)


@pytest.mark.parametrize("component", ["position", "rotation", "joint", "other"])
def test_gripper_representation_cannot_label_an_unrelated_component(component):
    declared, info = source()
    declared["field_annotations"][1].update(representation="gripper", component=component)
    with pytest.raises(ValueError, match="COMPOSITION_INVALID.*夹爪"):
        normalize_field_annotations(declared["field_annotations"])
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH.*夹爪"):
        verify_registered_dataset_profile(declared, info)


@pytest.mark.parametrize("representation", ["gripper", "unknown", "eef", "joint"])
def test_gripper_only_action_is_recordable_but_cannot_supply_complete_arm_actions(representation):
    declared, info = source()
    declared["field_annotations"][2].update(
        representation=representation, component="gripper", shape=[16, 1], horizon=16, names=["gripper_width"],
    )
    declared["action_dim"] = 1
    info["features"]["actions"].update(shape=[16, 1], names=["gripper_width"])
    assert verify_registered_dataset_profile(declared, info) == declared
    selected = deepcopy(profile()["processing"])
    selected["action_mode"] = "delta"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH.*不能仅用夹爪开度"):
        validate_composition_profile_input(declared, selected)


def test_gripper_label_does_not_relax_source_facts_or_action_horizon():
    declared, info = source()
    declared["field_annotations"][2].update(representation="gripper", component="gripper", horizon=1)
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH.*horizon"):
        verify_registered_dataset_profile(declared, info)
    declared["field_annotations"][2]["horizon"] = 16
    declared["field_annotations"][2]["shape"] = [16, 1]
    with pytest.raises(ValueError, match="DATASET_PROFILE_MISMATCH.*形状"):
        verify_registered_dataset_profile(declared, info)


@pytest.mark.parametrize("field", ["state_representation", "action_representation"])
def test_gripper_source_label_does_not_widen_composition_arm_representation(field):
    selected = deepcopy(profile()["processing"])
    selected[field] = "gripper"
    declared, _ = source()
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        validate_composition_profile_input(declared, selected)
