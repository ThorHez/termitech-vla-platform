from copy import deepcopy

import pytest

from vla_adapters.automatic_config import automatic_data_layout, normalize_auto_config
from vla_adapters.composition_spec import (normalize_composition, validate_composition_profile_input,
                                           verify_composition_metadata)
from vla_protocol import canonical_digest
from vla_protocol.norm_plan import _validate_composition

from test_composed_processing_contract import core, profile  # noqa: F401
from test_composite_source_fields import packed_source


def shared_spec(representation="eef", arm_mode="bimanual", normalize=True):
    spec = deepcopy(profile()["processing"])
    spec.update(action_representation=representation, arm_mode=arm_mode)
    spec["action_layout"] = [{"arm": "shared", "kind": "shared", "width": 14, "normalize": normalize}]
    return spec


@pytest.mark.parametrize("representation", ["eef", "joint"])
@pytest.mark.parametrize("arm_mode", ["single", "bimanual"])
@pytest.mark.parametrize("normalize", [True, False])
def test_shared_action_is_explicit_full_vector_in_both_validators(core, representation, arm_mode, normalize):
    spec = shared_spec(representation, arm_mode, normalize)
    # An opaque vector makes no claims about internal position/rotation blocks.
    spec["rotation_representation"] = "rotation_6d"
    original = deepcopy(spec)
    assert normalize_composition(spec) == core.normalize_composition(spec)
    assert normalize_composition(spec)["action_layout"] == original["action_layout"]
    assert core.normalization_masks(spec)["actions"] == (normalize,) * 14
    assert spec == original


@pytest.mark.parametrize("case", ["wrong_owner", "wrong_kind", "duplicate", "missing_arm", "incomplete_rotation", "mixed_joint_rotation"])
def test_shared_cannot_hide_invalid_ownership_or_incomplete_regular_blocks(core, case):
    spec = shared_spec()
    if case == "wrong_owner":
        spec["action_layout"][0]["arm"] = "left"
    elif case == "wrong_kind":
        spec["action_layout"][0]["kind"] = "joint"
    elif case == "duplicate":
        spec["action_layout"].append(deepcopy(spec["action_layout"][0]))
    else:
        regular = deepcopy(profile()["processing"]["action_layout"])
        spec["action_layout"].extend(regular)
        if case == "incomplete_rotation":
            spec["arm_mode"] = "single"
            spec["action_layout"] = [row for row in spec["action_layout"] if row["kind"] != "rotation"]
        elif case == "mixed_joint_rotation":
            spec.update(arm_mode="single", action_representation="joint", rotation_representation="axis_angle")
            spec["action_layout"] = [spec["action_layout"][0], {"arm": "left", "kind": "joint", "width": 6, "normalize": True}]
    for validate in [normalize_composition, core.normalize_composition]:
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            validate(spec)


def test_mixed_shared_and_complete_regular_layout_retains_every_width_and_mask(core):
    spec = deepcopy(profile()["processing"])
    spec["action_layout"].insert(1, {"arm": "shared", "kind": "shared", "width": 4, "normalize": False})
    assert normalize_composition(spec) == core.normalize_composition(spec)
    assert len(core.normalization_masks(spec)["actions"]) == 14
    assert core.normalization_masks(spec)["actions"][3:7] == (False,) * 4


def test_shared_does_not_bypass_source_width_horizon_or_model_dimension_checks():
    declared, info, spec = packed_source()
    spec["action_layout"] = [{"arm": "shared", "kind": "shared", "width": 14, "normalize": True}]
    spec["action_mode"] = "delta"
    action = next(row for row in declared["field_annotations"] if row["role"] == "action")
    action.update(shape=[16, 14], arm="shared")
    action.pop("names", None)
    info["features"]["actions"]["shape"] = [16, 14]
    assert validate_composition_profile_input(declared, spec) == normalize_composition(spec)
    assert verify_composition_metadata(spec, info, action_horizon=16) == normalize_composition(spec)
    changed = deepcopy(spec); changed["action_layout"][0]["width"] = 13
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH"):
        validate_composition_profile_input(declared, changed)
    with pytest.raises(ValueError, match="COMPOSITION"):
        verify_composition_metadata(spec, info, action_horizon=8)
    row = {"processor": {"name": "ComposableDataConfig", "parameters": {"composition": spec}}}
    with pytest.raises(ValueError, match="AUTO_CONFIG_DIMENSION_MISMATCH"):
        automatic_data_layout(normalize_auto_config({"action_dim": 13}), [row])


def test_norm_protocol_binds_shared_raw_vector_and_model_evidence_without_gripper_inference():
    spec = normalize_composition(shared_spec(), norm_stats=True)
    processor = {"name": "ComposableDataConfig", "parameters": {"composition": spec}}
    model = normalize_auto_config({"action_horizon": 16})
    evidence = {"schema_version": 1, "spec": spec, "digest": canonical_digest(spec),
        "training_stage": "sft", "model_type": "Pi0Config", "pi05": True,
        "action_dim": 32, "action_horizon": 16, "raw_action_dim": 14,
        "raw_state_dim": sum(field["width"] for field in spec["state_fields"]),
        "state_history": spec["state_history"], "normalization": spec["normalization"],
        "action_sequence_keys": [], "action_metadata_sha256": "a" * 64,
        "action_loss_mask": [1.] * 14 + [0.] * 18,
        "image_keys": [camera["slot"] for camera in spec["cameras"]], "depth_keys": [],
        "source": {"path": "src/openpi/training/composed_processing.py", "sha256": "b" * 64}}
    dataset = {"composition": evidence, "actions_shape": [16, 14], "metadata_sha256": "a" * 64}
    _validate_composition(dataset, processor, model)
    wrong = deepcopy(dataset); wrong["composition"]["raw_action_dim"] = 7
    with pytest.raises(ValueError, match="numeric contract"):
        _validate_composition(wrong, processor, model)
