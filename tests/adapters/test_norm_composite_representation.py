from copy import deepcopy

import pytest

from vla_adapters.composition_spec import (normalize_composition, composition_choice, composition_digest,
    validate_composition_profile_input, verify_composition_metadata)
from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_adapters.norm_datasets import normalize_norm_datasets
from vla_adapters import norm_entry
from vla_adapters.training_plan import dataset_contract

from test_composed_processing_contract import core, profile  # noqa: F401
from test_norm_automatic import automatic_norm, automatic, composed_module, processor_module, training  # noqa: F401


def opaque_source():
    spec = deepcopy(profile()["processing"])
    spec.update(state_representation="composite", action_representation="composite", arm_mode="bimanual",
                action_mode="absolute", cameras=[])
    spec["state_fields"] = [{"key": "observation.state", "width": 14, "arm": "shared", "kind": "other", "normalize": False}]
    spec["action_layout"] = [{"arm": "shared", "kind": "shared", "width": 14, "normalize": False}]
    declared = {"schema_version": 1, "format": "lerobot", "robot_type": spec["robot_model"],
        "arm_mode": "bimanual", "action_representation": "unknown", "cameras": [],
        "task_description": "Packed state and action", "state_dim": 14, "action_dim": 14,
        "field_annotations": [{"key": key, "role": role, "representation": "composite", "component": "other",
            "arm": "shared", "value_mode": "absolute", "shape": shape, "dtype": "float32"}
            for key, role, shape in [("observation.state", "state", [14]), ("actions", "action", [50, 14])]]}
    info = {"features": {"observation.state": {"dtype": "float32", "shape": [14]},
                         "actions": {"dtype": "float32", "shape": [50, 14]}}}
    return spec, declared, info


def test_numeric_statistics_preserve_explicit_composite_source_labels_without_guessing_joint_or_eef(core):
    spec, declared, info = opaque_source()
    original = deepcopy((spec, declared, info))
    clean = normalize_composition(spec, norm_stats=True)
    assert clean == core.normalize_composition(spec, norm_stats=True)
    assert clean["state_representation"] == clean["action_representation"] == "composite"
    assert validate_composition_profile_input(declared, clean, norm_stats=True) == clean
    assert verify_composition_metadata(clean, info, action_horizon=50, norm_stats=True) == clean
    assert verify_registered_dataset_profile(declared, info, selected_composition=clean, norm_stats=True) == declared
    choice = composition_choice({"processing": spec}, norm_stats=True)
    assert normalize_norm_datasets([{"dataset_asset_id": "data", "processor": choice}])[0]["processor"] == choice
    assert composition_digest(spec, norm_stats=True) == composition_digest(clean, norm_stats=True)
    assert (spec, declared, info) == original


@pytest.mark.parametrize("field", ["state_representation", "action_representation"])
def test_composite_representation_never_relaxes_training_normalizers(core, field):
    spec = deepcopy(profile()["processing"])
    spec[field] = "composite"
    for validate in [normalize_composition, core.normalize_composition]:
        with pytest.raises(ValueError, match=field):
            validate(spec)


@pytest.mark.parametrize("layout", [
    [{"arm": "left", "kind": "joint", "width": 14, "normalize": False}],
    [{"arm": "shared", "kind": "shared", "width": 7, "normalize": False},
     {"arm": "left", "kind": "joint", "width": 7, "normalize": False}],
])
def test_composite_action_requires_one_complete_shared_vector_in_both_runtimes(core, layout):
    spec, _, _ = opaque_source(); spec["action_layout"] = layout
    for validate in [normalize_composition, core.normalize_composition]:
        with pytest.raises(ValueError, match="完整共享动作块"):
            validate(spec, norm_stats=True)


@pytest.mark.parametrize("change", ["state_width", "state_kind", "state_arm", "action_width", "action_mode", "action_source", "state_source", "horizon"])
def test_composite_statistics_keep_source_identity_width_mode_and_horizon_checks(change):
    spec, declared, info = opaque_source()
    if change == "state_width": spec["state_fields"][0]["width"] = 13
    elif change == "state_kind": spec["state_fields"][0]["kind"] = "joint"
    elif change == "state_arm": spec["state_fields"][0]["arm"] = "left"
    elif change == "action_width": spec["action_layout"][0]["width"] = 13
    elif change == "action_mode": spec["action_mode"] = "delta"
    elif change == "action_source": declared["field_annotations"][1].update(representation="joint", component="joint")
    elif change == "state_source": declared["field_annotations"][0].update(representation="joint", component="joint")
    else: info["features"]["actions"]["shape"] = [49, 14]
    with pytest.raises(ValueError, match="COMPOSITION"):
        validate_composition_profile_input(declared, spec, norm_stats=True)
        verify_composition_metadata(spec, info, action_horizon=50, norm_stats=True)


def test_automatic_norm_config_and_dataset_contract_keep_real_composite_labels(automatic_norm):
    import json
    from pathlib import Path
    spec, _ = automatic_norm
    composition, declared, info = opaque_source()
    selected = spec["norm_datasets"][0]
    selected["processor"]["parameters"]["composition"] = composition
    source = spec["inputs"]["dataset_0"]
    source["dataset_profile"] = declared
    (Path(source["path"]) / "meta/info.json").write_text(json.dumps({**info, "total_frames": 20, "total_episodes": 2}))
    config = norm_entry.prepare_config(spec)
    evidence = dataset_contract(config.data, spec, declared, model_config=config.model, binding=selected, norm_stats=True)
    frozen = evidence["composition"]
    assert frozen["spec"]["state_representation"] == frozen["spec"]["action_representation"] == "composite"
    assert frozen["raw_state_dim"] == frozen["raw_action_dim"] == 14
    assert frozen["action_horizon"] == 50 and frozen["action_dim"] == 32
    assert frozen["image_keys"] == frozen["depth_keys"] == []
    assert evidence["registered_profile"]["action_representation"] == "unknown"
    with pytest.raises(ValueError, match="state_representation"):
        dataset_contract(config.data, spec, declared, model_config=config.model, binding=selected)
