"""Statistics context is numeric-only without weakening training contracts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vla_adapters.automatic_config import automatic_data_layout, normalize_auto_config
from vla_adapters.composition_spec import (
    composition_digest,
    normalize_composition,
    validate_composition_profile_input,
    verify_composition_metadata,
)
from vla_adapters.data_processors import normalize_processor
from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_adapters.norm_datasets import normalize_norm_datasets


@pytest.fixture
def numeric_composition():
    corpus = json.loads((Path(__file__).parents[1] / "fixtures/composed-processing-v1.json").read_text())
    spec = deepcopy(next(item for item in corpus["valid"] if len(item["action_layout"]) == 1
                         and item["action_layout"][0]["kind"] == "shared"))
    spec["normalization"] = "zscore"
    spec["state_fields"][0]["normalize"] = False
    spec["action_layout"][0]["normalize"] = False
    return spec


@pytest.fixture
def numeric_source(numeric_composition):
    spec = numeric_composition
    state_key = spec["state_fields"][0]["key"]
    state_width = spec["state_fields"][0]["width"]
    action_width = spec["action_layout"][0]["width"]
    annotations = [
        {"key": state_key, "role": "state", "representation": "composite", "component": "other",
         "arm": "shared", "value_mode": "absolute", "shape": [state_width], "dtype": "float32"},
        {"key": spec["action_key"], "role": "action", "representation": spec["action_representation"],
         "component": "other", "arm": "shared", "value_mode": "absolute",
         "shape": [16, action_width], "horizon": 16, "dtype": "float32"},
    ]
    profile = {"schema_version": 1, "format": "lerobot", "robot_type": spec["robot_model"],
               "arm_mode": spec["arm_mode"], "action_representation": spec["action_representation"],
               "action_mode": "absolute", "rotation_representation": spec["rotation_representation"],
               "task_description": "Independent numeric statistics", "cameras": [None, {"old": "camera"}],
               "action_dim": action_width, "state_dim": state_width, "field_annotations": annotations,
               "feature_keys": [state_key, spec["action_key"], "missing.video"]}
    info = {"features": {state_key: {"dtype": "float32", "shape": [state_width]},
                         spec["action_key"]: {"dtype": "float32", "shape": [16, action_width]},
                         "unreadable.video": {"dtype": "video", "shape": "invalid visual metadata"}}}
    return profile, info


@pytest.mark.parametrize("cameras", [[], [None], [{"key": "missing.video", "layout": "invalid"}], "old camera"])
def test_statistics_canonicalizes_old_or_empty_cameras_without_mutating_masks(numeric_composition, cameras):
    spec = deepcopy(numeric_composition)
    spec["cameras"] = cameras
    before = deepcopy(spec)
    clean = normalize_composition(spec, norm_stats=True)
    assert clean["cameras"] == []
    assert clean["normalization"] == "zscore"
    assert clean["state_fields"] == spec["state_fields"]
    assert clean["action_layout"] == [{"arm": "shared", "kind": "shared", "width": 14, "normalize": False}]
    assert spec == before
    with pytest.raises(ValueError):
        normalize_composition(spec)


def test_statistics_digest_ignores_visuals_but_retains_saved_numeric_configuration(numeric_composition):
    empty = {**numeric_composition, "cameras": []}
    invalid = {**numeric_composition, "cameras": [None]}
    assert composition_digest(empty, norm_stats=True) == composition_digest(invalid, norm_stats=True)
    changed = deepcopy(empty)
    changed["state_fields"][0]["normalize"] = True
    assert composition_digest(changed, norm_stats=True) != composition_digest(empty, norm_stats=True)
    assert normalize_composition(numeric_composition)["cameras"]
    with pytest.raises(ValueError):
        composition_digest(empty)


def test_processor_and_norm_rows_propagate_statistics_context(numeric_composition):
    choice = {"name": "ComposableDataConfig", "parameters": {"composition": {**numeric_composition, "cameras": []}}}
    normalized = normalize_processor(choice, norm_stats=True)
    assert normalized["parameters"]["composition"]["cameras"] == []
    rows = normalize_norm_datasets([{"dataset_asset_id": "source", "processor": choice}])
    assert rows == [{"dataset_asset_id": "source", "processor": normalized}]
    with pytest.raises(ValueError):
        normalize_processor(choice)


def test_statistics_profile_and_metadata_do_not_require_visual_or_task_sources(numeric_composition, numeric_source):
    profile, info = numeric_source
    spec = {**numeric_composition, "cameras": []}
    assert validate_composition_profile_input(profile, spec, norm_stats=True)["cameras"] == []
    assert verify_composition_metadata(spec, info, action_horizon=16, norm_stats=True)["cameras"] == []
    assert verify_registered_dataset_profile(profile, info, selected_composition=spec, norm_stats=True)
    with pytest.raises(ValueError):
        verify_composition_metadata(spec, info, action_horizon=16)
    with pytest.raises(ValueError):
        validate_composition_profile_input(profile, spec)


@pytest.mark.parametrize("target,shape", [("state", [13]), ("actions", [16, 13]), ("actions", [15, 14])])
def test_statistics_keeps_real_numeric_width_and_horizon_validation(numeric_composition, numeric_source, target, shape):
    _, info = numeric_source
    key = numeric_composition["state_fields"][0]["key"] if target == "state" else numeric_composition["action_key"]
    changed = deepcopy(info)
    changed["features"][key]["shape"] = shape
    with pytest.raises(ValueError):
        verify_composition_metadata({**numeric_composition, "cameras": []}, changed,
                                    action_horizon=16, norm_stats=True)


def test_statistics_keeps_source_mode_and_normalization_field_validation(numeric_composition, numeric_source):
    profile, _ = numeric_source
    spec = {**numeric_composition, "cameras": []}
    profile = deepcopy(profile)
    profile["field_annotations"][1]["value_mode"] = "delta"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH"):
        validate_composition_profile_input(profile, spec, norm_stats=True)
    malformed = deepcopy(spec)
    malformed["state_fields"][0]["normalize"] = "false"
    with pytest.raises(ValueError):
        normalize_composition(malformed, norm_stats=True)
    malformed = deepcopy(spec)
    malformed["action_layout"][0]["arm"] = "left"
    with pytest.raises(ValueError):
        normalize_composition(malformed, norm_stats=True)


def test_automatic_statistics_layout_has_no_image_or_depth_inputs(numeric_composition):
    row = {"processor": {"name": "ComposableDataConfig", "parameters": {
        "composition": {**numeric_composition, "cameras": [None]}}}}
    vision, padding = automatic_data_layout(normalize_auto_config({}), [row], norm_stats=True)
    assert vision == {"image_keys": (), "depth_keys": ()}
    assert padding == 14
    with pytest.raises(ValueError):
        automatic_data_layout(normalize_auto_config({}), [row])
