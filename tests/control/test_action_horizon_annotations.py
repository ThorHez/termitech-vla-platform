"""Source action horizon is a field fact, not a model prediction setting."""
from copy import deepcopy

import pytest

from vla_control.dataset_profiles import DatasetFieldAnnotation, normalize_dataset_profile


def annotation(**updates):
    return {"key": "actions", "role": "action", "representation": "unknown",
            "value_mode": "unknown", "component": "other", "arm": "unknown", **updates}


@pytest.mark.parametrize("shape,horizon", [([10], 1), ([16, 10], 16), ([1, 7], 1), ([2**31 - 1, 1], 2**31 - 1)])
def test_valid_source_horizon_matches_the_original_action_shape(shape, horizon):
    result = DatasetFieldAnnotation.model_validate(annotation(shape=shape, horizon=horizon))
    assert result.horizon == horizon
    assert result.shape == shape


@pytest.mark.parametrize("updates", [
    {"horizon": 0}, {"horizon": -1}, {"horizon": 1.5}, {"horizon": True},
    {"horizon": "16"}, {"horizon": None}, {"horizon": 2**31},
    {"role": "state", "horizon": 1}, {"role": "state", "horizon": None},
    {"shape": [10], "horizon": 16}, {"shape": [16, 10], "horizon": 10},
    {"shape": [2, 16, 10], "horizon": 16},
])
def test_invalid_or_non_action_horizon_is_rejected(updates):
    with pytest.raises(ValueError):
        DatasetFieldAnnotation.model_validate(annotation(**updates))


def test_legacy_annotations_omit_horizon_and_unknown_shape_can_be_confirmed_later():
    for row in (annotation(shape=[7]), annotation(shape=[2, 16, 7]),
                annotation(key="state", role="state", shape=[2, 7])):
        clean = DatasetFieldAnnotation.model_validate(row).model_dump(mode="json", exclude_none=True)
        assert "horizon" not in clean
    assert DatasetFieldAnnotation.model_validate(annotation(horizon=32)).horizon == 32


def test_registration_roundtrip_preserves_source_horizon_without_adding_model_configuration():
    source = {"schema_version": 1, "format": "lerobot", "robot_type": "umi", "arm_mode": "single",
              "action_representation": "unknown", "cameras": [{"name": "wrist", "key": "image", "kind": "rgb"}],
              "task_description": "pick", "field_annotations": [annotation(shape=[16, 10], horizon=16)]}
    before = deepcopy(source)
    result = normalize_dataset_profile(source)
    assert result["field_annotations"][0]["horizon"] == 16
    assert "action_horizon" not in result
    assert "processing" not in result
    assert source == before
