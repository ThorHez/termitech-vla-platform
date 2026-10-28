"""Source semantics are labels, independent of a training processing layout."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from vla_agent.dataset_inspection import inspect_dataset
from vla_protocol.dataset_inspection import normalize_dataset_inspection


def inspect(tmp_path, features, **metadata):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"features": features, **metadata}))
    return inspect_dataset({"roots": [str(tmp_path)]}, {
        "path": str(root), "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()})


def feature(width=3, **metadata):
    return {"dtype": "float32", "shape": [width], "names": [], **metadata}


def test_state_and_action_have_independent_source_representation_and_modes(tmp_path):
    result = inspect(tmp_path, {
        "observation.robot0_eef_pos": feature(value_mode="absolute"),
        "actions": feature(6, representation="joint", value_mode="delta"),
    })
    rows = {row["key"]: row for row in result["suggestions"]["field_annotations"]}
    assert rows["observation.robot0_eef_pos"] == {
        "key": "observation.robot0_eef_pos", "role": "state", "representation": "eef",
        "value_mode": "absolute", "component": "position", "arm": "left",
        "shape": [3], "dtype": "float32", "names": []}
    assert rows["actions"]["representation"] == "joint"
    assert rows["actions"]["value_mode"] == "delta"
    assert rows["actions"]["role"] == "action"
    assert "state_dim" not in result["suggestions"]
    assert all(not {"normalize", "slot", "width"} & row.keys() for row in rows.values())


def test_origin_relative_is_not_inferred_as_step_delta_and_gripper_does_not_inherit_pose_mode(tmp_path):
    result = inspect(tmp_path, {
        "observation.robot0_eef_pos_wrt_start": feature(),
        "observation.robot0_gripper": feature(1),
        "observation.robot0_eef_pos_delta": feature(),
        "actions": feature(4, names=["left_delta_x", "left_delta_y", "left_delta_z", "left_gripper"]),
    }, state_mode="delta", action_mode="delta", action_representation="eef")
    rows = {row["key"]: row for row in result["suggestions"]["field_annotations"]}
    assert rows["observation.robot0_eef_pos_wrt_start"]["value_mode"] == "unknown"
    assert rows["observation.robot0_gripper"]["value_mode"] == "unknown"
    assert rows["observation.robot0_gripper"]["representation"] == "gripper"
    assert rows["observation.robot0_eef_pos_delta"]["value_mode"] == "delta"
    assert rows["actions"]["value_mode"] == "unknown"
    assert any("起点" in warning for warning in result["warnings"])


@pytest.mark.parametrize("key,role,shape", [
    ("observation.robot0_gripper", "state", [1]),
    ("observation.robot0_eef_gripper", "state", [1]),
    ("actions.left_gripper", "action", [16, 1]),
])
def test_standalone_gripper_does_not_inherit_arm_representation_or_mode(tmp_path, key, role, shape):
    result = inspect(tmp_path, {key: feature(shape=shape)},
                     state_representation="eef", action_representation="joint", state_mode="absolute", action_mode="delta")
    row = result["suggestions"]["field_annotations"][0]
    assert row["role"] == role
    assert row["component"] == "gripper"
    assert row["representation"] == "gripper"
    assert row["value_mode"] == "unknown"
    if role == "action":
        assert row["horizon"] == 16
    assert normalize_dataset_inspection(result) == result


def test_one_dimensional_unknown_field_does_not_imply_gripper(tmp_path):
    result = inspect(tmp_path, {"state": feature(1)})
    row = result["suggestions"]["field_annotations"][0]
    assert row["component"] == "other" and row["representation"] == "unknown"


def test_dto_rejects_gripper_representation_on_other_components(tmp_path):
    result = inspect(tmp_path, {"observation.left_gripper": feature(1)})
    result["suggestions"]["field_annotations"][0]["component"] = "position"
    with pytest.raises(ValueError, match="INVALID_DATASET_INSPECTION"):
        normalize_dataset_inspection(result)


def test_inconsistent_gripper_declaration_is_reported_without_emitting_invalid_labels(tmp_path):
    result = inspect(tmp_path, {"observation.left_eef_pos": feature(representation="gripper")})
    row = result["suggestions"]["field_annotations"][0]
    assert row["representation"] == "unknown" and row["component"] == "position"
    assert any("夹爪开度表示" in warning for warning in result["warnings"])
    assert normalize_dataset_inspection(result) == result


def test_unhashable_optional_semantics_are_ignored_without_widening_reader(tmp_path):
    result = inspect(tmp_path, {"state": feature(7, role={}, semantic=[], component={}, arm=[], representation={}, value_mode={}),
                                "actions": feature(7)}, state_mode={})
    rows = result["suggestions"]["field_annotations"]
    assert len(rows) == 2 and all(row["representation"] == "unknown" for row in rows)
    assert all(row["value_mode"] == "unknown" for row in rows)


@pytest.mark.parametrize("change", [
    lambda row: row.update(key="not-a-real-field"),
    lambda row: row.update(shape=[99]),
    lambda row: row.update(dtype="float64"),
    lambda row: row.update(names=["unrelated"]),
    lambda row: row.update(normalize=True),
    lambda row: row.update(representation="eef_delta"),
    lambda row: row.update(value_mode="relative"),
    lambda row: row.update(shape=[]),
])
def test_source_annotation_dto_must_match_actual_feature_summary(tmp_path, change):
    result = inspect(tmp_path, {"state": feature(), "actions": feature()})
    candidate = deepcopy(result)
    change(candidate["suggestions"]["field_annotations"][0])
    with pytest.raises(ValueError, match="INVALID_DATASET_INSPECTION"):
        normalize_dataset_inspection(candidate)


def test_duplicate_annotation_keys_are_rejected(tmp_path):
    result = inspect(tmp_path, {"state": feature(), "actions": feature()})
    result["suggestions"]["field_annotations"].append(deepcopy(result["suggestions"]["field_annotations"][0]))
    with pytest.raises(ValueError):
        normalize_dataset_inspection(result)
