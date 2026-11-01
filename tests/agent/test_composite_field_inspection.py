"""Packed-vector metadata remains explicit and is never inferred from width."""

from copy import deepcopy
import json

import pytest

from vla_agent.dataset_inspection import inspect_dataset
from vla_protocol.dataset_inspection import normalize_dataset_inspection


def inspect(tmp_path, *, role="action", representation="composite", component=None, names=None):
    path = tmp_path / "dataset"
    (path / "meta").mkdir(parents=True)
    key = "actions" if role == "action" else "observation.eef_state"
    label = {"key": key, "role": role, "representation": representation}
    if component is not None:
        label["component"] = component
    info = {"action_mode": "delta", "state_mode": "absolute", "features": {
        key: {"dtype": "float32", "shape": [16, 10] if role == "action" else [2, 10],
              "names": names or []}}, "field_annotations": [label]}
    (path / "meta/info.json").write_text(json.dumps(info))
    from datetime import datetime, timedelta, timezone
    result = inspect_dataset({"roots": [str(tmp_path)]}, {"path": str(path),
        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()})
    return result


@pytest.mark.parametrize("role", ["state", "action"])
def test_explicit_composite_round_trips_without_claiming_a_single_representation(tmp_path, role):
    result = inspect(tmp_path, role=role)
    assert normalize_dataset_inspection(result) == result
    row = result["suggestions"]["field_annotations"][0]
    assert row["representation"] == "composite" and row["component"] == "other"
    assert row["value_mode"] == "unknown", "A global movement mode is not proof for a packed vector"
    assert row["shape"] == ([16, 10] if role == "action" else [2, 10])
    assert row.get("horizon") == (16 if role == "action" else None)


def test_explicit_composite_is_not_retyped_from_joint_coordinate_names(tmp_path):
    result = inspect(tmp_path, names=[f"joint_{index}" for index in range(10)])
    row = result["suggestions"]["field_annotations"][0]
    assert row["representation"] == "composite" and row["component"] == "other"


def test_conflicting_composite_single_component_is_reported_as_unconfirmed(tmp_path):
    result = inspect(tmp_path, component="position")
    row = result["suggestions"]["field_annotations"][0]
    assert row["representation"] == "unknown"
    assert any("组合向量" in warning for warning in result["warnings"])


def test_unknown_vector_width_does_not_invent_a_composite_tag_or_chunks(tmp_path):
    result = inspect(tmp_path, representation="unknown")
    row = result["suggestions"]["field_annotations"][0]
    assert row["representation"] == "unknown"
    assert "action_layout" not in result["processing"]


def test_protocol_rejects_composite_with_a_single_component_and_wrong_horizon(tmp_path):
    result = inspect(tmp_path)
    for patch in ({"component": "position"}, {"horizon": 50}):
        invalid = deepcopy(result)
        invalid["suggestions"]["field_annotations"][0].update(patch)
        with pytest.raises(ValueError):
            normalize_dataset_inspection(invalid)
