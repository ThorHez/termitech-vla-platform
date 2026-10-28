"""Inspection derives only a source sequence length and keeps runtime horizon separate."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from vla_agent.dataset_inspection import inspect_dataset
from vla_protocol.dataset_inspection import normalize_dataset_inspection


def inspect(tmp_path, shape, **changes):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    action = {"dtype": "float32", "shape": shape, "names": [], **changes.pop("action_metadata", {})}
    metadata = {"features": {"state": {"dtype": "float32", "shape": [7], "names": []},
                              "actions": action}, **changes}
    (root / "meta/info.json").write_text(json.dumps(metadata))
    return inspect_dataset({"roots": [str(tmp_path)]}, {
        "path": str(root), "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()})


@pytest.mark.parametrize("shape,horizon", [([10], 1), ([16, 10], 16), ([1, 7], 1)])
def test_inspection_reads_the_stored_action_sequence_length(tmp_path, shape, horizon):
    result = inspect(tmp_path, shape, action_horizon=50)
    rows = {row["key"]: row for row in result["suggestions"]["field_annotations"]}
    assert rows["actions"]["horizon"] == horizon
    assert "horizon" not in rows["state"]
    assert "action_horizon" not in result["suggestions"]
    assert normalize_dataset_inspection(result) == result


@pytest.mark.parametrize("horizon", [50, "16", None, True, [], {"value": 16}])
def test_conflicting_or_untrusted_horizon_declarations_do_not_override_shape(tmp_path, horizon):
    result = inspect(tmp_path, [16, 10], action_metadata={"horizon": horizon})
    action = next(row for row in result["suggestions"]["field_annotations"] if row["role"] == "action")
    assert action["horizon"] == 16
    assert any("horizon 声明" in warning for warning in result["warnings"])


def test_high_rank_actions_have_no_inferred_horizon(tmp_path):
    result = inspect(tmp_path, [2, 16, 10])
    action = next(row for row in result["suggestions"]["field_annotations"] if row["role"] == "action")
    assert "horizon" not in action
    assert any("不是 [D] 或 [H,D]" in warning for warning in result["warnings"])


def test_explicit_annotation_horizon_cannot_replace_the_actual_shape(tmp_path):
    result = inspect(tmp_path, [16, 10], field_annotations=[{"key": "actions", "horizon": 50}])
    action = next(row for row in result["suggestions"]["field_annotations"] if row["role"] == "action")
    assert action["horizon"] == 16
    assert any("horizon 声明" in warning for warning in result["warnings"])


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(horizon=1),
    lambda rows: rows[1].update(horizon=0),
    lambda rows: rows[1].update(horizon=True),
    lambda rows: rows[1].update(horizon=None),
    lambda rows: rows[1].update(horizon=50),
    lambda rows: (rows[1].pop("shape"), rows[1].update(horizon=50)),
])
def test_dto_rejects_non_action_invalid_or_forged_source_horizon(tmp_path, change):
    result = inspect(tmp_path, [16, 10]);candidate = deepcopy(result)
    change(candidate["suggestions"]["field_annotations"])
    with pytest.raises(ValueError, match="INVALID_DATASET_INSPECTION"):
        normalize_dataset_inspection(candidate)


def test_dto_keeps_old_annotations_without_horizon_compatible(tmp_path):
    result = inspect(tmp_path, [16, 10])
    for row in result["suggestions"]["field_annotations"]:
        row.pop("horizon", None)
    assert normalize_dataset_inspection(result) == result
