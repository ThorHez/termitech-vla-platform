from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from vla_agent import dataset_inspection as inspection
from vla_agent.safety import AgentError
from vla_protocol.dataset_inspection import normalize_dataset_inspection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/dataset_inspection"


def query(path, **changes):
    return {"path": str(path), "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(), **changes}


def dataset(tmp_path, name="complete"):
    path = tmp_path / "dataset";meta = path / "meta";meta.mkdir(parents=True)
    for source in (FIXTURES / name).iterdir():
        shutil.copyfile(source, meta / source.name)
    return path


def read(path):
    return inspection.inspect_dataset({"roots": [str(path.parent)]}, query(path))


def test_complete_metadata_fills_only_supported_defaults_with_evidence(tmp_path):
    path = dataset(tmp_path);result = read(path)
    assert result == normalize_dataset_inspection(result, expected_path=str(path))
    assert result["source"] == {key + "_sha256": hashlib.sha256((path / "meta" / filename).read_bytes()).hexdigest()
                                for key, filename in (("info", "info.json"), ("tasks", "tasks.jsonl"))}
    suggested = result["suggestions"]
    assert suggested["robot_type"] == "ARX5" and suggested["fps"] == 30.0
    assert suggested["action_dim"] == 7 and suggested["state_dim"] == 7
    assert suggested["arm_mode"] == "single" and suggested["action_representation"] == "joint"
    assert suggested["rotation_representation"] == "none"
    assert "红色方块" in suggested["task_description"]
    assert result["processing"]["prompt_key"] == "task"
    assert result["processing"]["state_fields"] == [{"key": "observation.state", "width": 7, "arm": "shared", "kind": "other"}]
    assert result["processing"]["action_layout"] == [{"arm": "left", "kind": "joint", "width": 6}, {"arm": "left", "kind": "gripper", "width": 1}]
    assert result["processing"]["cameras"][0]["layout"] == "chw"
    assert {row["kind"] for row in result["evidence"]} == {"metadata", "inferred"}
    assert "must-not-return" not in json.dumps(result)
    assert "private_token" not in json.dumps(result) and "private_camera_password" not in json.dumps(result)


def test_prepacked_history_and_bimanual_fields_are_not_flattened_or_converted(tmp_path):
    result = read(dataset(tmp_path, "history"));p = result["processing"];s = result["suggestions"]
    assert s["action_dim"] == 20 and "state_dim" not in s
    assert s["arm_mode"] == "bimanual" and s["action_representation"] == "eef"
    assert s["rotation_representation"] == "rotation_6d"
    assert p["state_history"] == 2 and p["state_representation"] == "eef"
    assert len(p["state_fields"]) == 6 and sum(x["width"] for x in p["action_layout"]) == 20
    assert p["cameras"][-1] == {"key": "observation.images.right_depth", "kind": "depth", "layout": "hwc"}
    assert any("时间轴" in warning for warning in result["warnings"])
    assert not {"action_mode", "coordinate_frame", "position_unit", "joint_unit", "gripper_range"} & (s.keys() | p.keys())
    assert all("normalize" not in field and "slot" not in field for field in p["state_fields"] + p["cameras"])


@pytest.mark.parametrize("width", [7, 10, 14, 20])
def test_ambiguous_widths_do_not_invent_robot_or_action_semantics(tmp_path, width):
    path = dataset(tmp_path, "ambiguous");source = path / "meta/info.json"
    info = json.loads(source.read_text());info["features"]["actions"]["shape"] = [width];source.write_text(json.dumps(info))
    result = read(path);s = result["suggestions"];p = result["processing"]
    assert s["action_dim"] == width
    assert not {"robot_type", "arm_mode", "action_representation", "rotation_representation", "action_mode"} & s.keys()
    assert not {"action_layout", "state_representation", "cameras"} & p.keys()
    assert any("维数不能证明" in warning for warning in result["warnings"])


def test_metadata_reader_never_opens_samples_or_imports_training_code(tmp_path, monkeypatch):
    path = dataset(tmp_path);(path / "data").mkdir();os.mkfifo(path / "data/huge.parquet")
    original = os.open;opened = []
    def tracked(file, *args, **kwargs):
        opened.append(str(file));return original(file, *args, **kwargs)
    monkeypatch.setattr(os, "open", tracked)
    before = {name for name in sys.modules if name.startswith("openpi")}
    read(path)
    assert not any("parquet" in name or "videos" in name for name in opened)
    assert {name for name in sys.modules if name.startswith("openpi")} == before


def test_directory_walk_uses_nofollow_for_ancestors_and_keeps_descriptor_budget_constant(tmp_path, monkeypatch):
    path = dataset(tmp_path)
    opened, maximum = set(), [0]
    original_open, original_close = os.open, os.close
    def tracked_open(file, flags, *args, **kwargs):
        if file not in {"info.json", "tasks.jsonl"}:
            assert flags & os.O_NOFOLLOW and flags & os.O_DIRECTORY
            assert str(file) == "/" or kwargs.get("dir_fd") is not None
        descriptor = original_open(file, flags, *args, **kwargs)
        opened.add(descriptor);maximum[0] = max(maximum[0], len(opened))
        return descriptor
    def tracked_close(descriptor):
        opened.discard(descriptor);return original_close(descriptor)
    monkeypatch.setattr(os, "open", tracked_open);monkeypatch.setattr(os, "close", tracked_close)
    read(path)
    # fdopen owns its file descriptors and closes them without calling os.close;
    # directory traversal itself must retain only parent+child at a time.
    assert maximum[0] <= 3


@pytest.mark.parametrize("field,value,fixture,removed", [
    ("action_representation", "eef", "complete", {"action_representation", "rotation_representation"}),
    ("arm_mode", "single", "history", {"arm_mode"}),
    ("rotation_representation", "quaternion", "complete", {"rotation_representation"}),
])
def test_contradictory_metadata_and_complete_action_names_withdraw_defaults(tmp_path, field, value, fixture, removed):
    path = dataset(tmp_path, fixture);source = path / "meta/info.json"
    info = json.loads(source.read_text());info[field] = value;source.write_text(json.dumps(info))
    result = read(path)
    assert not removed & result["suggestions"].keys()
    assert "action_layout" not in result["processing"]
    assert not removed & {row["field"] for row in result["evidence"]}
    assert any("矛盾" in warning for warning in result["warnings"])


def test_target_and_command_fields_are_excluded_from_state_candidates(tmp_path):
    path = dataset(tmp_path, "history");source = path / "meta/info.json"
    info = json.loads(source.read_text())
    for key in ("left_joint_target", "command_right_joint", "actions.left_joint", "left_joint_command"):
        info["features"][key] = {"dtype": "float32", "shape": [2, 6]}
    source.write_text(json.dumps(info))
    result = read(path)
    assert len(result["processing"]["state_fields"]) == 6
    assert "state_dim" not in result["suggestions"]
    assert all(row["role"] == "action" for row in result["suggestions"]["field_annotations"]
               if row["key"] in {"left_joint_target", "command_right_joint", "actions.left_joint", "left_joint_command"})


def test_repeated_or_incomplete_coordinate_names_do_not_produce_action_layouts(tmp_path):
    path = dataset(tmp_path);source = path / "meta/info.json"
    info = json.loads(source.read_text())
    for names in (["left_joint_0"] * 7, ["left_eef_pos_x", "left_eef_pos_y", "left_eef_rot_axis_angle_z"]):
        info["features"]["actions"].update(shape=[len(names)], names=names)
        source.write_text(json.dumps(info))
        assert "action_layout" not in read(path)["processing"]


def test_optional_tasks_are_bounded_by_lines_and_bytes_without_returning_extra_data(tmp_path):
    path = dataset(tmp_path);source = path / "meta/tasks.jsonl"
    source.write_bytes(b'x' * (inspection.MAX_TASK_LINE_BYTES + 1) + b'\n' +
                       json.dumps({"task": "bounded example", "private_extra": "never-return"}).encode())
    result = read(path)
    assert result["suggestions"]["task_description"] == "bounded example"
    assert any("单行过大" in warning for warning in result["warnings"])
    assert "never-return" not in json.dumps(result)
    source.write_text('\n'.join([json.dumps({"task": ""})] * inspection.MAX_TASK_LINES + [json.dumps({"task": "past-line-budget"})]))
    result = read(path)
    assert "task_description" not in result["suggestions"]
    assert any("固定数量" in warning for warning in result["warnings"])


def test_protocol_ecmascript_trim_and_nonfinite_values_remain_strict(tmp_path):
    result = read(dataset(tmp_path))
    result["suggestions"]["robot_type"] = "\ufeffARX5\ufeff"
    assert normalize_dataset_inspection(result)["suggestions"]["robot_type"] == "ARX5"
    for value in ("\x1cARX5", "\x85ARX5"):
        result["suggestions"]["robot_type"] = value
        with pytest.raises(ValueError):normalize_dataset_inspection(result)


@pytest.mark.parametrize("location", ["info", "tasks", "meta"])
def test_symlinks_are_rejected_and_cannot_escape_roots(tmp_path, location):
    allowed = tmp_path / "allowed";allowed.mkdir();path = dataset(allowed)
    outside = tmp_path / "outside";outside.mkdir();(outside / "info.json").write_text('{"features":{}}')
    target = path / "meta" if location == "meta" else path / "meta" / ("info.json" if location == "info" else "tasks.jsonl")
    if target.is_dir():shutil.rmtree(target)
    else:target.unlink()
    target.symlink_to(outside if location == "meta" else outside / "info.json", target_is_directory=location == "meta")
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code in {"PATH_OUTSIDE_ROOTS", "DATASET_INSPECT_INVALID_METADATA"}


@pytest.mark.parametrize("name", ["info.json", "tasks.jsonl"])
def test_fifo_metadata_is_rejected_without_reading(tmp_path, name):
    path = dataset(tmp_path);source = path / "meta" / name;source.unlink();os.mkfifo(source)
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_INVALID_METADATA"


def test_missing_outside_expired_and_permission_requests_have_safe_errors(tmp_path, monkeypatch):
    path = dataset(tmp_path)
    for request, code in ((query(path, expires_at="2000-01-01T00:00:00Z"), "DATASET_INSPECT_EXPIRED"),
                          (query(path, expires_at="not-a-time"), "INVALID_DATASET_INSPECTION"),
                          (query(path, limit=1), "INVALID_DATASET_INSPECTION"),
                          (query(path.parent / "missing"), "DATASET_INSPECT_NOT_FOUND"),
                          (query(path.parent.parent), "PATH_OUTSIDE_ROOTS")):
        with pytest.raises(AgentError) as failure:inspection.inspect_dataset({"roots": [str(path.parent)]}, request)
        assert failure.value.code == code
    original = os.open
    def denied(file, *args, **kwargs):
        if file == "info.json":raise PermissionError("arbitrary-secret-error")
        return original(file, *args, **kwargs)
    monkeypatch.setattr(os, "open", denied)
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_PERMISSION_DENIED"
    assert "arbitrary-secret-error" not in str(failure.value)


def test_file_feature_task_and_output_budgets(tmp_path):
    path = dataset(tmp_path);source = path / "meta/info.json";original = source.read_bytes()
    source.write_bytes(b" " * (inspection.MAX_INFO_BYTES + 1))
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_LIMIT"
    source.write_text(json.dumps({"features": {str(i): {"dtype": "float32", "shape": [1]} for i in range(257)}}))
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_LIMIT"
    source.write_bytes(original);(path / "meta/tasks.jsonl").write_bytes(b" " * (inspection.MAX_TASKS_BYTES + 1))
    result = read(path)
    assert result["source"]["tasks_sha256"] is None and "task_description" not in result["suggestions"]
    assert any("预算" in warning for warning in result["warnings"])
    source.write_text(json.dumps({"features": {"state": {"dtype": "float32", "shape": [1], "names": ["x" * 256] * 512},
                                              "actions": {"dtype": "float32", "shape": [1], "names": ["y" * 256] * 512}}}))
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_LIMIT"


@pytest.mark.parametrize("content", ["[]", "{}", '{"features":{}}', '{"features":{"x":{}}}',
                                     '{"features":{"x":{"dtype":"float32","shape":[false]}}}',
                                     '{"features":{"x":{}},"features":{}}', "{"])
def test_malformed_or_duplicate_info_is_rejected_without_returning_content(tmp_path, content):
    path = dataset(tmp_path);(path / "meta/info.json").write_text(content)
    with pytest.raises(AgentError) as failure:read(path)
    assert failure.value.code == "DATASET_INSPECT_INVALID_METADATA"


@pytest.mark.parametrize("mutation", [
    lambda r:r.update(extra={"secret": "no"}), lambda r:r.update(schema_version=True), lambda r:r.update(path="../other"),
    lambda r:r.update(features=[]),
    lambda r:r["source"].update(token="no"), lambda r:r["source"].update(info_sha256="0"),
    lambda r:r["features"][0].update(shape=[True]), lambda r:r["features"][0].update(shape=[0]),
    lambda r:r["features"][0].update(shape=[1] * 6), lambda r:r["features"][0].update(names="x"),
    lambda r:r["suggestions"].update(position_unit="m"), lambda r:r["suggestions"].update(fps=float("nan")),
    lambda r:r["processing"]["state_fields"][0].update(normalize=True),
    lambda r:r["processing"]["cameras"][0].update(slot="base_0_rgb"),
    lambda r:r["evidence"][0].update(kind="verified"), lambda r:r.update(warnings=["x"] * 33),
    lambda r:r.update(features=tuple(r["features"])), lambda r:r.update(source={"info_sha256": "0" * 64, "tasks_sha256": object()}),
])
def test_protocol_rejects_non_json_and_unapproved_dto_fields(tmp_path, mutation):
    result = read(dataset(tmp_path));mutation(result)
    with pytest.raises(ValueError):normalize_dataset_inspection(result)


def test_protocol_binds_path_and_rejects_controls_and_overlong_fields(tmp_path):
    result = read(dataset(tmp_path))
    with pytest.raises(ValueError):normalize_dataset_inspection(result, expected_path="/another")
    for field, value in (("robot_type", "\x85ARX5"), ("robot_type", "x" * 101), ("fps", True), ("action_dim", 0)):
        changed = deepcopy(result);changed["suggestions"][field] = value
        with pytest.raises(ValueError):normalize_dataset_inspection(changed)
