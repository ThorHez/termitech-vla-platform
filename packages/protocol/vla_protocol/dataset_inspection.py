"""Bounded JSON-only DTO shared by dataset inspection producers and consumers."""
from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
import re

MAX_FEATURES = 256
MAX_WARNINGS = 32
MAX_EVIDENCE = 64
MAX_RESULT_BYTES = 256 * 1024
TRIM = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
SUGGESTION_FIELDS = {"format", "robot_type", "arm_mode", "action_representation", "cameras", "task_description",
                     "action_dim", "state_dim", "fps", "action_mode", "rotation_representation", "feature_keys", "field_annotations"}
PROCESSING_FIELDS = {"action_key", "prompt_key", "state_history", "state_representation", "state_fields", "action_layout", "cameras"}


def _fail():
    raise ValueError("INVALID_DATASET_INSPECTION: 数据集检查结果不符合受限 JSON 协议")


def _plain(value, depth=0, budget=None):
    if budget is None:
        budget = [30000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 16:
        _fail()
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail()
        return
    if type(value) is list:
        for item in value:
            _plain(item, depth + 1, budget)
    elif type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item, depth + 1, budget)
    else:
        _fail()


def _obj(value, required, optional=()):
    if type(value) is not dict or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        _fail()
    return value


def _text(value, maximum=256, *, multiline=False):
    if type(value) is not str:
        _fail()
    value = value.strip(TRIM)
    if not 1 <= len(value) <= maximum or any((ord(c) < 32 and not (multiline and c in "\n\t")) or 127 <= ord(c) <= 159 for c in value):
        _fail()
    return value


def _int(value, maximum, minimum=1):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail()
    return value


def _enum(value, choices):
    if type(value) is not str or value not in choices:
        _fail()
    return value


def _list(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _fail()
    return value


def _strings(value, maximum, *, length=256):
    result = [_text(item, length) for item in _list(value, maximum)]
    if len(set(result)) != len(result):
        _fail()
    return result


def _camera(value, processing=False):
    required = {"key", "kind", "layout"} if processing else {"name", "key", "kind"}
    _obj(value, required)
    result = {"key": _text(value["key"]), "kind": _enum(value["kind"], {"rgb", "depth"})}
    if processing:
        result["layout"] = _enum(value["layout"], {"chw", "hwc"})
    else:
        result["name"] = _text(value["name"], 100)
    return result


def _annotation(value):
    _obj(value, {"key", "role", "representation", "value_mode", "component", "arm"}, {"shape", "dtype", "names", "horizon"})
    result = {"key": _text(value["key"]), "role": _enum(value["role"], {"state", "action"}),
              "representation": _enum(value["representation"], {"eef", "joint", "gripper", "unknown"}),
              "value_mode": _enum(value["value_mode"], {"absolute", "delta", "unknown"}),
              "component": _enum(value["component"], {"position", "rotation", "joint", "gripper", "other"}),
              "arm": _enum(value["arm"], {"left", "right", "shared", "unknown"})}
    if result["representation"] == "gripper" and result["component"] != "gripper":
        _fail()
    if "shape" in value:
        result["shape"] = [_int(n, 2**31 - 1) for n in _list(value["shape"], 5, 1)]
    if "dtype" in value:
        result["dtype"] = _text(value["dtype"], 64)
    if "names" in value:
        result["names"] = [_text(name) for name in _list(value["names"], 512)]
    if "horizon" in value:
        if result["role"] != "action":
            _fail()
        result["horizon"] = _int(value["horizon"], 2**31 - 1)
        if "shape" in result:
            shape = result["shape"]
            expected = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else None
            if result["horizon"] != expected:
                _fail()
    return result


def normalize_dataset_inspection(value, expected_path=None):
    """Reject extra data, non-JSON types, unsafe sizes and mismatched responses."""
    _plain(value)
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_RESULT_BYTES:
            _fail()
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("INVALID_DATASET_INSPECTION: 检查结果必须是有限且有界的 JSON") from exc
    _obj(value, {"schema_version", "path", "source", "features", "suggestions", "processing", "evidence", "warnings"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _fail()
    path = _text(value["path"], 4096)
    if not PurePosixPath(path).is_absolute() or ".." in path.split("/") or (expected_path is not None and path != expected_path):
        _fail()
    source = _obj(value["source"], {"info_sha256", "tasks_sha256"})
    for key, digest in source.items():
        if not (key == "tasks_sha256" and digest is None) and (type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            _fail()
    features = []
    for item in _list(value["features"], MAX_FEATURES, 1):
        _obj(item, {"key", "dtype", "shape", "names"})
        features.append({"key": _text(item["key"]), "dtype": _text(item["dtype"], 64),
                         "shape": [_int(n, 2**31 - 1) for n in _list(item["shape"], 5)],
                         "names": [_text(n) for n in _list(item["names"], 512)]})
    if len({f["key"] for f in features}) != len(features):
        _fail()
    suggestions = _obj(value["suggestions"], (), SUGGESTION_FIELDS)
    clean = {}
    enums = {"format": {"lerobot", "rlds", "other"}, "arm_mode": {"single", "bimanual"},
             "action_representation": {"eef", "joint", "unknown"}, "action_mode": {"absolute", "delta", "unknown"},
             "rotation_representation": {"axis_angle", "quaternion", "rotation_6d", "none", "unknown"}}
    for key, item in suggestions.items():
        if key in enums:
            clean[key] = _enum(item, enums[key])
        elif key in {"action_dim", "state_dim"}:
            clean[key] = _int(item, 512)
        elif key == "fps":
            if type(item) not in (float, int) or not 0 < item <= 1000:
                _fail()
            clean[key] = float(item)
        elif key == "feature_keys":
            clean[key] = _strings(item, 256)
        elif key == "field_annotations":
            clean[key] = [_annotation(row) for row in _list(item, MAX_FEATURES)]
            features_by_key = {row["key"]: row for row in features}
            if len({row["key"] for row in clean[key]}) != len(clean[key]):
                _fail()
            for row in clean[key]:
                feature = features_by_key.get(row["key"])
                if feature is None or any(name in row and row[name] != feature[name] for name in ("shape", "dtype", "names")):
                    _fail()
                if "horizon" in row:
                    shape = feature["shape"]
                    expected = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else None
                    if row["horizon"] != expected:
                        _fail()
        elif key == "cameras":
            clean[key] = [_camera(row) for row in _list(item, 16, 1)]
            if len({r["name"] for r in clean[key]}) != len(clean[key]) or len({r["key"] for r in clean[key]}) != len(clean[key]):
                _fail()
        else:
            clean[key] = _text(item, 2000 if key == "task_description" else 100, multiline=key == "task_description")
    processing = _obj(value["processing"], (), PROCESSING_FIELDS)
    patch = {}
    for key, item in processing.items():
        if key in {"action_key", "prompt_key"}:
            patch[key] = _text(item)
        elif key == "state_history":
            patch[key] = _int(item, 64)
        elif key == "state_representation":
            patch[key] = _enum(item, {"eef", "joint"})
        elif key == "cameras":
            patch[key] = [_camera(row, True) for row in _list(item, 16, 1)]
            if len({r["key"] for r in patch[key]}) != len(patch[key]):
                _fail()
        else:
            rows = []
            for row in _list(item, 32 if key == "state_fields" else 16, 1):
                state = key == "state_fields"
                _obj(row, {"key", "width", "arm", "kind"} if state else {"arm", "kind", "width"})
                result = {"width": _int(row["width"], 128), "arm": _enum(row["arm"], {"left", "right", "shared"} if state else {"left", "right"}),
                          "kind": _enum(row["kind"], {"position", "rotation", "joint", "gripper", "other"} if state else {"position", "rotation", "joint", "gripper"})}
                if state:
                    result["key"] = _text(row["key"])
                rows.append(result)
            if sum(row["width"] for row in rows) > 512:
                _fail()
            if key == "state_fields" and len({r["key"] for r in rows}) != len(rows):
                _fail()
            if key == "action_layout" and len({(r["arm"], r["kind"]) for r in rows}) != len(rows):
                _fail()
            patch[key] = rows
    evidence = []
    for row in _list(value["evidence"], MAX_EVIDENCE):
        _obj(row, {"field", "kind", "reason", "keys"})
        field = _text(row["field"])
        if field not in SUGGESTION_FIELDS | {"features"} | {"processing." + key for key in PROCESSING_FIELDS}:
            _fail()
        evidence.append({"field": field, "kind": _enum(row["kind"], {"metadata", "inferred"}),
                         "reason": _text(row["reason"], 1000), "keys": _strings(row["keys"], 256)})
    warnings = [_text(item, 1000) for item in _list(value["warnings"], MAX_WARNINGS)]
    return {"schema_version": 1, "path": path, "source": dict(source), "features": features,
            "suggestions": clean, "processing": patch, "evidence": evidence, "warnings": warnings}
