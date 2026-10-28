"""Portable data-only contract for the registered composable processing factory.

The training project implements the same versioned contract. Conformance tests
compare both validators; this module imports no model or training dependencies.
"""

from copy import deepcopy
import hashlib
import json
import math


COMPOSABLE_PROCESSOR = "ComposableDataConfig"
FIELDS = {"schema_version", "robot_model", "arm_mode", "state_representation", "state_fields", "state_history",
          "action_representation", "action_key", "action_layout", "action_mode", "rotation_representation",
          "position_unit", "joint_unit", "coordinate_frame", "gripper_range", "cameras", "prompt_key",
          "normalization", "model_transform", "aux_keys"}
OPTIONAL_FIELDS = {"position_unit", "joint_unit", "coordinate_frame", "gripper_range",
                   "action_mode", "rotation_representation", "prompt_key"}
SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
STATE_KINDS = {"position", "rotation", "joint", "gripper", "other"}
# Match ECMAScript String.trim exactly across the API, UI and training code.
# Python's default strip also removes C0 separators / U+0085 and keeps U+FEFF.
TRIM_CHARACTERS = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def canonical_text_trim(value):
    return value.strip(TRIM_CHARACTERS)


def _fail(message):
    raise ValueError("COMPOSITION_INVALID: " + message)


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        _fail(label + "字段不完整或包含未知字段")


def _text(value, label, maximum=256):
    if not isinstance(value, str):
        _fail(label + "需要字符串")
    result = canonical_text_trim(value)
    if not 1 <= len(result) <= maximum or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in result):
        _fail(label + "不能为空、过长或包含控制字符")
    return result


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label}需要 {minimum} 到 {maximum} 的整数")
    return value


def _key(value, label):
    result = _text(value, label)
    if (result.startswith(("/", "~")) or "\\" in result or ":" in result
            or any(part in {"", ".", ".."} for part in result.split("/"))
            or any(char in "()[]{}'\"`; $".replace(" ", "") for char in result)):
        _fail(label + "必须是数据字段名，不能是路径或表达式")
    return result


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        _fail(label + "需要有限数值")
    return float(value)


def _enum(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        _fail(label + "不在支持范围内")
    return value


def _optional_text(value, label, maximum=256):
    if value is None or isinstance(value, str) and not canonical_text_trim(value):
        return ""
    return _text(value, label, maximum)


def _optional_unit(value, choices, label):
    if value is None or isinstance(value, str) and not canonical_text_trim(value):
        return ""
    return _enum(value, choices, label)


def normalize_composition(value):
    """Normalize the v1 spec; never guess dimensions, encodings or declarations."""
    if (not isinstance(value, dict) or not FIELDS - OPTIONAL_FIELDS <= set(value)
            or set(value) - FIELDS):
        _fail("组合处理配置字段不完整或包含未知字段")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _fail("仅支持组合处理版本 1")
    result = {"schema_version": 1}
    for key, maximum in (("robot_model", 100), ("action_key", 256)):
        result[key] = _text(value[key], key, maximum)
    result["coordinate_frame"] = _optional_text(value.get("coordinate_frame"), "coordinate_frame", 100)
    for key, choices in (("position_unit", {"m", "mm"}), ("joint_unit", {"rad", "deg"})):
        result[key] = _optional_unit(value.get(key), choices, key)
    result["action_key"] = _key(value["action_key"], "action_key")
    prompt = _optional_text(value.get("prompt_key"), "prompt_key")
    result["prompt_key"] = _key(prompt, "prompt_key") if prompt else ""
    for key, choices in (("action_mode", {"absolute", "delta"}),
                         ("rotation_representation", {"axis_angle", "quaternion", "rotation_6d", "none"})):
        result[key] = _optional_unit(value.get(key), choices, key)
    for key, choices in (("arm_mode", {"single", "bimanual"}), ("state_representation", {"eef", "joint"}),
                         ("action_representation", {"eef", "joint"}),
                         ("normalization", {"quantile", "zscore"}), ("model_transform", {"standard"})):
        result[key] = _enum(value[key], choices, key)
    result["state_history"] = _integer(value["state_history"], 1, 64, "state_history")
    arms = {"left"} if result["arm_mode"] == "single" else {"left", "right"}
    state = value["state_fields"]
    if not isinstance(state, list) or not 1 <= len(state) <= 32:
        _fail("状态拼接需要 1 到 32 个字段")
    result["state_fields"] = []
    for item in state:
        _object(item, {"key", "width", "arm", "kind", "normalize"}, "状态字段")
        if type(item["normalize"]) is not bool:
            _fail("状态归一化选项需要布尔值")
        result["state_fields"].append({"key": _key(item["key"], "状态字段名"),
            "width": _integer(item["width"], 1, 128, "状态字段宽度"),
            "arm": _enum(item["arm"], arms | {"shared"}, "状态所属机械臂"),
            "kind": _enum(item["kind"], STATE_KINDS, "状态字段类型"), "normalize": item["normalize"]})
    if len({item["key"] for item in result["state_fields"]}) != len(state):
        _fail("状态字段名不能重复；本版不支持隐式切片")
    layout = value["action_layout"]
    if not isinstance(layout, list) or not 1 <= len(layout) <= 16:
        _fail("动作布局需要 1 到 16 个分块")
    result["action_layout"] = []
    for item in layout:
        _object(item, {"arm", "kind", "width", "normalize"}, "动作分块")
        if type(item["normalize"]) is not bool:
            _fail("动作归一化选项需要布尔值")
        result["action_layout"].append({"arm": _enum(item["arm"], arms, "动作所属机械臂"),
            "kind": _enum(item["kind"], STATE_KINDS - {"other"}, "动作分块类型"),
            "width": _integer(item["width"], 1, 128, "动作分块宽度"), "normalize": item["normalize"]})
    if len({(item["arm"], item["kind"]) for item in result["action_layout"]}) != len(layout):
        _fail("每个机械臂的动作分块类型不能重复")
    for arm in arms:
        blocks = {item["kind"]: item["width"] for item in result["action_layout"] if item["arm"] == arm}
        if result["action_representation"] == "joint":
            if ("joint" not in blocks or set(blocks) - {"joint", "gripper"}
                    or result["rotation_representation"] not in {"", "none"}):
                _fail("Joint 动作需要显式关节块，旋转编码可留空或为 none")
        else:
            rotation = result["rotation_representation"]
            width = {"axis_angle": 3, "quaternion": 4, "rotation_6d": 6}.get(rotation)
            if (blocks.get("position") != 3 or blocks.get("rotation") not in {3, 4, 6}
                    or rotation and (width is None or blocks.get("rotation") != width)
                    or set(blocks) - {"position", "rotation", "gripper"}):
                _fail("EEF 动作需要位置 3 维及与旋转编码一致的旋转分块")
    for name in ("state_fields", "action_layout"):
        if sum(item["width"] for item in result[name]) > 512:
            _fail(name + "总宽度不能超过 512")
    gripper = value.get("gripper_range")
    result["gripper_range"] = None
    if gripper is not None:
        if not isinstance(gripper, list) or len(gripper) != 2:
            _fail("夹爪范围需要最小值和最大值")
        result["gripper_range"] = [_number(item, "夹爪范围") for item in gripper]
        if result["gripper_range"][0] >= result["gripper_range"][1]:
            _fail("夹爪范围最小值必须小于最大值")
    cameras = value["cameras"]
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 3:
        _fail("标准模型支持 1 到 3 个视觉输入")
    result["cameras"] = []
    for item in cameras:
        _object(item, {"key", "slot", "kind", "layout", "time_index", "depth_min", "depth_max"}, "视觉输入")
        index = item["time_index"]
        if index is not None:
            index = _integer(index, -64, 63, "图像时间索引")
        low, high = _number(item["depth_min"], "深度最小值"), _number(item["depth_max"], "深度最大值")
        if low >= high:
            _fail("深度显示范围需要最小值小于最大值")
        result["cameras"].append({"key": _key(item["key"], "图像字段名"),
            "slot": _enum(item["slot"], set(SLOTS), "模型视觉输入槽"),
            "kind": _enum(item["kind"], {"rgb", "depth"}, "图像类型"),
            "layout": _enum(item["layout"], {"chw", "hwc"}, "图像布局"), "time_index": index,
            "depth_min": low, "depth_max": high})
    if len({item["key"] for item in result["cameras"]}) != len(cameras) or len({item["slot"] for item in result["cameras"]}) != len(cameras):
        _fail("视觉字段与模型输入槽不能重复")
    keys = [item["key"] for item in result["state_fields"] + result["cameras"]]
    if (len(set(keys)) != len(keys) or result["action_key"] in keys
            or result["prompt_key"] and (result["prompt_key"] in keys or result["action_key"] == result["prompt_key"])):
        _fail("状态、图像、动作和任务字段不能互相占用")
    aux = value["aux_keys"]
    if (not isinstance(aux, list) or len(aux) > 2 or len(set(item for item in aux if isinstance(item, str))) != len(aux)
            or any(item not in {"value_target", "is_positive"} for item in aux if isinstance(item, str))
            or any(not isinstance(item, str) for item in aux)):
        _fail("辅助字段仅支持 value_target 和 is_positive，且不能重复")
    if any(key in {*keys, result["action_key"], result["prompt_key"]} for key in aux):
        _fail("辅助字段不能与其它输入字段重名")
    result["aux_keys"] = list(aux)
    return result


def composition_digest(value):
    return hashlib.sha256(json.dumps(normalize_composition(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_profile_composition(profile):
    """Require the single asset description and composed layout to agree."""
    if profile.get("processing") is None:
        return
    if profile.get("format") != "lerobot":
        _fail("组件组合处理首版仅支持 LeRobot 格式")
    spec = normalize_composition(profile["processing"])
    for key, expected in (("robot_type", spec["robot_model"]), ("arm_mode", spec["arm_mode"]),
                          ("action_representation", spec["action_representation"])):
        if profile.get(key) != expected:
            _fail("组合配置与登记描述不一致：" + key)
    for key in ("action_mode", "rotation_representation"):
        if spec[key] and profile.get(key, "unknown") not in ("unknown", spec[key]):
            _fail("组合配置与登记编码不一致：" + key)
    for key, fields in (("state_dim", "state_fields"), ("action_dim", "action_layout")):
        if profile.get(key) is not None and profile[key] != sum(item["width"] for item in spec[fields]):
            _fail("组合配置与登记维度不一致：" + key)
    if {(c["key"], c["kind"]) for c in spec["cameras"]} != {(c["key"], c["kind"]) for c in profile.get("cameras", [])}:
        _fail("组合视觉输入与登记相机字段不一致")


def composition_choice(profile):
    spec = profile.get("processing") if isinstance(profile, dict) else None
    return None if spec is None else {"name": COMPOSABLE_PROCESSOR, "parameters": {"composition": deepcopy(normalize_composition(spec))}}


def normalize_field_annotations(value):
    """Source-field facts, independent of any model-specific concatenation."""
    if not isinstance(value, list) or len(value) > 256:
        _fail("字段标签需要不超过 256 项的列表")
    required = {"key", "role", "representation", "value_mode", "component", "arm"}
    optional = {"shape", "dtype", "names", "horizon"}
    result = []
    for item in value:
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - optional:
            _fail("字段标签缺少必填项或包含处理规则")
        row = {"key": _key(item["key"], "标签字段名")}
        for key, choices in (("role", {"state", "action"}),
                             ("representation", {"eef", "joint", "gripper", "unknown"}),
                             ("value_mode", {"absolute", "delta", "unknown"}),
                             ("component", STATE_KINDS),
                             ("arm", {"left", "right", "shared", "unknown"})):
            row[key] = _enum(item[key], choices, "字段标签 " + key)
        if row["representation"] == "gripper" and row["component"] != "gripper":
            _fail("夹爪开度表示仅适用于夹爪分量")
        if "shape" in item:
            shape = item["shape"]
            if not isinstance(shape, list) or not 1 <= len(shape) <= 5:
                _fail("字段标签形状需要 1 到 5 个正整数")
            row["shape"] = [_integer(size, 1, 2**31 - 1, "字段标签形状") for size in shape]
        if "dtype" in item:
            row["dtype"] = _text(item["dtype"], "字段标签类型", 64)
        if "names" in item:
            names = item["names"]
            if not isinstance(names, list) or len(names) > 512:
                _fail("字段标签坐标名最多 512 项")
            row["names"] = [_text(name, "字段标签坐标名") for name in names]
        if "horizon" in item:
            if row["role"] != "action":
                _fail("只有动作字段可以登记 horizon")
            row["horizon"] = _integer(item["horizon"], 1, 2**31 - 1, "原始动作序列长度 horizon")
            if "shape" in row:
                shape = row["shape"]
                expected = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else None
                if expected is None:
                    _fail("原始动作 horizon 仅能由 [D] 或 [H,D] 形状核验")
                if row["horizon"] != expected:
                    _fail("原始动作 horizon 与字段形状不一致")
        result.append(row)
    if len({row["key"] for row in result}) != len(result):
        _fail("字段标签不能重复")
    return result


def validate_composition_profile_input(profile, composition):
    """Check a run-owned plan against source labels, never against a legacy plan.

    The v1 composable processor repacks tensors; it does not convert EEF/Joint
    representations or absolute/delta encodings. A new field-label profile must
    therefore explicitly describe each selected state/action input.
    """
    spec = normalize_composition(composition)
    if not isinstance(profile, dict):
        return spec
    if profile.get("format") != "lerobot":
        _fail("组件组合处理首版仅支持 LeRobot 格式")
    cameras = {(row["key"], row["kind"]) for row in profile.get("cameras", [])}
    if any((camera["key"], camera["kind"]) not in cameras for camera in spec["cameras"]):
        raise ValueError("COMPOSITION_SOURCE_MISMATCH: 组合视觉字段未在数据集中登记或图像类型不一致")
    annotations = profile.get("field_annotations")
    if annotations is None:
        # Historical profiles have no per-field contract. Retain their known
        # action declarations without treating their saved processing as owned.
        for source, target in (("action_representation", "action_representation"),
                               ("action_mode", "action_mode")):
            if spec[target] and profile.get(source, "unknown") not in {"unknown", spec[target]}:
                raise ValueError("COMPOSITION_SOURCE_MISMATCH: 数据集原始动作声明与处理方案不一致：" + source)
        return spec
    annotations = {row["key"]: row for row in normalize_field_annotations(annotations)}
    for field in spec["state_fields"]:
        label = annotations.get(field["key"])
        if label is None or label["role"] != "state":
            raise ValueError("COMPOSITION_SOURCE_UNCONFIRMED: 请先确认状态字段标签：" + field["key"])
        if (label["value_mode"] == "unknown" or
                label["component"] not in {"gripper", "other"} and label["representation"] == "unknown"):
            raise ValueError("COMPOSITION_SOURCE_UNCONFIRMED: 状态字段的表示或绝对/增量标签待确认：" + field["key"])
        if (label["component"] not in {"gripper", "other"} and label["representation"] != spec["state_representation"]
                or label["component"] != "other" and label["component"] != field["kind"]
                or label["arm"] != "unknown" and label["arm"] != field["arm"]):
            raise ValueError("COMPOSITION_SOURCE_MISMATCH: 状态拼接与字段原始标签不一致：" + field["key"])
    action = annotations.get(spec["action_key"])
    if action is not None and action["component"] == "gripper":
        raise ValueError("COMPOSITION_SOURCE_MISMATCH: 当前组合处理器需要完整 EEF/Joint 动作字段，不能仅用夹爪开度字段：" + spec["action_key"])
    if action is None or action["role"] != "action" or action["representation"] == "unknown":
        raise ValueError("COMPOSITION_SOURCE_UNCONFIRMED: 请先确认动作字段的 EEF/Joint 标签：" + spec["action_key"])
    if (action["representation"] != spec["action_representation"] or
            spec["action_mode"] and action["value_mode"] not in {"unknown", spec["action_mode"]}):
        raise ValueError("COMPOSITION_SOURCE_MISMATCH: 处理器不支持自动转换动作字段的表示或绝对/增量编码")
    return spec


def verify_composition_metadata(value, info, *, derived_features=(), action_horizon=None):
    """Validate declared tensor layout against real metadata; no value conversion."""
    spec = normalize_composition(value)
    features = info.get("features") if isinstance(info, dict) else None
    if not isinstance(features, dict):
        _fail("数据集缺少 features 元信息")

    def shape(key):
        feature = features.get(key)
        dimensions = feature.get("shape") if isinstance(feature, dict) else None
        if (not isinstance(dimensions, list) or not dimensions
                or any(type(item) is not int or item <= 0 for item in dimensions)):
            _fail("缺少字段或维度元信息：" + key)
        return dimensions

    history = spec["state_history"]
    for field in spec["state_fields"]:
        actual = shape(field["key"])
        if actual != [history, field["width"]] and not (history == 1 and actual == [field["width"]]):
            _fail("状态字段维度与声明的历史长度/宽度不一致：" + field["key"])
    width = sum(field["width"] for field in spec["action_layout"])
    actions = shape(spec["action_key"])
    if actions != [width] and not (len(actions) == 2 and actions[-1] == width
                                  and (action_horizon is None or actions[0] == action_horizon)):
        _fail("动作字段需要与模型和布局一致的 [D] 或 [H,D]")
    for camera in spec["cameras"]:
        actual = shape(camera["key"])
        if camera["time_index"] is not None:
            if not -actual[0] <= camera["time_index"] < actual[0]:
                _fail("图像时间索引越界：" + camera["key"])
            actual = actual[1:]
        channels = actual[0 if camera["layout"] == "chw" else -1]
        if camera["kind"] == "rgb":
            if len(actual) != 3 or channels != 3:
                _fail("RGB 元信息须明确三通道和 CHW/HWC 布局：" + camera["key"])
        elif len(actual) not in (2, 3) or (len(actual) == 3 and channels != 1):
            _fail("深度元信息须明确单通道布局：" + camera["key"])
    for key in ([spec["prompt_key"]] if spec["prompt_key"] else []) + spec["aux_keys"]:
        if key not in features and key not in derived_features:
            _fail("数据集缺少任务或阶段辅助字段：" + key)
    return spec
