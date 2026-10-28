"""Inspect bounded LeRobot metadata; never import training code or scan samples."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from vla_protocol.dataset_inspection import MAX_FEATURES, MAX_RESULT_BYTES, TRIM, normalize_dataset_inspection

from .safety import AgentError, allowed_path

MAX_INFO_BYTES = 1024 * 1024
MAX_TASKS_BYTES = 256 * 1024
MAX_TASK_LINES = 256
MAX_TASK_LINE_BYTES = 8192
MAX_TASK_EXAMPLES = 3
NUMERIC = re.compile(r"(?:float|int|uint)(?:8|16|32|64)?|double|bfloat16")


def _text(value, maximum=256, multiline=False):
    if not isinstance(value, str):
        return None
    value = value.strip(TRIM)
    if not value or len(value) > maximum or any((ord(c) < 32 and not (multiline and c in "\n\t")) or 127 <= ord(c) <= 159 for c in value):
        return None
    return value


def _safe_key(key):
    return (_text(key) is not None and not key.startswith(("/", "~")) and "\\" not in key and ":" not in key
            and all(part not in {"", ".", ".."} for part in key.split("/"))
            and not any(char in key for char in "()[]{}'\"`;$"))


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON property")
        result[key] = value
    return result


def _json(content):
    def constant(_):
        raise ValueError("non-finite JSON constant")
    return json.loads(content, object_pairs_hook=_pairs, parse_constant=constant)


def _request(payload):
    if not isinstance(payload, dict) or set(payload) != {"path", "expires_at"}:
        raise AgentError("INVALID_DATASET_INSPECTION", "数据集检查需要路径及过期时间")
    path = payload["path"]
    if (_text(path, 4096) != path or not isinstance(path, str) or not path.startswith("/")
            or ".." in path.split("/")):
        raise AgentError("INVALID_DATASET_INSPECTION", "数据集路径必须是允许目录内的绝对路径")
    try:
        expiry = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        if expiry.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("UTC required")
    except (TypeError, AttributeError, ValueError) as exc:
        raise AgentError("INVALID_DATASET_INSPECTION", "数据集检查需要 UTC 过期时间") from exc
    if expiry <= datetime.now(timezone.utc):
        raise AgentError("DATASET_INSPECT_EXPIRED", "数据集检查已过期，请重新读取")
    return path


class _Reader:
    def __init__(self, path, roots, stack):
        # Resolve once for root authorization, then use directory descriptors
        # with NOFOLLOW for each descendant to reject symlink swaps/escapes.
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise AgentError("DATASET_INSPECT_FAILED", "当前系统缺少安全的目录读取支持")
        target = allowed_path(path, roots)
        trusted = [Path(root).resolve(strict=True) for root in roots]
        root = max((r for r in trusted if target == r or r in target.parents), key=lambda r: len(r.parts))
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        # A lexical meta symlink is never followed, including links inside roots.
        allowed_path(str(target / "meta"), roots)
        descriptor = os.open("/", flags)
        try:
            # Walk the trusted root's ancestors too: opening an absolute root
            # once would still follow a swapped ancestor symlink. Close each
            # parent immediately so a deep path cannot exhaust file descriptors.
            for part in (*root.parts[1:], *target.relative_to(root).parts, "meta"):
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
        self.meta = descriptor
        stack.callback(os.close, descriptor)

    def read(self, name, maximum, *, optional=False):
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=self.meta)
        except FileNotFoundError:
            if optional:
                return None
            raise
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AgentError("DATASET_INSPECT_INVALID_METADATA", "元信息必须是普通文件，不能是链接或设备")
            if before.st_size > maximum:
                if optional:
                    return None
                raise AgentError("DATASET_INSPECT_LIMIT", "数据集元信息超过读取预算")
            content = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
            if len(content) > maximum:
                raise AgentError("DATASET_INSPECT_LIMIT", "数据集元信息超过读取预算")
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(content) != after.st_size:
                raise AgentError("DATASET_INSPECT_FAILED", "元信息在读取期间发生变化，请重试")
            return content


def _tokens(key):
    return "_" + re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") + "_"


def _arm(key):
    key = _tokens(key)
    left = re.search(r"_(?:left|robot0|arm0)_", key) is not None
    right = re.search(r"_(?:right|robot1|arm1)_", key) is not None
    return "left" if left and not right else "right" if right and not left else None


def _kind(key):
    key = _tokens(key)
    if re.search(r"_(?:gripper|grip|finger)_", key):
        return "gripper"
    if re.search(r"_(?:joints?|joint\d+)_", key):
        return "joint"
    if re.search(r"_(?:rot|rotation|quaternion|quat|rotvec|rot6d)_", key) or "_axis_angle_" in key:
        return "rotation"
    if re.search(r"_(?:pos|position|translation)_", key):
        return "position"
    return "other"


def _representation(labels):
    kinds = {_kind(label) for label in labels}
    if "joint" in kinds and kinds <= {"joint", "gripper"}:
        return "joint"
    if kinds <= {"position", "rotation", "gripper"} and {"position", "rotation"} <= kinds:
        return "eef"
    return None


def _rotation(labels):
    encodings = set()
    rotations = [label for label in labels if _kind(label) == "rotation"]
    for label in rotations:
        token = _tokens(label)
        if "_rotation_6d_" in token or "_rot6d_" in token:
            encodings.add("rotation_6d")
        elif "_axis_angle_" in token or "_rotvec_" in token:
            encodings.add("axis_angle")
        elif "_quaternion_" in token or "_quat_" in token:
            encodings.add("quaternion")
        else:
            return None
    return next(iter(encodings)) if len(encodings) == 1 else None


def _field_annotations(info, features, warn):
    """Describe each original numeric field without choosing a training layout."""
    def label(value, choices):
        return value if type(value) is str and value in choices else None
    declared = info.get("field_annotations")
    declared = {row.get("key"): row for row in declared if isinstance(row, dict) and isinstance(row.get("key"), str)} if isinstance(declared, list) and len(declared) <= MAX_FEATURES else {}
    result = []
    for feature in features:
        key, names, shape = feature["key"], feature["names"], feature["shape"]
        if not NUMERIC.fullmatch(feature["dtype"]) or not _safe_key(key) or not shape:
            continue
        raw = info["features"][key]
        explicit = declared.get(key, {})
        token = _tokens(key)
        component = _kind(key)
        role = next((value for value in (explicit.get("role"), raw.get("role"), raw.get("semantic")) if label(value, {"state", "action"})), None)
        if role is None:
            if re.search(r"_(?:actions?|targets?|commands?|cmd)_", token):
                role = "action"
            elif key in {"state", "observation.state"} or (
                    component != "other" and (key.startswith("observation.") or _arm(key) is not None or re.search(r"_(?:eef|joints?|gripper)_", token))):
                role = "state"
        if role is None:
            continue
        component = explicit.get("component", raw.get("component", component))
        component = label(component, {"position", "rotation", "joint", "gripper", "other"}) or "other"
        # A standalone gripper is not an EEF pose, even when its key includes
        # "eef" or the dataset declares EEF movements for the arm.
        inferred_rep = "gripper" if component == "gripper" else "joint" if _kind(key) == "joint" else "eef" if "_eef_" in token else _representation(names)
        rep = next((value for value in (explicit.get("representation"), raw.get("representation"))
                    if label(value, {"eef", "joint", "gripper"})), None)
        if rep is None and component != "gripper":
            rep = next((value for value in (raw.get(role + "_representation"), info.get(role + "_representation"))
                        if label(value, {"eef", "joint"})), None)
        if rep == "gripper" and component != "gripper":
            warn("字段 " + key + " 的夹爪开度表示与分量声明冲突，请手动确认源字段标签。")
            rep = "unknown"
        elif rep and inferred_rep and rep != inferred_rep and component != "gripper":
            warn("字段 " + key + " 的表示声明与名称冲突，请手动确认源字段标签。")
            rep = "unknown"
        rep = rep or inferred_rep or "unknown"
        mode = next((value for value in (explicit.get("value_mode"), raw.get("value_mode"), raw.get(role + "_mode")) if label(value, {"absolute", "delta"})), None)
        # A dataset-wide movement declaration says nothing about a separate
        # gripper field, nor a packed vector mixing pose and gripper commands.
        mixed_gripper = any(_kind(name) == "gripper" for name in names)
        if mode is None and component != "gripper" and not mixed_gripper:
            mode = label(info.get(role + "_mode"), {"absolute", "delta"})
        origin_relative = bool(re.search(r"_(?:wrt_start|from_start|episode_start|relative_start)_", token))
        if origin_relative:
            if mode == "delta":
                mode = None
            warn("字段 " + key + " 以起点为参考，不能自动等同逐步增量；请确认数值语义。")
        elif mode is None:
            modes = {value for marker, value in (("delta", "delta"), ("increment", "delta"), ("incremental", "delta"), ("absolute", "absolute"), ("abs", "absolute")) if "_" + marker + "_" in token}
            if len(modes) == 1:
                mode = next(iter(modes))
        arm = explicit.get("arm", raw.get("arm", _arm(key)))
        annotation = {"key": key, "role": role, "representation": rep,
                       "value_mode": mode or "unknown", "component": component,
                       "arm": label(arm, {"left", "right", "shared"}) or "unknown",
                       "shape": list(shape), "dtype": feature["dtype"], "names": list(names)}
        if role == "action":
            horizon = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else None
            if horizon is not None:
                annotation["horizon"] = horizon
            else:
                warn("动作字段 " + key + " 的形状不是 [D] 或 [H,D]，未推断 horizon，请核对原始数据。")
            if "horizon" in explicit or "horizon" in raw:
                declared_horizon = explicit.get("horizon", raw.get("horizon"))
                if type(declared_horizon) is not int or declared_horizon != horizon:
                    warn("动作字段 " + key + " 的 horizon 声明与原始形状冲突，已沿用形状证据，请核对元信息。")
        result.append(annotation)
    if any(row["representation"] == "unknown" or row["value_mode"] == "unknown" for row in result):
        warn("部分源字段的 EEF/Joint/夹爪开度或绝对/增量语义无法从元信息确定，已标为未知，请确认标签；不会按维数猜测。")
    return result


def _infer(info, features, tasks, warn):
    suggestions, processing, evidence = {}, {}, []
    blocked = set()
    by_key = {row["key"]: row for row in features}
    def put(field, value, reason, keys=(), kind="inferred"):
        if value is None or field in blocked:
            return
        target, key = (processing, field[11:]) if field.startswith("processing.") else (suggestions, field)
        target[key] = value
        evidence.append({"field": field, "kind": kind, "reason": reason, "keys": list(keys)})
    def withdraw(*fields):
        for field in fields:
            blocked.add(field)
            target, key = (processing, field[11:]) if field.startswith("processing.") else (suggestions, field)
            target.pop(key, None)
        evidence[:] = [row for row in evidence if row["field"] not in blocked]
    put("format", "lerobot", "已读取 LeRobot meta/info.json 的特征声明", kind="metadata")
    put("feature_keys", list(by_key), "实际元信息中的有效字段名", by_key, "metadata")
    robot = _text(info.get("robot_type"), 100)
    put("robot_type", robot, "元信息直接声明机器人类型", kind="metadata")
    fps = info.get("fps")
    if type(fps) in (int, float) and 0 < fps <= 1000:
        put("fps", float(fps), "元信息直接声明采样频率", kind="metadata")
    direct_enums = {"arm_mode": {"single", "bimanual"}, "action_representation": {"eef", "joint"},
                    "action_mode": {"absolute", "delta"}, "rotation_representation": {"axis_angle", "quaternion", "rotation_6d", "none"}}
    for key, choices in direct_enums.items():
        if type(info.get(key)) is str and info[key] in choices:
            put(key, info[key], "元信息直接声明该语义，仍需与数据生成方核对", kind="metadata")
    if tasks:
        put("task_description", "\n".join(tasks)[:2000], "从任务元信息提取的任务文本示例，不能代替逐样本 prompt", kind="metadata")
    numeric = {key: row for key, row in by_key.items() if NUMERIC.fullmatch(row["dtype"]) and _safe_key(key)}
    candidates = [key for key in numeric if re.search(r"(?:^|[._/])actions?$", key)]
    action_key = next((key for key in ("actions", "action") if key in candidates), candidates[0] if len(candidates) == 1 else None)
    action = numeric.get(action_key)
    if action and 1 <= len(action["shape"]) <= 2 and action["shape"][-1] <= 512:
        put("processing.action_key", action_key, "动作字段名明确且具有一维或预分块二维数值形状", [action_key])
        put("action_dim", action["shape"][-1], "动作形状末轴直接给出每步宽度；维数不证明控制语义", [action_key], "metadata")
        names = action["names"]
        if names and len(names) == action["shape"][-1] and len(set(names)) == len(names):
            rep = _representation(names)
            if rep and "action_representation" in suggestions and suggestions["action_representation"] != rep:
                withdraw("action_representation", "rotation_representation", "processing.action_layout")
                warn("动作元信息声明与完整坐标名的 EEF/Joint 语义矛盾，相关默认值已撤回，请确认数据生成定义。")
            if "action_representation" not in suggestions:
                put("action_representation", rep, "完整动作坐标名明确标注了关节或末端字段", [action_key])
            layout = []
            for name in names:
                arm, kind = _arm(name), _kind(name)
                if arm is None and suggestions.get("arm_mode") == "single":
                    arm = "left"
                if arm is None or kind == "other":
                    layout = []
                    break
                if layout and (layout[-1]["arm"], layout[-1]["kind"]) == (arm, kind):
                    layout[-1]["width"] += 1
                else:
                    layout.append({"arm": arm, "kind": kind, "width": 1})
            encoding = "none" if rep == "joint" else _rotation(names)
            valid_layout = bool(layout) and rep is not None
            for arm in {row["arm"] for row in layout}:
                blocks = {row["kind"]: row["width"] for row in layout if row["arm"] == arm}
                if rep == "eef" and (blocks.get("position") != 3 or blocks.get("rotation") not in {3, 4, 6}
                        or (encoding is not None and blocks.get("rotation") != {"axis_angle": 3, "quaternion": 4, "rotation_6d": 6}.get(encoding))):
                    valid_layout = False
                if rep == "joint" and "joint" not in blocks:
                    valid_layout = False
            if (valid_layout and len(layout) <= 16 and all(row["width"] <= 128 for row in layout)
                    and len({(r["arm"], r["kind"]) for r in layout}) == len(layout)):
                arms = {row["arm"] for row in layout}
                named_arm_mode = "bimanual" if arms == {"left", "right"} else "single"
                if "arm_mode" in suggestions and suggestions["arm_mode"] != named_arm_mode:
                    withdraw("arm_mode", "processing.action_layout")
                    warn("机械臂数量声明与完整动作坐标名矛盾，机械臂及动作布局默认值已撤回。")
                if arms == {"right"}:
                    warn("动作只标注右臂，需确认单臂 canonical 映射；未自动重命名机械臂。")
                else:
                    put("processing.action_layout", layout, "完整动作坐标名按原顺序连续分组；未重排或转换数值", [action_key])
                if "arm_mode" not in suggestions:
                    put("arm_mode", "bimanual" if arms == {"left", "right"} else "single" if arms == {"left"} else None,
                        "完整动作字段明确标注了参与控制的机械臂", [action_key])
            rotation = encoding if valid_layout else "none" if rep == "joint" else None
            if rotation and "rotation_representation" in suggestions and rotation != suggestions["rotation_representation"]:
                withdraw("rotation_representation", "processing.action_layout")
                warn("旋转编码声明与完整动作坐标名矛盾，相关默认值已撤回。")
            if "rotation_representation" not in suggestions:
                put("rotation_representation", rotation, "完整动作坐标名明确声明旋转编码，未根据维数猜测", [action_key])
        if "action_representation" not in suggestions:
            warn("动作维数不能证明 EEF/Joint 或单/双臂；请确认动作坐标名及控制语义。")
    else:
        warn("未找到唯一且形状明确的动作字段，动作映射需要手动确认。")
    prompt = next((key for key in ("task", "prompt", "language_instruction", "instruction")
                   if key in by_key and by_key[key]["dtype"] in {"string", "str"} and _safe_key(key)), None)
    if prompt is None and "task_index" in by_key and tasks:
        prompt = "task"
    put("processing.prompt_key", prompt, "明确的文本字段或 task_index/tasks 元信息支持任务文本映射", [prompt] if prompt in by_key else ["task_index"] if prompt else [])
    unified = next((key for key in ("observation.state", "state") if key in numeric), None)
    state_candidates = [unified] if unified else [key for key in numeric if key != action_key and _kind(key) != "other"
        and not re.search(r"_(?:actions?|targets?|commands?|cmd)_", _tokens(key))
        and (_arm(key) is not None or re.search(r"_(?:eef|joints?|gripper)_", _tokens(key)))]
    fields, histories = [], []
    for key in state_candidates:
        row = numeric[key];shape = row["shape"]
        if len(shape) not in (1, 2) or not 1 <= shape[-1] <= 128 or (len(shape) == 2 and shape[0] > 64):
            continue
        fields.append({"key": key, "width": shape[-1], "arm": _arm(key) or "shared", "kind": _kind(key)})
        histories.append(shape[0] if len(shape) == 2 else 1)
    if fields and len(fields) <= 32 and len(set(histories)) == 1 and sum(row["width"] for row in fields) <= 512:
        keys = [row["key"] for row in fields]
        put("processing.state_fields", fields, "优先选择统一状态向量，否则列出语义明确的机器人数值字段；顺序是元信息原顺序", keys)
        put("processing.state_history", histories[0], "由状态特征形状给出历史轴候选，需要确认二维首轴确为时间", keys)
        if unified:
            put("state_dim", numeric[unified]["shape"][-1], "统一状态字段末轴给出原始宽度，不含历史轴展开", [unified], "metadata")
        labels = numeric[unified]["names"] if unified else keys
        put("processing.state_representation", _representation(labels), "状态字段或坐标名明确标注了表示方式；与动作表示分开推断", keys)
        if "arm_mode" not in suggestions and {row["arm"] for row in fields} >= {"left", "right"}:
            put("arm_mode", "bimanual", "状态字段明确包含左右两臂标识；仍需核对动作布局", keys)
        warn("状态拼接顺序、历史轴和每个字段的归一化策略必须与统计参数核对，不能仅凭字段名确认。")
    elif state_candidates:
        warn("状态字段的维度、数量或历史轴不一致，未生成可直接使用的状态拼接候选。")
    cameras, processing_cameras, names_seen = [], [], set()
    for row in features:
        if row["dtype"] not in {"image", "video"}:
            continue
        key = row["key"];raw = info["features"][key]
        detail = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        depth = raw.get("semantic") == "depth" or detail.get("is_depth") is True or re.search(r"_depth_", _tokens(key)) is not None
        kind = "depth" if depth else "rgb"
        name = re.split(r"[./]", key)[-1][:90]
        name = name or "camera"
        if name in names_seen:
            name += "_" + str(len(cameras) + 1)
        names_seen.add(name)
        cameras.append({"name": name, "key": key, "kind": kind})
        shape = row["shape"][-3:] if len(row["shape"]) == 4 else row["shape"]
        channel = 1 if depth else 3
        layout = ("chw" if shape[0] == channel else "hwc") if len(shape) == 3 and ((shape[0] == channel) != (shape[-1] == channel)) else None
        if layout and _safe_key(key):
            processing_cameras.append({"key": key, "kind": kind, "layout": layout})
        else:
            warn("部分视觉字段无法唯一判断 CHW/HWC 布局，请手动核对。")
        if len(row["shape"]) == 4:
            warn("视觉元信息包含额外时间轴，需要显式选择图像帧；本次不会自动丢帧。")
    if cameras and len(cameras) <= 16:
        put("cameras", cameras, "图像/视频 dtype 提供视觉字段证据；depth 名称或标记只用于保守语义提示", [row["key"] for row in cameras])
    elif cameras:
        warn("视觉字段超过支持的登记数量，请手动选择。")
    if processing_cameras and len(processing_cameras) <= 16:
        put("processing.cameras", processing_cameras, "仅在通道轴唯一时给出布局候选；槽位、深度范围和选帧需人工确认", [row["key"] for row in processing_cameras])
    if len(cameras) > 3:
        warn("组合处理当前最多使用三路 canonical 视觉槽，请明确选择需要训练的视觉。")
    warn("单位、坐标系、夹爪范围及归一化策略不会从维数或机器人名称自动推断。")
    annotations = _field_annotations(info, features, warn)
    if annotations:
        put("field_annotations", annotations, "仅标记原始状态与动作字段及其语义，不定义拼接顺序、归一化或模型映射", [row["key"] for row in annotations])
    return suggestions, processing, evidence


def inspect_dataset(config: dict, payload: dict) -> dict:
    """Return only the approved DTO, using this Agent's allowed roots."""
    path = _request(payload)
    warnings = []
    def warn(message):
        if message not in warnings and len(warnings) < 32:
            warnings.append(message)
    try:
        with ExitStack() as stack:
            reader = _Reader(path, config["roots"], stack)
            content = reader.read("info.json", MAX_INFO_BYTES)
            info = _json(content)
            raw_features = info.get("features") if isinstance(info, dict) else None
            if not isinstance(raw_features, dict) or not raw_features:
                raise AgentError("DATASET_INSPECT_INVALID_METADATA", "LeRobot 元信息缺少有效 features")
            if len(raw_features) > MAX_FEATURES:
                raise AgentError("DATASET_INSPECT_LIMIT", "数据集字段数超过读取预算")
            features = []
            for key, raw in raw_features.items():
                shape = raw.get("shape") if isinstance(raw, dict) else None
                dtype = _text(raw.get("dtype"), 64) if isinstance(raw, dict) else None
                if (_text(key) != key or dtype is None or not isinstance(shape, list) or len(shape) > 5
                        or any(type(n) is not int or not 1 <= n <= 2**31 - 1 for n in shape)):
                    warn("部分特征元信息类型或尺寸无效，已从返回摘要中排除。")
                    continue
                names = raw.get("names")
                if names is None:
                    names = []
                if not isinstance(names, list) or len(names) > 512 or any(_text(n) != n for n in names):
                    names = []
                    warn("部分坐标名称不是有界文本列表，未用于自动推断。")
                features.append({"key": key, "dtype": dtype, "shape": shape, "names": names})
            if not features:
                raise AgentError("DATASET_INSPECT_INVALID_METADATA", "元信息没有可返回的有效特征声明")
            task_bytes = reader.read("tasks.jsonl", MAX_TASKS_BYTES, optional=True)
        examples = []
        if task_bytes is not None:
            lines = task_bytes.splitlines()
            if len(lines) > MAX_TASK_LINES:
                warn("任务元信息条目较多，只读取固定数量的任务示例。")
            for line in lines[:MAX_TASK_LINES]:
                if not line.strip():
                    continue
                if len(line) > MAX_TASK_LINE_BYTES:
                    warn("部分任务元信息单行过大，已跳过。")
                    continue
                try:
                    row = _json(line)
                    text = _text(row.get("task"), 2000, True) if isinstance(row, dict) else None
                except (ValueError, UnicodeError, RecursionError):
                    text = None
                if text and text not in examples:
                    examples.append(text)
                if len(examples) == MAX_TASK_EXAMPLES:
                    break
        elif task_bytes is None:
            warn("未读取可选任务文本，可能未提供或超过固定读取预算。")
        if not examples:
            text = _text(info.get("task_description"), 2000, True)
            if text:
                examples.append(text)
        suggestions, processing, evidence = _infer(info, features, examples, warn)
        result = {"schema_version": 1, "path": path,
                  "source": {"info_sha256": hashlib.sha256(content).hexdigest(), "tasks_sha256": hashlib.sha256(task_bytes).hexdigest() if task_bytes is not None else None},
                  "features": features, "suggestions": suggestions, "processing": processing,
                  "evidence": evidence, "warnings": warnings}
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_RESULT_BYTES:
            raise AgentError("DATASET_INSPECT_LIMIT", "数据集检查摘要超过返回预算")
        return normalize_dataset_inspection(result, expected_path=path)
    except AgentError as exc:
        if exc.code == "PATH_UNAVAILABLE":
            raise AgentError("DATASET_INSPECT_NOT_FOUND", "数据集目录或必需元信息不存在") from exc
        raise
    except FileNotFoundError as exc:
        raise AgentError("DATASET_INSPECT_NOT_FOUND", "数据集目录或必需元信息不存在") from exc
    except PermissionError as exc:
        raise AgentError("DATASET_INSPECT_PERMISSION_DENIED", "没有读取该数据集元信息的权限") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AgentError("DATASET_INSPECT_INVALID_METADATA", "元信息不是受支持的有界 JSON 结构") from exc
    except OSError as exc:
        raise AgentError("DATASET_INSPECT_INVALID_METADATA", "元信息路径不可安全读取，链接或特殊文件不受支持") from exc
