"""Conservative input hints from processor source, without calling user code.

A hint describes the default factory's statically visible Repack inputs. It
does not prove the contents, action units or tensor shapes of a real dataset.
Dynamic mapping construction and ambiguous control flow deliberately get no
hint; the normal training preflight remains authoritative.
"""

import ast
from functools import lru_cache
import inspect
import json
import math
from pathlib import Path
import re

if __package__:
    from .composition_spec import canonical_text_trim, normalize_composition, normalize_field_annotations
else:
    from composition_spec import canonical_text_trim, normalize_composition, normalize_field_annotations


MAX_SOURCE_BYTES = 1024 * 1024
MAX_AST_NODES = 100_000
MAX_AST_DEPTH = 96
MAX_FEATURES = 256
MAX_CAMERAS = 64
_REQUIRED = {"arm_mode", "action_representation", "camera_keys", "required_features", "training_stage", "evidence"}
_OPTIONAL = {"action_mode", "rotation_representation", "model_family"}


def _text(value):
    return (isinstance(value, str) and 1 <= len(value) <= 256 and canonical_text_trim(value) == value
            and not any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value))


def normalize_dataset_contract(value):
    """Validate the bounded optional Agent catalog field at the API boundary."""
    if not isinstance(value, dict) or not _REQUIRED <= set(value) or set(value) - _REQUIRED - _OPTIONAL:
        raise ValueError("invalid dataset contract fields")
    enums = {
        "arm_mode": {"single", "bimanual"},
        "action_representation": {"eef", "joint"},
        "training_stage": {"sft", "value", "offline_rl"},
        "evidence": {"static_repack"},
        "action_mode": {"absolute", "delta", "unknown"},
        "rotation_representation": {"axis_angle", "quaternion", "rotation_6d", "none", "unknown"},
        "model_family": {"standard", "memory", "hybrid"},
    }
    result = {}
    for key, choices in enums.items():
        if key in value:
            if not isinstance(value[key], str) or value[key] not in choices:
                raise ValueError("invalid dataset contract enum")
            result[key] = value[key]
    for key, maximum in (("camera_keys", MAX_CAMERAS), ("required_features", MAX_FEATURES)):
        items = value[key]
        if (not isinstance(items, list) or not 1 <= len(items) <= maximum or not all(_text(item) for item in items)
                or len(set(items)) != len(items)):
            raise ValueError("invalid dataset contract feature keys")
        result[key] = sorted(items)
    if not set(result["camera_keys"]) <= set(result["required_features"]):
        raise ValueError("dataset contract cameras must be required inputs")
    return result


def verify_registered_dataset_profile(profile, info, *, derived_features=(), selected_composition=None,
                                      warnings=None):
    """Check declared fields against real LeRobot metadata, returning a copy.

    Robot identity, task prose and units are declarations, not proof from
    info.json. Per-field shapes are source facts; concatenated model widths
    belong to the run's processor and are verified separately.

    A valid declared fps takes precedence over valid metadata for this training
    process. Report a difference through the optional warning list, including
    both source values and the effective registered value. This validator never
    mutates either input, resamples data, rewrites timestamps or writes files;
    the caller applies the registered value to the loader's in-memory metadata.
    """
    if selected_composition is not None:
        selected_composition = normalize_composition(selected_composition)

    def fail(message, code="DATASET_PROFILE_MISMATCH"):
        raise ValueError(code + ": " + message)

    try:
        serialized = json.dumps(profile, allow_nan=False, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > 128 * 1024:
            fail("数据集登记描述过大")
        clean = json.loads(serialized)
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError) as exc:
        raise ValueError("DATASET_PROFILE_MISMATCH: 数据集登记描述必须为受限 JSON") from exc
    required = {"schema_version", "format", "robot_type", "arm_mode", "action_representation", "cameras", "task_description"}
    optional = {"action_dim", "state_dim", "fps", "action_mode", "rotation_representation", "feature_keys", "processing", "field_annotations"}
    if not isinstance(clean, dict) or not required <= set(clean) or set(clean) - required - optional:
        fail("数据集登记描述字段无效")
    if type(clean["schema_version"]) is not int or clean["schema_version"] != 1:
        fail("不支持的数据集登记描述版本")
    if clean["format"] != "lerobot":
        fail("当前训练预检仅支持 LeRobot 数据集描述")
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        fail("LeRobot meta/info.json 缺少 features")
    if (not isinstance(derived_features, (list, tuple)) or any(key != "task" for key in derived_features)
            or len(derived_features) > 1 or (derived_features and not isinstance(info["features"].get("task_index"), dict))):
        fail("派生任务字段缺少有效的 LeRobot task_index 证据")
    for key, choices in (("arm_mode", {"single", "bimanual"}), ("action_representation", {"eef", "joint", "unknown"}),
                         ("action_mode", {"absolute", "delta", "unknown"}),
                         ("rotation_representation", {"axis_angle", "quaternion", "rotation_6d", "none", "unknown"})):
        if key in clean and (not isinstance(clean[key], str) or clean[key] not in choices):
            fail("数据集登记描述枚举无效：" + key)
    for key, maximum in (("robot_type", 100), ("task_description", 2000)):
        value = clean[key]
        if (not isinstance(value, str) or not 1 <= len(value) <= maximum or value.strip() != value
                or any((ord(char) < 32 and not (key == "task_description" and char in "\n\t"))
                       or 127 <= ord(char) <= 159 for char in value)):
            fail("数据集登记文本无效：" + key)
    cameras = clean["cameras"]
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 16:
        fail("数据集相机登记数量无效")
    names, keys = set(), set()
    features = info["features"]
    for camera in cameras:
        if (not isinstance(camera, dict) or set(camera) != {"name", "key", "kind"}
                or not _text(camera.get("name")) or len(camera["name"]) > 100 or not _text(camera.get("key"))
                or camera.get("kind") not in ("rgb", "depth") or camera["name"] in names or camera["key"] in keys):
            fail("数据集相机登记字段无效")
        names.add(camera["name"]);keys.add(camera["key"])
        feature = features.get(camera["key"])
        if not isinstance(feature, dict) or feature.get("dtype") not in ("image", "video"):
            fail("登记的相机字段不存在或不是 image/video：" + camera["key"])
        detail = feature.get("info") if isinstance(feature.get("info"), dict) else {}
        depth = bool(re.search(r"(?:^|[._/])depth(?:$|[._/])", camera["key"], re.I)
                     or detail.get("is_depth") is True or feature.get("semantic") == "depth")
        if camera["kind"] == "depth" and not depth:
            fail("相机字段缺少明确的深度语义：" + camera["key"])
        if camera["kind"] == "rgb" and depth:
            fail("深度字段被登记为 RGB：" + camera["key"])
    declared = clean.get("feature_keys", [])
    if (not isinstance(declared, list) or len(declared) > MAX_FEATURES or not all(_text(key) for key in declared)
            or len(set(declared)) != len(declared)):
        fail("登记的数据字段列表无效")
    for key in declared:
        if key not in features and key not in derived_features:
            fail("登记的数据字段不在 meta/info.json features 中：" + key)

    def dimension(feature):
        shape = feature.get("shape") if isinstance(feature, dict) else None
        return shape[-1] if isinstance(shape, list) and shape and all(type(item) is int and item > 0 for item in shape) else None

    for key in ("action_dim", "state_dim"):
        if key in clean and (type(clean[key]) is not int or not 1 <= clean[key] <= 512):
            fail("登记维度无效：" + key)
    annotations = None
    if "field_annotations" in clean:
        try:
            annotations = normalize_field_annotations(clean["field_annotations"])
        except ValueError as exc:
            fail(str(exc))
        for row in annotations:
            feature = features.get(row["key"])
            if not isinstance(feature, dict) or not re.fullmatch(r"(?:float|int|uint)(?:8|16|32|64)?|double|bfloat16", str(feature.get("dtype", ""))):
                fail("状态或动作标签必须对应真实数值字段：" + row["key"])
            shape = feature.get("shape")
            if not isinstance(shape, list) or not shape or any(type(size) is not int or size <= 0 for size in shape):
                fail("标签字段缺少真实形状：" + row["key"])
            if "shape" in row and row["shape"] != shape:
                fail("字段标签形状与真实元信息不一致：" + row["key"])
            if "horizon" in row:
                horizon = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else None
                if horizon is None:
                    fail("动作 horizon 仅支持真实 [D] 或 [H,D] 字段：" + row["key"])
                if row["horizon"] != horizon:
                    fail("动作字段 horizon 与真实元信息不一致：" + row["key"])
            if "dtype" in row and row["dtype"] != feature.get("dtype"):
                fail("字段标签类型与真实元信息不一致：" + row["key"])
            if "names" in row:
                actual_names = feature.get("names")
                actual_names = (actual_names if isinstance(actual_names, list) and len(actual_names) <= 512
                                and all(_text(name) for name in actual_names) else [])
                if row["names"] != actual_names:
                    fail("字段标签坐标名与真实元信息不一致：" + row["key"])
        clean["field_annotations"] = annotations
    historical = None
    if clean.get("processing") is not None:
        # Historical assets retain their saved plan for import. It does not
        # constrain which fields/order/history the current training selects.
        historical = normalize_composition(clean["processing"])
        clean["processing"] = historical
    action_labels = [row for row in annotations or [] if row["role"] == "action"]
    action_key = (action_labels[0]["key"] if len(action_labels) == 1 else
                  historical["action_key"] if historical and "actions" not in features else "actions")
    if ("action_dim" in clean and (annotations is None or len(action_labels) == 1)
            and dimension(features.get(action_key)) != clean["action_dim"]):
        fail("登记 action_dim 与实际 actions 特征维度不一致")
    if "state_dim" in clean and annotations is None and historical is None:
        vectors = [features[key] for key in ("state", "observation.state") if key in features]
        if not vectors and selected_composition is None:
            fail("数据集没有统一 state/observation.state 特征，暂不能核验登记 state_dim；请留空或使用统一状态向量", "DATASET_PROFILE_UNVERIFIABLE")
        if any(dimension(feature) != clean["state_dim"] for feature in vectors):
            fail("登记 state_dim 与实际统一状态向量维度不一致")
    if "fps" in clean:
        supplied, actual = clean["fps"], info.get("fps")
        try:
            valid = (type(supplied) in (int, float) and 0 < supplied <= 1000 and math.isfinite(supplied)
                     and type(actual) in (int, float) and actual > 0 and math.isfinite(actual))
        except OverflowError:
            valid = False
        if not valid:
            fail("登记 fps 必须是 0–1000 范围内的有限正数，实际 meta/info.json fps 也必须是有限正数")
        if not math.isclose(supplied, actual, rel_tol=1e-9, abs_tol=1e-6) and warnings is not None:
            warnings.append({"code": "DATASET_FPS_MISMATCH", "supplied": supplied, "metadata": actual,
                "effective": supplied, "source": "registered",
                "message": f"登记采样率 {supplied:g} Hz 与 meta/info.json 的 {actual:g} Hz 不一致；本次训练进程内采用登记采样率 {supplied:g} Hz，不自动重采样、修改时间戳或写回元信息文件。"})
    return clean


@lru_cache(maxsize=4)
def _classes(source):
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return {}
    tree = ast.parse(source)
    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            return {}
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in result:
                result[node.name] = None
            else:
                result[node.name] = node
    return result


def _class_node(cls, root):
    filename = inspect.getsourcefile(cls)
    if not filename or "." in cls.__qualname__:
        return None
    path = Path(filename).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        return None
    source = path.read_bytes()
    if len(source) > MAX_SOURCE_BYTES:
        return None
    return _classes(source.decode("utf-8")).get(cls.__name__)


def _name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _assigned_field(nodes, name):
    """Read an inherited field's literal AST, never its default factory."""
    for node in nodes:
        values = []
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == name:
                values.append(statement.value)
            elif isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
                values.append(statement.value)
        if values:
            if len(values) != 1:
                return None
            value = values[0]
            if isinstance(value, ast.Call) and _name(value.func) == "field" and not value.args:
                defaults = [item for item in value.keywords if item.arg in {"default", "default_factory"}]
                if len(defaults) != 1:
                    return None
                value = defaults[0].value
                if defaults[0].arg == "default_factory":
                    if (not isinstance(value, ast.Lambda) or value.args.args or value.args.posonlyargs
                            or value.args.kwonlyargs or value.args.vararg or value.args.kwarg):
                        return None
                    value = value.body
            return value
    return None


def _mapping(node, prefix="", depth=0):
    """Flatten only small nested string dictionaries, rejecting ** and calls."""
    if not isinstance(node, ast.Dict) or not 1 <= len(node.keys) <= MAX_FEATURES or depth > 4:
        return None
    result = {}
    keys = set()
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not _text(key.value) or key.value in keys:
            return None
        keys.add(key.value)
        output = prefix + key.value
        if isinstance(value, ast.Constant) and _text(value.value):
            result[output] = value.value
        elif isinstance(value, ast.Dict):
            nested = _mapping(value, output + ".", depth + 1)
            if nested is None:
                return None
            result.update(nested)
        else:
            return None
        if len(result) > MAX_FEATURES:
            return None
    return result


def _repack_mapping(nodes, create):
    """Prove one direct assignment is used by the final returned DataConfig."""
    returns = [node for node in ast.walk(create) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0] not in create.body:
        return None
    returned = returns[0].value
    if (not isinstance(returned, ast.Call) or not isinstance(returned.func, ast.Attribute)
            or not isinstance(returned.func.value, ast.Name) or returned.func.value.id != "dataclasses"
            or returned.func.attr != "replace"):
        return None
    repack_values = [item.value for item in returned.keywords if item.arg == "repack_transforms"]
    if len(repack_values) != 1 or not isinstance(repack_values[0], ast.Name):
        return None
    name = repack_values[0].id
    assignments = [node for node in ast.walk(create) if isinstance(node, (ast.Assign, ast.AnnAssign)) and
                   any(isinstance(target, ast.Name) and target.id == name for target in
                       (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    if len(assignments) != 1 or assignments[0] not in create.body:
        return None
    assignment = assignments[0]
    calls = [node for node in ast.walk(create) if isinstance(node, ast.Call) and _name(node.func) == "RepackTransform"]
    if len(calls) != 1 or calls[0] not in list(ast.walk(assignment.value)):
        return None
    # Any extra access/mutation of this variable makes its final value unclear.
    if len([node for node in ast.walk(create) if isinstance(node, ast.Name) and node.id == name]) != 2:
        return None
    call = calls[0]
    if len(call.args) != 1 or call.keywords:
        return None
    value = call.args[0]
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self" and value.attr == "mapping":
        value = _assigned_field(nodes, "mapping")
    return _mapping(value)


def contract_from_class_nodes(nodes):
    """Extract a hint from bounded class ASTs ordered like an already known MRO."""
    if not nodes or len(nodes) > 16 or any(not isinstance(node, ast.ClassDef) for node in nodes):
        return None
    pending = [(node, 0) for node in nodes]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            return None
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    if any(re.search(r"inference|compute.?norm|norm.?stats", node.name, re.I) for node in nodes):
        return None
    create = next((item for node in nodes for item in node.body
                   if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "create"), None)
    if not isinstance(create, ast.FunctionDef):
        return None
    mapping = _repack_mapping(nodes, create)
    if not mapping:
        return None
    outputs = set(mapping)
    eef = {int(match[1]) for key in outputs if (match := re.fullmatch(r"robot([01])_eef_pos", key))}
    joint = {int(match[1]) for key in outputs if (match := re.fullmatch(r"robot([01])_joint_(?:pos|position|positions)", key))}
    if bool(eef) == bool(joint):
        return None
    arms = eef or joint
    if arms not in ({0}, {0, 1}):
        return None
    camera_keys = sorted({value for key, value in mapping.items()
                          if re.search(r"(?:^|[._/])(?:rgb|depth|images?)(?:$|[._/])", key, re.I)})
    if not camera_keys or len(camera_keys) > MAX_CAMERAS:
        return None
    result = {"arm_mode": "bimanual" if len(arms) == 2 else "single",
              "action_representation": "eef" if eef else "joint", "camera_keys": camera_keys,
              "required_features": sorted(set(mapping.values())), "evidence": "static_repack",
              "training_stage": "offline_rl" if "is_positive" in outputs else "value" if "value_target" in outputs else "sft"}
    calls = {_name(node.func) for node in ast.walk(create) if isinstance(node, ast.Call)}
    if "BuildVideoTensor" in calls or any(re.fullmatch(r"UmiInputs\w*Video\w*", name) for name in calls):
        result["model_family"] = "memory"
    elif "TokenizeHybridInput" in calls:
        result["model_family"] = "hybrid"
    elif "ModelTransformFactory" in calls or "TokenizePrompt" in calls:
        result["model_family"] = "standard"
    try:
        return normalize_dataset_contract(result)
    except ValueError:
        return None


def static_dataset_contract(cls, code_dir):
    """Return a default input hint only for traceable, unambiguous class code."""
    try:
        root = Path(code_dir).resolve(strict=True)
        lineage = []
        for ancestor in cls.__mro__:
            if ancestor is object:
                break
            lineage.append(ancestor)
            if ancestor.__name__ == "DataConfigFactory":
                break
        if len(lineage) > 16:
            return None
        nodes = []
        for ancestor in lineage:
            node = _class_node(ancestor, root)
            if node is None:
                return None
            nodes.append(node)
        owner = next((ancestor for ancestor in lineage if "create" in vars(ancestor)), None)
        method = vars(owner).get("create") if owner else None
        if (not inspect.isfunction(method) or method.__name__ != "create"
                or method.__qualname__ != owner.__qualname__ + ".create"):
            return None
        owner_node = nodes[lineage.index(owner)]
        methods = [item for item in owner_node.body if isinstance(item, ast.FunctionDef) and item.name == "create"]
        if len(methods) != 1 or not methods[0].lineno <= method.__code__.co_firstlineno + len(methods[0].decorator_list) <= methods[0].end_lineno:
            return None
        return contract_from_class_nodes(nodes)
    except (OSError, ValueError, TypeError, SyntaxError, UnicodeError, RecursionError):
        return None
