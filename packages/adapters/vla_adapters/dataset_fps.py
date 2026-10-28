"""Run-scoped LeRobot FPS declarations without changing dataset files.

Resolution uses only the standard library. Runtime adaptation keeps the original
Metadata class and constructor, including its normal timestamp/video checks.
Only the matched instance's in-memory info dictionary receives the registered
FPS. This corrects a declared time base; it does not resample stored rows.
"""

import ast
from contextlib import contextmanager
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
import textwrap
import threading


MODULE = "lerobot.common.datasets.lerobot_dataset"
HOOK = "_vla_registered_fps_hook_v1"
LOCK = "_vla_registered_fps_lock_v1"
MAX_INFO_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _fail(code, message):
    raise ValueError(code + ": " + message)


def _finite_positive(value):
    try:
        return type(value) in (int, float) and value > 0 and math.isfinite(value)
    except OverflowError:
        return False


def _canonical(value, spec):
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        _fail("DATASET_FPS_INVALID", "FPS 策略需要实际绑定的数据集绝对路径")
    path = Path(value).resolve(strict=True)
    roots = spec.get("allowed_roots")
    if roots is not None and (not isinstance(roots, list) or not roots or not any(
            path.is_relative_to(Path(root).resolve(strict=True)) for root in roots)):
        _fail("DATASET_FPS_INVALID", "FPS 数据源超出 Agent 受管目录")
    return path


def _read_info(path, spec):
    source = _canonical(path / "meta" / "info.json", spec)
    descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INFO_BYTES:
            _fail("DATASET_FPS_INVALID", "FPS 元信息必须是受限大小的普通 JSON 文件")
        content = stream.read(MAX_INFO_BYTES + 1)
    if len(content) > MAX_INFO_BYTES:
        _fail("DATASET_FPS_INVALID", "FPS 元信息超过读取上限")

    def nonfinite(_value):
        _fail("DATASET_FPS_INVALID", "FPS 元信息包含非有限 JSON 数值")

    document = json.loads(content, parse_constant=nonfinite)
    if not isinstance(document, dict):
        _fail("DATASET_FPS_INVALID", "FPS 元信息必须是 JSON 对象")
    return document, hashlib.sha256(content).hexdigest()


def _roles(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
        _fail("DATASET_FPS_INVALID", "FPS 策略缺少实际输入资产")
    inputs, bindings = spec["inputs"], spec.get("dataset_bindings")
    if bindings is None:
        return ["dataset"] if "dataset" in inputs else []
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 64:
        _fail("DATASET_FPS_INVALID", "FPS 策略需要 1 到 64 组实际数据集绑定")
    roles = []
    for index, binding in enumerate(bindings):
        role = "dataset_" + str(index)
        asset = inputs.get(role)
        if (not isinstance(binding, dict) or not isinstance(asset, dict)
                or not isinstance(binding.get("dataset_asset_id"), str)
                or asset.get("id") != binding["dataset_asset_id"]):
            _fail("DATASET_FPS_INVALID", "FPS 数据集角色与本次绑定的资产身份不一致")
        roles.append(role)
    return roles


def resolve_registered_fps(spec):
    """Return fresh, deterministic per-role facts; never borrow another role's FPS.

    Historical requests without a registered FPS keep their previous validation
    scope. Missing/invalid metadata FPS is recorded as None in that case; an
    explicit declaration still requires usable original metadata for comparison.
    """
    policy, targets = [], {}
    for role in _roles(spec):
        asset = spec["inputs"][role]
        if not isinstance(asset, dict):
            _fail("DATASET_FPS_INVALID", "FPS 数据集资产描述无效")
        path = _canonical(asset.get("path"), spec)
        if not path.is_dir():
            _fail("DATASET_FPS_INVALID", "FPS 数据源不是数据集目录")
        profile = asset.get("dataset_profile")
        if profile is not None and not isinstance(profile, dict):
            _fail("DATASET_FPS_INVALID", "FPS 登记描述无效")
        registered = profile.get("fps") if profile else None
        if registered is not None and (not _finite_positive(registered) or registered > 1000):
            _fail("DATASET_FPS_INVALID", "登记 fps 必须为不大于 1000 的有限正数")
        document, digest = _read_info(path, spec)
        metadata = document.get("fps")
        if registered is not None and not _finite_positive(metadata):
            _fail("DATASET_FPS_INVALID", "已登记 fps 的数据集必须有可核验的有限正数元信息 fps")
        metadata = metadata if _finite_positive(metadata) else None
        effective = registered if registered is not None else metadata
        key = str(path)
        if key in targets and targets[key] != effective:
            _fail("DATASET_FPS_POLICY_CONFLICT", "同一真实数据集路径被绑定了不同的生效 fps：" + key)
        targets[key] = effective
        policy.append({"role": role, "path": key, "registered_fps": registered,
                       "metadata_fps": metadata, "effective_fps": effective,
                       "source": "registered" if registered is not None else "metadata",
                       "info_sha256": digest})
    return policy


def _function_ast(function):
    try:
        parsed = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "无法核验 LeRobot FPS 实现来源")
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot FPS 实现不是受支持的普通方法")
    return functions[0]


def _runtime(module, original):
    cls = getattr(module, "LeRobotDatasetMetadata", None)
    if (not isinstance(cls, type) or cls.__module__ != MODULE
            or cls.__qualname__ != "LeRobotDatasetMetadata" or not inspect.isfunction(original)):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot 缺少受支持的 Metadata 类")
    parameters = list(inspect.signature(original).parameters.values())
    names = ["self", "repo_id", "root", "revision", "force_cache_sync"]
    if (len(parameters) != len(names) or [item.name for item in parameters] != names
            or any(item.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for item in parameters)
            or any(parameters[index].default is not expected for index, expected in ((2, None), (3, None), (4, False)))):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 构造器接口不支持本次 FPS 覆盖")
    fps = getattr(cls, "fps", None)
    if not isinstance(fps, property) or fps.fget is None:
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 缺少可核验的 fps 属性")
    body = _function_ast(fps.fget).body
    body = [node for node in body if not (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
    expected = ast.parse('self.info["fps"]', mode="eval").body
    if (len(body) != 1 or not isinstance(body[0], ast.Return)
            or ast.dump(body[0].value, include_attributes=False) != ast.dump(expected, include_attributes=False)):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot fps 并非直接读取实例 info，不能保证覆盖生效")
    initializer = _function_ast(original)
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
               and node.func.attr == "load_metadata" for node in ast.walk(initializer)):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 构造流程与已核验接口不一致")
    load = _function_ast(getattr(cls, "load_metadata", None))
    expected_load = ast.parse("self.info = load_info(self.root)").body[0]
    if not any(ast.dump(node, include_attributes=False) == ast.dump(expected_load, include_attributes=False)
               for node in load.body):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 未从实例目录加载独立 info")
    path = Path(getattr(module, "__file__", "")).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 来源超出核验范围")
    if any(Path(inspect.getsourcefile(function) or "").resolve() != path
           for function in (original, fps.fget, cls.load_metadata)):
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 方法来自不同的动态来源")
    content = path.read_bytes()
    if len(content) > MAX_SOURCE_BYTES:
        _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 来源超过读取上限")
    return {"module": MODULE, "module_path": str(path), "module_sha256": hashlib.sha256(content).hexdigest(),
            "metadata_class": "LeRobotDatasetMetadata", "constructor_signature": str(inspect.signature(original)),
            "fps_property": 'self.info["fps"]', "scope": "canonical_dataset_root",
            "timestamp_validation": "unchanged", "resampling": False, "writes_metadata": False}


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _install(spec):
    policy = resolve_registered_fps(spec)
    wanted = any(row["registered_fps"] is not None for row in policy)
    module = sys.modules.get(MODULE)
    if wanted and module is None:
        try:
            module = importlib.import_module(MODULE)
        except ImportError as exc:
            raise ValueError("DATASET_FPS_RUNTIME_UNSUPPORTED: 无法导入已登记训练环境的 LeRobot") from exc
    cls = getattr(module, "LeRobotDatasetMetadata", None) if module is not None else None
    if not wanted and (cls is None or not hasattr(cls, HOOK)):
        return {"schema_version": 1, "policy": policy, "runtime": None, "installed": False}, None
    lock = vars(module).setdefault(LOCK, threading.RLock())
    with lock:
        state = getattr(cls, HOOK, None)
        if state is not None:
            if (not isinstance(state, dict) or cls.__init__ is not state.get("wrapper")
                    or not callable(state.get("original")) or not isinstance(state.get("receipt"), str)):
                _fail("DATASET_FPS_POLICY_CONFLICT", "既有 FPS 适配状态已改变，不能覆盖")
            original = state["original"]
        else:
            if not isinstance(cls, type):
                _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 类不可用")
            original = cls.__init__
        runtime = _runtime(module, original)
        receipt = {"schema_version": 1, "policy": policy, "runtime": runtime, "installed": wanted}
        serialized = json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if state is not None:
            if state["receipt"] != serialized:
                _fail("DATASET_FPS_POLICY_CONFLICT", "当前进程已安装不同的 FPS 策略，不能覆盖或跨训练沿用")
            return _copy(receipt), None
        targets = {row["path"]: (row["metadata_fps"], row["effective_fps"], row["info_sha256"])
                   for row in policy if row["registered_fps"] is not None}
        # Copy roots, so mutating a caller's spec cannot widen the installed map.
        source_spec = {"allowed_roots": _copy(spec["allowed_roots"])} if "allowed_roots" in spec else {}

        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            root = getattr(self, "root", None)
            if not isinstance(root, (str, Path)):
                _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot Metadata 没有可核验的实例目录")
            root = Path(root).resolve(strict=True)
            target = targets.get(str(root))
            if target is None:
                return
            metadata_fps, effective_fps, expected_sha = target
            current, actual_sha = _read_info(root, source_spec)
            if actual_sha != expected_sha or current.get("fps") != metadata_fps:
                _fail("DATASET_FPS_METADATA_CHANGED", "数据集元信息已在 FPS 策略冻结后变化，请重新预检：" + str(root))
            info = getattr(self, "info", None)
            if not isinstance(info, dict) or info.get("fps") != metadata_fps:
                _fail("DATASET_FPS_RUNTIME_UNSUPPORTED", "LeRobot 已加载元信息与 FPS 来源证据不一致")
            self.info = {**info, "fps": effective_fps}

        state = {"original": original, "wrapper": initialize, "receipt": serialized}
        cls.__init__ = initialize
        setattr(cls, HOOK, state)
        return _copy(receipt), (module, cls, state)


def install_registered_fps(spec):
    """Persist an idempotent policy until its owning context exits or process ends.

    Useful for an exported build_config() whose loader is created after the
    builder returns. A different policy is rejected, never silently substituted.
    """
    receipt, _created = _install(spec)
    return receipt


@contextmanager
def registered_fps_context(spec):
    """Restore the exact constructor/state present on entry, including on errors.

    An already installed identical policy belongs to its original owner; nested
    contexts reuse it and do not remove that owner's persistent installation.
    """
    receipt, created = _install(spec)
    try:
        yield receipt
    finally:
        if created is not None:
            module, cls, state = created
            with vars(module)[LOCK]:
                if getattr(cls, HOOK, None) is not state or cls.__init__ is not state["wrapper"]:
                    _fail("DATASET_FPS_POLICY_CONFLICT", "FPS 适配状态在执行中被其他代码修改，拒绝覆盖未知状态")
                cls.__init__ = state["original"]
                delattr(cls, HOOK)
