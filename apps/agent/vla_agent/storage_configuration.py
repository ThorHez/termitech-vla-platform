"""Persist idle-node storage settings and bind cache paths to future launches."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .profile_rebinding import (MAX_CONFIG_BYTES, MAX_OPERATIONS, ProfileRebindError,
                                _directory, _persist, _read, profile_configuration_lock)
from .recipe_writeback import checkout_lock
from .safety import AgentError, allowed_path, safe_identifier


OPERATIONS_KEY = "_storage_configuration_operations"
HISTORY_KEY = "_output_root_history"
PAYLOAD_KEYS = {"expected_output_root", "expected_cache_root", "output_root", "cache_root", "expires_at"}


class StorageConfigurationError(AgentError):
    def __init__(self, code, message, *, persisted=False):
        super().__init__(code, message)
        self.persisted = persisted

    def as_dict(self):
        result = super().as_dict()
        if self.persisted is not None:
            result["persisted"] = self.persisted
        return result


def _fail(code, message, *, persisted=False):
    raise StorageConfigurationError("STORAGE_CONFIGURE_" + code, message, persisted=persisted)


def _path(value, *, nullable=False):
    if value is None and nullable:
        return None
    if (not isinstance(value, str) or not value or len(value) > 4096
            or not Path(value).is_absolute() or ".." in Path(value).parts
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
            or Path(value) == Path("/")):
        _fail("INVALID", "存储目录必须是完整绝对路径，不能使用根目录、上级目录或控制字符")
    return str(Path(value))


def _pair(config):
    return _path(config.get("output_root")), _path(config.get("cache_root"), nullable=True)


def cache_environment(config):
    """Explicit cache settings override all inherited library-specific caches."""
    value = _path(config.get("cache_root"), nullable=True)
    if value is None:
        return {}
    root = allowed_path(value, config["roots"], must_exist=False)
    if str(root) != value:
        _fail("INVALID", "缓存根目录必须使用实际目录的规范路径")
    hf = root / "huggingface"
    temporary = str(root / "tmp")
    hub = str(hf / "hub")
    return {"XDG_CACHE_HOME": str(root), "HF_HOME": str(hf),
            "HF_DATASETS_CACHE": str(hf / "datasets"), "HF_HUB_CACHE": hub,
            "HUGGINGFACE_HUB_CACHE": hub, "TRANSFORMERS_CACHE": hub,
            "JAX_COMPILATION_CACHE_DIR": str(root / "jax"), "OPENPI_DATA_HOME": str(root / "openpi"),
            "TMPDIR": temporary, "TEMP": temporary, "TMP": temporary}


def _probe_directory(value, roots):
    try:
        return _probe_writable_directory(value, roots)
    except PermissionError as exc:
        _fail("UNWRITABLE", "目标目录无法写入：" + value + ": " + str(exc))


def _probe_writable_directory(value, roots):
    path = allowed_path(value, roots, must_exist=False)
    if str(path) != value:
        _fail("INVALID", "存储目录必须使用实际目录的规范路径：" + str(path))
    path.mkdir(parents=True, exist_ok=True)
    # Resolve again after creation; a symlink outside the managed roots must
    # never become an accepted writable target.
    if allowed_path(value, roots) != path or not path.is_dir():
        _fail("INVALID", "存储目录不是受管本地目录")
    with tempfile.NamedTemporaryFile(prefix=".vla-storage-probe-", dir=path) as stream:
        stream.write(b"vla\n")
        stream.flush()
        os.fsync(stream.fileno())
    usage = shutil.disk_usage(path)
    return {"path": str(path), "free_bytes": usage.free, "total_bytes": usage.total}


@contextmanager
def _background_lock(config):
    path = Path(config["state_dir"]) / "preflight-worker.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            _fail("FAILED", "后台操作锁必须是普通文件")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("BUSY", "机器仍有预检、部署或代码操作，请等待完成")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _require_idle(config, state):
    if state.active_attempts():
        _fail("BUSY", "机器仍有未结束或状态未知的训练任务，请等待完成")
    from .policy_deployment import active_policy_checkouts
    if active_policy_checkouts(config):
        _fail("BUSY", "机器仍有本地推理部署，请停止部署后再修改存储目录")


def configure_storage(config, state, payload, *, operation_id):
    """Probe both targets, persist a recoverable CAS, then update live settings."""
    persisted = False
    try:
        if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
            _fail("INVALID", "存储配置参数不完整或包含未知字段")
        operation_id = safe_identifier(operation_id)
        request = {key: _path(payload[key], nullable=key.endswith("cache_root"))
                   for key in PAYLOAD_KEYS - {"expires_at"}}
        expires = payload["expires_at"]
        if not isinstance(expires, str) or len(expires) > 128:
            _fail("INVALID", "存储配置必须包含 UTC 过期时间")
        try:
            deadline = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
                raise ValueError("UTC required")
        except ValueError:
            _fail("INVALID", "存储配置过期时间必须是 UTC 时间")
        fingerprint = hashlib.sha256(json.dumps({**request, "expires_at": expires},
                                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = Path(_path(config.get("config_path")))
        with profile_configuration_lock(config, busy_code="STORAGE_CONFIGURE_BUSY"), _background_lock(config):
            try:
                original, disk = _read(path)
            except (OSError, ProfileRebindError):
                _fail("RECOVERY_PENDING", "Agent 配置暂时不可读取，等待同一存储操作重试确认", persisted=None)
            operations = disk.get(OPERATIONS_KEY, [])
            if (not isinstance(operations, list) or len(operations) > MAX_OPERATIONS
                    or any(not isinstance(item, dict) for item in operations)):
                _fail("CONFLICT", "存储配置持久操作记录无效")
            target = request["output_root"], request["cache_root"]
            prior = next((item for item in operations if item.get("operation_id") == operation_id), None)
            if prior is not None:
                result = prior.get("result")
                if (prior.get("fingerprint") != fingerprint or not isinstance(result, dict)
                        or result.get("persisted") is not True or _pair(disk) != target
                        or (result.get("output_root"), result.get("cache_root")) != target):
                    _fail("RECOVERY_PENDING", "已保存的存储配置或同一操作参数发生变化，等待确认", persisted=None)
                persisted = True
                config.update(output_root=disk["output_root"], cache_root=disk.get("cache_root"))
                config[HISTORY_KEY] = copy.deepcopy(disk.get(HISTORY_KEY, []))
                return copy.deepcopy(result)
            if deadline <= datetime.now(timezone.utc):
                _fail("EXPIRED", "存储配置请求已过期，请重试")
            expected = request["expected_output_root"], request["expected_cache_root"]
            if _pair(disk) != expected or _pair(config) != expected:
                _fail("CONFLICT", "存储配置已变化，请刷新后重试")
            with ExitStack() as locks:
                for directory in sorted({item["code_dir"] for item in config.get("profiles", [])}):
                    locks.enter_context(checkout_lock(config, directory))
                _require_idle(config, state)
                state_path = Path(config['state_dir']).resolve()
                config_path = path.resolve()
                for value in target:
                    if value is None:
                        continue
                    candidate = allowed_path(value, config['roots'], must_exist=False)
                    if (candidate == state_path or candidate in state_path.parents or state_path in candidate.parents
                            or candidate == config_path or candidate in config_path.parents):
                        _fail('INVALID', '训练输出和缓存目录必须与 Agent 状态及配置目录分开')
                output = _probe_directory(request["output_root"], config["roots"])
                cache = (_probe_directory(request["cache_root"], config["roots"])
                         if request["cache_root"] is not None else None)
                if cache is not None:
                    for directory in set(cache_environment({**config, "cache_root": request["cache_root"]}).values()):
                        _probe_directory(directory, config["roots"])
                _require_idle(config, state)
                if _pair(config) != expected:
                    _fail("CONFLICT", "运行中的存储配置已变化，请刷新后重试")
                result = {"persisted": True, "output_root": request["output_root"],
                          "cache_root": request["cache_root"], "directories": {"output": output, "cache": cache}}
                history = disk.get(HISTORY_KEY, [])
                if not isinstance(history, list) or any(not isinstance(item, str) for item in history):
                    _fail("CONFLICT", "历史输出目录记录无效")
                disk[HISTORY_KEY] = list(dict.fromkeys([*history, disk["output_root"]]))
                disk.update(output_root=request["output_root"], cache_root=request["cache_root"])
                disk[OPERATIONS_KEY] = [*operations, {"operation_id": operation_id,
                                                     "fingerprint": fingerprint, "result": result}][-MAX_OPERATIONS:]
                content = (json.dumps(disk, ensure_ascii=True, indent=2) + "\n").encode()
                if len(content) > MAX_CONFIG_BYTES:
                    _fail("FAILED", "更新后的 Agent 配置超过大小上限")
                backup = _directory(config) / ("before-storage-" + hashlib.sha256(operation_id.encode() + original).hexdigest() + ".json")
                _persist(path, original, content, backup, deadline=deadline)
                persisted = True
                config.update(output_root=disk["output_root"], cache_root=disk.get("cache_root"))
                config[HISTORY_KEY] = copy.deepcopy(disk[HISTORY_KEY])
                return result
    except StorageConfigurationError:
        raise
    except ProfileRebindError as exc:
        code = exc.code.removeprefix("PROFILE_REBIND_").removeprefix("STORAGE_CONFIGURE_")
        _fail(code, str(exc), persisted=exc.persisted)
    except AgentError as exc:
        _fail("BUSY" if exc.code in {"RECIPE_WRITE_BUSY", "RECIPE_CHECKOUT_BUSY"} else "INVALID", str(exc))
    except Exception as exc:
        if persisted:
            _fail("RECOVERY_PENDING", "配置已保存，等待同一操作重试同步运行状态", persisted=None)
        _fail("UNWRITABLE" if isinstance(exc, PermissionError) else "FAILED", "存储目录配置失败：" + str(exc))
