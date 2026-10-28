"""Rebind one idle training profile without restarting or deploying the Agent."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .recipe_writeback import _idle, _repository_root, checkout_lock
from .runtime_checks import check_runtime
from .safety import AgentError, allowed_path, safe_identifier


MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_OPERATIONS = 32
OPERATIONS_KEY = "_profile_rebind_operations"
PAYLOAD_KEYS = {"profile_id", "expected_code_dir", "expected_python", "code_dir", "python", "expires_at"}


class ProfileRebindError(AgentError):
    def __init__(self, code: str, message: str, *, persisted=False, runtime_check=None):
        super().__init__(code, message)
        self.persisted = persisted
        self.runtime_check = runtime_check

    def as_dict(self):
        result = super().as_dict()
        if self.persisted is not None:
            result["persisted"] = self.persisted
        if self.runtime_check is not None:
            result["runtime_check"] = self.runtime_check
        return result


def _fail(code, message, **kwargs):
    raise ProfileRebindError("PROFILE_REBIND_" + code, message, **kwargs)


def _path(value, label):
    if (not isinstance(value, str) or not 1 <= len(value) <= 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
            or not Path(value).is_absolute() or ".." in Path(value).parts):
        _fail("INVALID", label + "必须是无控制字符或 .. 的完整绝对路径")
    return str(Path(value))


def _directory(config):
    path = Path(config["state_dir"]) / "profile-rebind"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        _fail("FAILED", "运行环境配置锁目录不可用")
    path.chmod(0o700)
    return path


@contextmanager
def profile_configuration_lock(config, *, busy_code="PROFILE_REBIND_BUSY"):
    """Shared with START before its profile lookup; processes use the same lock."""
    try:
        descriptor = os.open(_directory(config) / "configuration.lock",
                             os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "a+b") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _fail("FAILED", "运行环境配置锁不是普通文件")
            os.fchmod(stream.fileno(), 0o600)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ProfileRebindError(busy_code, "运行环境正在切换，请稍后重试") from exc
            try:
                yield
            finally:
                try:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                except OSError:
                    pass  # Closing the descriptor also releases the lock.
    except ProfileRebindError:
        raise
    except OSError as exc:
        _fail("FAILED", "无法锁定 Agent 运行环境配置：" + str(exc))


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONFIG_BYTES:
            _fail("FAILED", "Agent 配置必须是大小受限的普通文件")
        content = stream.read(MAX_CONFIG_BYTES + 1)
    if len(content) > MAX_CONFIG_BYTES:
        _fail("FAILED", "Agent 配置超过读取上限")
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("profiles"), list):
            raise ValueError("Invalid profiles")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        _fail("FAILED", "Agent 配置不是有效的配置对象")
    return content, parsed


def _profile(config, profile_id):
    profiles = config.get("profiles", [])
    found = [item for item in profiles if isinstance(item, dict) and item.get("id") == profile_id]
    if len(found) != 1:
        _fail("INVALID", "运行环境不存在或标识重复")
    profile = found[0]
    if profile.get("adapter") != "openpi-sft" or profile.get("execution_mode", "native") != "native":
        _fail("UNSUPPORTED", "仅支持切换本机 OpenPI 运行环境")
    return profile


def _temporary(parent, content):
    descriptor, temporary = tempfile.mkstemp(prefix=".profile-rebind-", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return Path(temporary)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sync_parent(path):
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _persist(path, old, new, backup, *, deadline):
    """Publish one private config, with a prewritten rollback for failed fsync."""
    temporary = rollback = None
    published = False
    replace_attempted = False
    try:
        temporary = _temporary(path.parent, new)
        rollback = _temporary(path.parent, old)
        before = _temporary(backup.parent, old)
        try:
            os.replace(before, backup)
            _sync_parent(backup)
        finally:
            before.unlink(missing_ok=True)
        if _read(path)[0] != old:
            _fail("CONFLICT", "Agent 配置已被其他操作修改，请刷新后重试")
        if deadline <= datetime.now(timezone.utc):
            _fail("EXPIRED", "保存配置前切换请求已过期")
        replace_attempted = True
        os.replace(temporary, path)
        published = True
        _sync_parent(path)
    except BaseException as exc:
        if replace_attempted:
            rollback_durable = not published
            if published:
                try:
                    os.replace(rollback, path)
                    _sync_parent(path)
                    rollback_durable = True
                except OSError:
                    pass
            try:
                restored = _read(path)[0] == old
            except (OSError, AgentError):
                restored = False
            if not restored or not rollback_durable:
                _fail("RECOVERY_PENDING", "配置可能已保存，等待同一操作重试确认", persisted=None)
        if isinstance(exc, ProfileRebindError):
            raise
        if not isinstance(exc, Exception):
            raise
        _fail("FAILED", "未能保存运行环境配置：" + str(exc))
    finally:
        for item in (temporary, rollback):
            if item is not None:
                try:
                    item.unlink(missing_ok=True)
                except OSError:
                    pass  # Cleanup cannot turn a committed operation into a false failure.


def _active(config, state, profile_id, old_checkout, new_checkout):
    attempts = state.active_attempts()
    for attempt in attempts:
        payload = attempt.get("payload") or {}
        frozen = payload.get("profile") or {}
        if payload.get("profile_id") == profile_id or frozen.get("id") == profile_id:
            _fail("BUSY", "该运行环境仍有未结束或状态未知的训练任务")
        candidates = [frozen.get("code_dir"), (payload.get("snapshot") or {}).get("code_dir"),
                      (payload.get("code_expectation") or {}).get("code_dir")]
        configured = next((item for item in config["profiles"] if item.get("id") == payload.get("profile_id")), {})
        candidates.append(configured.get("code_dir"))
        if not any(candidates) and not (payload.get("code_expectation") or {}).get("repository_root"):
            _fail("BUSY", "存在无法确认代码目录的训练任务，请先确认其结束状态")
    if old_checkout is None and attempts:
        _fail("BUSY", "旧代码目录不可用且仍有训练任务，无法安全确认目录占用")
    for checkout in {old_checkout, new_checkout} - {None}:
        try:
            _idle(config, state, checkout)
        except AgentError as exc:
            _fail("BUSY", str(exc))


def _project_contract(candidate, roots, runtime):
    """An editable install must select this project, including within a monorepo."""
    try:
        project = allowed_path(candidate["code_dir"], roots)
        package = allowed_path(str(project / "src/openpi"), roots)
        source = allowed_path(str(project / "src/openpi/training/config.py"), [str(package)])
        origin = allowed_path(runtime["module_origin"], [str(package)])
        if package != project / "src/openpi" or source != project / "src/openpi/training/config.py":
            raise ValueError("所选项目必须使用标准 src/openpi 布局，源码目录或 config.py 不支持额外符号链接")
        if not source.is_file() or not os.access(source, os.R_OK):
            raise ValueError("所选项目的 src/openpi/training/config.py 不可读取")
        if not origin.is_file() or not os.access(origin, os.R_OK):
            raise ValueError("训练 Python 实际导入的 OpenPI 文件不可读取")
    except (AgentError, KeyError, TypeError, ValueError, OSError) as exc:
        failed = {**runtime, "status": "FAILED", "error": "训练 Python 必须导入所选项目中的 OpenPI：" + str(exc)}
        failed["checks"] = [*runtime.get("checks", []), {"name": "selected_project", "ok": False, "message": failed["error"]}]
        _fail("RUNTIME_UNAVAILABLE", failed["error"], runtime_check=failed)
    runtime["checks"] = [*runtime.get("checks", []),
                         {"name": "selected_project", "ok": True, "message": str(source)},
                         {"name": "selected_project_import", "ok": True, "message": str(origin)}]


def rebind_profile(config, state, payload, *, operation_id):
    """Probe, CAS and persist one profile; only then expose it in live memory."""
    persisted = False
    try:
        if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
            _fail("INVALID", "运行环境切换参数不完整或包含未知字段")
        try:
            operation_id = safe_identifier(operation_id)
        except AgentError as exc:
            _fail("INVALID", str(exc))
        profile_id = payload["profile_id"]
        if (not isinstance(profile_id, str) or not 1 <= len(profile_id) <= 200
                or not profile_id.strip() or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in profile_id)):
            _fail("INVALID", "运行环境标识必须是长度不超过 200 的无控制字符文本")
        request = {key: _path(payload[key], key) for key in ("expected_code_dir", "expected_python", "code_dir", "python")}
        expires = payload["expires_at"]
        if not isinstance(expires, str) or len(expires) > 128:
            _fail("INVALID", "切换请求必须包含 UTC 过期时间")
        try:
            deadline = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
                raise ValueError("UTC required")
        except ValueError:
            _fail("INVALID", "切换请求过期时间必须是 UTC 时间")
        fingerprint = hashlib.sha256(json.dumps({**request, "profile_id": profile_id, "expires_at": expires},
                                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = Path(_path(config.get("config_path"), "Agent 配置路径"))
        with profile_configuration_lock(config):
            try:
                old, disk = _read(path)
            except (OSError, ProfileRebindError):
                _fail("RECOVERY_PENDING", "Agent 配置暂时不可读取，无法确认是否已保存；等待同一操作重试", persisted=None)
            operations = disk.get(OPERATIONS_KEY, [])
            if (not isinstance(operations, list) or len(operations) > MAX_OPERATIONS
                    or any(not isinstance(item, dict) for item in operations)):
                _fail("CONFLICT", "运行环境持久操作记录无效")
            prior = next((item for item in operations if item.get("operation_id") == operation_id), None)
            if prior is not None:
                result = prior.get("result")
                try:
                    disk_profile = _profile(disk, profile_id)
                    _profile(config, profile_id)
                except AgentError:
                    _fail("RECOVERY_PENDING", "已保存的运行环境发生变化，等待确认本次切换结果", persisted=None)
                if (prior.get("fingerprint") != fingerprint or not isinstance(result, dict)
                        or result.get("profile_id") != profile_id or result.get("code_dir") != request["code_dir"]
                        or result.get("python") != request["python"] or result.get("persisted") is not True
                        or any(_path(disk_profile.get(key), key) != request[key] for key in ("code_dir", "python"))):
                    _fail("RECOVERY_PENDING", "同一切换操作的参数或已保存目标发生变化，等待确认", persisted=None)
                persisted = True
                config["profiles"] = copy.deepcopy(disk["profiles"])
                return copy.deepcopy(result)
            disk_profile = _profile(disk, profile_id)
            memory_profile = _profile(config, profile_id)
            if deadline <= datetime.now(timezone.utc):
                _fail("EXPIRED", "运行环境切换请求已过期")
            if any(_path(profile.get(key), key) != request["expected_" + key]
                   for profile in (disk_profile, memory_profile) for key in ("code_dir", "python")):
                _fail("CONFLICT", "运行环境已变更，请刷新后重试")
            candidate = {**copy.deepcopy(disk_profile), "code_dir": request["code_dir"], "python": request["python"]}
            try:
                checkout = allowed_path(candidate["code_dir"], config["roots"])
                if not checkout.is_dir():
                    _fail("INVALID", "新代码目录必须是已存在的目录")
                new_checkout = _repository_root(config, checkout, strict=True)
                try:
                    old_checkout = _repository_root(config, Path(request["expected_code_dir"]), strict=True)
                except AgentError:
                    old_checkout = None
                interpreter = Path(candidate["python"])
                if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                    _fail("INVALID", "训练 Python 必须是 Agent 用户可执行的现有文件")
                with ExitStack() as locks:
                    for directory in sorted({str(item) for item in (old_checkout, new_checkout) if item is not None}):
                        locks.enter_context(checkout_lock(config, directory, require_git=True))
                    _active(config, state, profile_id, old_checkout, new_checkout)
                    runtime = check_runtime(candidate, config["roots"], timeout=10)
                    if not isinstance(runtime, dict) or runtime.get("status") != "READY":
                        _fail("RUNTIME_UNAVAILABLE", "新训练环境检查失败：" + str(runtime.get("error", "未就绪") if isinstance(runtime, dict) else "无效检查结果"),
                              runtime_check=runtime)
                    _project_contract(candidate, config["roots"], runtime)
                    if deadline <= datetime.now(timezone.utc):
                        _fail("EXPIRED", "环境检查完成时切换请求已过期")
                    _active(config, state, profile_id, old_checkout, new_checkout)
                    if any(_path(_profile(config, profile_id).get(key), key) != request["expected_" + key] for key in ("code_dir", "python")):
                        _fail("CONFLICT", "运行中的环境配置已变更，请刷新后重试")
                    result = {"profile_id": profile_id, "code_dir": candidate["code_dir"], "python": candidate["python"],
                              "runtime_check": runtime, "persisted": True}
                    disk_profile.update(code_dir=candidate["code_dir"], python=candidate["python"])
                    disk[OPERATIONS_KEY] = [*operations, {"operation_id": operation_id, "fingerprint": fingerprint,
                                                          "result": result}][-MAX_OPERATIONS:]
                    content = (json.dumps(disk, ensure_ascii=True, indent=2) + "\n").encode()
                    if len(content) > MAX_CONFIG_BYTES:
                        _fail("FAILED", "更新后的 Agent 配置超过大小上限")
                    backup = _directory(config) / ("before-" + hashlib.sha256(operation_id.encode() + old).hexdigest() + ".json")
                    _persist(path, old, content, backup, deadline=deadline)
                    persisted = True
                    config["profiles"] = copy.deepcopy(disk["profiles"])
                    return result
            except ProfileRebindError:
                raise
            except AgentError as exc:
                _fail("BUSY" if exc.code in {"RECIPE_CHECKOUT_BUSY", "RECIPE_WRITE_BUSY"} else "INVALID", str(exc))
    except ProfileRebindError:
        raise
    except Exception as exc:
        if persisted:
            _fail("RECOVERY_PENDING", "配置已保存，等待同一操作重试同步运行状态", persisted=None)
        _fail("FAILED", "运行环境切换失败：" + str(exc))
