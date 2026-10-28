"""Durable, asynchronous export of a verified training inference configuration."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import time
import signal

from .config import get_profile
from .policy_deployment import _environment, _frozen, _read, _record, _tail
from .safety import AgentError, allowed_path, identity_matches, process_identity, safe_identifier, signal_owned_group


WORKER = Path(__file__).with_name("policy_export_worker.py").resolve()
CHILDREN = {}


def _export_environment():
    """Export configuration on CPU even if the Agent inherits GPU-only settings."""
    environment = _environment({})
    environment.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    return environment


def _cancelled():
    return {"state": "FAILED", "error": {"code": "POLICY_EXPORT_CANCELLED", "message": "Inference configuration export was cancelled"}}


def _stop_export(directory):
    identity = _read(directory / "identity.json")
    if identity_matches(identity):
        signal_owned_group(identity, signal.SIGTERM)
        deadline = time.monotonic() + .3
        while identity_matches(identity) and time.monotonic() < deadline:
            time.sleep(.02)
        if identity_matches(identity):
            signal_owned_group(identity, signal.SIGKILL)
    _record(directory / "result.json", _cancelled())
    return _cancelled()


@contextmanager
def _directory(config, payload):
    directory = Path(config["state_dir"]) / "policy-exports" / safe_identifier(payload.get("export_id"))
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (directory / "operation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AgentError("POLICY_EXPORT_BUSY", "Inference configuration export is already being prepared") from exc
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _status(directory):
    for pid, child in list(CHILDREN.items()):
        if child.poll() is not None:
            del CHILDREN[pid]
    if (directory / "cancelled.json").exists():
        return _cancelled()
    result = _read(directory / "result.json")
    if result is not None:
        manifest = _read(directory / "manifest.json")
        if manifest and manifest.get("source_warnings"):
            result = dict(result, warnings=manifest["source_warnings"])
        return result
    manifest = _read(directory / "manifest.json")
    if manifest is None:
        raise AgentError("POLICY_EXPORT_NOT_FOUND", "Inference configuration export was not started")
    identity = _read(directory / "identity.json")
    if not identity_matches(identity):
        return {"state": "FAILED", "error": {"code": "POLICY_EXPORT_INTERRUPTED",
            "message": "Export process exited before recording a result; retry the transfer"}, "logs": _tail(directory / "export.log")}
    result = {"state": "RUNNING"}
    if manifest.get("source_warnings"):
        result["warnings"] = manifest["source_warnings"]
    return result


def handle_export_command(config, command_type, payload):
    if command_type == "POLICY_EXPORT_CANCEL":
        # Cancellation cannot wait for the checkout snapshot or resolver. A
        # persisted tombstone also prevents an older queued EXPORT from launch.
        directory = Path(config["state_dir"]) / "policy-exports" / safe_identifier(payload.get("export_id"))
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        _record(directory / "cancelled.json", {"cancelled": True})
        return _stop_export(directory)
    with _directory(config, payload) as directory:
        if (directory / "cancelled.json").exists():
            return _cancelled()
        if command_type == "POLICY_EXPORT_STATUS":
            return _status(directory)
        if command_type != "POLICY_EXPORT":
            raise AgentError("COMMAND_UNSUPPORTED", "Unknown policy export command")
        if (directory / "manifest.json").exists():
            return _status(directory)
        profile = get_profile(config, payload["profile_id"])
        if profile.get("adapter") != "openpi-sft":
            raise AgentError("POLICY_PROFILE_UNSUPPORTED", "Export requires a native OpenPI environment")
        code_dir = allowed_path(profile["code_dir"], config["roots"])
        checkpoint = allowed_path(payload["checkpoint_path"], config["roots"])
        if not code_dir.is_dir() or not checkpoint.is_dir():
            raise AgentError("POLICY_CHECKPOINT_INVALID", "Source code and checkpoint must be directories")
        interpreter = Path(profile["python"])
        if not interpreter.is_absolute() or not interpreter.is_file():
            raise AgentError("POLICY_RUNTIME_UNAVAILABLE", "Registered OpenPI interpreter is unavailable")
        if payload.get("source_training") is None:
            raise AgentError("POLICY_TRAINING_CONFIG_INVALID", "A confirmed source training configuration is required")
        from .profile_rebinding import profile_configuration_lock
        from .recipe_writeback import checkout_lock
        from vla_adapters.portable_policy import _code_files
        with profile_configuration_lock(config, busy_code="POLICY_EXPORT_BUSY"):
            with checkout_lock(config, str(code_dir), require_git=True):
                baseline = _code_files(code_dir)
                frozen = _frozen(config, profile, payload)
                if _code_files(code_dir) != baseline:
                    raise AgentError("POLICY_EXPORT_CODE_CHANGED", "Source checkout changed while validating its training snapshot")
        import vla_adapters
        manifest = {"export_id": payload["export_id"], "python": str(interpreter), "code_dir": str(code_dir),
            "checkpoint_path": str(checkpoint), "adapter_root": str(Path(vla_adapters.__file__).resolve().parent.parent),
            "source_run_id": payload["source_training"].get("run_id"), "started_at": time.time(),
            "code_files_baseline": baseline, **frozen}
        if (directory / "cancelled.json").exists():
            return _cancelled()
        _record(directory / "manifest.json", manifest)
        with (directory / "export.log").open("ab") as output:
            child = subprocess.Popen([str(interpreter), "-u", str(WORKER), str(directory / "manifest.json")],
                cwd=code_dir, env=_export_environment(), stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        CHILDREN[child.pid] = child
        identity = process_identity(child.pid)
        if identity is not None:
            _record(directory / "identity.json", identity)
        if (directory / "cancelled.json").exists():
            return _stop_export(directory)
        result = {"state": "RUNNING"}
        if frozen.get("source_warnings"):
            result["warnings"] = frozen["source_warnings"]
        return result


def execute_export_command(config, command_type, payload):
    try:
        return handle_export_command(config, command_type, payload)
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "POLICY_EXPORT_FAILED", "message": str(exc)[:2000]}
        return {"state": "FAILED", "error": error}
