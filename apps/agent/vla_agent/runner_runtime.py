"""Proof that an active training supervisor is independent of Agent updates.

The installer publishes a new concrete runtime for each release. Existing
Runners retain their interpreter, package files and a private config snapshot;
only the Agent's convenience symlink changes during an update.
"""
from __future__ import annotations

import hashlib
import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile

import psutil

from .safety import AgentError, identity_matches, safe_identifier


PROTOCOL_VERSION = 1
MARKER = "vla-training-runtime.json"
MAX_CONFIG_BYTES = 8 * 1024 * 1024


@contextmanager
def training_start_lock(state_dir: str):
    """Do not register or spawn training while the installer switches releases."""
    directory = Path(state_dir) / "locks"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Training lock directory cannot be a symlink")
    descriptor = os.open(directory / "maintenance.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Training maintenance lock must be a regular file")
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AgentError("AGENT_MAINTENANCE", "Agent 正在升级，训练启动命令将在升级完成后重试") from exc
        yield


def _concrete(value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("Runtime paths must be strings")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
        raise ValueError("Runtime paths must be concrete absolute paths")
    if ".venv" in path.parts:
        raise ValueError("Mutable Agent aliases are not training runtimes")
    return path


def _trusted_directory(path: Path, owners: set[int], *, private: bool = False):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in owners
            or info.st_mode & (0o077 if private else 0o022)):
        raise ValueError("Runtime metadata directory has unsafe ownership or permissions")


def _read_file(path: Path, owners: set[int], *, private: bool = False,
               maximum: int = MAX_CONFIG_BYTES) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid not in owners
                or info.st_mode & (0o077 if private else 0o022)
                or not 0 < info.st_size <= maximum):
            raise ValueError("Runtime metadata file has unsafe ownership, permissions or size")
        content = stream.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError("Runtime metadata exceeds its size limit")
        return content


def _runtime(value: dict) -> dict:
    if (not isinstance(value, dict) or type(value.get("protocol_version")) is not int
            or value["protocol_version"] != PROTOCOL_VERSION):
        raise ValueError("Unsupported training runtime protocol")
    runtime = _concrete(value.get("runtime_path"))
    if runtime.parent.name != "runtimes":
        raise ValueError("Training runtime is not an immutable release directory")
    owners = {0, os.geteuid()}
    for directory in (runtime.parent.parent, runtime.parent, runtime, runtime / "bin"):
        _trusted_directory(directory, owners)
    interpreter = runtime / "bin/python"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("Training runtime interpreter is unavailable")
    marker = json.loads(_read_file(runtime / MARKER, owners, maximum=4096))
    if (not isinstance(marker, dict) or type(marker.get("schema_version")) is not int
            or marker["schema_version"] != 1 or marker.get("runtime_path") != str(runtime)):
        raise ValueError("Training runtime marker does not match its release directory")
    build = marker.get("build_id")
    if build is not None and (not isinstance(build, str) or not re.fullmatch(r"[0-9a-f]{64}", build)):
        raise ValueError("Training runtime marker has an invalid build identity")
    if value.get("build_id") != build:
        raise ValueError("Training runtime build identity changed")
    return {"protocol_version": PROTOCOL_VERSION, "runtime_path": str(runtime), "build_id": build}


def installed_runtime() -> dict | None:
    """Return immutable-runtime proof, or None for editable/legacy installs."""
    try:
        runtime = Path(sys.prefix).resolve(strict=True)
        marker = json.loads(_read_file(runtime / MARKER, {0, os.geteuid()}, maximum=4096))
        return _runtime({"protocol_version": PROTOCOL_VERSION, "runtime_path": str(runtime),
                         "build_id": marker.get("build_id")})
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
        return None


def _independent_profiles(config: dict, runtime: dict):
    prefix = Path(runtime["runtime_path"]).parent.parent
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Runner snapshot has no training profiles")
    for profile in profiles:
        for key in ("python", "code_dir"):
            raw = Path(profile[key])
            if not raw.is_absolute():
                raise ValueError("Training environment must use absolute paths")
            resolved = raw.resolve(strict=True)
            # Check both spelling and destination: a Python symlink inside the
            # mutable Agent prefix still depends on that prefix surviving.
            if raw.is_relative_to(prefix) or resolved.is_relative_to(prefix):
                raise ValueError("Training environment shares the Agent installation")
            if key == "code_dir" and prefix.is_relative_to(resolved):
                raise ValueError("Training source contains the Agent installation")
        if not Path(profile["code_dir"]).is_dir():
            raise ValueError("Training source directory is unavailable")
        if not Path(profile["python"]).is_file() or not os.access(profile["python"], os.X_OK):
            raise ValueError("Training interpreter is unavailable")


def snapshot_runner_config(config: dict, attempt_id: str, runtime: dict) -> dict:
    """Publish one private, immutable config; duplicate identical starts reuse it."""
    safe_identifier(attempt_id)
    metadata = _runtime(runtime)
    _independent_profiles(config, metadata)
    state_dir = _concrete(config["state_dir"])
    owner = state_dir.stat().st_uid
    if owner != os.geteuid():
        raise ValueError("Runner state directory belongs to a different user")
    directory = state_dir / "runner-configs"
    directory.mkdir(mode=0o700, exist_ok=True)
    _trusted_directory(directory, {owner}, private=True)
    path = directory / (attempt_id + ".json")
    snapshot = {key: value for key, value in config.items() if key != "enrollment_token"}
    snapshot["config_path"] = str(path)
    content = (json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    if len(content) > MAX_CONFIG_BYTES:
        raise ValueError("Runner config snapshot exceeds its size limit")
    descriptor, temporary = tempfile.mkstemp(prefix=".runner-config-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # link() is an atomic exclusive publish: another START cannot
            # replace a config already being consumed by a running supervisor.
            os.link(temporary, path)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            if _read_file(path, {owner}, private=True) != content:
                raise ValueError("Attempt already has a different runner config snapshot")
    finally:
        os.unlink(temporary)
    return {**metadata, "config_path": str(path), "config_sha256": hashlib.sha256(content).hexdigest()}


def _prove_attempt(attempt: dict, state_dir: Path, owner: int):
    metadata = attempt.get("runner_runtime")
    runtime = _runtime(metadata)
    attempt_id = safe_identifier(attempt["id"])
    config_path = _concrete(metadata.get("config_path"))
    if config_path != state_dir / "runner-configs" / (attempt_id + ".json"):
        raise ValueError("Runner snapshot is outside its private attempt directory")
    _trusted_directory(config_path.parent, {owner}, private=True)
    content = _read_file(config_path, {owner}, private=True)
    if metadata.get("config_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("Runner snapshot digest changed")
    config = json.loads(content)
    if (not isinstance(config, dict) or config.get("state_dir") != str(state_dir)
            or config.get("config_path") != str(config_path) or "enrollment_token" in config):
        raise ValueError("Runner snapshot does not match its recorded state directory")
    _independent_profiles(config, runtime)
    for kind in ("runner", "child"):
        identity = attempt.get(kind + "_identity")
        if kind == "child" and identity is None:
            continue
        if not identity_matches(identity):
            raise ValueError("Runner or training process identity is unavailable")
        process = psutil.Process(identity["pid"])
        if process.uids().effective != owner:
            raise ValueError("Runner or training process belongs to another user")
        if kind == "runner":
            command = process.cmdline()
            expected = [str(Path(runtime["runtime_path"]) / "bin/python"), "-I", "-m", "vla_agent.runner",
                        "--config", str(config_path), "--attempt-id", attempt_id]
            if command != expected:
                raise ValueError("Live supervisor does not use its recorded immutable runtime and config")


def upgrade_status(state_dir: str) -> dict:
    """Inspect durable attempts without changing the database or process state.

    This is a point-in-time proof. The upgrade workflow must also block new
    START dispatch and check again immediately before switching the Agent.
    """
    result = {"protocol_version": PROTOCOL_VERSION, "active_attempts": 0,
              "active_attempt_ids": [], "upgrade_safe": False, "reason": ""}
    try:
        directory = _concrete(state_dir)
        database = directory / "agent.sqlite3"
        if not database.exists():
            result.update(upgrade_safe=True, reason="No local training attempts")
            return result
        info = database.lstat()
        owner = info.st_uid
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022
                or owner not in {0, os.geteuid()} or directory.stat().st_uid != owner):
            raise ValueError("Training state ownership is unsafe")
        _trusted_directory(directory, {owner})
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5) as db:
            db.row_factory = sqlite3.Row
            attempts = [dict(row) for row in db.execute(
                "SELECT * FROM attempts WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED') ORDER BY id")]
        result.update(active_attempts=len(attempts), active_attempt_ids=[attempt["id"] for attempt in attempts])
        for attempt in attempts:
            for field in ("runner_runtime", "runner_identity", "child_identity"):
                value = attempt.get(field)
                attempt[field] = json.loads(value) if value else None
            _prove_attempt(attempt, directory, owner)
        result.update(upgrade_safe=True, reason="All active training supervisors use preserved immutable runtimes"
                      if attempts else "No local training attempts")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, sqlite3.Error, psutil.Error) as exc:
        result["reason"] = str(exc)
    return result
