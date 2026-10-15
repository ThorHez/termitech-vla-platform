from __future__ import annotations

import os
import re
import signal
from pathlib import Path

import psutil


class AgentError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self)}


def allowed_path(value: str, roots: list[str], *, must_exist: bool = True) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise AgentError("INVALID_PATH", "Paths must be absolute")
    try:
        result = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise AgentError("PATH_UNAVAILABLE", f"Path is unavailable: {candidate}") from exc
    if not any(result == Path(root).resolve() or Path(root).resolve() in result.parents for root in roots):
        raise AgentError("PATH_OUTSIDE_ROOTS", f"Path is outside configured roots: {candidate}")
    return result


def safe_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise AgentError("INVALID_IDENTIFIER", "Unsafe attempt identifier")
    return value


def process_identity(pid: int) -> dict | None:
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {"pid": pid, "created_at": process.create_time()}
    except (psutil.Error, OSError):
        return None


def identity_matches(identity: dict | None) -> bool:
    if not identity:
        return False
    current = process_identity(identity["pid"])
    return current is not None and abs(current["created_at"] - identity["created_at"]) < 0.001


def signal_owned_group(identity: dict, sig: int = signal.SIGTERM) -> bool:
    """Never signal a reused PID or somebody else's process group."""
    if not identity_matches(identity):
        return False
    try:
        if os.getpgid(identity["pid"]) != identity["pid"]:
            return False
        os.killpg(identity["pid"], sig)
        return True
    except ProcessLookupError:
        return False

