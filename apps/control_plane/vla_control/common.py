from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import uuid


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
ACTIVE = {"STARTING", "RUNNING", "CANCELLING", "UNKNOWN"}
INPUT_KINDS = {"dataset": "DATASET", "model": "MODEL", "norm": "NORM"}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def offset_time(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def uid() -> str:
    return str(uuid.uuid4())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def absolute_path(path: str) -> bool:
    p = PurePosixPath(path)
    return p.is_absolute() and ".." not in p.parts and "\x00" not in path


def under_root(path: str, roots: list[str]) -> bool:
    if not absolute_path(path):
        return False
    p = PurePosixPath(path)
    return any(p == PurePosixPath(root) or PurePosixPath(root) in p.parents for root in roots)


def require_path(path: str, roots: list[str] | None = None) -> None:
    if not absolute_path(path) or (roots is not None and not under_root(path, roots)):
        raise APIError(422, "PATH_OUTSIDE_ROOTS", "Path must be an absolute path inside a registered root")


def snapshot_identity(value: dict) -> dict:
    return {key: value.get(key) for key in ("code_dir", "repository_root", "commit_sha", "dirty", "changes_digest", "untracked_digest", "module_origin", "submodules", "runtime_digest", "automatic_preflight", "requested_entrypoint", "requested_entrypoint_sha256", "dataset_bindings", "training_stage", "stage_config")}


def valid_snapshot(value) -> bool:
    if not isinstance(value, dict):
        return False
    commit = value.get("commit_sha", "")
    return (isinstance(commit, str) and len(commit) in {40, 64}
            and all(c in "0123456789abcdef" for c in commit.lower())
            and value.get("complete") is True and isinstance(value.get("dirty"), bool)
            and bool(value.get("code_dir")) and bool(value.get("module_origin"))
            and bool(value.get("changes_digest")) and bool(value.get("untracked_digest")))
