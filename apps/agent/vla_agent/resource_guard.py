"""Optional, read-only protection for a test Agent sharing a production host.

The local controller's reservation tables remain separate. This guard rejects
resources currently reserved or allocated by the configured authoritative DB.
Explicitly listed reservations may be shared by their configured owner; active
allocations are always blocked. GPU flock files serialize Runner/serve processes.
It is a conservative snapshot, not an atomic cross-controller lease protocol.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid

from .safety import AgentError


def _human_reservations(db, node_id):
    columns = {row[1] for row in db.execute("PRAGMA table_info(gpu_reservations)")}
    users = {row[1] for row in db.execute("PRAGMA table_info(users)")}
    if not ({"created_at", "expires_at"} <= columns and {"id", "enabled", "role"} <= users):
        # Older authoritative controllers have no expiry/account policy. Keep
        # their reservations protected; never infer that a legacy row is free.
        return list(db.execute("SELECT gpu_uuid,user_id FROM gpu_reservations WHERE node_id=?", (node_id,)))
    must_change = "u.must_change_password" if "must_change_password" in users else "0"
    rows = db.execute("SELECT r.gpu_uuid,r.user_id,r.created_at,r.expires_at,u.enabled,u.role," + must_change +
        " FROM gpu_reservations r LEFT JOIN users u ON u.id=r.user_id WHERE r.node_id=?", (node_id,))
    current = datetime.now(timezone.utc)
    result = []
    for gpu, owner, created_at, expires_at, enabled, role, change_password in rows:
        # Corrupt or unresolvable upgraded rows must fail closed. This guard
        # only reads the authority and never deletes or repairs its records.
        expiry = datetime.fromisoformat(expires_at) if expires_at is not None else (
            datetime.fromisoformat(created_at) + timedelta(seconds=28800))
        if expiry.tzinfo is None or enabled not in (0, 1) or role is None or change_password not in (0, 1):
            raise ValueError()
        if expiry > current and enabled and role in {"admin", "operator"} and not change_password:
            result.append((gpu, owner))
    return result


def _policy(config):
    value = config.get("resource_guard")
    if value is None:
        return None
    try:
        if (not isinstance(value, dict) or not {"database", "node_id"} <= set(value)
                or set(value) - {"database", "node_id", "lock_directory", "shared_reservations"}):
            raise ValueError()
        path = Path(value["database"])
        if (not path.is_absolute() or ".." in path.parts
                or any(parent.is_symlink() for parent in (path, *path.parents))
                or not path.is_file()):
            raise ValueError()
        node_id = str(uuid.UUID(value["node_id"]))
        if node_id != value["node_id"]:
            raise ValueError()
        shared = None
        if "shared_reservations" in value:
            shared = value["shared_reservations"]
            if (not isinstance(shared, dict) or set(shared) != {"owner_id", "gpu_uuids"}
                    or not isinstance(shared["owner_id"], str)
                    or str(uuid.UUID(shared["owner_id"])) != shared["owner_id"]):
                raise ValueError()
            devices = shared["gpu_uuids"]
            if (not isinstance(devices, list) or not 1 <= len(devices) <= 256
                    or any(not isinstance(gpu, str) or not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9/-]{1,196}", gpu)
                           for gpu in devices)
                    or len(devices) != len(set(devices))):
                raise ValueError()
            shared = shared["owner_id"], frozenset(devices)
        return path, node_id, shared
    except (OSError, TypeError, ValueError, AttributeError, KeyError):
        raise AgentError("RESOURCE_GUARD_UNAVAILABLE", "无法核验正式环境的显卡预约，测试任务暂不能使用 GPU") from None


def protected_gpus(config):
    policy = _policy(config)
    if policy is None:
        return None
    path, node_id, shared = policy
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.5)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            node = db.execute("SELECT gpus FROM nodes WHERE id=?", (node_id,)).fetchone()
            if node is None:
                raise ValueError()
            inventory = json.loads(node[0])
            known = {gpu["uuid"] for gpu in inventory}
            if len(known) != len(inventory) or not all(isinstance(item, str) for item in known):
                raise ValueError()
            reservations = _human_reservations(db, node_id)
            if shared is None:
                blocked = {row[0] for row in reservations}
            else:
                owner, devices = shared
                if not devices <= known:
                    raise ValueError()
                # Only these explicitly listed cards belonging to this exact
                # production account may share a human reservation. A later
                # ownership change blocks them immediately on the next check.
                blocked = {gpu for gpu, user_id in reservations
                    if user_id != owner or gpu not in devices}
            blocked.update(row[0] for row in db.execute(
                "SELECT gpu_uuid FROM gpu_allocations WHERE node_id=?", (node_id,)))
            blocked.update(row[0] for row in db.execute(
                "SELECT gpu_uuid FROM policy_deployments WHERE node_id=? "
                "AND state IN ('PENDING','STARTING','RUNNING','STOPPING','UNKNOWN') "
                "AND gpu_uuid IS NOT NULL", (node_id,)))
            if not blocked <= known:
                raise ValueError()
            if config["resource_guard"].get("lock_directory") is not None:
                source = Path(config["resource_guard"]["lock_directory"])
                target = Path(config["state_dir"]) / "locks"
                if (not source.is_absolute() or ".." in source.parts
                        or any(p.is_symlink() for p in (source, target, *source.parents, *target.parents))):
                    raise ValueError()
                for gpu in known:
                    filename = "gpu-" + hashlib.sha256(gpu.encode()).hexdigest() + ".lock"
                    a, b = (source / filename).lstat(), (target / filename).lstat()
                    if (not all(stat.S_ISREG(s.st_mode) and s.st_uid in {0, os.geteuid()}
                                and not s.st_mode & 0o022 for s in (a, b))
                            or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino)):
                        raise ValueError()
            return known, blocked
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, OverflowError):
        raise AgentError("RESOURCE_GUARD_UNAVAILABLE", "无法核验正式环境的显卡预约，测试任务暂不能使用 GPU") from None


def require_gpu_access(config, requested):
    if not requested:
        return
    policy = protected_gpus(config)
    if policy is None:
        return
    known, blocked = policy
    if any(gpu not in known for gpu in requested):
        raise AgentError("RESOURCE_GUARD_SCOPE_MISMATCH", "测试 GPU 不在正式机器的已核验资源范围内")
    if blocked.intersection(requested):
        raise AgentError("GPU_RESERVED_BY_PRODUCTION", "所选 GPU 已由正式环境预约或分配，请选择其他显卡")


def filter_inventory(config, inventory):
    """Only advertise resources eligible for this test environment's scheduler."""
    if config.get("resource_guard") is None:
        return inventory, None
    try:
        known, blocked = protected_gpus(config)
        if any(gpu.get("uuid") not in known for gpu in inventory):
            raise AgentError("RESOURCE_GUARD_SCOPE_MISMATCH", "测试 GPU 与正式机器资源清单不一致")
        eligible = [gpu for gpu in inventory if gpu["uuid"] not in blocked]
        return eligible, {"status": "AVAILABLE", "source": "production", "total_gpus": len(inventory),
                          "eligible_gpus": len(eligible), "protected_gpus": len(inventory) - len(eligible),
                          "message": ("测试环境遵守正式环境的显卡预约及分配限制"
                                      if "shared_reservations" in config["resource_guard"] else
                                      "测试环境仅分配正式环境未预约、未分配的显卡")}
    except AgentError as exc:
        return [], {"status": "UNAVAILABLE", "source": "production", "code": exc.code,
                    "message": str(exc), "eligible_gpus": 0}
