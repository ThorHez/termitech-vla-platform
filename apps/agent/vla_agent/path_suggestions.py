"""Bounded, read-only completion of paths inside this machine's allowed roots."""
from __future__ import annotations

import os
import posixpath
import stat
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .config import get_profile
from .safety import AgentError, allowed_path


MAX_SCANNED_ENTRIES = 5000
SCAN_BUDGET_SECONDS = 0.2


def _normalized(value: str) -> str:
    return "/" + posixpath.normpath(value).lstrip("/")


def _inside(candidate: Path, roots: list[Path]) -> bool:
    return any(candidate == root or root in candidate.parents for root in roots)


def _accessible_root(value: str) -> Path | None:
    try:
        resolved = Path(value).resolve(strict=True)
        if resolved.is_dir() and os.access(resolved, os.R_OK | os.X_OK):
            return resolved
    except (OSError, RuntimeError):
        pass
    return None


def _validate_request(payload: dict) -> tuple[str, int]:
    prefix = payload.get("prefix")
    if (not isinstance(prefix, str) or len(prefix) > 4096
            or any(unicodedata.category(char) == "Cc" for char in prefix)
            or ".." in prefix.split("/") or (prefix and not prefix.startswith("/"))):
        raise AgentError("INVALID_PATH", "Path prefix must be absolute, without traversal or control characters, and at most 4096 characters")
    limit = payload.get("limit", 50)
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise AgentError("INVALID_PATH", "Path suggestion limit must be an integer")
    expiry = payload.get("expires_at")
    try:
        expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if expires_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("UTC timestamp required")
    except (AttributeError, TypeError, ValueError) as exc:
        raise AgentError("INVALID_PATH", "Path query requires an expires_at timestamp in UTC") from exc
    if expires_at <= datetime.now(timezone.utc):
        raise AgentError("PATH_QUERY_EXPIRED", "Path suggestion query has expired; submit a new query")
    return prefix, min(50, max(1, limit))


def suggest_paths(config: dict, payload: dict) -> dict:
    """Return one directory's names and types, never file contents or descendants.

    Paths returned to the UI use the configured spelling, including a symlinked
    root. Real targets are checked separately so symlinks cannot reveal entries
    outside the roots. A small scan budget keeps ordinary large directories from
    occupying the Agent's command loop for an entire heartbeat interval.
    """
    prefix, limit = _validate_request(payload)
    scoped = any(key in payload for key in ("profile_id", "base_path", "python_only"))
    profile_id = payload.get("profile_id")
    base_path = payload.get("base_path")
    if scoped:
        if not isinstance(profile_id, str) or not profile_id or payload.get("python_only") is not True:
            raise AgentError("INVALID_PATH", "Project path queries require a runtime profile and Python file filter")
        try:
            profile = get_profile(config, profile_id)
        except ValueError as exc:
            raise AgentError("RUNTIME_UNAVAILABLE", str(exc)) from exc
        # Bind to the current registered project, rather than trusting a base
        # path supplied by the control plane or an older browser selection.
        if base_path != profile.get("code_dir"):
            raise AgentError("PATH_SCOPE_MISMATCH", "Path query project differs from the current runtime profile; reload the project")
        if (not isinstance(base_path, str) or not base_path.startswith("/")
                or ".." in base_path.split("/")
                or any(unicodedata.category(char) == "Cc" for char in base_path)):
            raise AgentError("INVALID_PATH", "Runtime project must be an absolute path without traversal or control characters")
        project = allowed_path(base_path, config["roots"])
        if not project.is_dir():
            raise AgentError("PATH_UNAVAILABLE", "Runtime project is not an accessible directory")
    started = time.monotonic()
    effective_prefix = prefix or (base_path.rstrip("/") + "/" if scoped else "")
    normalized = (_normalized(effective_prefix) if effective_prefix.endswith("/") else
                  posixpath.join(_normalized(posixpath.dirname(effective_prefix)), posixpath.basename(effective_prefix))) if effective_prefix else ""
    lexical_roots = [_normalized(base_path)] if scoped else list(dict.fromkeys(_normalized(root) for root in config["roots"]))
    resolved_roots = {root: _accessible_root(root) for root in lexical_roots}
    allowed_targets = [root for root in resolved_roots.values() if root is not None]
    root_matches = [root for root in lexical_roots if root.startswith(normalized)]
    result = {"prefix": prefix, "parent": None, "entries": [], "truncated": False}
    if scoped:
        result.update(profile_id=profile_id, base_path=base_path)
    # Without a trailing slash an exact root is offered as a completion. With
    # one, it is opened and its immediate children are returned instead.
    requested = Path(normalized)
    lexical_paths = [Path(root) for root in lexical_roots]
    if scoped and not _inside(requested, lexical_paths):
        raise AgentError("PATH_OUTSIDE_PROJECT", "Path prefix is outside the selected training project")
    root_completion = (not scoped and (not prefix or (normalized == "/" and "/" not in lexical_roots)
                       or (normalized in lexical_roots and not prefix.endswith("/"))
                       or not _inside(requested, lexical_paths)))
    if root_matches and root_completion:
        roots = [root for root in root_matches if resolved_roots[root] is not None]
        roots.sort(key=lambda root: (Path(root).name, root))
        result["entries"] = [{"name": Path(root).name or "/", "path": root.rstrip("/") + "/", "kind": "directory"}
                             for root in roots[:limit]]
        result["truncated"] = len(roots) > limit
        return result

    if not _inside(requested, lexical_paths):
        raise AgentError("PATH_OUTSIDE_ROOTS", "Path prefix is outside this machine's configured roots")
    open_project = scoped and requested == lexical_paths[0]
    parent = requested if effective_prefix.endswith("/") or open_project else Path(posixpath.dirname(normalized))
    name_prefix = "" if effective_prefix.endswith("/") or open_project else posixpath.basename(normalized)
    if not _inside(parent, lexical_paths):
        raise AgentError("PATH_OUTSIDE_ROOTS", "Path parent is outside this machine's configured roots")
    try:
        resolved_parent = parent.resolve(strict=True)
        if not _inside(resolved_parent, allowed_targets):
            if not any(resolved_roots[root] is not None and _inside(parent, [Path(root)]) for root in lexical_roots):
                raise AgentError("PATH_UNAVAILABLE", "Configured root is unavailable or inaccessible")
            raise AgentError("PATH_OUTSIDE_PROJECT" if scoped else "PATH_OUTSIDE_ROOTS", "Path parent resolves outside the selected training project" if scoped else "Path parent resolves outside this machine's configured roots")
        if not os.access(resolved_parent, os.R_OK | os.X_OK):
            raise PermissionError("Directory is not readable and searchable")
        entries = []
        scanned = 0
        with os.scandir(resolved_parent) as listing:
            for entry in listing:
                if scanned >= MAX_SCANNED_ENTRIES or time.monotonic() - started >= SCAN_BUDGET_SECONDS:
                    result["truncated"] = True
                    break
                scanned += 1
                if not entry.name.startswith(name_prefix) or (entry.name.startswith(".") and not name_prefix.startswith(".")):
                    continue
                if any(unicodedata.category(char) == "Cc" for char in entry.name):
                    continue
                path = str(parent / entry.name)
                if len(path) + 1 > 4096:
                    continue
                try:
                    target = Path(entry.path).resolve(strict=True)
                    if not _inside(target, allowed_targets):
                        continue
                    mode = target.stat().st_mode
                    kind = "directory" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else None
                    if kind is None:
                        continue
                    if scoped and kind == "file" and (not entry.name.endswith(".py") or target.suffix != ".py"):
                        continue
                except (OSError, RuntimeError):
                    # Dangling, inaccessible, or concurrently removed children
                    # are not useful completions and reveal no additional paths.
                    continue
                entries.append({"name": entry.name, "path": path + "/" if kind == "directory" else path, "kind": kind})
        entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"]))
        result.update(parent=str(parent), entries=entries[:limit], truncated=result["truncated"] or len(entries) > limit)
        return result
    except PermissionError as exc:
        raise AgentError("PATH_PERMISSION_DENIED", "Agent cannot read this directory") from exc
    except (OSError, RuntimeError) as exc:
        raise AgentError("PATH_UNAVAILABLE", "Path parent does not exist or is not a readable directory") from exc
