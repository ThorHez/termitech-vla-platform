"""Pinned, static details for a literal recipe in the local OpenPI registry."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import time
import tokenize
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .recipe_suggestions import RECIPE_NAME, SOURCE_PATHS, _Reader, _unknown
from .safety import AgentError, allowed_path


MAX_SOURCE_BYTES = 1024 * 1024
MAX_AST_NODES = 100_000
MAX_SOURCE_CODE_CHARS = 32768
MAX_FIELDS = 500
MAX_EXPRESSION_CHARS = 2048
MAX_FIELD_TEXT_CHARS = 65536
MAX_FIELD_DEPTH = 12
MAX_GIT_DIFF_BYTES = 4 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 3
DETAIL_BUDGET_SECONDS = 1.0
_UNRESOLVED = object()


def _request(payload: dict) -> tuple[str, str]:
    profile_id, recipe = payload.get("profile_id"), payload.get("recipe")
    if (not isinstance(profile_id, str) or not 1 <= len(profile_id) <= 200
            or any(unicodedata.category(char) == "Cc" for char in profile_id)
            or not isinstance(recipe, str) or not RECIPE_NAME.fullmatch(recipe)):
        raise AgentError("INVALID_RECIPE_DETAIL", "Recipe details require a configured profile and safe recipe name")
    try:
        expires_at = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        if expires_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("UTC timestamp required")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise AgentError("INVALID_RECIPE_DETAIL", "Recipe details require an expires_at timestamp in UTC") from exc
    if expires_at <= datetime.now(timezone.utc):
        raise AgentError("RECIPE_QUERY_EXPIRED", "Recipe detail query has expired; submit a new query")
    return profile_id, recipe


def _identity(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _source(path: Path, checkout: Path, roots: list[str]) -> tuple[bytes, tuple, Path]:
    resolved = allowed_path(str(path), roots)
    if checkout not in resolved.parents:
        raise AgentError("PATH_OUTSIDE_ROOTS", "Recipe source must stay inside its configured checkout")
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "Recipe source must be a readable regular file")
        if before.st_size > MAX_SOURCE_BYTES:
            raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded detail size")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(MAX_SOURCE_BYTES + 1)
        after = os.fstat(descriptor)
        if len(content) > MAX_SOURCE_BYTES:
            raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded detail size")
        if _identity(before) != _identity(after) or len(content) != after.st_size:
            raise AgentError("RECIPE_SOURCE_CHANGED", "Recipe source changed while details were being read")
        return content, _identity(after), resolved
    finally:
        os.close(descriptor)


def _git(checkout: Path, *arguments: str, max_bytes: int = 8192) -> bytes:
    """Bound both elapsed time and captured output; never run filters or hooks."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GIT_", "VLA_"))}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    process = None
    try:
        process = subprocess.Popen(["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
                                    "-C", str(checkout), *arguments], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, env=environment, start_new_session=True)
        chunks, length = [], 0
        deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Git inspection exceeded its time limit")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), min(65536, max_bytes - length + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    length += len(chunk)
                    if length > max_bytes:
                        raise ValueError("Git inspection exceeded its output limit")
                    chunks.append(chunk)
        if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
            raise ValueError("Git inspection failed")
        return b"".join(chunks)
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Cannot obtain a bounded Git version for this checkout") from exc
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Some desktop sandboxes allow signaling an owned child,
                    # but prohibit signaling its separately created group.
                    process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()


def _git_pin(checkout: Path, roots: list[str]) -> dict:
    version = _git(checkout, "rev-parse", "--show-toplevel", "--verify", "HEAD").decode("utf-8", "strict").splitlines()
    if len(version) != 2 or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", version[1]):
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Checkout has no valid Git commit")
    repository = allowed_path(version[0], roots)
    if repository != checkout and repository not in checkout.parents:
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Checkout is outside the inspected Git repository")
    status = _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=no", max_bytes=131072)
    staged = _git(repository, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary", "--", max_bytes=MAX_GIT_DIFF_BYTES)
    unstaged = _git(repository, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--", max_bytes=MAX_GIT_DIFF_BYTES - len(staged))
    return {"commit_sha": version[1], "dirty": bool(status),
            "changes_digest": hashlib.sha256(b"STAGED\0" + staged + b"\0UNSTAGED\0" + unstaged).hexdigest()}


class _PrimaryReader(_Reader):
    """Reuse registry binding semantics, without resolving imported recipes."""
    def __init__(self, checkout: Path, module_root: Path, roots: list[str], tree: ast.Module):
        super().__init__(checkout, module_root, roots)
        self.tree = tree

    def read(self, path: Path, *, primary: bool = False) -> ast.Module:
        return self.tree

    def imported(self, reference, depth: int):
        return _unknown()


def _literal(node: ast.expr, depth: int = 0):
    if depth > 4:
        return _UNRESOLVED
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int) and abs(value) <= 2 ** 53 - 1:
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if isinstance(value, str) and len(value) <= 1024:
            return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand, depth + 1)
        if value is not _UNRESOLVED and isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 64:
        values = [_literal(item, depth + 1) for item in node.elts]
        if all(value is not _UNRESOLVED for value in values) and len(json.dumps(values, ensure_ascii=False)) <= 4096:
            return values
    return _UNRESOLVED


def _display(value: str) -> str:
    return "".join(f"\\u{ord(char):04x}" if (unicodedata.category(char) in {"Cc", "Cf"} and char not in "\n\t") else char
                   for char in value)


class _Fields:
    def __init__(self, text: str):
        self.lines = [line.encode("utf-8") for line in text.splitlines(keepends=True)]
        self.fields: list[dict] = []
        self.warnings: set[str] = set()
        self.characters = 0
        self.started = time.monotonic()

    def segment(self, node: ast.expr) -> str:
        if node.lineno == node.end_lineno:
            raw = self.lines[node.lineno - 1][node.col_offset:node.end_col_offset]
        else:
            raw = b"".join([self.lines[node.lineno - 1][node.col_offset:],
                            *self.lines[node.lineno:node.end_lineno - 1], self.lines[node.end_lineno - 1][:node.end_col_offset]])
        value = raw.decode("utf-8")
        displayed = _display(value)
        if displayed != value:
            self.warnings.add("DETAIL_TEXT_SANITIZED")
        return displayed

    def add(self, path: str, node: ast.expr, origin: str, depth: int = 0) -> None:
        if (len(self.fields) >= MAX_FIELDS or len(path) > 512 or depth > MAX_FIELD_DEPTH
                or self.characters >= MAX_FIELD_TEXT_CHARS or time.monotonic() - self.started > DETAIL_BUDGET_SECONDS):
            self.warnings.add("DETAIL_FIELDS_TRUNCATED")
            return
        expression = self.segment(node)
        if len(expression) > MAX_EXPRESSION_CHARS:
            expression = expression[:MAX_EXPRESSION_CHARS - 1] + "…"
            self.warnings.add("DETAIL_EXPRESSION_TRUNCATED")
        if self.characters + len(expression) + len(path) > MAX_FIELD_TEXT_CHARS:
            self.warnings.add("DETAIL_FIELDS_TRUNCATED")
            return
        self.characters += len(expression) + len(path)
        value = _literal(node)
        field = {"path": path, "kind": "expression" if value is _UNRESOLVED else "literal", "expression": expression, "origin": origin}
        if value is not _UNRESOLVED:
            field["value"] = value
        self.fields.append(field)
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is not None:
                    self.add(f"{path}.{keyword.arg}", keyword.value, origin, depth + 1)
                else:
                    self.warnings.add("UNRESOLVED_DEFAULTS")
        elif isinstance(node, (ast.List, ast.Tuple)) and value is _UNRESOLVED:
            for index, item in enumerate(node.elts):
                if index >= 64:
                    self.warnings.add("DETAIL_FIELDS_TRUNCATED")
                    break
                self.add(f"{path}[{index}]", item, origin, depth + 1)


def recipe_details(config: dict, payload: dict) -> dict:
    """Read a pinned recipe snippet and declared values, never training imports."""
    profile_id, recipe = _request(payload)
    profile = next((item for item in config["profiles"] if item.get("id") == profile_id), None)
    if profile is None or profile.get("adapter") != "openpi-sft" or profile.get("execution_mode", "native") != "native":
        raise AgentError("PROFILE_UNAVAILABLE", "Recipe details require a configured native OpenPI environment")
    try:
        checkout = allowed_path(profile["code_dir"], config["roots"])
        for source in SOURCE_PATHS:
            path = checkout / source
            if path.exists() or path.is_symlink():
                break
        else:
            raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "OpenPI training config.py was not found in this checkout")
        pin = _git_pin(checkout, config["roots"])
        content, identity, resolved = _source(path, checkout, config["roots"])
        try:
            tree = ast.parse(content)
            if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
                raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded syntax size")
            encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
            text = content.decode(encoding)
        except (SyntaxError, ValueError, UnicodeError, RecursionError) as exc:
            raise AgentError("RECIPE_SOURCE_INVALID", "Recipe source cannot be parsed statically") from exc
        reader = _PrimaryReader(checkout, checkout / "src" if source.startswith("src/") else checkout, config["roots"], tree)
        environment = reader.module("openpi.training.config", primary=True)
        registry = reader.value(ast.Name(id="_CONFIGS"), environment or {}, set(), 0)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == "TrainConfig")
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == "TrainConfig"))
                 and any(keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and keyword.value.value == recipe
                         for keyword in node.keywords)]
        if recipe not in registry.names:
            if registry.warnings or calls:
                raise AgentError("RECIPE_DETAIL_UNSUPPORTED", "This recipe cannot be resolved from a primary-file literal registry")
            raise AgentError("RECIPE_NOT_FOUND", "Recipe was not found in the configured checkout")
        if len(calls) != 1 or registry.names.count(recipe) != 1:
            raise AgentError("RECIPE_DETAIL_UNSUPPORTED", "Recipe declaration is ambiguous in this source")
        selected = calls[0]
        if selected.args or any(keyword.arg is None for keyword in selected.keywords):
            raise AgentError("RECIPE_DETAIL_UNSUPPORTED", "Recipe details require explicit keyword fields")
        keywords = {keyword.arg: keyword.value for keyword in selected.keywords}
        if len(keywords) != len(selected.keywords):
            raise AgentError("RECIPE_DETAIL_UNSUPPORTED", "Recipe declaration contains duplicate fields")
        fields = _Fields(text)
        snippet = fields.segment(selected)
        if len(snippet) > MAX_SOURCE_CODE_CHARS:
            raise AgentError("RECIPE_DETAIL_UNSUPPORTED", "Recipe declaration exceeds the bounded snippet size")
        defaults = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"]
        for name, node in keywords.items():
            fields.add(name, node, "recipe")
        if len(defaults) == 1:
            for statement in defaults[0].body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None:
                    if statement.target.id not in keywords:
                        fields.add(statement.target.id, statement.value, "default")
                        if _literal(statement.value) is _UNRESOLVED:
                            fields.warnings.add("UNRESOLVED_DEFAULTS")
        else:
            fields.warnings.add("UNRESOLVED_DEFAULTS")
        current, current_identity, current_resolved = _source(path, checkout, config["roots"])
        if current != content or current_identity != identity or current_resolved != resolved or _git_pin(checkout, config["roots"]) != pin:
            raise AgentError("RECIPE_SOURCE_CHANGED", "Recipe source or tracked code changed while details were being read")
        return {"profile_id": profile_id, "recipe": recipe,
                "source": {"path": source, "sha256": hashlib.sha256(content).hexdigest(), "line": selected.lineno,
                           "end_line": selected.end_lineno, "code_dir": str(checkout), **pin},
                "fields": fields.fields, "source_code": snippet, "warnings": sorted(fields.warnings | registry.warnings)}
    except AgentError:
        raise
    except (KeyError, OSError, RuntimeError, UnicodeError) as exc:
        raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "Recipe details are unavailable or unreadable") from exc
