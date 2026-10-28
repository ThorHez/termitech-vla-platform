"""Discover registry-bound OpenPI recipe literals without importing training code."""
from __future__ import annotations

import ast
import os
import re
import stat
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vla_protocol.recipe_search import recipe_matches, recipe_sort_key

from .safety import AgentError, allowed_path


MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILES = 8
MAX_AST_NODES = 100_000
MAX_EVALUATION_DEPTH = 40
MAX_COLLECTED_NAMES = 5000
SCAN_BUDGET_SECONDS = 0.5
RECIPE_NAME = re.compile(r"[A-Za-z0-9_-]{1,255}\Z", re.ASCII)
SOURCE_PATHS = ("src/openpi/training/config.py", "openpi/training/config.py")
RECIPE_PART_MODULES = frozenset(f"openpi.training.recipes.{family}" for family in ("sft", "recap", "mem", "rlt", "uot"))
RECIPE_PART_PATHS = tuple(f"{prefix}openpi/training/recipes/{family}.py"
                         for prefix in ("src/", "") for family in ("sft", "recap", "mem", "rlt", "uot"))


@dataclass
class _Value:
    names: list[str] = field(default_factory=list)
    warnings: set[str] = field(default_factory=set)
    truncated: bool = False

    def include(self, other: _Value) -> None:
        remaining = max(0, MAX_COLLECTED_NAMES - len(self.names))
        exceeded = len(other.names) > remaining
        # Slice before extending: self/alias extension cannot grow exponentially.
        self.names.extend(other.names[:remaining])
        self.warnings.update(other.warnings)
        self.truncated = self.truncated or other.truncated or exceeded
        if exceeded:
            self.warnings.add("DISCOVERY_LIMIT_REACHED")


@dataclass
class _Imported:
    module: str
    attribute: str | None = None


@dataclass
class _Binding:
    expression: ast.expr
    environment: dict
    cached: _Value | None = None


def _unknown() -> _Value:
    return _Value(warnings={"DYNAMIC_REGISTRY_UNSUPPORTED"})


def _limited() -> _Value:
    return _Value(warnings={"DISCOVERY_LIMIT_REACHED"}, truncated=True)


def _validate_request(payload: dict) -> tuple[str, str, int]:
    profile_id, prefix, limit = payload.get("profile_id"), payload.get("prefix"), payload.get("limit", 50)
    if (not isinstance(profile_id, str) or not 1 <= len(profile_id) <= 200
            or any(unicodedata.category(char) == "Cc" for char in profile_id)
            or not isinstance(prefix, str) or len(prefix) > 255
            or (prefix and not RECIPE_NAME.fullmatch(prefix))
            or not isinstance(limit, int) or isinstance(limit, bool)):
        raise AgentError("INVALID_RECIPE_QUERY", "Recipe query requires a safe profile, prefix and integer limit")
    try:
        expires_at = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        if expires_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("UTC timestamp required")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise AgentError("INVALID_RECIPE_QUERY", "Recipe query requires an expires_at timestamp in UTC") from exc
    if expires_at <= datetime.now(timezone.utc):
        raise AgentError("RECIPE_QUERY_EXPIRED", "Recipe suggestion query has expired; submit a new query")
    return profile_id, prefix, min(50, max(1, limit))


class _Reader:
    def __init__(self, checkout: Path, module_root: Path, roots: list[str]):
        self.checkout, self.module_root, self.roots = checkout, module_root, roots
        self.started = time.monotonic()
        self.total_bytes = self.total_nodes = 0
        self.modules: dict[str, dict] = {}
        self.loading: set[str] = set()
        self.limit_reached = False

    def budget_exhausted(self) -> bool:
        return time.monotonic() - self.started > SCAN_BUDGET_SECONDS

    def read(self, path: Path, *, primary: bool = False) -> ast.Module:
        resolved = allowed_path(str(path), self.roots)
        if resolved != self.checkout and self.checkout not in resolved.parents:
            raise AgentError("PATH_OUTSIDE_ROOTS", "Recipe source must stay inside its configured checkout")
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "Recipe source must be a readable regular file")
            if info.st_size > MAX_SOURCE_BYTES or self.total_bytes + info.st_size > MAX_TOTAL_BYTES:
                raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded discovery size")
            # The capped read also covers a regular file growing after fstat.
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                content = handle.read(MAX_SOURCE_BYTES + 1)
            if len(content) > MAX_SOURCE_BYTES or self.total_bytes + len(content) > MAX_TOTAL_BYTES:
                raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded discovery size")
            self.total_bytes += len(content)
        finally:
            os.close(descriptor)
        try:
            tree = ast.parse(content)
            for _ in ast.walk(tree):
                self.total_nodes += 1
                if self.total_nodes > MAX_AST_NODES:
                    raise AgentError("RECIPE_SOURCE_TOO_LARGE", "Recipe source exceeds the bounded syntax size")
            return tree
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise AgentError("RECIPE_SOURCE_INVALID", "Recipe source cannot be parsed statically") from exc

    def module(self, name: str, *, primary: bool = False) -> dict | None:
        if name in self.loading:
            return None
        if name in self.modules:
            return self.modules[name]
        if (len(self.modules) + len(self.loading) >= MAX_SOURCE_FILES or self.budget_exhausted()
                or not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part, re.ASCII) for part in name.split("."))):
            self.limit_reached = True
            return None
        path = self.module_root.joinpath(*name.split(".")).with_suffix(".py")
        self.loading.add(name)
        try:
            tree = self.read(path, primary=primary)
            env: dict = {}
            for statement in tree.body:
                if self.budget_exhausted():
                    value = self.value(ast.Name(id="_CONFIGS"), env, set(), 0)
                    value.include(_limited())
                    env["_CONFIGS"] = value
                    break
                if isinstance(statement, ast.ImportFrom) and statement.module is not None:
                    package = name.split(".")[:-1]
                    module = ".".join(package[:len(package) - statement.level + 1] + statement.module.split(".")) if statement.level else statement.module
                    for alias in statement.names:
                        env[alias.asname or alias.name] = _Imported(module, alias.name)
                elif isinstance(statement, ast.Import):
                    for alias in statement.names:
                        # Unaliased dotted imports bind their first component.
                        env[alias.asname or alias.name.split(".")[0]] = _Imported(alias.name if alias.asname else alias.name.split(".")[0])
                elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    expression = statement.value
                    if expression is not None:
                        # Freeze referenced bindings at assignment time. Laziness
                        # avoids reading unrelated imports, while shared cached
                        # values preserve list alias mutation semantics.
                        captured = {node.id: env[node.id] for node in ast.walk(expression)
                                    if isinstance(node, ast.Name) and node.id in env}
                        binding = _Binding(expression, captured)
                        eager = (any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in targets)
                                 or isinstance(expression, (ast.List, ast.Tuple))
                                 or (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add))
                                 or (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                                     and expression.func.id == "TrainConfig"))
                        if eager:
                            binding.cached = self.value(expression, captured, set(), 0)
                        for target in targets:
                            if isinstance(target, ast.Name):
                                env[target.id] = binding.cached if binding.cached is not None else binding
                elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                    previous = self.value(statement.target, env, set(), 0)
                    if isinstance(statement.op, ast.Add):
                        previous.include(self.value(statement.value, env, set(), 0))
                        env[statement.target.id] = previous
                    else:
                        env[statement.target.id] = _unknown()
                elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                      and isinstance(statement.value.func, ast.Attribute) and isinstance(statement.value.func.value, ast.Name)):
                    call = statement.value
                    target = call.func.value.id
                    if target in env:
                        previous = self.value(call.func.value, env, set(), 0)
                        if call.func.attr in {"append", "extend"} and len(call.args) == 1 and not call.keywords:
                            previous.include(self.value(call.args[0], env, set(), 0))
                        else:
                            previous.include(_unknown())
                        env[target] = previous
                elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    # Conditional registry changes cannot be established without
                    # executing code. Keep only entries already unconditional.
                    if any((isinstance(node, ast.Name) and node.id == "_CONFIGS" and isinstance(node.ctx, ast.Store))
                           or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                               and isinstance(node.func.value, ast.Name) and node.func.value.id == "_CONFIGS")
                           for node in ast.walk(statement)):
                        value = self.value(ast.Name(id="_CONFIGS"), env, set(), 0)
                        value.include(_unknown())
                        env["_CONFIGS"] = value
            self.modules[name] = env
            return env
        except (AgentError, OSError, RuntimeError) as exc:
            if primary:
                raise
            if isinstance(exc, AgentError) and exc.code == "RECIPE_SOURCE_TOO_LARGE":
                self.limit_reached = True
            return None
        finally:
            self.loading.remove(name)

    def imported(self, reference: _Imported, depth: int) -> _Value:
        if reference.attribute is None:
            return _unknown()
        env = self.module(reference.module)
        if env is None:
            return _limited() if self.limit_reached else _unknown()
        return self.value(ast.Name(id=reference.attribute), env, set(), depth + 1)

    def value(self, expression: ast.expr, env: dict, visiting: set[int], depth: int) -> _Value:
        if isinstance(expression, ast.Name) and isinstance(env.get(expression.id), _Value):
            return env[expression.id]
        if self.budget_exhausted() or depth >= MAX_EVALUATION_DEPTH:
            return _limited()
        if isinstance(expression, ast.Name):
            value = env.get(expression.id)
            if isinstance(value, _Value):
                return value
            if isinstance(value, _Imported):
                return self.imported(value, depth)
            if isinstance(value, _Binding):
                if value.cached is not None:
                    return value.cached
                if id(value) in visiting:
                    return _unknown()
                value.cached = self.value(value.expression, value.environment, visiting | {id(value)}, depth + 1)
                return value.cached
            return _unknown()
        if isinstance(expression, ast.Attribute):
            parts = []
            base: ast.expr = expression
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            reference = env.get(base.id) if isinstance(base, ast.Name) else None
            if isinstance(reference, _Imported) and reference.attribute is None:
                return self.imported(_Imported(".".join([reference.module, *reversed(parts[1:])]), parts[0]), depth)
            return _unknown()
        if isinstance(expression, (ast.List, ast.Tuple)):
            result = _Value()
            for item in expression.elts:
                result.include(self.value(item.value if isinstance(item, ast.Starred) else item, env, visiting, depth + 1))
            return result
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            first = self.value(expression.left, env, visiting, depth + 1)
            result = _Value(list(first.names), set(first.warnings), first.truncated)
            result.include(self.value(expression.right, env, visiting, depth + 1))
            return result
        if isinstance(expression, ast.Call) and isinstance(expression.func, (ast.Name, ast.Attribute)):
            constructor = expression.func.id if isinstance(expression.func, ast.Name) else expression.func.attr
            if constructor == "TrainConfig":
                name = next((item.value for item in expression.keywords if item.arg == "name"), None)
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    return _Value([name.value]) if RECIPE_NAME.fullmatch(name.value) else _Value(warnings={"INVALID_RECIPE_NAME"})
        return _unknown()


def suggest_recipes(config: dict, payload: dict) -> dict:
    """Return literal registry recipe names; discovery never grants validation."""
    profile_id, prefix, limit = _validate_request(payload)
    profile = next((profile for profile in config["profiles"] if profile.get("id") == profile_id), None)
    if profile is None or profile.get("adapter") != "openpi-sft" or profile.get("execution_mode", "native") != "native":
        raise AgentError("PROFILE_UNAVAILABLE", "Recipe discovery requires a configured native OpenPI training environment")
    try:
        checkout = allowed_path(profile["code_dir"], config["roots"])
        if not checkout.is_dir():
            raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "Training checkout is unavailable")
        for source in SOURCE_PATHS:
            path = checkout / source
            if path.exists() or path.is_symlink():
                break
        else:
            raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "OpenPI training config.py was not found in this checkout")
        module_root = checkout / "src" if source.startswith("src/") else checkout
        reader = _Reader(checkout, module_root, config["roots"])
        env = reader.module("openpi.training.config", primary=True)
        discovered = reader.value(ast.Name(id="_CONFIGS"), env or {}, set(), 0)
        names = sorted({name for name in discovered.names if recipe_matches(name, prefix)},
                       key=lambda name: recipe_sort_key(name, prefix))
        return {"profile_id": profile_id, "prefix": prefix, "entries": [{"name": name} for name in names[:limit]],
                "truncated": discovered.truncated or len(names) > limit, "source": source,
                "warnings": sorted(discovered.warnings)}
    except AgentError:
        raise
    except (KeyError, OSError, RuntimeError) as exc:
        raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "Recipe source is unavailable or unreadable") from exc
