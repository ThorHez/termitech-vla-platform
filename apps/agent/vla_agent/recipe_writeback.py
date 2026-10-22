"""Write one statically declared recipe and commit only its config.py change."""
from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import tokenize
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vla_adapters.recipe_definitions import (override_path_tokens, validate_definition,
                                            validate_overrides_against_fields)

from .recipe_details import MAX_AST_NODES, MAX_SOURCE_BYTES, _git, _git_pin, recipe_details
from .safety import AgentError, allowed_path, identity_matches, process_identity, safe_identifier


def _directory(config: dict) -> Path:
    directory = Path(config["state_dir"]) / "recipe-writeback"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _repository_root(config: dict, checkout: Path, *, strict: bool = False) -> Path:
    try:
        return allowed_path(_git(checkout, "rev-parse", "--show-toplevel").decode().strip(), config["roots"])
    except AgentError:
        # START may refer to a non-Git smoke environment. A write still requires
        # the stricter source pin, but its lock must not break such environments.
        if strict:
            raise
        return checkout.resolve()


@contextmanager
def checkout_lock(config: dict, code_dir: str, *, blocking: bool = False, require_git: bool = False):
    """Shared with START so an idle checkout cannot become busy during a write."""
    key = hashlib.sha256(str(_repository_root(config, Path(code_dir).resolve(), strict=require_git)).encode()).hexdigest()
    path = _directory(config) / (key + ".lock")
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise AgentError("RECIPE_WRITE_BUSY", "该代码目录正在保存配方，请稍后重试") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _write_checkout_lock(config: dict, checkout: Path, receipt_path: Path):
    try:
        with checkout_lock(config, str(checkout), require_git=True):
            yield
    except AgentError as exc:
        if exc.code == "CODE_VERSION_UNAVAILABLE" and receipt_path.exists():
            raise AgentError("RECIPE_WRITE_RECOVERY_PENDING", "配方操作已有持久记录，Git 状态暂时无法确认；将自动重试同一操作") from exc
        raise


def _atomic(path: Path, content: bytes, mode: int = 0o600):
    descriptor, temporary = tempfile.mkstemp(prefix=".recipe-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record(path: Path, value: dict):
    _atomic(path, json.dumps(value, ensure_ascii=True, sort_keys=True).encode())


def _git_write(checkout: Path, *arguments: str, input: bytes | None = None,
               index: Path | None = None, allow_missing=False) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "VLA_"))}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    command = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
               "-C", str(checkout), *arguments]
    try:
        result = subprocess.run(command, input=input, capture_output=True, env=environment, timeout=10)
        if result.returncode and not (allow_missing and result.returncode == 1):
            raise ValueError("Git command failed")
        if len(result.stdout) > MAX_SOURCE_BYTES * 2 or len(result.stderr) > 65536:
            raise ValueError("Git command exceeded output bounds")
        return result.stdout
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise AgentError("RECIPE_COMMIT_FAILED", "配方 Git 操作失败；未提交的本次源码改动会回滚") from exc


def _idle(config: dict, state, checkout: Path):
    profiles = {item.get("id"): item for item in config.get("profiles", [])}
    repository = _repository_root(config, checkout, strict=True)
    for attempt in state.active_attempts():
        payload = attempt.get("payload") or {}
        profile = payload.get("profile") or profiles.get(payload.get("profile_id")) or {}
        candidates = [profile.get("code_dir"), (payload.get("snapshot") or {}).get("code_dir"),
                      (payload.get("code_expectation") or {}).get("code_dir")]
        frozen_repository = (payload.get("code_expectation") or {}).get("repository_root")
        if (frozen_repository and Path(frozen_repository).resolve() == repository
                or any(code_dir and _repository_root(config, Path(code_dir).resolve()) == repository for code_dir in candidates)):
            raise AgentError("RECIPE_CHECKOUT_BUSY", "该代码目录仍有训练任务，请等待任务结束后回写配方")


def _offset(text: str, line: int, column: int) -> int:
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[:line - 1])) + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))


def _literal(value) -> ast.expr:
    return ast.parse(repr(value), mode="eval").body


def _set(node: ast.expr, tokens: tuple, value) -> None:
    token, *tail = tokens
    if isinstance(token, str) and isinstance(node, ast.Call) and not node.args and all(keyword.arg is not None for keyword in node.keywords):
        matches = [keyword for keyword in node.keywords if keyword.arg == token]
        if len(matches) > 1:
            raise AgentError("RECIPE_WRITE_UNSUPPORTED", "配方包含重复参数，无法安全回写")
        if not tail:
            if matches:
                matches[0].value = _literal(value)
            else:
                node.keywords.append(ast.keyword(arg=token, value=_literal(value)))
            return
        if len(matches) == 1:
            _set(matches[0].value, tuple(tail), value)
            return
    elif isinstance(token, int) and isinstance(node, (ast.List, ast.Tuple)) and token < len(node.elts):
        if tail:
            _set(node.elts[token], tuple(tail), value)
        else:
            node.elts[token] = _literal(value)
        return
    raise AgentError("RECIPE_WRITE_UNSUPPORTED", "嵌套参数不属于静态构造器，无法安全回写；请在源码中显式定义该参数")


def _default_constructor(node: ast.expr, tree: ast.Module) -> ast.expr:
    """Materialize a zero-argument dataclass default factory as static syntax."""
    modules, fields = {"dataclasses"}, {"field"}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            modules.update(alias.asname or alias.name for alias in statement.names if alias.name == "dataclasses")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "dataclasses":
            fields.update(alias.asname or alias.name for alias in statement.names if alias.name == "field")
    is_field = isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in fields
        or isinstance(node.func, ast.Attribute) and node.func.attr == "field"
        and isinstance(node.func.value, ast.Name) and node.func.value.id in modules)
    if not is_field:
        return copy.deepcopy(node)
    arguments = {keyword.arg: keyword.value for keyword in node.keywords}
    if "default" in arguments and isinstance(arguments["default"], ast.Call):
        return copy.deepcopy(arguments["default"])
    factory = arguments.get("default_factory")
    if isinstance(factory, (ast.Name, ast.Attribute)):
        return ast.Call(func=copy.deepcopy(factory), args=[], keywords=[])
    if (isinstance(factory, ast.Lambda) and not factory.args.args and not factory.args.posonlyargs
            and not factory.args.kwonlyargs and not factory.args.vararg and not factory.args.kwarg
            and isinstance(factory.body, ast.Call)):
        return copy.deepcopy(factory.body)
    raise AgentError("RECIPE_WRITE_UNSUPPORTED", "默认工厂无法静态展开，请在基础配方中显式声明该构造器")


def _render(content: bytes, base_recipe: str, config_name: str, overrides: dict) -> bytes:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
        text = content.decode(encoding)
        tree = ast.parse(text)
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError("Oversized syntax")
        registries = []
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in statement.targets):
                registries.append(statement.value)
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == "_CONFIGS":
                registries.append(statement.value)
        if len(registries) != 1 or not isinstance(registries[0], (ast.List, ast.Tuple)):
            raise AgentError("RECIPE_WRITE_UNSUPPORTED", "回写要求 config.py 中存在唯一静态 _CONFIGS 列表或元组")
        registry = registries[0]
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == "TrainConfig")
                      or isinstance(node.func, ast.Attribute) and node.func.attr == "TrainConfig")]
        def named(node, name):
            return any(keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and keyword.value.value == name for keyword in node.keywords)
        if any(named(node, config_name) for node in calls):
            raise AgentError("RECIPE_NAME_CONFLICT", "该 config 名称已存在，请使用新的名称")
        bases = [node for node in calls if named(node, base_recipe)]
        if len(bases) != 1 or bases[0] not in registry.elts:
            raise AgentError("RECIPE_WRITE_UNSUPPORTED", "基础配方必须是 _CONFIGS 中直接声明的唯一 TrainConfig")
        clone = copy.deepcopy(bases[0])
        # Scalar defaults may be added as explicit keyword fields. Nested edits
        # need a declared constructor, never an evaluated default factory.
        defaults = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"]
        declared = {keyword.arg for keyword in clone.keywords}
        default_values = {}
        if len(defaults) == 1:
            default_values = {statement.target.id: statement.value for statement in defaults[0].body
                              if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None}
        _set(clone, ("name",), config_name)
        for path, value in overrides.items():
            tokens = override_path_tokens(path)
            if len(tokens) > 1 and tokens[0] not in declared and tokens[0] in default_values:
                clone.keywords.append(ast.keyword(arg=tokens[0], value=_default_constructor(default_values[tokens[0]], tree)))
                declared.add(tokens[0])
            _set(clone, tokens, value)
        snippet = ast.unparse(ast.fix_missing_locations(clone))
        # Preserve every original byte/comment. Only the appended constructor is
        # formatted; this deliberately avoids reprinting the training module.
        end = _offset(text, registry.end_lineno, registry.end_col_offset) - 1
        previous = registry.elts[-1]
        previous_end = _offset(text, previous.end_lineno, previous.end_col_offset)
        tail = text[previous_end:end]
        tokens = list(tokenize.generate_tokens(io.StringIO(tail + "\n").readline))
        has_comma = any(token.type == tokenize.OP and token.string == "," for token in tokens)
        prefix = text[:previous_end] + ("" if has_comma else ",") + text[previous_end:end]
        result = (prefix + "\n    " + snippet.replace("\n", "\n    ") + ",\n" + text[end:]).encode(encoding)
        compiled = ast.parse(result)
        if len(result) > MAX_SOURCE_BYTES or sum(1 for _ in ast.walk(compiled)) > MAX_AST_NODES:
            raise ValueError("Oversized generated source")
        return result
    except AgentError:
        raise
    except (SyntaxError, ValueError, UnicodeError, RecursionError, tokenize.TokenError) as exc:
        raise AgentError("RECIPE_WRITE_UNSUPPORTED", "无法为该静态配方生成语法有效的配置") from exc


def _request(config: dict, payload: dict):
    try:
        operation_id = safe_identifier(payload["operation_id"])
        definition = validate_definition(payload["definition"])
        name = payload["config_name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", name):
            raise ValueError("Unsafe config name")
        if payload["profile_id"] != definition["profile_id"]:
            raise ValueError("Profile identity differs")
        expires = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        if expires.utcoffset() != timedelta(0):
            raise ValueError("UTC expiration required")
        profile = next((item for item in config["profiles"] if item.get("id") == payload["profile_id"]), None)
        if not profile or profile.get("adapter") != "openpi-sft" or profile.get("execution_mode", "native") != "native":
            raise ValueError("Unavailable profile")
        checkout = allowed_path(profile["code_dir"], config["roots"])
        if str(checkout) != definition["source"]["code_dir"]:
            raise ValueError("Checkout identity differs")
        return operation_id, definition, name, checkout, expires
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration) as exc:
        raise AgentError("INVALID_RECIPE_WRITE", "回写请求无效、已过期或与训练环境不匹配") from exc


def _clean_source(checkout: Path, relative: str):
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "--", relative)
    tracked = _git(checkout, "ls-files", "--stage", "--", relative).decode().splitlines()
    if status or len(tracked) != 1 or not tracked[0].startswith(("100644 ", "100755 ")):
        raise AgentError("RECIPE_SOURCE_DIRTY", "config.py 有未提交或暂存的改动，请先单独处理这些改动后再回写")
    # status may hide edits behind assume-unchanged/skip-worktree flags. Compare
    # the actual file and index blob with HEAD rather than trusting that hint.
    entry = tracked[0].split("\t", 1)[0].split()
    head_blob = _git(checkout, "rev-parse", f"HEAD:./{relative}").decode().strip()
    original = _git(checkout, "show", f"HEAD:./{relative}", max_bytes=MAX_SOURCE_BYTES)
    source = checkout / relative
    expected_mode = "100755" if source.stat().st_mode & 0o111 else "100644"
    if (len(entry) != 3 or entry[2] != "0" or entry[1] != head_blob or entry[0] != expected_mode
            or source.read_bytes() != original):
        raise AgentError("RECIPE_SOURCE_DIRTY", "config.py 与 HEAD 或索引不一致，请先单独提交已有改动")


def _same_source(path: Path, expected: bytes):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES or path.read_bytes() != expected:
        raise AgentError("RECIPE_SOURCE_CHANGED", "config.py 在保存期间发生变化，请重新读取配方")


def _index_lock(checkout: Path, receipt: dict, receipt_path: Path):
    index = Path(_git(checkout, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    lock = Path(str(index) + ".lock")
    previous = receipt.get("index_lock")
    if lock.exists() and previous:
        info = lock.stat()
        if [info.st_dev, info.st_ino] == previous["identity"] and not identity_matches(previous["owner"]):
            lock.unlink()
    # Persist the inode's ownership before exposing Git's lock name. A worker
    # killed between open() and journaling must never leave an unowned index.lock.
    descriptor, prepared_name = tempfile.mkstemp(prefix=".vla-recipe-index-", dir=index.parent)
    prepared = Path(prepared_name)
    info = os.fstat(descriptor)
    identity = [info.st_dev, info.st_ino]
    try:
        receipt["index_lock"] = {"identity": identity, "owner": process_identity(os.getpid())}
        _record(receipt_path, receipt)
        try:
            os.link(prepared, lock)
        except FileExistsError as exc:
            raise AgentError("RECIPE_GIT_BUSY", "Git 索引正在使用，请等待其他 Git 操作结束后重试") from exc
        prepared.unlink()
        return index, lock, descriptor
    except BaseException:
        os.close(descriptor)
        if lock.exists():
            current = lock.stat()
            if [current.st_dev, current.st_ino] == identity:
                lock.unlink()
        if prepared.exists():
            prepared.unlink()
        raise


def _result(config: dict, payload: dict, receipt: dict) -> dict:
    result = {key: receipt[key] for key in ("operation_id", "definition_digest", "profile_id", "base_recipe", "config_name",
                                            "previous_commit_sha", "commit_sha", "changed_paths")}
    try:
        result["detail"] = recipe_details(config, {"profile_id": payload["profile_id"], "recipe": payload["config_name"],
                                                   "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()})
    except AgentError as exc:
        # The commit is already durable. A subsequent reader error must never
        # cause a second commit, a reverted branch, or an ambiguous retry.
        result["detail"] = None
        result["warning"] = {"code": exc.code, "message": "源码已提交，但详情刷新失败，请重新导入该 config 名称"}
    return result


def write_recipe(config: dict, payload: dict, state) -> dict:
    """Idempotently append a recipe, then create one local, source-only commit."""
    operation_id, definition, name, checkout, expires = _request(config, payload)
    identity = hashlib.sha256(json.dumps({"profile_id": payload["profile_id"], "definition": definition["digest"],
                                         "config_name": name}, sort_keys=True).encode()).hexdigest()
    directory = _directory(config)
    receipt_path = directory / (hashlib.sha256(operation_id.encode()).hexdigest() + ".json")
    with _write_checkout_lock(config, checkout, receipt_path):
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if receipt and receipt.get("request_digest") != identity:
            raise AgentError("RECIPE_OPERATION_CONFLICT", "该回写操作 ID 已被其他配方内容使用")
        if receipt and receipt.get("result"):
            return copy.deepcopy(receipt["result"])
        if receipt is None and expires <= datetime.now(timezone.utc):
            raise AgentError("RECIPE_QUERY_EXPIRED", "配方回写请求已过期，请重新提交")
        relative = definition["source"]["path"]
        source = checkout / relative
        if source.is_symlink() or allowed_path(str(source), config["roots"]) != source:
            raise AgentError("RECIPE_SOURCE_CHANGED", "config.py 必须是训练代码目录中的普通文件")
        _idle(config, state, checkout)
        if receipt is None:
            detail = recipe_details(config, {"profile_id": payload["profile_id"], "recipe": definition["base_recipe"],
                                             "expires_at": payload["expires_at"]})
            if detail["source"] != definition["source"]:
                raise AgentError("RECIPE_SOURCE_CHANGED", "基础配方或 Git 版本已变化，请重新读取并保存版本")
            clean = validate_overrides_against_fields(definition["overrides"], detail["fields"])
            _clean_source(checkout, relative)
            original = source.read_bytes()
            proposed = _render(original, definition["base_recipe"], name, clean)
            mode = stat.S_IMODE(source.stat().st_mode)
            # Prepare the isolated commit tree without touching the real index,
            # worktree, branch, or unrelated staged/unstaged files.
            repository = Path(_git(checkout, "rev-parse", "--show-toplevel").decode().strip())
            repository_relative = source.relative_to(repository).as_posix()
            previous = definition["source"]["commit_sha"]
            blob = _git_write(checkout, "hash-object", "-w", "--no-filters", "--stdin", input=proposed).decode().strip()
            with tempfile.TemporaryDirectory(prefix="recipe-tree-", dir=directory) as temporary:
                commit_index = Path(temporary) / "index"
                _git_write(checkout, "read-tree", previous, index=commit_index)
                file_mode = "100755" if mode & 0o111 else "100644"
                _git_write(checkout, "update-index", "--add", "--cacheinfo", file_mode, blob, repository_relative, index=commit_index)
                tree = _git_write(checkout, "write-tree", index=commit_index).decode().strip()
            git_identity = []
            for key, fallback in (("user.name", "VLA Agent"), ("user.email", "vla-agent@localhost")):
                value = _git_write(checkout, "config", "--get", key, allow_missing=True).decode().strip() or fallback
                git_identity.extend(["-c", f"{key}={value}"])
            message = f"Add training recipe {name}\n\nVLA-Recipe-Operation: {operation_id}\nVLA-Recipe-Digest: {definition['digest']}\n"
            commit = _git_write(checkout, *git_identity, "commit-tree", tree, "-p", previous, input=message.encode()).decode().strip()
            key = receipt_path.stem
            _atomic(directory / (key + ".original"), original)
            _atomic(directory / (key + ".proposed"), proposed)
            receipt = {"request_digest": identity, "operation_id": operation_id, "definition_digest": definition["digest"],
                       "profile_id": payload["profile_id"], "base_recipe": definition["base_recipe"], "config_name": name,
                       "previous_commit_sha": previous, "commit_sha": commit, "changed_paths": [relative],
                       "repository_relative": repository_relative, "blob": blob, "mode": mode, "file_mode": file_mode}
            _record(receipt_path, receipt)
        original = (directory / (receipt_path.stem + ".original")).read_bytes()
        proposed = (directory / (receipt_path.stem + ".proposed")).read_bytes()
        previous, commit = receipt["previous_commit_sha"], receipt["commit_sha"]
        try:
            head = _git(checkout, "rev-parse", "HEAD").decode().strip()
        except AgentError as exc:
            raise AgentError("RECIPE_WRITE_RECOVERY_PENDING", "配方操作已有持久记录，Git 状态暂时无法确认；将自动重试同一操作") from exc
        if head not in {previous, commit}:
            raise AgentError("RECIPE_SOURCE_CHANGED", "Git HEAD 在保存期间变化，请检查已记录的配方提交")
        if head == commit:
            _same_source(source, proposed)
        elif source.read_bytes() == original:
            _clean_source(checkout, relative)
            if _git_pin(checkout, config["roots"]) != {key: definition["source"][key] for key in ("commit_sha", "dirty", "changes_digest")}:
                raise AgentError("RECIPE_SOURCE_CHANGED", "训练代码的 Git 状态已变化，请重新读取配方")
        else:
            _same_source(source, proposed)  # Recover a crash between source write and ref update.
        try:
            index, lock, descriptor = _index_lock(checkout, receipt, receipt_path)
        except Exception as exc:
            if head == commit:
                raise AgentError("RECIPE_WRITE_RECOVERY_PENDING", "配方已提交，正在等待 Git 索引修复；将自动重试同一操作") from exc
            raise
        published = head == commit
        publication_attempted = False
        try:
            if not published and source.read_bytes() == original:
                _clean_source(checkout, relative)
            with tempfile.TemporaryDirectory(prefix="recipe-index-", dir=directory) as temporary:
                new_index = Path(temporary) / "index"
                if index.exists():
                    _atomic(new_index, index.read_bytes())
                else:
                    _git_write(checkout, "read-tree", previous, index=new_index)
                _git_write(checkout, "update-index", "--add", "--cacheinfo", receipt["file_mode"], receipt["blob"],
                           receipt["repository_relative"], index=new_index)
                _idle(config, state, checkout)
                if not published:
                    current = source.read_bytes()
                    if current not in (original, proposed):
                        raise AgentError("RECIPE_SOURCE_CHANGED", "config.py 在保存期间发生变化，请重新读取配方")
                    if current == original:
                        _same_source(source, original)
                        _atomic(source, proposed, receipt["mode"])
                    _same_source(source, proposed)
                    publication_attempted = True
                    _git_write(checkout, "update-ref", "-m", f"VLA recipe {operation_id}", "HEAD", commit, previous)
                    published = True
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(new_index.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(lock, index)
            receipt["result"] = _result(config, payload, receipt)
            _record(receipt_path, receipt)
            return copy.deepcopy(receipt["result"])
        except BaseException as exc:
            # update-ref can succeed before a transport/IO failure is reported.
            # Never cache a terminal rejection for a published, repairable commit.
            observed_head, unknown_head = None, False
            if not published:
                try:
                    observed_head = _git(checkout, "rev-parse", "HEAD").decode().strip()
                    published = observed_head == commit
                except AgentError:
                    unknown_head = True
            if not published and source.exists() and source.read_bytes() == proposed:
                if observed_head == previous:
                    _atomic(source, original, receipt["mode"])
            if (published or publication_attempted and unknown_head) and isinstance(exc, Exception):
                raise AgentError("RECIPE_WRITE_RECOVERY_PENDING", "配方已提交，正在修复操作记录和 Git 索引；将自动重试同一操作") from exc
            raise
        finally:
            os.close(descriptor)
            if lock.exists():
                info = lock.stat()
                if [info.st_dev, info.st_ino] == receipt["index_lock"]["identity"]:
                    lock.unlink()
