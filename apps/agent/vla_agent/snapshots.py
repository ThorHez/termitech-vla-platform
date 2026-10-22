from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from vla_protocol import canonical_digest

from .safety import AgentError, allowed_path
from .state import utc_now


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _entrypoint_digest(path: Path) -> str:
    """Read bounded source content and refuse a concurrently substituted link."""
    maximum = 2 * 1024 * 1024
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise AgentError("INVALID_ENTRYPOINT", "Selected training entrypoint must be a regular Python file no larger than 2 MiB")
            content = source.read(maximum + 1)
            if len(content) > maximum:
                raise AgentError("INVALID_ENTRYPOINT", "Selected training entrypoint exceeds 2 MiB")
    except OSError as exc:
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Selected training entrypoint cannot be read") from exc
    return digest(content)


def git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError("CODE_VERSION_UNAVAILABLE", f"Cannot inspect Git repository at {root}") from exc


def sanitize_remote(value: str) -> str:
    if "://" not in value:
        return value.split("@", 1)[-1] if "@" in value else value
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def capture_code(profile: dict, roots: list[str], evidence_dir: Path | None = None, *, recipe: str | None = None,
                 recipe_definition: dict | None = None, entrypoint: str | None = None,
                 dataset_bindings: list[dict] | None = None, training_stage: str | None = None,
                 stage_config: dict | None = None) -> dict:
    root = allowed_path(profile["code_dir"], roots)
    from vla_adapters.dataset_bindings import normalize_dataset_bindings
    from vla_adapters.entrypoints import validate_entrypoint
    from vla_adapters.recap import validate_training_stage
    try:
        training_stage, stage_config = validate_training_stage("recap" if training_stage is not None else profile.get("adapter"), training_stage, stage_config)
    except ValueError as exc:
        raise AgentError("TRAINING_STAGE_INVALID", str(exc)) from exc
    try:
        dataset_bindings = normalize_dataset_bindings(dataset_bindings)
    except ValueError as exc:
        raise AgentError("DATASET_BINDINGS_INVALID", str(exc)) from exc
    try:
        entrypoint = validate_entrypoint(entrypoint)
    except ValueError as exc:
        raise AgentError("INVALID_ENTRYPOINT", str(exc)) from exc
    entrypoint_sha256 = None
    if entrypoint is not None:
        selected_entry = allowed_path(str(root / entrypoint), [str(root)])
        if not selected_entry.is_file() or selected_entry.suffix != ".py":
            raise AgentError("INVALID_ENTRYPOINT", "Selected training entrypoint must be a Python file inside the training project")
        entrypoint_sha256 = _entrypoint_digest(selected_entry)
    repository_root = allowed_path(git(root, "rev-parse", "--show-toplevel").decode().strip(), roots)
    commit = git(root, "rev-parse", "--verify", "HEAD").decode().strip()
    if len(commit) not in (40, 64):
        raise AgentError("CODE_VERSION_UNAVAILABLE", "No valid Git commit")
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    if branch == "HEAD":
        branch = None
    try:
        remote = sanitize_remote(git(root, "remote", "get-url", "origin").decode().strip())
    except AgentError:
        remote = None
    changed = git(repository_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git(repository_root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
    staged = git(repository_root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary", "--")
    unstaged = git(repository_root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--")
    untracked: list[dict] = []
    untracked_contents: dict[str, bytes] = {}
    untracked_names = git(repository_root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    complete = True
    missing: list[str] = []
    total_bytes = 0
    for name in sorted(n for n in untracked_names if n):
        relative = os.fsdecode(name)
        path = repository_root / relative
        try:
            # Refuse links escaping the repository. Their contents are not training provenance.
            resolved = path.resolve(strict=True)
            if repository_root not in resolved.parents:
                raise ValueError("untracked link escapes code directory")
            size = path.stat().st_size
            total_bytes += size
            if total_bytes > 100 * 1024 * 1024:
                raise ValueError("untracked files exceed 100 MiB capture limit")
            content = path.read_bytes()
            untracked_contents[relative] = content
            untracked.append({"path": relative, "sha256": digest(content), "size": len(content),
                              "symlink": os.readlink(path) if path.is_symlink() else None})
        except (OSError, ValueError) as exc:
            complete = False
            missing.append(f"{relative}: {exc}")
    submodules = []
    submodule_patches: dict[str, dict[str, bytes]] = {}
    for line in git(repository_root, "submodule", "status", "--recursive").decode().splitlines():
        if not line:
            continue
        # Git's status starts with one status character, then SHA and path.
        fields = line[1:].split()
        if len(fields) < 2:
            complete = False
            missing.append("unrecognized submodule status")
            continue
        relative = fields[1]
        if line[0] == "-":
            complete = False
            missing.append(f"uninitialized submodule: {relative}")
            continue
        subroot = allowed_path(str(repository_root / relative), [str(repository_root)])
        substatus = git(subroot, "status", "--porcelain=v1", "-z")
        subdiff = git(subroot, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
        substaged = git(subroot, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary", "--")
        subunstaged = git(subroot, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--")
        submodule_patches[relative] = {"code.diff": subdiff, "staged.diff": substaged, "unstaged.diff": subunstaged}
        submodules.append({"path": relative, "commit_sha": git(subroot, "rev-parse", "HEAD").decode().strip(),
                           "dirty": bool(substatus), "changes_digest": digest(b"STAGED\0" + substaged + b"\0UNSTAGED\0" + subunstaged)})
        if git(subroot, "ls-files", "--others", "--exclude-standard", "-z").strip(b"\0"):
            complete = False
            missing.append(f"untracked submodule files not captured: {relative}")
    if profile.get("adapter") == "smoke":
        origin = allowed_path(str(root / "smoke_training.py"), [str(repository_root)])
    else:
        import vla_adapters
        wrapper_dir = str(Path(vla_adapters.__file__).resolve().parent)
        # Match script execution's sys.path[0], not `python -c`'s implicit cwd.
        probe = (f"import importlib.util,json,sys; sys.path[0]={wrapper_dir!r}; s=importlib.util.find_spec('openpi'); "
                 "print(json.dumps({'origin': s.origin if s else None}))")
        try:
            result = subprocess.run([profile["python"], "-c", probe], cwd=root, check=True,
                                    capture_output=True, text=True, timeout=30)
            located = json.loads(result.stdout.strip())["origin"]
            if not located:
                raise ValueError("openpi could not be located")
            origin = allowed_path(located, [str(repository_root)])
        except (KeyError, ValueError, OSError, subprocess.SubprocessError, AgentError) as exc:
            raise AgentError("IMPORT_PATH_MISMATCH", "Training interpreter does not resolve openpi inside configured repository") from exc
    # Recheck identity after collecting files so a concurrent edit cannot silently pass.
    changed_during_capture = (
        git(root, "rev-parse", "HEAD").decode().strip() != commit
        or git(repository_root, "status", "--porcelain=v1", "-z", "--untracked-files=all") != changed
        or git(repository_root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary", "--") != staged
        or git(repository_root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--") != unstaged
        or git(repository_root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") != untracked_names
    )
    for entry in untracked:
        try:
            current_file = repository_root / entry["path"]
            resolved = allowed_path(str(current_file), [str(repository_root)])
            if digest(resolved.read_bytes()) != entry["sha256"]:
                changed_during_capture = True
        except (OSError, AgentError):
            changed_during_capture = True
    if entrypoint is not None:
        try:
            current_entry = allowed_path(str(root / entrypoint), [str(root)])
            if _entrypoint_digest(current_entry) != entrypoint_sha256:
                changed_during_capture = True
        except (OSError, AgentError):
            changed_during_capture = True
    if changed_during_capture:
        complete = False
        missing.append("code changed during provenance capture")
    runtime = {key: profile.get(key) for key in (
        "python", "recipe", "adapter", "execution_mode", "docker_image", "metadata")}
    # Preserve legacy digests for queued tasks without a per-task recipe. Bind both
    # the original registered profile and task selection when one was frozen.
    if recipe is not None:
        runtime["training_recipe"] = recipe
    if entrypoint is not None:
        runtime["requested_entrypoint"] = entrypoint
    if dataset_bindings is not None:
        runtime["dataset_bindings"] = dataset_bindings
    if training_stage is not None:
        runtime.update(training_stage=training_stage, stage_config=stage_config)
    if recipe_definition is not None:
        from vla_adapters.recipe_definitions import validate_definition
        recipe_definition = validate_definition(recipe_definition)
        runtime["recipe_definition_digest"] = recipe_definition["digest"]
    selected_recipe = recipe if recipe is not None else profile.get("recipe")
    if recipe is None and profile.get("adapter") == "smoke":
        selected_recipe = "smoke"
    snapshot = {
        "repo_url": remote, "code_dir": str(root), "repository_root": str(repository_root),
        "branch": branch, "commit_sha": commit,
        "commit_message": git(root, "log", "-1", "--format=%s").decode().strip(),
        "committed_at": git(root, "log", "-1", "--format=%cI").decode().strip(),
        "dirty": bool(changed), "changes_digest": digest(b"STAGED\0" + staged + b"\0UNSTAGED\0" + unstaged),
        "untracked_digest": digest(json.dumps(untracked, sort_keys=True).encode()),
        "module_origin": str(origin), "captured_at": utc_now(), "complete": complete,
        "recipe": selected_recipe,
        "requested_entrypoint": entrypoint,
        "requested_entrypoint_sha256": entrypoint_sha256,
        "missing": missing, "submodules": submodules,
        "evidence_scope": ("Git commit, staged/unstaged patches, untracked file content, and selected entrypoint SHA-256 at launch; other ignored files and runtime dependencies are outside this snapshot"
                           if entrypoint is not None else
                           "Git commit, staged/unstaged patches, and untracked file content at launch; ignored files and runtime dependencies are outside this snapshot"),
        "untracked_contents_saved": evidence_dir is not None and complete,
        "runtime_digest": digest(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()),
    }
    if recipe_definition is not None:
        from .recipe_execution import verify_recipe_source
        verify_recipe_source(profile, roots, recipe_definition, snapshot)
        snapshot["recipe_definition"] = recipe_definition
    if dataset_bindings is not None:
        snapshot["dataset_bindings"] = dataset_bindings
    if training_stage is not None:
        snapshot.update(training_stage=training_stage, stage_config=stage_config)
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "code.diff").write_bytes(diff)
        (evidence_dir / "staged.diff").write_bytes(staged)
        (evidence_dir / "unstaged.diff").write_bytes(unstaged)
        (evidence_dir / "untracked.json").write_text(json.dumps(untracked, indent=2))
        for relative, content in untracked_contents.items():
            saved = evidence_dir / "untracked" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(content)
        for relative, patches in submodule_patches.items():
            for name, content in patches.items():
                saved = evidence_dir / "submodules" / relative / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(content)
        (evidence_dir / "code-snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return snapshot


def compare_code(expected: dict, actual: dict, *, compare_automatic: bool = True):
    if not expected.get("complete") or not actual.get("complete"):
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Code snapshot is incomplete; repeat preflight after resolving missing evidence")
    fields = ("commit_sha", "code_dir", "repository_root", "dirty", "changes_digest", "untracked_digest", "module_origin", "submodules", "runtime_digest", "requested_entrypoint", "requested_entrypoint_sha256", "dataset_bindings", "training_stage", "stage_config")
    changed = [field for field in fields if expected.get(field) != actual.get(field)]
    if changed:
        raise AgentError("CODE_CHANGED_SINCE_PREFLIGHT", "Code changed after preflight: " + ", ".join(changed))
    if compare_automatic and "automatic_preflight" in expected:
        frozen = expected["automatic_preflight"]
        current = actual.get("automatic_preflight")
        try:
            valid = isinstance(frozen, dict) and isinstance(current, dict)
            valid = valid and frozen.get("schema_version") == current.get("schema_version") == 1
            valid = valid and frozen.get("scope") == current.get("scope") == "configuration_and_local_assets"
            valid = valid and isinstance(frozen.get("plan"), dict) and isinstance(current.get("plan"), dict)
            valid = valid and frozen.get("plan_digest") == canonical_digest(frozen["plan"])
            valid = valid and current.get("plan_digest") == canonical_digest(current["plan"])
            valid = valid and frozen["plan_digest"] == current["plan_digest"]
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise AgentError("ASSETS_CHANGED_SINCE_PREFLIGHT", "训练配置或本机数据、归一化统计、模型在预检后发生变化，请重新预检")
