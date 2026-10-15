from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .safety import AgentError, allowed_path
from .state import utc_now


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def capture_code(profile: dict, roots: list[str], evidence_dir: Path | None = None) -> dict:
    root = allowed_path(profile["code_dir"], roots)
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
    if changed_during_capture:
        complete = False
        missing.append("code changed during provenance capture")
    snapshot = {
        "repo_url": remote, "code_dir": str(root), "repository_root": str(repository_root),
        "branch": branch, "commit_sha": commit,
        "commit_message": git(root, "log", "-1", "--format=%s").decode().strip(),
        "committed_at": git(root, "log", "-1", "--format=%cI").decode().strip(),
        "dirty": bool(changed), "changes_digest": digest(b"STAGED\0" + staged + b"\0UNSTAGED\0" + unstaged),
        "untracked_digest": digest(json.dumps(untracked, sort_keys=True).encode()),
        "module_origin": str(origin), "captured_at": utc_now(), "complete": complete,
        "missing": missing, "submodules": submodules,
        "evidence_scope": "Git commit, staged/unstaged patches, and untracked file content at launch; ignored files and runtime dependencies are outside this snapshot",
        "untracked_contents_saved": evidence_dir is not None and complete,
        "runtime_digest": digest(json.dumps({key: profile.get(key) for key in (
            "python", "recipe", "adapter", "execution_mode", "docker_image", "metadata")},
            sort_keys=True, separators=(",", ":")).encode()),
    }
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


def compare_code(expected: dict, actual: dict):
    if not expected.get("complete") or not actual.get("complete"):
        raise AgentError("CODE_VERSION_UNAVAILABLE", "Code snapshot is incomplete; repeat preflight after resolving missing evidence")
    fields = ("commit_sha", "code_dir", "repository_root", "dirty", "changes_digest", "untracked_digest", "module_origin", "submodules", "runtime_digest")
    changed = [field for field in fields if expected.get(field) != actual.get(field)]
    if changed:
        raise AgentError("CODE_CHANGED_SINCE_PREFLIGHT", "Code changed after preflight: " + ", ".join(changed))
