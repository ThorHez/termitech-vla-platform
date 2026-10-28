"""Check or restore the pinned ARX5 and UMI source snapshots without running them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATHS = {"arx5-sdk": "third_party/arx5-sdk", "umi-arx-deploy": "third_party/umi-arx-deploy"}
MANIFEST = "third_party/robot-sources.lock.json"


class SourceError(ValueError):
    pass


def relative_path(value):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or value.startswith("/") or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(part in {"", ".", ".."} or part.casefold() == ".git" for part in value.split("/"))
            or PurePosixPath(value).as_posix() != value):
        raise SourceError("Unsafe source path: " + repr(value))
    return value


def local_path(root, value):
    candidate = root / relative_path(value)
    current = candidate
    while current != root:
        if current.is_symlink():
            raise SourceError("Symbolic links are not accepted in managed source paths: " + value)
        current = current.parent
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise SourceError("Source path escapes its project: " + value)
    return candidate


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git(project, *arguments, check=True):
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    result = subprocess.run(["git", "-C", str(project), *arguments], capture_output=True,
                            env=environment, timeout=60)
    if check and result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()[:2000]
        raise SourceError("Git operation failed: " + message)
    return result


def load_projects(root, selected=None):
    manifest_path = local_path(root, MANIFEST)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise SourceError("Robot source lock file is missing or too large")
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("projects"), list)):
        raise SourceError("Unsupported robot source lock schema")
    projects = manifest["projects"]
    names = [item.get("name") for item in projects if isinstance(item, dict)]
    if len(names) != len(projects) or len(names) != len(set(names)) or set(names) != set(PROJECT_PATHS):
        raise SourceError("The lock file must describe exactly arx5-sdk and umi-arx-deploy")
    if selected is not None and selected not in PROJECT_PATHS:
        raise SourceError("Unknown robot source project")
    return [item for item in projects if selected is None or item["name"] == selected]


def patch_paths(project, patch):
    content = patch.read_bytes()
    if not content or len(content) > 64 * 1024 * 1024:
        raise SourceError("Source patch is empty or exceeds the size limit")
    # Do not permit a patch to create links or nested repositories, even if a
    # future lock file accidentally includes such a change.
    for mode in re.findall(rb"^(?:old mode|new mode|new file mode|deleted file mode) ([0-9]+)$", content, re.MULTILINE):
        if mode not in {b"100644", b"100755"}:
            raise SourceError("Source patches may contain only regular files")
    output = git(project, "apply", "--numstat", "-z", str(patch)).stdout.split(b"\0")
    paths, index = set(), 0
    while index < len(output):
        record = output[index]
        index += 1
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise SourceError("Invalid source patch file inventory")
        if fields[2]:
            paths.add(relative_path(fields[2].decode("utf-8")))
        else:
            # Git's NUL format uses two separate path records for renames.
            if index + 1 >= len(output):
                raise SourceError("Invalid source patch rename inventory")
            for name in output[index:index + 2]:
                paths.add(relative_path(name.decode("utf-8")))
            index += 2
    if not paths:
        raise SourceError("Source patch contains no files")
    return paths


def prepare(root, item):
    name = item["name"]
    if item.get("path") != PROJECT_PATHS[name]:
        raise SourceError("Robot sources cannot target a different project directory")
    project = local_path(root, item["path"])
    if not project.is_dir() or git(project, "rev-parse", "--show-toplevel", check=False).returncode:
        raise SourceError(f"Initialize {item['path']} with git submodule update --init first")
    top = Path(git(project, "rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    if top != project.resolve():
        raise SourceError(f"{item['path']} is not an initialized independent repository")
    base = item.get("base_commit")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base):
        raise SourceError("Source base commit must be a full Git SHA")
    head = git(project, "rev-parse", "HEAD").stdout.decode().strip()
    if head != base:
        raise SourceError(f"{name}: HEAD differs from the pinned base; no checkout or reset was performed")
    patch_info = item.get("patch")
    if not isinstance(patch_info, dict) or not isinstance(patch_info.get("path"), str):
        raise SourceError("Source patch identity is missing")
    if not patch_info["path"].startswith("third_party/patches/") or not patch_info["path"].endswith(".patch"):
        raise SourceError("Source patches must be stored under third_party/patches")
    patch = local_path(root, patch_info["path"])
    if not patch.is_file() or sha256(patch) != patch_info.get("sha256"):
        raise SourceError(f"{name}: source patch SHA-256 differs from the lock file")
    excluded = set()
    for entry in item.get("excluded", []):
        value = entry.get("path") if isinstance(entry, dict) else entry
        excluded.add(relative_path(value))
    files = item.get("files")
    if not isinstance(files, list) or not files or len(files) > 50000:
        raise SourceError("Source inventory must contain a bounded nonempty file list")
    expected = {}
    for row in files:
        if not isinstance(row, dict):
            raise SourceError("Invalid source inventory entry")
        path = relative_path(row.get("path"))
        if (path in expected or path in excluded or row.get("mode") not in {"100644", "100755"}
                or type(row.get("size")) is not int or row["size"] < 0
                or not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise SourceError("Invalid or duplicate source file identity: " + path)
        local_path(project, path)
        expected[path] = row
    changed = patch_paths(project, patch)
    if changed & excluded:
        raise SourceError("Source patch modifies an explicitly excluded file")
    for path in changed:
        local_path(project, path)
    return {"name": name, "project": project, "patch": patch, "base_commit": base,
            "files": expected, "excluded": excluded}


def snapshot_matches(plan):
    project = plan["project"]
    for name, expected in plan["files"].items():
        path = local_path(project, name)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        mode = "100755" if info.st_mode & 0o111 else "100644"
        if (not stat.S_ISREG(info.st_mode) or info.st_size != expected["size"]
                or mode != expected["mode"] or sha256(path) != expected["sha256"]):
            return False
    tracked = git(project, "ls-files", "-z").stdout.split(b"\0")
    for raw in tracked:
        if not raw:
            continue
        name = relative_path(raw.decode("utf-8"))
        if name not in plan["files"] and name not in plan["excluded"] and local_path(project, name).exists():
            return False
    # The reverse check also verifies deletions, renames and executable modes;
    # matching only remaining file hashes could miss a deleted source file.
    return git(project, "apply", "--reverse", "--check", str(plan["patch"]), check=False).returncode == 0


def run(root, operation, selected=None):
    plans = [prepare(root, item) for item in load_projects(root, selected)]
    pending = []
    for plan in plans:
        if snapshot_matches(plan):
            continue
        if git(plan["project"], "status", "--porcelain=v1", "--untracked-files=no").stdout.strip():
            raise SourceError(f"{plan['name']}: tracked local changes differ from the locked snapshot; nothing was overwritten")
        git(plan["project"], "apply", "--check", str(plan["patch"]))
        pending.append(plan)
    if operation == "apply":
        # Preflight every selected project before mutating any of them.
        for plan in pending:
            git(plan["project"], "apply", str(plan["patch"]))
            if not snapshot_matches(plan):
                raise SourceError(f"{plan['name']}: applied files differ from the lock; inspect the checkout without resetting it")
    return {"operation": operation, "projects": [
        {"name": plan["name"], "status": "patch_required" if operation == "check" and plan in pending else "ready",
         "base_commit": plan["base_commit"], "expected_files": len(plan["files"]),
         "verified_files": 0 if operation == "check" and plan in pending else len(plan["files"])} for plan in plans]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "apply"))
    parser.add_argument("--project", choices=tuple(PROJECT_PATHS))
    arguments = parser.parse_args()
    try:
        result = run(ROOT, arguments.operation, arguments.project)
    except (SourceError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(row["status"] != "ready" for row in result["projects"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
