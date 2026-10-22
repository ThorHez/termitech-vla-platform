"""Package this working tree and render private Linux controller service files.

This utility never connects to a machine, stops a service, or enables a unit.
State snapshots require an explicitly quiesced source controller. Training code,
datasets, models, and Agent state are deliberately outside the controller package.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "pyproject.toml", "uv.lock", "README.md", "apps/control_plane", "apps/agent",
    "apps/web", "packages/protocol", "packages/adapters", "deploy", "scripts", "docs",
)
EXCLUDED_PARTS = {"__pycache__", "node_modules", ".venv", ".git", ".local", ".state"}
EXCLUDED_NAMES = {".env", ".DS_Store", ".env.local", ".env.production"}


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Private output directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def regular_files(directory: Path):
    for parent, directories, files in os.walk(directory, followlinks=False):
        base = Path(parent)
        directories[:] = sorted(
            name for name in directories
            if name not in EXCLUDED_PARTS and not (base / name).is_symlink()
        )
        for name in sorted(files):
            source = base / name
            if (name in EXCLUDED_NAMES or name.endswith((".pyc", ".pyo", ".tsbuildinfo"))
                    or name.startswith(".env.") or source.is_symlink()):
                continue
            if source.is_file():
                yield source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_archive(output: Path, stage: Path, *, excluded: frozenset[str] = frozenset()) -> dict:
    output = output.absolute()
    private_directory(output.parent)
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to replace an existing archive")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for source in sorted(stage.rglob("*")):
            if source.is_file() and source.relative_to(stage).as_posix() not in excluded:
                archive.add(source, arcname=source.relative_to(stage).as_posix(), recursive=False)
    return {"archive": str(output), "sha256": sha256(output), "bytes": output.stat().st_size}


def package(args) -> dict:
    source_root = args.root.resolve()
    if args.build:
        subprocess.run(["pnpm", "--dir", str(source_root / "apps/web"), "build"], check=True)
        subprocess.run([shutil.which("uv") or "uv", "run", "python",
                        str(source_root / "scripts/build_agent_bundle.py")], cwd=source_root, check=True)
    bundle = (args.agent_bundle or source_root / ".local/agent-dist/vla-agent-bundle.tar.gz").resolve()
    if not (source_root / "apps/web/dist/index.html").is_file() or not bundle.is_file():
        raise ValueError("Build the web distribution and Agent bundle before packaging, or use --build")
    with tempfile.TemporaryDirectory(prefix="vla-control-release-") as temporary:
        stage = Path(temporary)
        for name in SOURCE_ROOTS:
            source = source_root / name
            if not source.exists() or source.is_symlink():
                raise ValueError(f"Required source path is missing or a symlink: {name}")
            candidates = [source] if source.is_file() else regular_files(source)
            for candidate in candidates:
                destination = stage / candidate.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
        artifacts = stage / "artifacts"
        artifacts.mkdir()
        shutil.copyfile(bundle, artifacts / "vla-agent-bundle.tar.gz")
        # Reuse the already locked wheel and requirements produced by the Agent
        # builder. The same wheel contains controller, Agent, protocol and adapters.
        with tarfile.open(bundle, "r:gz") as archive:
            entries = archive.getmembers()
            manifest_entry = next((entry for entry in entries if entry.name == "manifest.json"), None)
            if manifest_entry is None or not manifest_entry.isfile():
                raise ValueError("Agent bundle is missing its integrity manifest")
            manifest = json.load(archive.extractfile(manifest_entry))
            wheels = [entry for entry in entries if entry.name.endswith(".whl")]
            requirements = [entry for entry in entries if entry.name == "requirements.txt"]
            if len(wheels) != 1 or len(requirements) != 1:
                raise ValueError("Agent bundle must contain one platform wheel and locked requirements")
            for entry in wheels + requirements:
                if not entry.isfile() or Path(entry.name).name != entry.name:
                    raise ValueError("Unsafe bundle artifact")
                data = archive.extractfile(entry).read()
                if manifest.get(entry.name) != hashlib.sha256(data).hexdigest():
                    raise ValueError("Bundle artifact does not match its integrity manifest")
                (artifacts / entry.name).write_bytes(data)
            for bundle_name, source_name in {
                "install-agent.sh": "deploy/install-agent.sh",
                "install_agent.py": "deploy/install_agent.py",
                "agent_ownership.py": "deploy/agent_ownership.py",
                "agent-bootstrap.sh": "apps/control_plane/vla_control/agent_bootstrap.sh",
            }.items():
                if manifest.get(bundle_name) != sha256(stage / source_name):
                    raise ValueError("Agent bundle is older than the working tree; rebuild it before packaging")
        with zipfile.ZipFile(artifacts / wheels[0].name) as wheel:
            for relative in ("apps/control_plane/vla_control", "apps/agent/vla_agent",
                             "packages/protocol/vla_protocol", "packages/adapters/vla_adapters"):
                package_root = stage / relative
                for source in package_root.rglob("*.py"):
                    archive_name = source.relative_to(package_root.parent).as_posix()
                    try:
                        packaged = wheel.read(archive_name)
                    except KeyError as error:
                        raise ValueError("Runtime wheel is missing current source; rebuild the Agent bundle") from error
                    if hashlib.sha256(packaged).hexdigest() != sha256(source):
                        raise ValueError("Runtime wheel is older than current source; rebuild the Agent bundle")
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "working-tree-snapshot",
            "source_roots": list(SOURCE_ROOTS),
            "files": {str(path.relative_to(stage)): sha256(path)
                      for path in sorted(stage.rglob("*")) if path.is_file()},
        }
        try:
            provenance["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
            provenance["git_status"] = subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=source_root, text=True, stderr=subprocess.DEVNULL,
            ).splitlines()
        except subprocess.CalledProcessError:
            provenance["git_head"] = None
        (stage / "release-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        result = new_archive(args.output, stage)
        result["source_files"] = len(provenance["files"])
        return result


def credentials(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Credentials must be an existing regular private file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("enrollment_token"), str) or not value["enrollment_token"].strip():
        raise ValueError("Credentials must contain a nonempty enrollment token")
    legacy = value.get("admin_token", "")
    if not isinstance(legacy, str) or legacy == value["enrollment_token"]:
        raise ValueError("Legacy administrator and enrollment credentials must be different")
    # Preserve an old value for state snapshots; human login no longer requires
    # this credential and the generated service explicitly disables its use.
    return {"admin_token": legacy, "enrollment_token": value["enrollment_token"]}


def snapshot(args) -> dict:
    if not args.controller_stopped:
        raise ValueError("Stop the source controller before the final snapshot; pass --controller-stopped")
    database = args.database.resolve(strict=True)
    remote_state = args.remote_state.resolve(strict=True)
    credential_value = credentials(args.credentials)
    if not remote_state.is_dir() or not (remote_state / "credentials.key").is_file():
        raise ValueError("Copy the SSH credential key and remote-management directory with the database")
    with tempfile.TemporaryDirectory(prefix="vla-control-state-") as temporary:
        stage = Path(temporary)
        # sqlite3.Connection.__exit__ commits/rolls back but does not close the
        # handle. Explicitly close both handles before hashing or archiving, and
        # convert the backup to rollback-journal mode so it is one portable file
        # even when the source uses WAL and still has open readers.
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(stage / "control.db")) as destination:
                source.backup(destination)
                destination.commit()
                if destination.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                    raise ValueError("SQLite backup could not be made self-contained")
                destination.commit()
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("SQLite backup integrity check failed")
        transient_files = frozenset({"control.db-wal", "control.db-shm"})
        if any((stage / name).exists() for name in transient_files):
            raise ValueError("SQLite backup still has transient WAL state after closing")
        private_write(stage / "credentials.json", (json.dumps(credential_value, indent=2) + "\n").encode())
        copied_state = stage / "remote-management"
        copied_state.mkdir(mode=0o700)
        for source in regular_files(remote_state):
            destination = copied_state / source.relative_to(remote_state)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            private_write(destination, source.read_bytes())
        manifest = {str(path.relative_to(stage)): sha256(path)
                    for path in sorted(stage.rglob("*")) if path.is_file()
                    and path.relative_to(stage).as_posix() not in transient_files}
        private_write(stage / "state-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        return new_archive(args.output, stage, excluded=transient_files)


def linux_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) == "/" or ".." in path.parts
            or re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is None):
        raise argparse.ArgumentTypeError("Expected an absolute, dedicated Linux path without whitespace or specifiers")
    return str(path)


def systemd_quote(value: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError("Service settings must not contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def env_quote(value: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError("Environment values must not contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$") + '"'


def render(args) -> dict:
    credential_value = credentials(args.credentials)
    output = args.output_dir.absolute()
    private_directory(output)
    release, state, python = (linux_path(value) for value in
                              (args.release_dir, args.state_dir, args.python))
    values = {
        "PYTHONUNBUFFERED": "1",
        "VLA_ADMIN_TOKEN": credential_value["admin_token"],
        "VLA_LEGACY_ADMIN_ENABLED": "false",
        "VLA_SESSION_COOKIE_SECURE": "true",
        "VLA_SESSION_TTL_SECONDS": "7200",
        "VLA_SESSION_IDLE_SECONDS": "900",
        "VLA_SESSION_REAUTH_SECONDS": "300",
        "VLA_ENROLLMENT_TOKEN": credential_value["enrollment_token"],
        "VLA_DATABASE_URL": "sqlite:///" + state + "/control.db",
        "VLA_HOST": args.bind,
        "VLA_PORT": str(args.port),
        "VLA_WEB_DIST": release + "/apps/web/dist",
        "VLA_AGENT_BUNDLE": release + "/artifacts/vla-agent-bundle.tar.gz",
        "VLA_REMOTE_STATE_DIR": state + "/remote-management",
        "VLA_AGENT_CONTROL_PLANE_URL": f"http://127.0.0.1:{args.port}",
        "VLA_TUNNEL_TARGET_HOST": "127.0.0.1",
        "VLA_TUNNEL_TARGET_PORT": str(args.port),
        "VLA_SCHEDULER_ENABLED": "true",
        "VLA_SCHEDULER_INTERVAL_SECONDS": "0.5",
    }
    environment = "".join(key + "=" + env_quote(value) + "\n" for key, value in values.items())
    environment_file = output / "control.env"
    private_write(environment_file, environment.encode())
    # EnvironmentFile must use the final server path rather than the local
    # staging path used to render this unit.
    final_environment = linux_path(args.environment_file or state + "/control.env")
    unit = "\n".join((
        "[Unit]", "Description=Termitech VLA controller and web",
        "Wants=network-online.target", "After=network-online.target", "", "[Service]",
        "Type=simple", "User=root", "Group=root", "UMask=0077",
        "WorkingDirectory=" + release,
        "EnvironmentFile=" + final_environment,
        "ExecStart=" + systemd_quote(python) + " -m vla_control.main",
        "Restart=on-failure", "RestartSec=3", "TimeoutStopSec=30", "KillMode=mixed",
        "NoNewPrivileges=true", "PrivateTmp=true", "ProtectSystem=full", "LimitNOFILE=65536",
        "", "[Install]", "WantedBy=multi-user.target", "",
    ))
    private_write(output / "termitech-vla-control.service", unit.encode())
    return {"unit": str(output / "termitech-vla-control.service"), "environment": str(environment_file),
            "final_environment": final_environment, "worker_count": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("package", help="Archive current source, built web, wheel and Agent bundle")
    source.add_argument("--root", type=Path, default=ROOT)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--agent-bundle", type=Path)
    source.add_argument("--build", action="store_true")
    source.set_defaults(run=package)
    state = commands.add_parser("snapshot", help="Snapshot a quiesced SQLite database and SSH metadata")
    state.add_argument("--database", type=Path, default=ROOT / ".local/control.db")
    state.add_argument("--credentials", type=Path, default=ROOT / ".local/dev-credentials.json")
    state.add_argument("--remote-state", type=Path, default=ROOT / ".local/remote-management")
    state.add_argument("--output", type=Path, required=True)
    state.add_argument("--controller-stopped", action="store_true")
    state.set_defaults(run=snapshot)
    service = commands.add_parser("render", help="Write private systemd unit and environment files")
    service.add_argument("--credentials", type=Path, required=True)
    service.add_argument("--release-dir", type=linux_path, required=True)
    service.add_argument("--state-dir", type=linux_path, required=True)
    service.add_argument("--python", type=linux_path, required=True)
    service.add_argument("--environment-file", type=linux_path)
    service.add_argument("--bind", default="127.0.0.1")
    service.add_argument("--port", type=int, default=8000)
    service.add_argument("--output-dir", type=Path, required=True)
    service.set_defaults(run=render)
    args = parser.parse_args()
    if hasattr(args, "port") and not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        print(json.dumps(args.run(args), indent=2))
    except (OSError, ValueError, sqlite3.Error, tarfile.TarError) as error:
        # Error messages deliberately contain no credential contents.
        parser.exit(1, f"Migration preparation failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
