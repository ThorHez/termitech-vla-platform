import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

from vla_protocol.agent_release import read_agent_bundle


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("agent_bundle_builder_fixture", ROOT / "scripts/build_agent_bundle.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def stage(tmp_path, label, *, source=b"# Agent code\n", requirements=b"# pinned requirements\n"):
    root = tmp_path / label
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    (root / "uv.lock").write_text("locked fixture\n")
    output = root / "stage"
    output.mkdir()
    (output / "requirements.txt").write_bytes(requirements)
    wheel = output / "platform.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vla_agent/main.py", source)
        archive.writestr("platform.dist-info/METADATA", "Version: 0.1.0\n")
        archive.writestr("platform.dist-info/RECORD", "platform.dist-info/RECORD,,\n")
    return root, output, wheel


def test_build_identity_tracks_packaged_source_and_locked_dependencies(tmp_path):
    releases = []
    for name, kwargs in (("a", {}), ("b", {}), ("c", {"source": b"# Changed Agent code\n"}),
                         ("d", {"requirements": b"# new dependencies\n"})):
        root, output, wheel = stage(tmp_path, name, **kwargs)
        releases.append(builder.stamp_release(wheel, output, root))
    assert releases[0] == releases[1]
    assert len({row["build_id"] for row in (releases[0], releases[2], releases[3])}) == 3


def test_wheel_and_bundle_share_identity_with_valid_record_without_checkout_writes(tmp_path):
    root, output, wheel = stage(tmp_path, "build")
    original = (root / "pyproject.toml").read_bytes()
    release = builder.stamp_release(wheel, output, root)
    assert json.loads((output / "release.json").read_text()) == release
    with zipfile.ZipFile(wheel) as archive:
        embedded = archive.read(builder.RELEASE_MEMBER)
        assert json.loads(embedded) == release
        records = list(csv.reader(io.StringIO(archive.read("platform.dist-info/RECORD").decode())))
    row = next(row for row in records if row[0] == builder.RELEASE_MEMBER)
    assert row[1] == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(embedded).digest()).decode().rstrip("=")
    assert int(row[2]) == len(embedded)
    assert (root / "pyproject.toml").read_bytes() == original
    assert not (root / "apps").exists()


def test_two_bundle_builds_omit_export_header_containing_temporary_paths(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    (root / "uv.lock").write_text("fixture lock\n")
    for name in ("deploy/install-agent.sh", "deploy/install_agent.py", "deploy/agent_ownership.py",
                 "apps/control_plane/vla_control/agent_bootstrap.sh"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder.shutil, "which", lambda _: "/test/uv")
    exports = []

    def run(arguments, **kwargs):
        if arguments[1] == "build":
            wheel = Path(arguments[arguments.index("--out-dir") + 1]) / "agent.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("vla_agent/main.py", "# source\n")
                archive.writestr("platform.dist-info/RECORD", "platform.dist-info/RECORD,,\n")
        else:
            exports.append(arguments)
            # uv's default header contains the full invocation, including --output-file.
            header = "" if "--no-header" in arguments else "# " + " ".join(arguments) + "\n"
            Path(arguments[arguments.index("--output-file") + 1]).write_text(header + "# locked dependency\n")

    monkeypatch.setattr(builder.subprocess, "run", run)
    releases = []
    for label in ("first", "second"):
        destination = tmp_path / label
        monkeypatch.setattr("sys.argv", ["build_agent_bundle.py", "--output-dir", str(destination)])
        builder.main()
        releases.append(read_agent_bundle(destination / "vla-agent-bundle.tar.gz"))
    assert releases[0] == releases[1]
    assert len(exports) == 2 and all("--no-header" in arguments for arguments in exports)
