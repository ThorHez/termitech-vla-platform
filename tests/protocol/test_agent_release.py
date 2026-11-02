import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from vla_protocol import agent_release


RELEASE = {"schema_version": 1, "version": "0.1.0", "build_id": "a" * 64}


def bundle(tmp_path, *, release=None, embedded=None, tampered=False, extra=()):
    release = RELEASE if release is None else release
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(agent_release.RELEASE_MEMBER, json.dumps(release if embedded is None else embedded))
    files = {name: b"fixture" for name in ("requirements.txt", "install_agent.py", "install-agent.sh",
                                          "agent-bootstrap.sh", "agent_ownership.py")}
    files.update({"release.json": json.dumps(release).encode(), "agent.whl": wheel.getvalue()})
    files["manifest.json"] = json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in files.items()}).encode()
    if tampered:
        files["agent.whl"] += b"changed"
    path = tmp_path / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [*files.items(), *extra]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_target_identity_is_checked_against_manifest_and_embedded_wheel(tmp_path):
    path = bundle(tmp_path)
    assert agent_release.read_agent_bundle(path) == {"version": "0.1.0", "build_id": "a" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        agent_release.read_agent_bundle(bundle(tmp_path, tampered=True))
    with pytest.raises(ValueError, match="不一致"):
        agent_release.read_agent_bundle(bundle(tmp_path, embedded={**RELEASE, "build_id": "b" * 64}))


@pytest.mark.parametrize("release", [{}, {**RELEASE, "schema_version": True}, {**RELEASE, "build_id": "latest"}])
def test_legacy_or_invalid_release_cannot_be_upgrade_target(tmp_path, release):
    with pytest.raises(ValueError, match="构建身份"):
        agent_release.read_agent_bundle(bundle(tmp_path, release=release))


@pytest.mark.parametrize("extra", [[("release.json", b"duplicate")], [("../escape", b"escape")]])
def test_duplicate_or_nested_tar_members_are_rejected_without_extraction(tmp_path, extra):
    with pytest.raises(ValueError, match="清单无效"):
        agent_release.read_agent_bundle(bundle(tmp_path, extra=extra))
    assert not (tmp_path.parent / "escape").exists()


def test_installed_release_uses_embedded_metadata_and_dev_has_no_build_id(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_release.resources, "files", lambda _: tmp_path)
    monkeypatch.setattr(agent_release.metadata, "version", lambda _: "0.1.0")
    assert agent_release.installed_agent_release() == {"version": "0.1.0", "build_id": None}
    (tmp_path / "agent-release.json").write_text(json.dumps(RELEASE))
    assert agent_release.installed_agent_release() == {"version": "0.1.0", "build_id": "a" * 64}
