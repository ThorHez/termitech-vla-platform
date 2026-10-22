"""Connection configuration can browse before an Agent or install path exists."""

import errno
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vla_control.remote_models import RemoteConnection
from vla_control.ssh_remote import RemoteError, SSHRemote

from .conftest import ADMIN
from .test_remote_deployment import FakeRemote, PASSWORD, create, payload, remote


def connect(platform):
    return platform.client.post("/api/v1/ssh-connections/connect", headers=ADMIN,
        json={"host": "192.0.2.7", "username": "operator", "password": PASSWORD})


def suggest(platform, identifier, **changes):
    return platform.client.post(f"/api/v1/ssh-connections/{identifier}/path-suggestions", headers=ADMIN,
        json={"prefix": "/data1/", **changes})


def test_preinstall_authentication_browses_and_then_checks_install_using_vault(remote, monkeypatch):
    platform, manager = remote
    requests = []

    def paths(self, prefix, **options):
        requests.append((prefix, options))
        return {"prefix": prefix, "parent": "/data1", "entries": [
            {"name": "project", "path": "/data1/project/", "kind": "directory"}], "truncated": False}

    monkeypatch.setattr(FakeRemote, "path_suggestions", paths, raising=False)
    response = connect(platform)
    assert response.status_code == 200, response.text
    connected = response.json()
    identifier = connected["connection_id"]
    assert "can_install" not in connected and PASSWORD not in response.text
    assert not create(platform, payload(identifier)).is_success
    result = suggest(platform, identifier, directory_only=False)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "READY"
    assert result.json()["connection_id"] == identifier
    assert result.json()["entries"][0]["path"] == "/data1/project/"
    assert requests == [("/data1/", {"limit": 30, "directory_only": False})]
    assert FakeRemote.calls[-1].options["expected_fingerprint"] == connected["fingerprint"]
    assert FakeRemote.calls[-1].closed
    checked = platform.client.post(f"/api/v1/ssh-connections/{identifier}/test", headers=ADMIN,
        json={"install_dir": "/data2/vla-agent"})
    assert checked.status_code == 200, checked.text
    assert checked.json()["connection_id"] == identifier and checked.json()["can_install"]
    assert FakeRemote.calls[-1].options["expected_fingerprint"] == connected["fingerprint"]
    assert FakeRemote.calls[-1].options["password"] == PASSWORD
    assert create(platform, payload(identifier)).status_code == 202
    assert checked.headers["cache-control"] == result.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("suffix,body", [("connect", {"host": "host", "username": "root", "password": "x"}),
    ("missing/path-suggestions", {"prefix": "/"}), ("missing/test", {"install_dir": "/data1/agent"})])
def test_new_connection_routes_require_admin(remote, suffix, body):
    platform, _ = remote
    assert platform.client.post("/api/v1/ssh-connections/" + suffix, json=body).status_code == 401
    assert not FakeRemote.calls


@pytest.mark.parametrize("field,value,code", [("expires_at", "2000-01-01T00:00:00+00:00", "SSH_CONNECTION_EXPIRED"),
    ("encrypted_password", "", "SSH_CONNECTION_REVOKED"), ("fingerprint", "", "SSH_CONNECTION_UNVERIFIED")])
def test_invalid_stored_proof_never_reconnects(remote, field, value, code):
    platform, manager = remote
    identifier = connect(platform).json()["connection_id"]
    with manager.db.write() as session:
        setattr(session.get(RemoteConnection, identifier), field, value)
    count = len(FakeRemote.calls)
    result = suggest(platform, identifier)
    assert result.status_code == 409 and result.json()["error"]["code"] == code
    checked = platform.client.post(f"/api/v1/ssh-connections/{identifier}/test", headers=ADMIN,
        json={"install_dir": "/data1/agent"})
    assert checked.status_code == 409
    assert len(FakeRemote.calls) == count


@pytest.mark.parametrize("prefix", ["../data", "/data/../root", "/data\n", "relative"])
def test_unsafe_prefix_rejected_without_ssh(remote, prefix):
    platform, _ = remote
    identifier = connect(platform).json()["connection_id"]
    count = len(FakeRemote.calls)
    response = suggest(platform, identifier, prefix=prefix)
    assert response.status_code == 422
    assert len(FakeRemote.calls) == count


def test_path_error_is_sanitized_and_releases_slot(remote, monkeypatch):
    platform, manager = remote
    identifier = connect(platform).json()["connection_id"]

    def paths(self, *args, **kwargs):
        raise RuntimeError(PASSWORD + " ssh://root:secret@host")

    monkeypatch.setattr(FakeRemote, "path_suggestions", paths, raising=False)
    response = suggest(platform, identifier)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "SSH_OPERATION_FAILED"
    assert PASSWORD not in response.text and "secret@host" not in response.text
    assert FakeRemote.calls[-1].closed
    assert manager._path_queries.acquire(blocking=False)
    manager._path_queries.release()


def test_host_identity_failure_remains_explicit(remote):
    platform, _ = remote
    identifier = connect(platform).json()["connection_id"]
    FakeRemote.error = RemoteError("SSH_HOST_KEY_MISMATCH", "SSH 主机指纹与已验证的机器不一致")
    response = suggest(platform, identifier)
    assert response.status_code == 502 and response.json()["error"]["code"] == "SSH_HOST_KEY_MISMATCH"
    assert FakeRemote.calls[-1].closed


def test_path_browse_concurrency_is_bounded_before_connecting(remote):
    platform, manager = remote
    identifier = connect(platform).json()["connection_id"]
    for _ in range(4):
        assert manager._path_queries.acquire(blocking=False)
    count = len(FakeRemote.calls)
    try:
        response = suggest(platform, identifier)
        assert response.status_code == 429 and response.json()["error"]["code"] == "SSH_PATH_BUSY"
        assert len(FakeRemote.calls) == count
    finally:
        for _ in range(4):
            manager._path_queries.release()


def sftp_remote(tmp_path, entries, parent_mode=stat.S_IFDIR | 0o755):
    ssh = SSHRemote("192.0.2.7", 22, "operator", PASSWORD, tmp_path)
    ssh.client = Mock()
    sftp = ssh.client.open_sftp.return_value
    sftp.stat.return_value = SimpleNamespace(st_mode=parent_mode)
    sftp.listdir_iter.return_value = iter(entries)
    return ssh, sftp


def item(name, mode):
    return SimpleNamespace(filename=name, st_mode=mode)


def test_readonly_sftp_suggestions_include_dirs_and_executables_and_follow_python_symlinks(tmp_path):
    entries = [item("python", stat.S_IFLNK | 0o777), item("private.txt", stat.S_IFREG | 0o644),
        item("folder", stat.S_IFDIR | 0o755), item(".venv", stat.S_IFDIR | 0o755),
        item("pipe", stat.S_IFIFO | 0o777), item("bad\nname", stat.S_IFDIR | 0o755)]
    ssh, sftp = sftp_remote(tmp_path, entries)
    sftp.stat.side_effect = lambda path: SimpleNamespace(st_mode=(stat.S_IFREG | 0o755) if path.endswith("python") else stat.S_IFDIR | 0o755)
    result = ssh.path_suggestions("/data1/project/", directory_only=False)
    assert result["entries"] == [
        {"name": ".venv", "path": "/data1/project/.venv/", "kind": "directory"},
        {"name": "folder", "path": "/data1/project/folder/", "kind": "directory"},
        {"name": "python", "path": "/data1/project/python", "kind": "file"}]
    sftp.listdir_iter.assert_called_once_with("/data1/project", read_aheads=1)
    ssh.client.exec_command.assert_not_called()
    sftp.open.assert_not_called()
    sftp.close.assert_called_once()


def test_directory_completion_and_truncation_do_not_open_the_typed_directory(tmp_path):
    ssh, sftp = sftp_remote(tmp_path, [item("project", stat.S_IFDIR | 0o755),
        item("project-other", stat.S_IFDIR | 0o755), item("project-python", stat.S_IFREG | 0o755)])
    result = ssh.path_suggestions("/data1/project", limit=1)
    assert result["parent"] == "/data1"
    assert result["entries"] == [{"name": "project", "path": "/data1/project/", "kind": "directory"}]
    assert result["truncated"] is True
    sftp.listdir_iter.assert_called_once_with("/data1", read_aheads=1)


@pytest.mark.parametrize("failure,code", [(PermissionError(errno.EACCES, "private"), "PATH_PERMISSION_DENIED"),
    (FileNotFoundError(errno.ENOENT, "private"), "PATH_NOT_FOUND")])
def test_sftp_errors_never_expose_private_exception_text(tmp_path, failure, code):
    ssh, sftp = sftp_remote(tmp_path, [])
    sftp.stat.side_effect = failure
    with pytest.raises(RemoteError) as caught:
        ssh.path_suggestions("/data1/")
    assert caught.value.code == code and "private" not in str(caught.value)
    sftp.close.assert_called_once()


def test_sftp_scans_only_a_bounded_number_of_entries(tmp_path):
    ssh, _ = sftp_remote(tmp_path, [item(f"dir-{number:05}", stat.S_IFDIR | 0o755) for number in range(6000)])
    result = ssh.path_suggestions("/", limit=50)
    assert len(result["entries"]) == 50 and result["truncated"]
