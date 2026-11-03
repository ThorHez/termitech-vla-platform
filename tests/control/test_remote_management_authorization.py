"""SSH jobs retain their initiator and cannot start with withdrawn privileges."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, func, select

from vla_control.auth_models import UserSession
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.common import offset_time

from .test_remote_deployment import FakeRemote, connection, create, payload, remote  # noqa: F401
from .test_agent_upgrade import connected, upgrade  # noqa: F401


PASSWORD = "remote-management-authority-password"


def accounts(platform):
    users = {name: platform.app.state.auth.create_user(name, PASSWORD, "admin")
             for name in ("remote-admin", "backup-admin")}
    client = TestClient(platform.app, base_url="https://testserver")
    logged = client.post("/api/v1/auth/login", json={"username": "remote-admin", "password": PASSWORD})
    assert logged.status_code == 200
    client.headers["X-CSRF-Token"] = logged.json()["csrf_token"]
    assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
    return users, client


@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "reauth-expired"])
def test_remote_enqueue_rechecks_original_request_before_persisting(remote, monkeypatch, change):
    platform, _ = remote
    identifier = connection(platform).json()["connection_id"]
    users, client = accounts(platform)
    original = platform.app.state.auth.require
    fired = []
    def checked(request, permission=None):
        principal = original(request, permission)
        if request.url.path == "/api/v1/remote-deployments" and not fired:
            fired.append(True)
            if change in {"disabled", "demoted"}:
                platform.app.state.auth.update_user(None, users["remote-admin"]["id"],
                    {"enabled": False} if change == "disabled" else {"role": "operator"})
            else:
                with platform.app.state.db.write() as session:
                    if change == "logout":
                        session.execute(delete(UserSession).where(UserSession.user_id == users["remote-admin"]["id"]))
                    else:
                        session.scalar(select(UserSession).where(UserSession.user_id == users["remote-admin"]["id"])).reauthenticated_at = offset_time(-301)
        return principal
    monkeypatch.setattr(platform.app.state.auth, "require", checked)
    try:
        result = client.post("/api/v1/remote-deployments", headers={"Idempotency-Key": "new-job"}, json=payload(identifier))
        assert fired and result.status_code in {401, 403}, result.text
        with platform.app.state.db.read() as session:
            assert session.scalar(select(func.count()).select_from(RemoteDeployment)) == 0
    finally:
        client.close()


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "operator"}])
def test_install_not_started_when_queued_initiator_is_revoked(remote, change):
    platform, manager = remote
    identifier = connection(platform).json()["connection_id"]
    users, client = accounts(platform)
    try:
        queued = client.post("/api/v1/remote-deployments", headers={"Idempotency-Key": "queued-job"}, json=payload(identifier))
        assert queued.status_code == 202, queued.text
        job = queued.json()
        assert job["actor_user_id"] == users["remote-admin"]["id"]
        platform.app.state.auth.update_user(None, users["remote-admin"]["id"], change)
        calls = len(FakeRemote.calls)
        manager._execute(job["id"])
        assert len(FakeRemote.calls) == calls
        result = manager.get(job["id"])
        assert result["state"] == "INTERRUPTED" and result["started_at"] is None
        assert result["error"]["code"] == "REMOTE_AUTHORIZATION_REVOKED"
        with platform.app.state.db.read() as session:
            assert session.get(RemoteConnection, identifier).encrypted_password
    finally:
        client.close()


@pytest.mark.parametrize("operation", ["upgrade", "uninstall"])
def test_privileged_machine_jobs_bind_actor_and_revoke_before_network(connected, operation):
    platform, manager, node_id, *_ = connected
    users, client = accounts(platform)
    try:
        queued = client.post(f"/api/v1/nodes/{node_id}/{operation}", headers={"Idempotency-Key": "queue-" + operation})
        assert queued.status_code == 202, queued.text
        platform.app.state.auth.update_user(None, users["remote-admin"]["id"], {"role": "operator"})
        calls = len(FakeRemote.calls)
        manager._execute(queued.json()["id"])
        assert len(FakeRemote.calls) == calls
        result = manager.get(queued.json()["id"])
        assert result["state"] == "INTERRUPTED"
        assert result["actor_user_id"] == users["remote-admin"]["id"]
    finally:
        client.close()


def test_role_lost_during_ssh_probe_cannot_begin_install(remote, monkeypatch):
    platform, manager = remote
    identifier = connection(platform).json()["connection_id"]
    users, client = accounts(platform)
    original = FakeRemote.probe
    installed = []
    def revoke_during_probe(remote_object, prefix):
        value = original(remote_object, prefix)
        platform.app.state.auth.update_user(None, users["remote-admin"]["id"], {"enabled": False})
        return value
    monkeypatch.setattr(FakeRemote, "probe", revoke_during_probe)
    monkeypatch.setattr(FakeRemote, "install", lambda *args: installed.append(True))
    try:
        queued = client.post("/api/v1/remote-deployments", headers={"Idempotency-Key": "probe-job"}, json=payload(identifier))
        assert queued.status_code == 202, queued.text
        manager._execute(queued.json()["id"])
        assert installed == []
        assert manager.get(queued.json()["id"])["state"] == "INTERRUPTED"
    finally:
        client.close()


def test_connection_probe_cannot_persist_credentials_after_logout(remote, monkeypatch):
    platform, _ = remote
    users, client = accounts(platform)
    original = FakeRemote.probe
    def logout_during_probe(remote_object, prefix):
        result = original(remote_object, prefix)
        with platform.app.state.db.write() as session:
            session.execute(delete(UserSession).where(UserSession.user_id == users["remote-admin"]["id"]))
        return result
    monkeypatch.setattr(FakeRemote, "probe", logout_during_probe)
    try:
        result = client.post("/api/v1/ssh-connections/test", json={"host": "192.0.2.169", "port": 22,
            "username": "root", "password": "fake-ssh-password", "install_dir": "/data2/vla-agent"})
        assert result.status_code == 401, result.text
        with platform.app.state.db.read() as session:
            assert session.scalar(select(func.count()).select_from(RemoteConnection)) == 0
    finally:
        client.close()
