"""A slow existing SSH probe cannot overwrite credential revocation or rotation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from vla_control.auth_models import UserSession
from vla_control.common import now, offset_time
from vla_control.models import Node
from vla_control.remote_models import RemoteConnection

from .test_remote_deployment import FakeRemote, connection, managed_node, payload, remote
from .test_remote_management_authorization import accounts


@pytest.fixture
def probe_account(remote):
    platform, manager = remote
    info = connection(platform).json()
    users, client = accounts(platform)
    try:
        yield SimpleNamespace(platform=platform, manager=manager, client=client, users=users, info=info)
    finally:
        client.close()


@pytest.fixture
def managed_probe(remote):
    platform, manager = remote
    node_id, _, info = managed_node(platform, manager)
    FakeRemote.probe_result["existing_agent"] = {"node_id": node_id, "profiles": []}
    users, client = accounts(platform)
    try:
        yield SimpleNamespace(platform=platform, manager=manager, client=client, users=users, info=info, node_id=node_id)
    finally:
        client.close()


def stored(context):
    with context.platform.app.state.db.read() as session:
        row = session.get(RemoteConnection, context.info["connection_id"])
        return None if row is None else {field: deepcopy(getattr(row, field)) for field in (
            "id", "host", "port", "username", "encrypted_password", "fingerprint", "service_name",
            "created_at", "expires_at", "install_dir", "probe_result")}


def reprobe(context, **changes):
    return context.client.post(f"/api/v1/ssh-connections/{context.info['connection_id']}/test",
        json={"install_dir": "/data2/vla-agent", **changes})


def during_probe(monkeypatch, action):
    original = FakeRemote.probe
    fired = []
    def probe(remote, prefix):
        value = original(remote, prefix)
        if not fired:
            fired.append(True)
            action()
        return value
    monkeypatch.setattr(FakeRemote, "probe", probe)
    return fired


@pytest.mark.parametrize("operation,terminal", [("disconnect", "DISCONNECTED"), ("uninstall", "REMOVED")])
def test_existing_probe_cannot_resurrect_credentials_revoked_by_machine_removal(managed_probe, monkeypatch, operation, terminal):
    context = managed_probe
    revoked = []
    def remove():
        response = context.client.post(f"/api/v1/nodes/{context.node_id}/{operation}",
            headers={"Idempotency-Key": "remove-during-probe"})
        assert response.status_code == (200 if operation == "disconnect" else 202), response.text
        if operation == "uninstall":
            context.manager._execute(response.json()["id"])
            assert context.manager.get(response.json()["id"])["state"] == "UNINSTALLED"
        revoked.append(stored(context))
        assert revoked[0]["encrypted_password"] == "" and revoked[0]["expires_at"] <= now()
    fired = during_probe(monkeypatch, remove)
    response = reprobe(context)
    assert fired and response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "SSH_CONNECTION_REVOKED"
    assert stored(context) == revoked[0]
    with context.platform.app.state.db.read() as session:
        assert session.get(Node, context.node_id).status == terminal
    reuse = context.client.post("/api/v1/remote-deployments", headers={"Idempotency-Key": "reuse-revoked"},
        json=payload(context.info["connection_id"]))
    assert reuse.status_code == 409 and reuse.json()["error"]["code"] == "SSH_CONNECTION_REVOKED"
    # A deliberate new password submission creates a new credential identity.
    fresh = context.client.post("/api/v1/ssh-connections/test", json={"host": "192.0.2.169", "port": 22,
        "username": "root", "password": "new-explicit-fixture-password", "install_dir": "/data2/vla-agent"})
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["connection_id"] != context.info["connection_id"]
    assert stored(context) == revoked[0]


@pytest.mark.parametrize("change,code,status", [
    ("cleared", "SSH_CONNECTION_REVOKED", 409), ("expired", "SSH_CONNECTION_EXPIRED", 409),
    ("password", "SSH_CONNECTION_CHANGED", 409), ("host", "SSH_CONNECTION_CHANGED", 409),
    ("port", "SSH_CONNECTION_CHANGED", 409), ("username", "SSH_CONNECTION_CHANGED", 409),
    ("fingerprint", "SSH_CONNECTION_CHANGED", 409), ("service_name", "SSH_CONNECTION_CHANGED", 409),
    ("deleted", "SSH_CONNECTION_NOT_FOUND", 404),
])
def test_existing_probe_refuses_changed_or_deleted_current_credential(probe_account, monkeypatch, change, code, status):
    context = probe_account
    replacement = []
    def mutate():
        with context.platform.app.state.db.write() as session:
            row = session.get(RemoteConnection, context.info["connection_id"])
            if change == "deleted":
                session.delete(row)
            else:
                values = {"cleared": ("encrypted_password", ""), "expired": ("expires_at", offset_time(-1)),
                    "password": ("encrypted_password", context.manager.vault.encrypt("replacement-fixture-password")),
                    "host": ("host", "192.0.2.170"), "port": ("port", 2222), "username": ("username", "different-user"),
                    "fingerprint": ("fingerprint", "SHA256:replacement-fixture"),
                    "service_name": ("service_name", "termitech-vla-agent-other")}
                field, value = values[change]
                setattr(row, field, value)
        replacement.append(stored(context))
    fired = during_probe(monkeypatch, mutate)
    response = reprobe(context)
    assert fired and response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert stored(context) == replacement[0]  # Deleted IDs are never reinserted.


def test_normal_reprobe_updates_only_probe_fields_and_preserves_shortened_expiry(probe_account, monkeypatch):
    context = probe_account
    before = stored(context)
    updated = {"expires_at": offset_time(60), "created_at": offset_time(-3600)}
    def shorten():
        with context.platform.app.state.db.write() as session:
            row = session.get(RemoteConnection, context.info["connection_id"])
            row.expires_at, row.created_at = updated["expires_at"], updated["created_at"]
    during_probe(monkeypatch, shorten)
    response = reprobe(context, install_dir="/data2/another-agent", service_name="termitech-vla-agent-reviewed")
    assert response.status_code == 200, response.text
    after = stored(context)
    for field in ("id", "host", "port", "username", "encrypted_password", "fingerprint"):
        assert after[field] == before[field]
    assert after["expires_at"] == response.json()["expires_at"] == updated["expires_at"]
    assert after["created_at"] == updated["created_at"]
    assert after["install_dir"] == "/data2/another-agent" and after["service_name"] == "termitech-vla-agent-reviewed"


def test_namespace_cannot_change_if_a_deployment_claimed_connection_during_probe(probe_account, monkeypatch):
    context = probe_account
    def claim():
        response = context.client.post("/api/v1/remote-deployments", headers={"Idempotency-Key": "claim-during-probe"},
            json=payload(context.info["connection_id"]))
        assert response.status_code == 202, response.text
    during_probe(monkeypatch, claim)
    response = reprobe(context, service_name="termitech-vla-agent-other")
    assert response.status_code == 409 and response.json()["error"]["code"] == "SSH_SERVICE_CHANGED"
    assert stored(context)["service_name"] == context.info["service_name"]


def test_concurrent_probes_preserve_current_security_fields(probe_account, monkeypatch):
    context = probe_account
    original = FakeRemote.probe
    before = stored(context)
    barrier, lock, changed = threading.Barrier(2), threading.Lock(), []
    updated = {"expires_at": offset_time(120), "created_at": offset_time(-7200)}
    def probe(remote, prefix):
        value = original(remote, prefix)
        with lock:
            if not changed:
                with context.platform.app.state.db.write() as session:
                    row = session.get(RemoteConnection, context.info["connection_id"])
                    row.expires_at, row.created_at = updated["expires_at"], updated["created_at"]
                changed.append(True)
        barrier.wait(timeout=5)
        return {**value, "hostname": prefix.rsplit("/", 1)[-1]}
    monkeypatch.setattr(FakeRemote, "probe", probe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda suffix: reprobe(context, install_dir="/data2/" + suffix), ["probe-one", "probe-two"]))
    assert all(response.status_code == 200 for response in results), [response.status_code for response in results]
    after = stored(context)
    for field in ("host", "port", "username", "encrypted_password", "fingerprint", "service_name"):
        assert after[field] == before[field]
    assert after["expires_at"] == updated["expires_at"] and after["created_at"] == updated["created_at"]
    assert after["probe_result"]["hostname"] == after["install_dir"].rsplit("/", 1)[-1]


@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "session-expired", "reauth-expired", "reset-password"])
def test_existing_probe_keeps_original_session_and_recent_password_revalidation(probe_account, monkeypatch, change):
    context = probe_account
    before = stored(context)
    user_id = context.users["remote-admin"]["id"]
    def revoke():
        if change in {"disabled", "demoted"}:
            context.platform.app.state.auth.update_user(None, user_id,
                {"enabled": False} if change == "disabled" else {"role": "operator"})
        elif change == "logout":
            assert context.client.post("/api/v1/auth/logout").status_code == 204
        elif change == "reset-password":
            context.platform.app.state.auth.reset_password(user_id, "replacement-account-fixture-password")
        else:
            with context.platform.app.state.db.write() as session:
                row = session.scalar(select(UserSession).where(UserSession.user_id == user_id))
                if change == "session-expired":
                    row.last_activity_at = offset_time(-context.platform.settings.session_idle_seconds - 1)
                else:
                    row.reauthenticated_at = offset_time(-context.platform.settings.session_reauth_seconds - 1)
    during_probe(monkeypatch, revoke)
    response = reprobe(context, install_dir="/data2/should-not-persist")
    assert response.status_code in {401, 403}, response.text
    assert stored(context) == before
