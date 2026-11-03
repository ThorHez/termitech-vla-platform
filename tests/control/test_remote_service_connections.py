"""Independent Agent service namespaces keep maintenance on the selected instance."""

import os
from pathlib import Path

import pytest
from sqlalchemy import select

from vla_control.common import digest
from vla_control.db import Database
from vla_control.models import Node
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN
from .test_remote_deployment import FakeRemote, agent_online, connection, create, payload, remote  # noqa: F401
from .test_agent_upgrade import RELEASE, write_bundle


TEST_SERVICE = "termitech-vla-agent-mac-test"


@pytest.mark.parametrize("name", ["", "x.service", "../canonical", "unit;reboot", "x y", "x\n", "a" * 101])
def test_invalid_names_reject_before_any_ssh(remote, name):
    platform, _ = remote
    result = connection(platform, service_name=name)
    assert result.status_code == 422
    assert not FakeRemote.calls


def test_namespace_survives_authentication_probe_deployment_and_management(remote):
    platform, manager = remote
    authenticated = platform.client.post("/api/v1/ssh-connections/connect", headers=ADMIN, json={
        "host": "192.0.2.169", "username": "root", "password": "unit-secret", "service_name": TEST_SERVICE,
    })
    assert authenticated.status_code == 200, authenticated.text
    info = authenticated.json()
    assert info["service_name"] == TEST_SERVICE
    # Older callers that omit the optional name during the directory check keep
    # the namespace already selected during authentication.
    checked = platform.client.post(f'/api/v1/ssh-connections/{info["connection_id"]}/test',
        headers=ADMIN, json={"install_dir": "/data2/vla-agent"})
    assert checked.status_code == 200, checked.text
    assert checked.json()["service_name"] == TEST_SERVICE
    job = create(platform, payload(info["connection_id"])).json()
    assert job["service_name"] == TEST_SERVICE
    assert all(call.options["service_name"] == TEST_SERVICE for call in FakeRemote.calls)
    with platform.app.state.db.read() as session:
        row = session.get(RemoteDeployment, job["id"])
        expected = digest({"host": info["host"], "port": 22, "install_dir": "/data2/vla-agent",
                           "service_name": TEST_SERVICE})
        assert row.target_key == expected and row.agent_config["_service_name"] == TEST_SERVICE


def test_deployment_namespace_cannot_be_redirected_by_rechecking_connection(remote):
    platform, _ = remote
    info = connection(platform, service_name=TEST_SERVICE).json()
    assert create(platform, payload(info["connection_id"])).status_code == 202
    before = len(FakeRemote.calls)
    changed = platform.client.post(f'/api/v1/ssh-connections/{info["connection_id"]}/test', headers=ADMIN,
        json={"install_dir": "/data2/vla-agent", "service_name": "termitech-vla-agent"})
    assert changed.status_code == 409 and changed.json()["error"]["code"] == "SSH_SERVICE_CHANGED"
    assert len(FakeRemote.calls) == before  # Reject before any remote operation.
    with platform.app.state.db.read() as session:
        assert session.get(RemoteConnection, info["connection_id"]).service_name == TEST_SERVICE


def test_canonical_target_key_and_existing_default_stay_compatible(remote):
    platform, _ = remote
    info = connection(platform).json()
    assert info["service_name"] == "termitech-vla-agent"
    job = create(platform, payload(info["connection_id"])).json()
    with platform.app.state.db.read() as session:
        row = session.get(RemoteDeployment, job["id"])
        assert row.target_key == digest({"host": info["host"], "port": 22, "install_dir": "/data2/vla-agent"})
        assert "_service_name" not in row.agent_config


def test_same_prefix_or_unit_cannot_be_installed_by_competing_jobs(remote):
    platform, _ = remote
    first = connection(platform).json()
    assert create(platform, payload(first["connection_id"]), "first").status_code == 202
    other = connection(platform, service_name=TEST_SERVICE).json()
    conflict = create(platform, payload(other["connection_id"]), "same-prefix-other-unit")
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "REMOTE_TARGET_MANAGED"
    other_prefix = connection(platform, install_dir="/data2/second-agent").json()
    body = payload(other_prefix["connection_id"])
    body["install_dir"] = "/data2/second-agent"
    conflict = create(platform, body, "same-unit-other-prefix")
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "REMOTE_TARGET_MANAGED"


def test_custom_instance_upgrade_and_uninstall_keep_selected_service(remote):
    platform, manager = remote
    info = connection(platform, service_name=TEST_SERVICE).json()
    first = create(platform, payload(info["connection_id"])).json()
    FakeRemote.installer = lambda config: agent_online(platform, config)
    manager._execute(first["id"])
    installed = manager.get(first["id"])
    assert installed["state"] == "SUCCEEDED", installed
    node_id = installed["node_id"]
    FakeRemote.probe_result["existing_agent"] = {**installed["agent"], "node_id": node_id}
    write_bundle(Path(os.environ["VLA_AGENT_BUNDLE"]))

    def upgrade_callback(config):
        agent_online(platform, config, node_id=node_id)
        with manager.db.write() as session:
            node = session.get(Node, node_id)
            node.metrics = {**node.metrics, "agent": dict(RELEASE)}

    FakeRemote.installer = upgrade_callback
    job = platform.client.post(f"/api/v1/nodes/{node_id}/upgrade",
        headers={**ADMIN, "Idempotency-Key": "custom-upgrade"}).json()
    manager._execute(job["id"])
    upgraded = manager.get(job["id"])
    assert upgraded["state"] == "SUCCEEDED" and upgraded["service_name"] == TEST_SERVICE, upgraded
    assert manager.management(node_id)["service_name"] == TEST_SERVICE
    with manager.db.read() as session:
        assert session.get(RemoteDeployment, job["id"]).agent_config["_service_name"] == TEST_SERVICE
    removal = platform.client.post(f"/api/v1/nodes/{node_id}/uninstall",
        headers={**ADMIN, "Idempotency-Key": "custom-uninstall"})
    assert removal.status_code == 202, removal.text
    manager._execute(removal.json()["id"])
    assert manager.get(removal.json()["id"])["state"] == "UNINSTALLED"
    assert all(call.options["service_name"] == TEST_SERVICE for call in FakeRemote.calls)


def test_old_database_adds_default_without_rewriting_connection_or_job_identity(tmp_path):
    db = Database("sqlite:///" + str(tmp_path / "old.db"))
    db.initialize()
    with db.engine.begin() as c:
        c.exec_driver_sql("ALTER TABLE remote_connections DROP COLUMN service_name")
        c.exec_driver_sql("INSERT INTO remote_connections "
            "(id,host,port,username,encrypted_password,fingerprint,probe_result,install_dir,created_at,expires_at) "
            "VALUES ('old','192.0.2.1',22,'root','ciphertext','SHA256:old','{}','/data2/agent','before','later')")
        before = c.exec_driver_sql("SELECT * FROM remote_connections").one()
    db.initialize()
    db.initialize()
    with db.engine.connect() as c:
        after = c.exec_driver_sql("SELECT * FROM remote_connections").one()
        assert tuple(after[:-1]) == tuple(before) and after[-1] == "termitech-vla-agent"
    with db.read() as session:
        assert session.scalar(select(RemoteConnection)).service_name == "termitech-vla-agent"
    db.close()
