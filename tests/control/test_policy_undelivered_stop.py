"""Stopping an undispatched deployment cannot leave a deferred start behind."""

import pytest
from sqlalchemy import select

from vla_control.models import Command
from vla_control.policy_deployment_models import PolicyDeployment

from .test_policy_deployment_restart import enable_restart
from .test_policy_deployments import ack, action, create
from .test_recipe_details import register
from .test_resource_ownership import accounts, login


def starts(platform, deployment):
    with platform.app.state.db.read() as session:
        return list(session.scalars(select(Command).where(Command.type == "POLICY_DEPLOY_START",
            Command.payload["deployment_id"].as_string() == deployment["id"]).order_by(Command.created_at)))


def test_admin_stop_retires_revoked_undelivered_start_and_frees_gpu(platform):
    node = register(platform)
    enable_restart(platform, node)
    users = accounts(platform)
    login(platform, "alice")
    response = create(platform, node)
    assert response.status_code == 202, response.text
    deployment = response.json()
    initial = starts(platform, deployment)[0]
    assert initial.delivered_at is None
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    pending = platform.commands(node)
    assert not any(command["type"] == "POLICY_DEPLOY_START" for command in pending)
    login(platform, "owner-admin")
    stopped = action(platform, deployment, "stop")
    assert stopped.status_code == 202, stopped.text
    assert stopped.json()["state"] == "STOPPED" and stopped.json()["pending_command"] is None
    assert stopped.json()["owner_user_id"] == users["alice"]
    with platform.app.state.db.read() as session:
        old = session.get(Command, initial.id)
        assert old.status == "REJECTED" and old.delivered_at is None
        assert old.result["state"] == "STOPPED" and old.result["execution_id"] is None
        assert old.result["error"]["code"] == "POLICY_DEPLOY_CANCELLED_BEFORE_DISPATCH"
        assert not list(session.scalars(select(Command).where(Command.type == "POLICY_DEPLOY_STOP")))
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": True})
    assert platform.commands(node) == []
    login(platform, "bob")
    another = create(platform, node, port=8001)
    assert another.status_code == 202, another.text  # The implicit GPU allocation was released.
    assert action(platform, another.json(), "stop").json()["state"] == "STOPPED"
    login(platform, "owner-admin")
    restarted = action(platform, deployment, "start")
    assert restarted.status_code == 202, restarted.text
    assert restarted.json()["owner_user_id"] == users["alice"]
    assert restarted.json()["execution_user_id"] == users["owner-admin"]
    new = platform.commands(node)
    assert len(new) == 1 and new[0]["type"] == "POLICY_DEPLOY_START"
    assert new[0]["id"] != initial.id and new[0]["payload"]["execution_id"]


@pytest.mark.parametrize("delivery", ["observed", "migration"])
def test_delivered_or_migrated_start_still_waits_for_agent_stop_receipt(platform, delivery):
    node = register(platform)
    users = accounts(platform)
    login(platform, "alice")
    response = create(platform, node)
    assert response.status_code == 202, response.text
    deployment = response.json()
    initial = starts(platform, deployment)[0]
    if delivery == "observed":
        assert any(command["id"] == initial.id for command in platform.commands(node))
    else:
        with platform.app.state.db.write() as session:
            row = session.get(Command, initial.id)
            row.delivered_at = row.created_at
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    login(platform, "owner-admin")
    stopped = action(platform, deployment, "stop")
    assert stopped.status_code == 202, stopped.text
    assert stopped.json()["state"] == "STOPPING"
    commands = platform.commands(node)
    stop = next(command for command in commands if command["type"] == "POLICY_DEPLOY_STOP")
    with platform.app.state.db.read() as session:
        assert session.get(Command, initial.id).status == "PENDING"
        assert session.get(PolicyDeployment, deployment["id"]).state == "STOPPING"
    assert ack(platform, node, stop, {"state": "STOPPED"}).status_code == 200
    assert platform.client.get("/api/v1/policy-deployments/" + deployment["id"]).json()["state"] == "STOPPED"
