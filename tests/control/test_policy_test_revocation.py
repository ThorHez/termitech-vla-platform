"""Queued inference tests retain their own caller and recheck that caller."""

import pytest
from sqlalchemy import select

from vla_control.models import Command
from vla_control.policy_deployment_models import PolicyDeployment

from .test_policy_deployments import action, ready
from .test_recipe_details import register
from .test_resource_ownership import PASSWORD, accounts, login


def queued_test(platform, deployment):
    response = action(platform, deployment, "test", observation_path="/machine/observation.json")
    assert response.status_code == 202, response.text
    assert response.json()["execution_user_id"] == deployment["execution_user_id"]
    with platform.app.state.db.read() as session:
        command = session.scalar(select(Command).where(Command.type == "POLICY_DEPLOY_TEST",
            Command.payload["deployment_id"].as_string() == deployment["id"]))
        assert command is not None and command.delivered_at is None
        return command


@pytest.mark.parametrize("changes", [{"enabled": False}, {"role": "viewer"}])
def test_revoked_owner_cannot_receive_a_queued_inference_test(platform, changes):
    node = register(platform)
    users = accounts(platform)
    login(platform, "alice")
    deployment = ready(platform, node)
    command = queued_test(platform, deployment)
    assert command.actor_user_id == users["alice"] and command.required_permission == "training.write"
    platform.app.state.auth.update_user(None, users["alice"], changes)
    assert not any(item["id"] == command.id for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        assert session.get(Command, command.id).status == "REJECTED"
        row = session.get(PolicyDeployment, deployment["id"])
        assert row.state == "RUNNING" and row.execution_user_id == users["alice"]
        assert row.test_result["status"] == "FAILED"
        assert row.test_result["error"]["code"] == "COMMAND_AUTHORIZATION_REVOKED"


def test_admin_testing_another_owners_service_must_still_be_admin_at_delivery(platform):
    node = register(platform)
    users = accounts(platform)
    platform.app.state.auth.create_user("remaining-admin", PASSWORD, "admin")
    login(platform, "alice")
    deployment = ready(platform, node)
    login(platform, "owner-admin")
    command = queued_test(platform, deployment)
    assert command.actor_user_id == users["owner-admin"]
    platform.app.state.auth.update_user(None, users["owner-admin"], {"role": "operator"})
    assert not any(item["id"] == command.id for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        row = session.get(PolicyDeployment, deployment["id"])
        assert row.state == "RUNNING" and row.owner_user_id == users["alice"]
        assert session.get(Command, command.id).status == "REJECTED"


def test_admin_test_uses_its_caller_even_when_service_owner_is_disabled(platform):
    node = register(platform)
    users = accounts(platform)
    login(platform, "alice")
    deployment = ready(platform, node)
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    login(platform, "owner-admin")
    command = queued_test(platform, deployment)
    assert command.actor_user_id == users["owner-admin"]
    assert any(item["id"] == command.id for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        assert session.get(PolicyDeployment, deployment["id"]).execution_user_id == users["alice"]


def test_previously_delivered_test_keeps_retry_semantics_after_account_revocation(platform):
    node = register(platform)
    users = accounts(platform)
    login(platform, "alice")
    deployment = ready(platform, node)
    command = queued_test(platform, deployment)
    assert any(item["id"] == command.id for item in platform.commands(node))
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    assert any(item["id"] == command.id for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        row = session.get(Command, command.id)
        assert row.status == "PENDING" and row.delivered_at is not None
