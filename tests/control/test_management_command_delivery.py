"""Sensitive commands authorize their persisted initiator at first delivery."""

import pytest
from sqlalchemy import select

from vla_control.common import offset_time
from vla_control.models import Command
from vla_control.auth_models import UserSession

from .test_management_permission_revalidation import management  # noqa: F401


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "operator"}])
def test_never_delivered_sensitive_command_is_rejected_when_account_loses_rights(management, change):
    platform, path, body, _, _, users, clients, node = management
    created = clients["admin-one"].post(path, json=body)
    assert created.status_code == 202, created.text
    identifier = created.json()["id"]
    platform.app.state.auth.update_user(None, users["admin-one"]["id"], change)
    assert identifier not in {item["id"] for item in platform.commands(node)}
    with platform.app.state.db.read() as session:
        command = session.get(Command, identifier)
        assert command.status == "REJECTED" and command.delivered_at is None
        assert command.result["error"]["code"] == "COMMAND_AUTHORIZATION_REVOKED"


def test_unknown_initiator_is_not_adopted_by_another_admin(management):
    platform, path, body, _, _, _, clients, node = management
    created = clients["admin-one"].post(path, json=body)
    identifier = created.json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, identifier)
        command.actor_user_id = command.required_permission = None
    assert identifier not in {item["id"] for item in platform.commands(node)}
    with platform.app.state.db.read() as session:
        command = session.get(Command, identifier)
        assert command.actor_user_id is None
        assert command.result["error"]["code"] == "COMMAND_ACTOR_REQUIRED"
    retried = clients["admin-two"].post(path, json=body)
    assert retried.status_code == 202 and retried.json()["id"] != identifier, retried.text


def test_already_delivered_command_keeps_retry_and_original_initiator(management):
    platform, path, body, _, _, users, clients, node = management
    created = clients["admin-one"].post(path, json=body)
    identifier = created.json()["id"]
    first = next(item for item in platform.commands(node) if item["id"] == identifier)
    platform.app.state.auth.update_user(None, users["admin-one"]["id"], {"role": "operator"})
    assert next(item for item in platform.commands(node) if item["id"] == identifier) == first
    with platform.app.state.db.read() as session:
        assert session.get(Command, identifier).actor_user_id == users["admin-one"]["id"]


@pytest.mark.parametrize("expiry", ["logout", "reauth-expired"])
def test_validly_accepted_background_command_survives_web_session_expiry(management, expiry):
    platform, path, body, _, _, users, clients, node = management
    created = clients["admin-one"].post(path, json=body)
    identifier = created.json()["id"]
    if expiry == "logout":
        assert clients["admin-one"].post("/api/v1/auth/logout").status_code == 204
    else:
        with platform.app.state.db.write() as session:
            row = session.scalar(select(UserSession).where(UserSession.user_id == users["admin-one"]["id"]))
            row.reauthenticated_at = offset_time(-301)
    assert identifier in {item["id"] for item in platform.commands(node)}
