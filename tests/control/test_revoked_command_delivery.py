"""Revoke first delivery without guessing that a delivered process has exited."""

import pytest
from sqlalchemy import select

from vla_control.common import uid
from vla_control.models import Allocation, Attempt, Command, Run

from .test_multiuser_task_permissions import actor, team  # noqa: F401


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}])
def test_undelivered_start_is_held_after_revocation_and_resumes_when_authorized(team, change):
    platform, node, users, clients = team
    run = actor(team).run(node)
    assert platform.schedule() == 1
    with platform.app.state.db.read() as session:
        command = session.scalar(select(Command).where(Command.type == "START"))
        command_id = command.id
        assert command.delivered_at is None
    platform.app.state.auth.update_user(None, users["alice"]["id"], change)
    commands = platform.commands(node)
    assert command_id not in {row["id"] for row in commands}
    with platform.app.state.db.read() as session:
        assert session.get(Command, command_id).delivered_at is None
        assert session.get(Run, run["id"]).state == "STARTING"
        assert session.get(Run, run["id"]).queue_reason == "ACCOUNT_EXECUTION_REVOKED"
        assert len(list(session.scalars(select(Allocation)))) == 1
    platform.app.state.auth.update_user(None, users["alice"]["id"], {"enabled": True, "role": "operator"})
    assert command_id in {row["id"] for row in platform.commands(node)}
    with platform.app.state.db.read() as session:
        assert session.get(Command, command_id).delivered_at is not None


def test_delivered_start_retains_retry_and_late_events_after_account_disable(team):
    platform, node, users, _ = team
    own = actor(team)
    run = own.run(node)
    assert platform.schedule() == 1
    first = next(row for row in platform.commands(node) if row["type"] == "START")
    platform.app.state.auth.update_user(None, users["alice"]["id"], {"enabled": False})
    retry = next(row for row in platform.commands(node) if row["type"] == "START")
    assert retry == first
    with platform.app.state.db.read() as session:
        attempt = session.scalar(select(Attempt).where(Attempt.run_id == run["id"]))
        attempt_id = attempt.id
    result = platform.emit(node, attempt_id, 1, "STARTED", {
        "pid": 1234, "code_snapshot": run["code_expectation"], "output_dir": "/machine/output/" + attempt_id})
    assert result.status_code == 200, result.text
    with platform.app.state.db.read() as session:
        assert session.get(Run, run["id"]).state == "RUNNING"
        assert len(list(session.scalars(select(Allocation)))) == 1


def test_held_starts_do_not_hide_admin_cancellation_or_starve_delivery_limit(team):
    platform, node, users, clients = team
    own = actor(team)
    run = own.run(node)
    assert platform.schedule() == 1
    platform.app.state.auth.update_user(None, users["alice"]["id"], {"enabled": False})
    cancelled = clients["team-admin"].post(f"/api/v1/runs/{run['id']}/cancel")
    assert cancelled.status_code == 202 and cancelled.json()["state"] == "CANCELLED"
    commands = platform.commands(node)
    assert not any(row["type"] == "START" for row in commands)
    assert not any(row["type"] == "CANCEL" for row in commands)
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Allocation))) == []
        assert session.scalar(select(Attempt).where(Attempt.run_id == run["id"])).state == "CANCELLED"
        start = session.scalar(select(Command).where(Command.type == "START"))
        assert start.status == "REJECTED" and start.result["error"]["code"] == "CANCELLED_BEFORE_DISPATCH"
    next_run = actor(team, "bob").run(node)
    assert platform.schedule() == 1
    assert clients["bob"].get(f"/api/v1/runs/{next_run['id']}").json()["state"] == "STARTING"


@pytest.mark.parametrize("kind,key", [("START", "run_id"), ("PREFLIGHT", "preview_id"),
                                    ("POLICY_DEPLOY_START", "deployment_id"), ("POLICY_EXPORT", "transfer_id")])
def test_orphan_execution_commands_cannot_skip_account_authorization(team, kind, key):
    from vla_control.common import now
    platform, node, _, _ = team
    identifier = uid()
    with platform.app.state.db.write() as session:
        session.add(Command(id=identifier, node_id=node["node_id"], type=kind,
                            status="PENDING", payload={key: uid()}, created_at=now()))
    assert identifier not in {row["id"] for row in platform.commands(node)}
    with platform.app.state.db.read() as session:
        assert session.get(Command, identifier).delivered_at is None
