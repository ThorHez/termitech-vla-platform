"""Cancellation and first delivery serialize without releasing a live lease."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from sqlalchemy import select

from vla_control.models import Allocation, Attempt, Command, Run

from .test_multiuser_task_permissions import actor, team  # noqa: F401


def test_first_agent_poll_racing_admin_cancel_has_one_safe_serialized_outcome(team):
    platform, node, _, clients = team
    run = actor(team).run(node)
    assert platform.schedule() == 1
    with platform.app.state.db.read() as session:
        attempt = session.scalar(select(Attempt).where(Attempt.run_id == run["id"]))
        attempt_id = attempt.id
        start_id = session.scalar(select(Command.id).where(Command.type == "START"))
        assert session.get(Command, start_id).delivered_at is None
    gate = Barrier(2)

    def poll():
        gate.wait(timeout=5)
        return platform.client.get("/agent/v1/commands", headers=node["headers"],
                                   params={"node_id": node["node_id"]})

    def cancel():
        gate.wait(timeout=5)
        return clients["team-admin"].post(f"/api/v1/runs/{run['id']}/cancel")

    with ThreadPoolExecutor(max_workers=2) as pool:
        polled, cancelled = pool.submit(poll), pool.submit(cancel)
        delivery, cancellation = polled.result(timeout=15), cancelled.result(timeout=15)
    assert delivery.status_code == 200, delivery.text
    assert cancellation.status_code == 202, cancellation.text
    delivered = start_id in {command["id"] for command in delivery.json()}
    with platform.app.state.db.read() as session:
        current = session.get(Run, run["id"])
        attempt = session.get(Attempt, attempt_id)
        start = session.get(Command, start_id)
        allocations = list(session.scalars(select(Allocation).where(Allocation.attempt_id == attempt_id)))
        cancels = [command for command in session.scalars(select(Command).where(Command.type == "CANCEL"))
                   if command.payload.get("attempt_id") == attempt_id]
        if delivered:
            # A response may already be in the Agent's hands. Retain its lease
            # until the ordinary Agent terminal event confirms process exit.
            assert current.state == attempt.state == "CANCELLING"
            assert start.status == "PENDING" and start.delivered_at is not None
            assert len(allocations) == len(cancels) == 1
        else:
            # Cancellation won the first-delivery guard. No Agent ever received
            # this execution, so the reservation can be released atomically.
            assert current.state == attempt.state == "CANCELLED"
            assert current.finished_at is not None and attempt.finished_at is not None
            assert start.status == "REJECTED" and start.delivered_at is None
            assert allocations == cancels == []
    subsequent = platform.commands(node)
    if not delivered:
        assert start_id not in {command["id"] for command in subsequent}
