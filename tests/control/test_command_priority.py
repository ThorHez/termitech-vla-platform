"""Lifecycle commands remain deliverable behind a large preflight backlog."""

from sqlalchemy import select

from vla_control.common import offset_time
from vla_control.models import Command

from .conftest import ADMIN


def test_large_preflight_backlog_cannot_starve_cancel_start_or_lightweight_commands(platform):
    node = platform.register()
    run = platform.run(node)
    assert platform.schedule() == 1
    response = platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN)
    assert response.status_code == 202, response.text
    with platform.app.state.db.write() as session:
        lifecycle = {row.type: row.id for row in session.scalars(select(Command).where(
            Command.node_id == node["node_id"], Command.status == "PENDING"))}
        # Same timestamps deliberately exercise deterministic ID tie-breaking;
        # insertion order must not affect which 100 commands are delivered.
        backlog_time = offset_time(-1200)
        for index in reversed(range(105)):
            session.add(Command(id=f"backlog-{index:03d}", node_id=node["node_id"], type="PREFLIGHT",
                                payload={"preview_id": f"pending-{index}"}, status="PENDING",
                                created_at=backlog_time))
        lightweight_time = offset_time(-1800)
        for identifier, kind in (("light-z", "PATH_SUGGEST"), ("light-a", "RECIPE_DETAIL")):
            session.add(Command(id=identifier, node_id=node["node_id"], type=kind, payload={},
                                status="PENDING", created_at=lightweight_time))
        # Acknowledged commands never enter this queue.
        session.add(Command(id="already-applied-cancel", node_id=node["node_id"], type="CANCEL",
                            payload={}, status="APPLIED", created_at=offset_time(-3600)))
    commands = platform.commands(node)
    assert len(commands) == 100
    assert [command["id"] for command in commands[:4]] == [
        lifecycle["START"], lifecycle["CANCEL"], "light-a", "light-z"]
    assert [command["id"] for command in commands[4:]] == [f"backlog-{index:03d}" for index in range(96)]
    assert commands == platform.commands(node)
    with platform.app.state.db.read() as session:
        assert all(session.get(Command, command["id"]).status == "PENDING" for command in commands)
    acknowledgement = {"node_id": node["node_id"], "status": "APPLIED", "result": {"cancel_requested": True}}
    for _ in range(2):
        response = platform.client.post(f"/agent/v1/commands/{lifecycle['CANCEL']}/ack", headers=node["headers"],
                                        json=acknowledgement)
        assert response.status_code == 200, response.text
    remaining = platform.commands(node)
    assert remaining[0]["id"] == lifecycle["START"]
    assert lifecycle["CANCEL"] not in {command["id"] for command in remaining}
    assert platform.get_run(run)["state"] == "CANCELLING"
