"""Pending source mutations retain queued work without reserving its GPUs."""

from vla_control.common import now, uid
from vla_control.models import Command


def pending_write(platform, node, code_dir):
    identifier = uid()
    with platform.app.state.db.write() as session:
        session.add(Command(id=identifier, node_id=node["node_id"], type="RECIPE_WRITE", status="PENDING",
                            created_at=now(), payload={"definition": {"source": {"code_dir": code_dir}}}))
    return identifier


def test_pending_writeback_blocks_only_matching_checkout(platform):
    node = platform.register()
    run = platform.run(node)
    identifier = pending_write(platform, node, "/machine/code")
    assert platform.schedule() == 0
    waiting = platform.get_run(run)
    assert waiting["state"] == "QUEUED" and waiting["attempts"] == []
    assert waiting["queue_reason"] == "RECIPE_WRITEBACK_PENDING"
    with platform.app.state.db.write() as session:
        session.get(Command, identifier).status = "APPLIED"
    assert platform.schedule() == 1
    assert platform.get_run(run)["state"] == "STARTING"


def test_writeback_in_other_checkout_does_not_block_queue(platform):
    node = platform.register()
    run = platform.run(node)
    pending_write(platform, node, "/machine/other-code")
    assert platform.schedule() == 1
    assert platform.get_run(run)["state"] == "STARTING"
