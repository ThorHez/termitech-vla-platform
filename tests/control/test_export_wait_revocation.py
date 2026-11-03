"""Undispatched revoked exports promptly release bounded transfer workers."""

import threading
import time

from sqlalchemy import select

from vla_control.checkpoint_transfer_models import CheckpointTransfer
from vla_control.models import Command

from .test_checkpoint_transfers import ack_export, complete, create, transfer
from .test_resource_ownership import accounts, login


def states(platform, items):
    with platform.app.state.db.read() as session:
        return [session.get(CheckpointTransfer, item["id"]).state for item in items]


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), "Transfer workers did not reach the expected state"


def test_two_revoked_undelivered_exports_release_both_slots_for_another_user(transfer, monkeypatch):
    platform, manager, source, *_ = transfer
    users = accounts(platform)
    login(platform, "alice")
    alice = [create(transfer, target_directory=f"/machine/alice-models-{index}").json() for index in range(2)]
    login(platform, "bob")
    bob = create(transfer, target_directory="/machine/bob-models").json()

    class NoNetwork:
        def connect(self):
            raise AssertionError("Held exports must never connect to a real host")
        def close(self):
            pass

    monkeypatch.setattr(manager.remote_manager, "_remote", lambda _: NoNetwork())
    threads = [threading.Thread(target=manager._run, args=(item["id"],), daemon=True) for item in [*alice, bob]]
    try:
        for thread in threads[:2]:
            thread.start()
        wait_for(lambda: states(platform, alice) == ["EXPORTING", "EXPORTING"])
        threads[2].start()
        assert states(platform, [bob]) == ["QUEUED"]
        platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
        wait_for(lambda: states(platform, alice) == ["INTERRUPTED", "INTERRUPTED"])
        wait_for(lambda: states(platform, [bob]) == ["EXPORTING"])
        for thread in threads[:2]:
            thread.join(timeout=1)
            assert not thread.is_alive()
        with platform.app.state.db.read() as session:
            for item in alice:
                row = session.get(CheckpointTransfer, item["id"])
                assert row.error["code"] == "ACCOUNT_EXECUTION_REVOKED"
                command = session.scalar(select(Command).where(Command.type == "POLICY_EXPORT",
                    Command.payload["transfer_id"].as_string() == item["id"]))
                assert command.status == "REJECTED" and command.delivered_at is None
        delivered = platform.commands(source)
        assert any(item["type"] == "POLICY_EXPORT" and item["payload"]["transfer_id"] == bob["id"] for item in delivered)
        assert not any(item["type"] == "POLICY_EXPORT" and item["payload"]["transfer_id"] in {row["id"] for row in alice}
                       for item in delivered)
    finally:
        manager._stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=3)


def test_already_delivered_export_can_finish_after_its_account_is_disabled(transfer):
    platform, _, source, *_ = transfer
    users = accounts(platform)
    login(platform, "alice")
    response = create(transfer)
    assert response.status_code == 202, response.text
    item = response.json()
    command = next(value for value in platform.commands(source) if value["type"] == "POLICY_EXPORT")
    assert ack_export(transfer, command).status_code == 200
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    login(platform, "owner-admin")
    complete(transfer, item)
    with platform.app.state.db.read() as session:
        row = session.get(CheckpointTransfer, item["id"])
        assert row.state == "SUCCEEDED" and row.owner_user_id == users["alice"]
        assert not list(session.scalars(select(Command).where(Command.type == "POLICY_EXPORT_CANCEL")))


def test_delivered_export_without_receipt_keeps_running_after_revocation(transfer):
    platform, manager, source, *_ = transfer
    users = accounts(platform)
    login(platform, "alice")
    item = create(transfer).json()
    command = next(value for value in platform.commands(source) if value["type"] == "POLICY_EXPORT")
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    worker = threading.Thread(target=manager._run, args=(item["id"],), daemon=True)
    try:
        worker.start()
        wait_for(lambda: states(platform, [item]) == ["EXPORTING"])
        # Give the waiting-loop revocation check a pass as well as the claim.
        manager._stop.wait(.25)
        assert worker.is_alive() and states(platform, [item]) == ["EXPORTING"]
        with platform.app.state.db.read() as session:
            original = session.get(Command, command["id"])
            assert original.status == "PENDING" and original.delivered_at is not None
            assert not list(session.scalars(select(Command).where(Command.type == "POLICY_EXPORT_CANCEL")))
    finally:
        manager._stop.set()
        worker.join(timeout=3)
