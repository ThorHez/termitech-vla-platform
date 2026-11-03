"""Expired reservations never inflate totals or block idle machine management."""

import pytest
from sqlalchemy import select

from vla_control.auth_models import User
from vla_control.common import APIError, now, offset_time
from vla_control.models import Allocation, Attempt, GPUReservation

from .test_gpu_reservations import accounts, login, reserve
from .test_gpu_reservation_lifecycle import queued


@pytest.mark.parametrize("revocation", ["expiry", "disable", "viewer"])
def test_all_reservation_readers_ignore_inactive_rows_without_needing_a_save(platform, revocation):
    node = platform.register()
    accounts(platform)
    owner = login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    assert platform.client.get("/api/v1/overview").json()["gpus_reserved"] == 1
    with platform.app.state.db.write() as session:
        if revocation == "expiry":
            session.get(GPUReservation, (node["node_id"], "GPU-1")).expires_at = offset_time(-1)
        elif revocation == "disable":
            session.get(User, owner).enabled = False
        else:
            session.get(User, owner).role = "viewer"
    login(platform.client, "alice")
    overview = platform.client.get("/api/v1/overview").json()
    assert overview["gpus_reserved"] == 0 and overview["gpus_available"] == 1
    with platform.app.state.db.write() as session:
        assert session.get(GPUReservation, (node["node_id"], "GPU-1")) is not None
        platform.app.state.remote_manager._assert_idle(session, node["node_id"])


def test_actual_allocation_still_blocks_management_when_reservation_expires(platform):
    node = platform.register()
    accounts(platform)
    owner = login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    run_id = queued(platform, node, owner, state="RUNNING")
    with platform.app.state.db.write() as session:
        session.get(GPUReservation, (node["node_id"], "GPU-1")).expires_at = offset_time(-1)
        session.add(Attempt(id="running-attempt", run_id=run_id, node_id=node["node_id"], number=1,
            execution_token="protected-execution", state="RUNNING", created_at=now(), gpu_uuids=["GPU-1"]))
        session.flush()
        session.add(Allocation(node_id=node["node_id"], gpu_uuid="GPU-1", attempt_id="running-attempt", created_at=now()))
    overview = platform.client.get("/api/v1/overview").json()
    assert overview["gpus_reserved"] == 0 and overview["gpus_available"] == 0
    with platform.app.state.db.write() as session:
        with pytest.raises(APIError) as rejected:
            platform.app.state.remote_manager._assert_idle(session, node["node_id"])
        assert rejected.value.code == "NODE_BUSY"
        assert len(list(session.scalars(select(Allocation)))) == 1
