import pytest
from sqlalchemy import update

import vla_control.main as main_module
from vla_control.common import offset_time
from vla_control.models import Node

from .conftest import ADMIN


@pytest.mark.parametrize("state", ["DISCONNECTED", "REMOVED"])
def test_retired_node_cannot_reconnect_and_does_not_count_as_compute(platform, state):
    node = platform.register()
    with platform.app.state.db.write() as session:
        stored = session.get(Node, node["node_id"])
        stored.status = state
        stored.last_seen = offset_time(-3600)
    assert platform.client.get("/api/v1/nodes", headers=ADMIN).json() == []
    historical = platform.client.get("/api/v1/nodes?include_inactive=true", headers=ADMIN).json()
    assert historical[0]["status"] == state
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).status_code == 200
    overview = platform.client.get("/api/v1/overview", headers=ADMIN).json()
    assert overview["nodes_total"] == 0
    assert overview["gpus_total"] == 0
    assert overview["gpus_available"] == 0
    for response in (
        platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={
            "node_id": node["node_id"], "gpus": [], "metrics": {}, "active_attempts": [],
        }),
        platform.client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]}),
        platform.client.post("/agent/v1/events/batch", headers=node["headers"], json={"node_id": node["node_id"], "events": []}),
    ):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "NODE_CONNECTION_REMOVED"
    platform.schedule()
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == state


@pytest.mark.parametrize("state", ["DISCONNECTED", "REMOVED"])
def test_preflight_rechecks_retirement_after_initial_validation(platform, monkeypatch, state):
    node = platform.register()
    original = main_module.validate_request

    def retired_after_validation(session, data):
        result = original(session, data)
        # Model another transaction committing retirement after the first
        # validation read while the ORM still caches the original ONLINE row.
        session.execute(update(Node).where(Node.id == node["node_id"]).values(status=state),
                        execution_options={"synchronize_session": False})
        assert result[0].status == "ONLINE"
        return result

    monkeypatch.setattr(main_module, "validate_request", retired_after_validation)
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={
        "name": "race", "node_id": node["node_id"], "profile_id": "default",
        "template_id": "smoke", "inputs": {}, "parameters": {"steps": 1}, "gpu_count": 0,
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NODE_OFFLINE"
    assert platform.commands(node) == []
