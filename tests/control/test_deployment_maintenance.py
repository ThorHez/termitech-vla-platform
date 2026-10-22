from vla_control.common import now, offset_time, uid
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN


def maintenance(platform, node):
    connection_id, deployment_id = uid(), uid()
    with platform.app.state.db.write() as session:
        session.add(RemoteConnection(id=connection_id, host="worker.example.com", port=22,
            username="root", encrypted_password="unused-test-value", fingerprint="test-fingerprint",
            probe_result={}, install_dir="/machine/agent", created_at=now(), expires_at=offset_time(3600)))
        session.flush()
        session.add(RemoteDeployment(id=deployment_id, connection_id=connection_id,
            idempotency_key=uid(), request_digest="x" * 64, target_key="y" * 64,
            onboarding_id=uid(), node_id=node["node_id"], state="RUNNING", stage="INSTALLING",
            logs=[], history=[], agent_config={}, install_dir="/machine/agent",
            control_plane_url="https://control.example.com", connection_mode="direct",
            tunnel_state="NOT_REQUIRED", created_at=now(), updated_at=now()))
    return deployment_id


def test_deployment_blocks_new_preflight(platform):
    node = platform.register()
    maintenance(platform, node)
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={
        "name": "blocked", "node_id": node["node_id"], "profile_id": "default",
        "template_id": "smoke", "inputs": {}, "parameters": {"steps": 3}, "gpu_count": 0,
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NODE_MAINTENANCE"
    assert platform.commands(node) == []


def test_deployment_blocks_submission_from_previous_preflight(platform):
    node = platform.register()
    preview = platform.preview(node, count=0)
    maintenance(platform, node)
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": uid()},
        json={"preflight_id": preview["id"]})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NODE_MAINTENANCE"
    assert platform.client.get("/api/v1/runs", headers=ADMIN).json() == []


def test_scheduler_retains_queue_until_deployment_finishes(platform):
    node = platform.register()
    run = platform.run(node, count=0)
    deployment_id = maintenance(platform, node)
    assert platform.schedule() == 0
    waiting = platform.get_run(run)
    assert waiting["state"] == "QUEUED"
    assert waiting["queue_reason"] == "NODE_MAINTENANCE"
    assert waiting["attempts"] == []
    with platform.app.state.db.write() as session:
        session.get(RemoteDeployment, deployment_id).state = "FAILED"
    assert platform.schedule() == 1
    assert platform.get_run(run)["state"] == "STARTING"
