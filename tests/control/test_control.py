from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vla_control.common import now, offset_time
from vla_control.main import create_app
from vla_control.models import Allocation, Asset, Node, Preview, RunInput
from vla_control.settings import Settings

from .conftest import ADMIN, gpu, snapshot


def allocations(platform):
    with platform.app.state.db.read() as session:
        return list(session.scalars(select(Allocation)))


def add_asset(platform, node, kind="DATASET", path="/machine/dataset"):
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "local asset", "kind": kind,
        "node_id": node["node_id"], "path": path, "version": "v1", "metadata": {}})
    assert response.status_code == 201, response.text
    return response.json()


def test_credentials_are_required_and_isolated(platform):
    with pytest.raises(ValueError):
        create_app(Settings(scheduler_enabled=False))
    assert platform.client.get("/healthz").status_code == 200
    assert platform.client.get("/api/v1/nodes").status_code == 401
    a = platform.register("worker-a")
    b = platform.register("worker-b")
    assert platform.client.get("/api/v1/nodes", headers=a["headers"]).status_code == 401
    assert platform.client.get("/agent/v1/commands", headers=ADMIN, params={"node_id": a["node_id"]}).status_code == 401
    assert platform.client.get("/agent/v1/commands", headers=a["headers"], params={"node_id": b["node_id"]}).status_code == 403
    assert platform.client.post("/agent/v1/heartbeat", headers=a["headers"], json={"node_id": b["node_id"]}).status_code == 403


def test_registration_is_idempotent_and_profile_ids_are_node_scoped(platform):
    a = platform.register("worker-a", key="persistent-instance-a")
    again = platform.register("worker-a", key="persistent-instance-a")
    b = platform.register("worker-b", key="persistent-instance-b")
    assert a["node_id"] == again["node_id"] and a["token"] == again["token"]
    assert a["node_id"] != b["node_id"]
    assert len(platform.client.get("/api/v1/profiles", headers=ADMIN).json()) == 2


def test_assets_are_local_and_cross_node_inputs_rejected_by_api_and_database(platform):
    a = platform.register("worker-a")
    b = platform.register("worker-b")
    foreign = add_asset(platform, b)
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={"name": "bad", "node_id": a["node_id"],
        "profile_id": "default", "template_id": "smoke", "gpu_count": 0, "inputs": {"dataset": foreign["id"]}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ASSET_NODE_MISMATCH"
    run = platform.run(a, count=0)
    with pytest.raises(IntegrityError), platform.app.state.db.write() as session:
        session.add(RunInput(run_id=run["id"], node_id=a["node_id"], role="dataset", asset_id=foreign["id"]))
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "escape", "node_id": a["node_id"],
        "kind": "DATASET", "path": "/machine/../etc/passwd"})
    assert response.status_code == 422


def test_preflight_run_submission_freezes_code_and_is_idempotent(platform):
    node = platform.register()
    preview = platform.preview(node, code=snapshot(dirty=True, changes_digest="d" * 64))
    headers = {**ADMIN, "Idempotency-Key": "submit-key"}
    first = platform.client.post("/api/v1/runs", headers=headers, json={"preflight_id": preview["id"]})
    repeat = platform.client.post("/api/v1/runs", headers=headers, json={"preflight_id": preview["id"]})
    assert first.status_code == repeat.status_code == 201
    assert first.json()["id"] == repeat.json()["id"]
    assert first.json()["code_expectation"] == preview["code_snapshot"]
    duplicate = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "another-key"}, json={"preflight_id": preview["id"]})
    assert duplicate.status_code == 409
    other = platform.preview(node)
    conflict = platform.client.post("/api/v1/runs", headers=headers, json={"preflight_id": other["id"]})
    assert conflict.status_code == 409
    with platform.app.state.db.write() as session:
        session.get(Preview, other["id"]).expires_at = offset_time(-1)
    expired = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "expired"}, json={"preflight_id": other["id"]})
    assert expired.status_code == 409 and expired.json()["error"]["code"] == "PREFLIGHT_EXPIRED"


def test_invalid_code_snapshot_cannot_be_used_to_submit(platform):
    node = platform.register()
    preview = platform.preview(node, code=snapshot(commit_sha="unknown"))
    assert preview["status"] == "FAILED"
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "invalid-code"}, json={"preflight_id": preview["id"]})
    assert response.status_code == 409


def test_two_schedulers_allocate_atomically_without_double_booking(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2")])
    first = platform.run(node, count=2)
    second = platform.run(node, count=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        dispatched = list(pool.map(lambda _: platform.schedule(), range(2)))
    assert sum(dispatched) == 1
    assert len(allocations(platform)) == 2
    assert platform.get_run(first)["state"] == "STARTING"
    queued = platform.get_run(second)
    assert queued["state"] == "QUEUED" and queued["attempts"] == []
    commands = [c for c in platform.commands(node) if c["type"] == "START"]
    assert len(commands) == 1
    assert platform.commands(node) == platform.commands(node)
    command = commands[0]
    platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"], json={"node_id": node["node_id"], "status": "APPLIED", "result": {"accepted": True}})
    assert platform.get_run(first)["state"] == "STARTING"
    assert not [c for c in platform.commands(node) if c["type"] == "START"]


def test_external_or_unknown_gpu_usage_blocks_allocation(platform):
    node = platform.register(gpus=[gpu(processes=[{"pid": 111, "memory_mb": 2}]), gpu("GPU-2", processes=None)])
    run = platform.run(node, count=1)
    assert platform.schedule() == 0
    assert platform.get_run(run)["queue_reason"] == "GPU_UNAVAILABLE"
    assert platform.client.get("/api/v1/overview", headers=ADMIN).json()["gpus_available"] == 0


def test_cancellation_requires_exit_before_releasing_gpu(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    response = platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN)
    assert response.status_code == 202 and response.json()["state"] == "CANCELLING"
    assert len(allocations(platform)) == 1
    cancel = next(c for c in platform.commands(node) if c["type"] == "CANCEL")
    platform.client.post(f"/agent/v1/commands/{cancel['id']}/ack", headers=node["headers"], json={"node_id": node["node_id"], "status": "APPLIED", "result": {}})
    assert len(allocations(platform)) == 1
    assert platform.emit(node, attempt["id"], 2, "CANCELLED", {"exit_code": None}).status_code == 422
    assert len(allocations(platform)) == 1
    assert platform.emit(node, attempt["id"], 2, "CANCELLED", {"exit_code": -15}).status_code == 200
    assert allocations(platform) == []
    assert platform.get_run(run)["state"] == "CANCELLED"


def test_disconnect_retains_allocations_and_reconciles_only_known_execution(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-100)
    platform.schedule()
    assert platform.get_run(run)["state"] == "UNKNOWN"
    assert len(allocations(platform)) == 1
    platform.heartbeat(node, attempts=[])
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == "RECOVERING"
    assert platform.get_run(run)["state"] == "UNKNOWN"
    platform.heartbeat(node, attempts=[{"attempt_id": attempt["id"], "pid": 1234, "state": "RUNNING"}])
    assert platform.get_run(run)["state"] == "RUNNING"
    assert len(allocations(platform)) == 1


def test_events_are_deduplicated_and_actual_code_is_recorded(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    stamp = now()
    for _ in range(2):
        assert platform.emit(node, attempt["id"], 2, "LOG", {"text": "step 1"}, stamp).status_code == 200
    conflict = platform.emit(node, attempt["id"], 2, "LOG", {"text": "changed text"}, stamp)
    assert conflict.status_code == 409
    assert platform.emit(node, attempt["id"], 3, "METRIC", {"step": 1, "total_steps": 3, "loss": 0.2}).status_code == 200
    assert platform.emit(node, attempt["id"], 4, "SUCCEEDED", {"exit_code": 0, "outputs": [], "code_changed_during_run": False}).status_code == 200
    result = platform.get_run(run)
    assert result["state"] == "SUCCEEDED"
    assert result["attempts"][0]["code_snapshot"] == run["code_expectation"]
    assert result["attempts"][0]["latest_metrics"]["loss"] == 0.2
    logs = platform.client.get(f"/api/v1/runs/{run['id']}/logs", headers=ADMIN).json()
    assert len(logs["items"]) == 1 and logs["next_seq"] == 2
    assert allocations(platform) == []


def test_prelaunch_failure_confirms_no_process_and_releases_gpu(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    changed = snapshot(commit_sha="e" * 40)
    response = platform.emit(node, attempt["id"], 1, "FAILED", {"exit_code": None, "process_started": False,
        "error": {"code": "CODE_CHANGED_SINCE_PREFLIGHT", "message": "source changed"}, "code_snapshot": changed})
    assert response.status_code == 200, response.text
    assert allocations(platform) == []
    result = platform.get_run(run)
    assert result["state"] == "FAILED" and result["attempts"][0]["code_snapshot"] == changed


def test_wrong_running_code_requests_stop_and_waits_for_exit(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": snapshot(commit_sha="e" * 40),
        "output_dir": "/machine/output/attempt"})
    assert response.status_code == 200
    assert platform.get_run(run)["state"] == "CANCELLING"
    assert len(allocations(platform)) == 1
    assert [c for c in platform.commands(node) if c["type"] == "CANCEL"]
    assert platform.emit(node, attempt["id"], 2, "CANCELLED", {"exit_code": -15}).status_code == 200
    assert platform.get_run(run)["state"] == "FAILED"
    assert allocations(platform) == []


def test_invalid_output_cannot_be_registered_on_success(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0,
        "outputs": [{"kind": "MODEL", "name": "escaped", "path": "/another-machine/checkpoint"}]})
    assert response.status_code == 200
    result = platform.get_run(run)
    assert result["state"] == "FAILED" and result["outputs"] == []
    assert result["error"]["code"] == "OUTPUT_INVALID"
    assert allocations(platform) == []


def test_active_input_cannot_be_archived_and_relationships_remain_visible(platform):
    node = platform.register()
    asset = add_asset(platform, node)
    run = platform.run(node, inputs={"dataset": asset["id"]}, count=0)
    response = platform.client.post(f"/api/v1/assets/{asset['id']}/archive", headers=ADMIN)
    assert response.status_code == 409
    detail = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert detail["assets"][0]["state"] == "READY"
    assert detail["runs"][0]["input_assets"][0]["id"] == asset["id"]
    platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN)
    assert platform.client.post(f"/api/v1/assets/{asset['id']}/archive", headers=ADMIN).status_code == 200


def test_state_survives_control_plane_restart(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    first_command = next(c for c in platform.commands(node) if c["type"] == "START")
    restarted_app = create_app(platform.settings)
    with TestClient(restarted_app) as client:
        result = client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN)
        assert result.status_code == 200 and result.json()["state"] == "STARTING"
        commands = client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]}).json()
        assert next(c for c in commands if c["type"] == "START") == first_command


def test_unimplemented_algorithms_and_arbitrary_parameters_are_rejected(platform):
    node = platform.register(gpus=[])
    body = {"name": "planned", "node_id": node["node_id"], "profile_id": "default", "template_id": "mem", "gpu_count": 0}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "TEMPLATE_UNAVAILABLE"
    body.update(template_id="smoke", parameters={"command": "rm -rf /"})
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=body)
    assert response.status_code == 422
    body.update(parameters={}, code_snapshot=snapshot())
    assert platform.client.post("/api/v1/run-previews", headers=ADMIN, json=body).status_code == 422


def test_profile_sync_preserves_active_runtime_and_accepts_telemetry(platform):
    node = platform.register()
    run = platform.run(node, count=0)
    original = platform.client.get("/api/v1/profiles", headers=ADMIN).json()[0]
    fields = {key: value for key, value in original.items() if key not in {"active", "node_id"}}
    changed = {**fields, "recipe": "new_recipe"}
    extra = {**fields, "id": "new-profile", "name": "new runtime"}
    body = {"node_id": node["node_id"], "gpus": [gpu()], "metrics": {"cpu_pct": 42}, "active_attempts": [], "profiles": [changed, extra]}
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json=body)
    assert response.status_code == 200
    assert response.json()["profile_sync"]["status"] == "BLOCKED"
    assert response.json()["profile_sync"]["updated_profile_ids"] == ["new-profile"]
    profiles = platform.client.get("/api/v1/profiles", headers=ADMIN).json()
    assert next(p for p in profiles if p["id"] == "default")["recipe"] == original["recipe"]
    detail = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert detail["status"] == "ONLINE" and detail["metrics"]["cpu_pct"] == 42
    platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN)
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json=body)
    assert response.json()["profile_sync"]["status"] == "APPLIED"
    assert next(p for p in platform.client.get("/api/v1/profiles", headers=ADMIN).json() if p["id"] == "default")["recipe"] == "new_recipe"
    # Retire a profile without deleting historical run references.
    body["profiles"] = [extra]
    assert platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json=body).status_code == 200
    assert len(platform.client.get("/api/v1/profiles", headers=ADMIN).json()) == 1
    assert platform.get_run(run)["profile_id"] == "default"


def test_profile_change_invalidates_pending_preflight(platform):
    node = platform.register()
    preview = platform.preview(node, count=0)
    original = platform.client.get("/api/v1/profiles", headers=ADMIN).json()[0]
    fields = {key: value for key, value in original.items() if key not in {"active", "node_id"}}
    fields["python"] = "/other/environment/bin/python"
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={"node_id": node["node_id"], "gpus": [gpu()], "profiles": [fields]})
    assert response.status_code == 200
    result = platform.client.get(f"/api/v1/run-previews/{preview['id']}", headers=ADMIN).json()
    assert result["status"] == "FAILED" and result["error"]["code"] == "PROFILE_CHANGED_SINCE_PREFLIGHT"


def test_node_cannot_ack_or_publish_another_nodes_execution(platform):
    a, b = platform.register("a"), platform.register("b")
    run = platform.run(a)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    assert platform.emit(b, attempt["id"], 1, "FAILED", {"exit_code": None, "process_started": False}).status_code == 403
    command = next(c for c in platform.commands(a) if c["type"] == "START")
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=b["headers"], json={"node_id": b["node_id"], "status": "APPLIED", "result": {}})
    assert response.status_code == 403
    assert len(allocations(platform)) == 1


@pytest.mark.parametrize("reported_state", ["UNKNOWN", None])
def test_online_agent_with_lost_runner_marks_execution_unknown(platform, reported_state):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    active = [{"attempt_id": attempt["id"], "pid": None, "state": reported_state}] if reported_state else []
    platform.heartbeat(node, attempts=active)
    assert platform.get_run(run)["state"] == "UNKNOWN"
    assert len(allocations(platform)) == 1
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == "RECOVERING"
    # Only the delayed exit event, not the live Agent, can release reservations.
    assert platform.emit(node, attempt["id"], 2, "FAILED", {"exit_code": -9, "error": {"code": "WORKER_DIED", "message": "exit confirmed"}}).status_code == 200
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == "ONLINE"
    platform.heartbeat(node, attempts=[{"attempt_id": attempt["id"], "pid": None, "state": "UNKNOWN"}])
    assert platform.get_run(run)["state"] == "FAILED"
    assert allocations(platform) == []


def test_starting_dispatch_gap_does_not_make_node_unhealthy(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    platform.heartbeat(node, attempts=[])
    assert platform.get_run(run)["state"] == "STARTING"
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == "ONLINE"


def test_terminal_event_does_not_revive_stale_node(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-100)
    platform.schedule()
    assert platform.emit(node, attempt["id"], 2, "CANCELLED", {"exit_code": -15}).status_code == 200
    assert platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()["status"] == "OFFLINE"
    assert allocations(platform) == []
