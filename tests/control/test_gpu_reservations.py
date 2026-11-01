"""Personal reservations constrain both training and inference admission."""

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.common import now
from vla_control.models import Allocation, GPUReservation, Node, Run
from vla_control.policy_deployment_models import PolicyDeployment

from .conftest import ADMIN, gpu
from .test_dataset_norm_migration import _legacy_table
from .test_policy_deployment_restart import enable_restart, terminal
from .test_policy_deployments import action, create
from .test_recipe_details import register
from .test_specific_gpu_training import ready_preview, selected_run, submit


PASSWORD = "gpu-reservation-password-123"


def accounts(platform):
    platform.client.base_url = "https://testserver"
    for username, role in (("alice", "admin"), ("bob", "operator"), ("reader", "viewer")):
        platform.app.state.auth.create_user(username, PASSWORD, role)


def login(client, username):
    client.cookies.clear()
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()["user"]["id"]


def path(node):
    return f"/api/v1/nodes/{node['node_id']}/gpu-reservations"


def reserve(platform, node, uuids):
    return platform.client.put(path(node), json={"gpu_uuids": uuids})


def test_account_identity_permissions_serialization_and_admin_release(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2")])
    assert platform.client.put(path(node), headers=ADMIN, json={"gpu_uuids": ["GPU-1"]}).status_code == 403
    accounts(platform)
    alice = login(platform.client, "alice")
    response = reserve(platform, node, ["GPU-1"])
    assert response.status_code == 200, response.text
    row = response.json()[0]
    assert row == {"node_id": node["node_id"], "gpu_uuid": "GPU-1", "user_id": alice,
                   "username": "alice", "display_name": "alice", "created_at": row["created_at"]}
    assert reserve(platform, node, ["GPU-1"]).json() == [row]
    detail = platform.client.get(f"/api/v1/nodes/{node['node_id']}").json()
    assert detail["gpus"][0]["user_reservation"] == row
    assert detail["gpus"][0]["reserved_by"] is None
    assert detail["gpus"][1]["user_reservation"] is None
    overview = platform.client.get("/api/v1/overview").json()
    assert overview["gpus_reserved"] == 1 and overview["gpus_available"] == 1
    login(platform.client, "reader")
    assert platform.client.get(path(node)).json() == [row]
    assert reserve(platform, node, []).status_code == 403
    assert platform.client.delete(path(node) + "/GPU-1").status_code == 403
    bob = login(platform.client, "bob")
    spoof = platform.client.put(path(node), json={"gpu_uuids": ["GPU-2"], "user_id": alice})
    assert spoof.status_code == 422
    assert reserve(platform, node, ["GPU-1"]).json()["error"]["code"] == "GPU_ALREADY_RESERVED"
    assert platform.client.delete(path(node) + "/GPU-1").status_code == 403
    assert reserve(platform, node, ["GPU-2"]).status_code == 200
    assert next(r for r in platform.client.get(path(node)).json() if r["gpu_uuid"] == "GPU-2")["user_id"] == bob
    login(platform.client, "alice")
    assert platform.client.delete(path(node) + "/GPU-2").json()["deleted"] is True
    assert platform.client.get(path(node)).json() == [row]


@pytest.mark.parametrize("selection", [["GPU-1", "GPU-1"], [""], ["GPU-1\n"], [True], ["unknown"], None, "GPU-1"])
def test_invalid_reservation_does_not_replace_existing_set(platform, selection):
    node = platform.register()
    accounts(platform)
    login(platform.client, "alice")
    original = reserve(platform, node, ["GPU-1"]).json()
    response = reserve(platform, node, selection)
    assert response.status_code == 422, response.text
    assert platform.client.get(path(node)).json() == original


def test_replace_is_atomic_and_release_survives_offline_missing_gpu(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2"), gpu("GPU-3")])
    accounts(platform)
    login(platform.client, "alice")
    mine = reserve(platform, node, ["GPU-1"]).json()[0]
    login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-2"]).status_code == 200
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-2", "GPU-3"]).status_code == 409
    assert mine in platform.client.get(path(node)).json()
    assert len(platform.client.get(path(node)).json()) == 2
    with platform.app.state.db.write() as session:
        record = session.get(Node, node["node_id"])
        record.status, record.gpus = "OFFLINE", []
    released = reserve(platform, node, [])
    assert released.status_code == 200 and [r["gpu_uuid"] for r in released.json()] == ["GPU-2"]
    assert platform.client.delete(path(node) + "/GPU-2").status_code == 200


@pytest.mark.parametrize("telemetry", [{"processes": None}, {"processes": [{"pid": 123}]}, {"memory_used_mb": 4096}])
def test_busy_or_unknown_non_platform_gpu_cannot_be_reserved(platform, telemetry):
    node = platform.register(gpus=[gpu("GPU-1", **telemetry)])
    accounts(platform)
    login(platform.client, "bob")
    result = reserve(platform, node, ["GPU-1"])
    assert result.status_code == 409 and result.json()["error"]["code"] == "GPU_IN_USE"


def test_automatic_scheduler_uses_owners_reserved_cards_first_and_protects_idle_reservation(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2"), gpu("GPU-3")])
    accounts(platform)
    alice = login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-2", "GPU-3"]).status_code == 200
    login(platform.client, "bob")
    first = platform.run(node)
    waiting = platform.run(node)
    assert platform.schedule() == 1
    assert platform.get_run(first)["attempts"][0]["gpu_uuids"] == ["GPU-1"]
    assert platform.get_run(waiting)["queue_reason"] == "GPU_RESERVED"
    login(platform.client, "alice")
    owned = platform.run(node)
    assert owned["owner_user_id"] == alice
    assert platform.schedule() == 1
    assert platform.get_run(owned)["attempts"][0]["gpu_uuids"] == ["GPU-2"]
    # CPU-only tasks retain their zero-GPU behavior.
    login(platform.client, "bob")
    cpu = platform.run(node, count=0)
    assert platform.schedule() == 1
    assert platform.get_run(cpu)["attempts"][0]["gpu_uuids"] == []


def test_queued_explicit_job_rechecks_new_reservation_and_resumes_after_release(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2")])
    accounts(platform)
    login(platform.client, "bob")
    queued = selected_run(platform, node, selected=["GPU-1"])
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    assert platform.schedule() == 0
    assert platform.get_run(queued)["queue_reason"] == "REQUESTED_GPU_RESERVED"
    assert reserve(platform, node, []).status_code == 200
    assert platform.schedule() == 1
    assert platform.get_run(queued)["attempts"][0]["gpu_uuids"] == ["GPU-1"]


def test_specific_preflight_and_submission_reject_another_users_reservation(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2")])
    accounts(platform)
    login(platform.client, "bob")
    preview = ready_preview(platform, node, selected=["GPU-1"])
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    login(platform.client, "bob")
    response = submit(platform, preview)
    assert response.status_code == 409 and response.json()["error"]["code"] == "GPU_RESERVED_BY_OTHER_USER"
    response = platform.client.post("/api/v1/run-previews", json={"name": "blocked", "node_id": node["node_id"],
        "profile_id": "default", "template_id": "smoke", "inputs": {}, "parameters": {"steps": 3},
        "gpu_count": 1, "requested_gpu_uuids": ["GPU-1"]})
    assert response.status_code == 409 and response.json()["error"]["code"] == "GPU_RESERVED_BY_OTHER_USER"


def test_active_training_is_never_preempted_and_owner_can_reserve_then_release(platform):
    node = platform.register()
    accounts(platform)
    login(platform.client, "bob")
    active = platform.run(node)
    assert platform.schedule() == 1
    login(platform.client, "alice")
    response = reserve(platform, node, ["GPU-1"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "GPU_IN_USE"
    login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    assert reserve(platform, node, []).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Run, active["id"]).state == "STARTING"
        assert session.get(Allocation, (node["node_id"], "GPU-1")) is not None


@pytest.mark.parametrize("mode", ["exclusive", "shared"])
def test_inference_cannot_bypass_other_users_reservation_including_shared_mode(platform, mode):
    node = register(platform, gpus=[gpu("GPU-1"), gpu("GPU-2")])
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).metrics = {"capabilities": ["POLICY_DEPLOY_SHARED_GPU_V1"]}
    accounts(platform)
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    login(platform.client, "bob")
    extra = {"gpu_mode": mode, "gpu_memory_mb": 1024 if mode == "shared" else None}
    response = create(platform, node, gpu_uuid="GPU-1", **extra)
    assert response.status_code == 409 and response.json()["error"]["code"] == "GPU_RESERVED_BY_OTHER_USER"
    automatic = create(platform, node)
    assert automatic.status_code == 202 and automatic.json()["gpu_uuid"] == "GPU-2"
    login(platform.client, "alice")
    mine = create(platform, node, port=8001)
    assert mine.status_code == 202 and mine.json()["gpu_uuid"] == "GPU-1"


def test_restart_does_not_inherit_original_users_reservation_rights(platform):
    node = register(platform)
    enable_restart(platform, node)
    accounts(platform)
    alice = login(platform.client, "alice")
    deployment, _ = terminal(platform, node)
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    login(platform.client, "bob")
    rejected = action(platform, deployment, "start")
    assert rejected.status_code == 409 and rejected.json()["error"]["code"] == "GPU_RESERVED_BY_OTHER_USER"
    login(platform.client, "alice")
    restarted = action(platform, deployment, "start")
    assert restarted.status_code == 202 and restarted.json()["owner_user_id"] == alice


def test_restart_transfers_resource_ownership_to_the_current_caller(platform):
    node = register(platform)
    enable_restart(platform, node)
    accounts(platform)
    login(platform.client, "alice")
    deployment, _ = terminal(platform, node)
    bob = login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    restarted = action(platform, deployment, "start")
    assert restarted.status_code == 202 and restarted.json()["owner_user_id"] == bob
    assert reserve(platform, node, []).status_code == 200
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-1"]).json()["error"]["code"] == "GPU_IN_USE"


def test_concurrent_personal_reservations_have_exactly_one_owner(platform):
    node = platform.register()
    accounts(platform)
    clients = [TestClient(platform.app, base_url="https://testserver"), TestClient(platform.app, base_url="https://testserver")]
    try:
        for client, username in zip(clients, ("alice", "bob")):
            login(client, username)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda client: client.put(path(node), json={"gpu_uuids": ["GPU-1"]}), clients))
        assert sorted(result.status_code for result in results) == [200, 409]
        with platform.app.state.db.read() as session:
            assert len(list(session.scalars(select(GPUReservation)))) == 1
    finally:
        for client in clients:
            client.close()


def test_additive_owner_migration_keeps_legacy_jobs_unprivileged(platform):
    node = register(platform)
    with platform.app.state.db.write() as session:
        row = Run(id="legacy-run", name="legacy", node_id=node["node_id"], profile_id="default", template_id="smoke",
            state="QUEUED", created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1, code_expectation={})
        session.add(row)
        session.add(PolicyDeployment(id="legacy-policy", name="legacy", node_id=node["node_id"], profile_id="default",
            checkpoint_path="/machine/checkpoint", config_name="pi0_base", port=8000,
            state="STOPPED", created_at=now(), updated_at=now()))
    database = platform.app.state.db
    raw = database.engine.raw_connection()
    try:
        raw.commit()
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        _legacy_table(cursor, "runs", {"owner_user_id"})
        _legacy_table(cursor, "policy_deployments", {"owner_user_id"})
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    database.initialize()
    with database.read() as session:
        assert session.get(Run, "legacy-run").owner_user_id is None
        assert session.get(PolicyDeployment, "legacy-policy").owner_user_id is None
        assert session.get(Run, "legacy-run").name == "legacy"
    for table in ("runs", "policy_deployments"):
        assert next(column for column in inspect(database.engine).get_columns(table) if column["name"] == "owner_user_id")["nullable"]
