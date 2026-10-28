"""Policy deployment commands preserve local ownership and real readiness."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from sqlalchemy import select

from vla_control.common import now, offset_time, uid
from vla_control.models import Asset, Attempt, Command, Node, Run
from vla_control.policy_deployments import CHECKPOINT_SOURCE_WARNING
from vla_control.policy_deployment_models import PolicyDeployment
from vla_control.scheduler import tick

from .conftest import ADMIN, generated_config
from .test_recipe_details import register


BASE = "/api/v1/policy-deployments"


def create(platform, node, **changes):
    return platform.client.post(BASE, headers=ADMIN, json={"name": "本地抓取测试", "node_id": node["node_id"],
        "profile_id": "default", "checkpoint_path": "/machine/output/training/10000", "config_name": "pi0_base",
        "port": 8000, **changes})


def get(platform, deployment):
    return platform.client.get(BASE + "/" + deployment["id"], headers=ADMIN).json()


def action(platform, deployment, verb, **body):
    return platform.client.post(BASE + "/" + deployment["id"] + "/" + verb, headers=ADMIN, json=body)


def command(platform, node, kind):
    return next(item for item in platform.commands(node) if item["type"] == kind)


def ack(platform, node, command, result, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": status, "result": result})


def running(**changes):
    return {"state": "RUNNING", "pid": 2345, "endpoint": "ws://127.0.0.1:8000", "logs": "model loaded\n",
        "checks": [{"name": "websocket_metadata", "ok": True, "message": "deployment metadata received"}], **changes}


def ready(platform, node):
    response = create(platform, node)
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), running()).status_code == 200
    return deployment


def add_run(session, node, **changes):
    code = {"recipe": "pi0_base", "code_dir": "/machine/code",
        "automatic_preflight": {"plan": {"datasets": [{"path": "/machine/data"}], "generated_config": generated_config()}},
        "confirmed_config_sha256": generated_config()["sha256"]}
    row = Run(id=uid(), name="training", node_id=node["node_id"], profile_id="default", template_id="openpi-sft",
        state="SUCCEEDED", created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1,
        code_expectation=code)
    for key, value in changes.items():
        setattr(row, key, value)
    session.add(row)
    session.flush()
    return row


def test_create_routes_local_command_and_only_metadata_proves_readiness(platform):
    node = register(platform)
    response = create(platform, node)
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment["state"] == "PENDING" and deployment["endpoint"] is None
    assert deployment["gpu_uuid"] == "GPU-1"
    assert deployment["connection_state"] == "ONLINE"
    cmd = command(platform, node, "POLICY_DEPLOY_START")
    assert now() < cmd["payload"]["expires_at"] <= offset_time(121)
    assert {key: value for key, value in cmd["payload"].items() if key != "expires_at"} == {"deployment_id": deployment["id"], "profile_id": "default",
        "checkpoint_path": "/machine/output/training/10000", "config_name": "pi0_base", "inference_backend": "auto", "port": 8000,
        "gpu_uuid": "GPU-1", "gpu_mode": "exclusive", "gpu_memory_mb": None, "default_prompt": None}
    result = {"state": "STARTING", "pid": 2345, "logs": "loading weights", "checks": []}
    assert ack(platform, node, cmd, result).status_code == 200
    assert get(platform, deployment)["state"] == "STARTING"
    assert action(platform, deployment, "refresh").status_code == 202
    cmd = command(platform, node, "POLICY_DEPLOY_STATUS")
    assert ack(platform, node, cmd, running()).status_code == 200
    current = get(platform, deployment)
    assert current["state"] == "RUNNING" and current["endpoint"] == "ws://127.0.0.1:8000"
    assert current["pending_command"] is None
    assert platform.client.get(BASE, headers=ADMIN).json()[0] == current


@pytest.mark.parametrize("receipt", [running(checks=[]), running(pid=None), running(endpoint="ws://other:8000"), {"state": "RUNNING"}])
def test_bad_ready_receipt_retains_uncertain_resources(platform, receipt):
    node = register(platform)
    deployment = create(platform, node).json()
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), receipt).status_code == 200
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN"
    assert current["error"]["code"] == "POLICY_DEPLOY_INVALID_RESULT"
    assert create(platform, node, port=8001).status_code == 409


def test_refresh_is_deduplicated_get_is_read_only_and_test_is_async(platform):
    node = register(platform)
    deployment = ready(platform, node)
    for _ in range(3):
        assert action(platform, deployment, "refresh").status_code == 202
        assert get(platform, deployment)["state"] == "RUNNING"
    pending = platform.commands(node)
    assert len(pending) == 1 and pending[0]["type"] == "POLICY_DEPLOY_STATUS"
    assert ack(platform, node, pending[0], running()).status_code == 200
    assert action(platform, deployment, "test", observation_path="/machine/observation.json").status_code == 202
    test_command = command(platform, node, "POLICY_DEPLOY_TEST")
    assert test_command["payload"]["observation_path"] == "/machine/observation.json"
    assert action(platform, deployment, "test", observation_path="/machine/observation.json").status_code == 409
    receipt = running(test_result={"status": "RUNNING", "observation_path": "/machine/observation.json"})
    assert ack(platform, node, test_command, receipt).status_code == 200
    assert action(platform, deployment, "test", observation_path="/machine/observation.json").status_code == 409
    assert action(platform, deployment, "refresh").status_code == 202
    test_result = {"status": "SUCCEEDED", "observation_path": "/machine/observation.json", "action_shape": [16, 14],
                   "latency_ms": 81.2, "finite": True, "output_path": "/machine/state/test.json"}
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_STATUS"), running(test_result=test_result)).status_code == 200
    assert get(platform, deployment)["test_result"] == test_result


def test_stop_waits_for_agent_receipt_and_delayed_status_cannot_resurrect(platform):
    node = register(platform)
    deployment = ready(platform, node)
    action(platform, deployment, "refresh")
    status_command = command(platform, node, "POLICY_DEPLOY_STATUS")
    assert action(platform, deployment, "stop").json()["state"] == "STOPPING"
    assert action(platform, deployment, "stop").status_code == 202
    stop_command = command(platform, node, "POLICY_DEPLOY_STOP")
    assert len([cmd for cmd in platform.commands(node) if cmd["type"] == "POLICY_DEPLOY_STOP"]) == 1
    assert create(platform, node, port=8001).status_code == 409
    assert ack(platform, node, stop_command, {"state": "STOPPED", "pid": None, "endpoint": None}).status_code == 200
    assert ack(platform, node, status_command, running()).status_code == 200
    assert get(platform, deployment)["state"] == "STOPPED"
    assert get(platform, deployment)["endpoint"] is None
    assert create(platform, node).status_code == 202


def test_unsupported_agent_and_cross_machine_receipt(platform):
    node = register(platform)
    other = register(platform, name="other")
    deployment = create(platform, node).json()
    cmd = command(platform, node, "POLICY_DEPLOY_START")
    assert ack(platform, other, cmd, running()).status_code == 403
    assert ack(platform, node, cmd, {"error": {"code": "COMMAND_UNSUPPORTED", "message": "old agent"}}, "REJECTED").status_code == 200
    current = get(platform, deployment)
    assert current["state"] == "FAILED" and current["error"]["code"] == "POLICY_DEPLOY_AGENT_UNSUPPORTED"


def test_offline_has_unknown_state_and_writes_do_not_fake_stop(platform):
    node = register(platform)
    deployment = ready(platform, node)
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-3600)
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN" and current["reported_state"] == "RUNNING"
    assert current["connection_state"] == "OFFLINE"
    assert action(platform, deployment, "stop").status_code == 409
    assert action(platform, deployment, "refresh").status_code == 409
    assert create(platform, node).status_code == 409
    with platform.app.state.db.read() as session:
        assert session.get(PolicyDeployment, deployment["id"]).state == "RUNNING"


@pytest.mark.parametrize("changes,code", [({"checkpoint_path": "/outside/weights"}, "PATH_OUTSIDE_ROOTS"),
    ({"checkpoint_path": "/machine/../etc/passwd"}, "PATH_OUTSIDE_ROOTS"),
    ({"checkpoint_path": "/machine/data\n"}, "INVALID_PATH"),
    ({"profile_id": "missing"}, "PROFILE_NODE_MISMATCH"),
    ({"gpu_uuid": "GPU-NONEXISTENT"}, "POLICY_DEPLOY_GPU_UNAVAILABLE")])
def test_create_rejects_invalid_local_contract(platform, changes, code):
    node = register(platform)
    response = create(platform, node, **changes)
    assert response.status_code in {409, 422}
    assert response.json()["error"]["code"] == code
    assert platform.commands(node) == []


def test_only_supported_runtime_and_safe_port_are_accepted(platform):
    node = platform.register()
    assert create(platform, node).json()["error"]["code"] == "POLICY_DEPLOY_PROFILE_UNSUPPORTED"
    node = register(platform, name="openpi")
    for changes in ({"port": 22}, {"port": True}, {"config_name": "recipe; rm -rf"}, {"name": "  "}):
        assert create(platform, node, **changes).status_code == 422


def test_source_run_uses_confirmed_generated_config_and_rejects_ambiguous_variants(platform):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
        source_id, expectation = run.id, deepcopy(run.code_expectation)
    deployment = create(platform, node, source_run_id=source_id).json()
    cmd = command(platform, node, "POLICY_DEPLOY_START")
    assert cmd["payload"]["source_training"] == {"run_id": source_id, "code_expectation": expectation,
                                                "checkpoint_roots": ["/machine/output/training"]}
    assert deployment["source_run_id"] == source_id
    assert ack(platform, node, cmd, {"state": "FAILED", "error": {"code": "LOAD_FAILED", "message": "fixture"}}).status_code == 200
    mismatch = create(platform, node, source_run_id=source_id, checkpoint_path="/machine/output/unrelated/10000")
    assert mismatch.status_code == 202, mismatch.text
    assert mismatch.json()["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    cmd = command(platform, node, "POLICY_DEPLOY_START")
    assert cmd["payload"]["source_training"]["checkpoint_roots"] == ["/machine/output/training"]
    assert cmd["payload"]["source_training"]["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    assert ack(platform, node, cmd, {"state": "FAILED", "error": {"code": "LOAD_FAILED", "message": "fixture"}}).status_code == 200
    for modifier, code in ((lambda c: c.update(training_stage="value"), "POLICY_DEPLOY_VALUE_UNSUPPORTED"),
        (lambda c: c["automatic_preflight"]["plan"].update(datasets=[{}, {}]), "POLICY_DEPLOY_MULTI_DATASET_UNSUPPORTED"),
        (lambda c: c.update(confirmed_config_sha256="0" * 64), "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH"),
        (lambda c: c["automatic_preflight"]["plan"].pop("generated_config"), "POLICY_DEPLOY_SOURCE_CONFIG_MISSING")):
        code_value = deepcopy(expectation)
        modifier(code_value)
        with platform.app.state.db.write() as session:
            session.get(Run, source_id).code_expectation = code_value
        response = create(platform, node, source_run_id=source_id)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == code


def test_unrecorded_checkpoint_warns_without_creating_training_ownership_or_relaxing_machine_roots(platform):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        run_id, expected = run.id, deepcopy(run.code_expectation)
        asset_id = uid()
        session.add(Asset(id=asset_id, name="externally trained", node_id=node["node_id"], kind="MODEL",
            path="/machine/external-model/59999", version="1", state="READY", metadata_json={}, created_at=now()))
    invalid = create(platform, node, source_run_id=run_id, checkpoint_path="/outside/model")
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "PATH_OUTSIDE_ROOTS"
    response = create(platform, node, source_run_id=run_id, checkpoint_path="/machine/external-model/59999")
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    assert get(platform, deployment)["warnings"] == deployment["warnings"]
    payload = command(platform, node, "POLICY_DEPLOY_START")["payload"]
    assert payload["source_training"] == {"run_id": run_id, "code_expectation": expected,
        "checkpoint_roots": [], "warnings": [CHECKPOINT_SOURCE_WARNING]}
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Attempt).where(Attempt.run_id == run_id))) == []
        assert session.get(Asset, asset_id).origin_attempt_id is None
        assert session.get(Run, run_id).code_expectation == expected


@pytest.mark.parametrize("kind", ["POLICY_DEPLOY_START", "POLICY_DEPLOY_STATUS"])
@pytest.mark.parametrize("recorded", [True, False])
def test_accepted_agent_attribution_warning_is_canonical_and_persisted_without_rewriting_source(platform, kind, recorded):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        run_id, expected = run.id, deepcopy(run.code_expectation)
        if recorded:
            session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
                execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
    deployment = create(platform, node, source_run_id=run_id).json()
    initial = command(platform, node, "POLICY_DEPLOY_START")
    frozen = deepcopy(initial["payload"]["source_training"])
    if kind == "POLICY_DEPLOY_STATUS":
        ack(platform, node, initial, running())
        action(platform, deployment, "refresh")
    receipt = command(platform, node, kind)
    response = ack(platform, node, receipt, running(warnings=[
        {"code": CHECKPOINT_SOURCE_WARNING["code"], "message": "untrusted text", "private": "discard"},
        {"code": "UNSUPPORTED_WARNING", "message": "discard"}, CHECKPOINT_SOURCE_WARNING]))
    assert response.status_code == 200, response.text
    assert get(platform, deployment)["warnings"] == [CHECKPOINT_SOURCE_WARNING]
    with platform.app.state.db.read() as session:
        assert session.get(Command, receipt["id"]).result["warnings"] == [CHECKPOINT_SOURCE_WARNING]
        assert session.get(Command, initial["id"]).payload["source_training"] == frozen
        assert session.get(Run, run_id).code_expectation == expected


@pytest.mark.parametrize("case", ["pending", "rejected", "other-deployment", "other-node", "other-type", "unknown-code"])
def test_unaccepted_or_unrelated_agent_warning_never_pollutes_deployment(platform, case):
    node, other = register(platform), register(platform, name="other")
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        run_id = run.id
        session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
    deployment = create(platform, node, source_run_id=run_id).json()
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=other["node_id"] if case == "other-node" else node["node_id"],
            type="POLICY_DEPLOY_TEST" if case == "other-type" else "POLICY_DEPLOY_STATUS",
            status={"pending": "PENDING", "rejected": "REJECTED"}.get(case, "APPLIED"),
            payload={"deployment_id": uid() if case == "other-deployment" else deployment["id"]},
            result={"warnings": [{"code": "OTHER"}] if case == "unknown-code" else [CHECKPOINT_SOURCE_WARNING]}, created_at=now()))
    assert get(platform, deployment)["warnings"] == []


@pytest.mark.parametrize("supplied", [{}, {"config_name": None}, {"config_name": "unrelated_existing_recipe"}])
@pytest.mark.parametrize("recipe_location", ["code", "plan"])
def test_training_deployment_derives_config_name_from_frozen_source(platform, supplied, recipe_location):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        expectation = deepcopy(run.code_expectation)
        expectation.pop("recipe")
        if recipe_location == "code":
            expectation["recipe"] = "source_training_only"
        else:
            expectation["automatic_preflight"]["plan"]["recipe"] = "source_training_only"
        run.code_expectation = expectation
        source_id = run.id
        session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
    response = platform.client.post(BASE, headers=ADMIN, json={"name": "自动推理配置", "node_id": node["node_id"],
        "profile_id": "default", "checkpoint_path": "/machine/output/training/10000", "source_run_id": source_id, **supplied})
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment["config_name"] == deployment["source_training_recipe"] == "source_training_only"
    assert deployment["config_source"] == "training"
    payload = command(platform, node, "POLICY_DEPLOY_START")["payload"]
    assert payload["config_name"] == "source_training_only"
    assert payload["source_training"]["code_expectation"] == expectation


@pytest.mark.parametrize("modifier,code", [
    (lambda value: value.pop("recipe"), "POLICY_DEPLOY_SOURCE_CONFIG_MISSING"),
    (lambda value: value.update(automatic_preflight={}), "POLICY_DEPLOY_SOURCE_CONFIG_MISSING"),
    (lambda value: value["automatic_preflight"]["plan"].update(recipe="another_recipe"), "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH"),
    (lambda value: value.update(confirmed_config_sha256="0" * 64), "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH"),
])
@pytest.mark.parametrize("checkpoint_path", ["/machine/output/training/10000", "/machine/external/59999"])
def test_training_source_never_falls_back_to_user_config(platform, modifier, code, checkpoint_path):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        expectation = deepcopy(run.code_expectation)
        modifier(expectation)
        run.code_expectation = expectation
        source_id = run.id
        session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
    response = create(platform, node, source_run_id=source_id, config_name="pi0_base", checkpoint_path=checkpoint_path)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == code
    assert platform.commands(node) == []


@pytest.mark.parametrize("snapshot,expected", [
    ({"recipe": "frozen_recipe"}, "frozen_recipe"),
    ({"automatic_preflight": {"plan": {"recipe": "frozen_recipe"}}}, "frozen_recipe"),
    ({"recipe": "frozen_recipe", "automatic_preflight": {"plan": {"recipe": "conflicting_recipe"}}}, None),
    ({}, None), (None, None),
])
def test_historical_training_response_uses_frozen_start_recipe_not_client_label_or_mutable_run(platform, snapshot, expected):
    node = register(platform)
    with platform.app.state.db.write() as session:
        run = add_run(session, node)
        run_id = run.id
        session.add(Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1,
            execution_token=uid(), state="SUCCEEDED", created_at=now(), output_dir="/machine/output/training"))
    deployment = create(platform, node, source_run_id=run_id).json()
    start_id = command(platform, node, "POLICY_DEPLOY_START")["id"]
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).config_name = "legacy_client_label"
        session.get(Run, run_id).code_expectation = {"recipe": "later_mutable_recipe"}
        start = session.get(Command, start_id)
        payload = deepcopy(start.payload)
        if snapshot is None:
            payload.pop("source_training")
        else:
            payload["source_training"]["code_expectation"] = snapshot
        start.payload = payload
    response = get(platform, deployment)
    assert response["config_name"] == "legacy_client_label"
    assert response["config_source"] == "training"
    assert response["source_training_recipe"] == expected


@pytest.mark.parametrize("supplied", [{}, {"config_name": None}])
def test_manual_deployment_requires_existing_config_name(platform, supplied):
    node = register(platform)
    response = platform.client.post(BASE, headers=ADMIN, json={"name": "手工权重", "node_id": node["node_id"],
        "profile_id": "default", "checkpoint_path": "/machine/output/training/10000", **supplied})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "POLICY_DEPLOY_CONFIG_REQUIRED"
    assert platform.commands(node) == []
    manual = create(platform, node).json()
    assert manual["config_name"] == "pi0_base" and manual["config_source"] == "existing"
    assert manual["source_training_recipe"] is None


def test_training_gpu_reservations_block_deploy_and_deploy_blocks_future_scheduler(platform):
    node = register(platform)
    deployment = create(platform, node).json()
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == "deployment:" + deployment["id"]
    assert platform.client.get("/api/v1/overview", headers=ADMIN).json()["gpus_available"] == 0
    with platform.app.state.db.write() as session:
        run = add_run(session, node, state="QUEUED")
        run_id = run.id
    assert tick(platform.app.state.db, platform.settings) == 0
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).queue_reason == "GPU_UNAVAILABLE"
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"),
        {"state": "FAILED", "error": {"code": "LOAD_FAILED", "message": "fixture"}}).status_code == 200
    assert tick(platform.app.state.db, platform.settings) == 1
    assert create(platform, node).json()["error"]["code"] == "POLICY_DEPLOY_GPU_UNAVAILABLE"


def test_prelaunch_rejection_releases_resources_but_stop_failure_preserves_unknown(platform):
    node = register(platform)
    deployment = create(platform, node).json()
    cmd = command(platform, node, "POLICY_DEPLOY_START")
    assert ack(platform, node, cmd, {"state": "FAILED", "error": {"code": "CHECKPOINT_MISSING", "message": "missing"}},
               "REJECTED").status_code == 200
    assert get(platform, deployment)["state"] == "FAILED"
    assert platform.client.get("/api/v1/overview", headers=ADMIN).json()["gpus_available"] == 1
    deployment = ready(platform, node)
    action(platform, deployment, "stop")
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_STOP"),
               {"error": {"code": "PERMISSION_DENIED", "message": "cannot signal owned process"}}, "REJECTED").status_code == 200
    assert get(platform, deployment)["state"] == "UNKNOWN"
    assert platform.client.get("/api/v1/overview", headers=ADMIN).json()["gpus_available"] == 0
    action(platform, deployment, "stop")
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_STOP"), {"state": "STOPPED"}).status_code == 200
    assert platform.client.get("/api/v1/overview", headers=ADMIN).json()["gpus_available"] == 1


@pytest.mark.parametrize("test_result", [{"status": "SUCCEEDED", "finite": True, "action_shape": "16,14"},
    {"status": "SUCCEEDED", "finite": False, "action_shape": [16, 14]},
    {"status": "SUCCEEDED", "finite": True, "action_shape": [16, 14], "latency_ms": -10}])
def test_malformed_test_shape_or_nonfinite_output_never_becomes_success(platform, test_result):
    node = register(platform)
    deployment = ready(platform, node)
    action(platform, deployment, "refresh")
    assert ack(platform, node, command(platform, node, "POLICY_DEPLOY_STATUS"),
               running(test_result=test_result)).status_code == 200
    current = get(platform, deployment)
    assert current["test_result"] is None
    assert current["error"]["code"] == "POLICY_DEPLOY_INVALID_RESULT"


def test_concurrent_create_cannot_claim_same_port_or_gpu(platform):
    node = register(platform)
    barrier = Barrier(2)

    def submit():
        barrier.wait()
        return create(platform, node).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(statuses) == [202, 409]
    with platform.app.state.db.read() as session:
        assert len(list(session.scalars(select(PolicyDeployment)))) == 1


def test_scheduler_keeps_policy_status_current_without_a_browser_and_deduplicates(platform):
    node = register(platform)
    deployment = ready(platform, node)
    assert tick(platform.app.state.db, platform.settings) == 0
    assert platform.commands(node) == []
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).updated_at = offset_time(-16)
    for _ in range(3):
        assert tick(platform.app.state.db, platform.settings) == 0
    assert len(platform.commands(node)) == 1
    cmd = command(platform, node, "POLICY_DEPLOY_STATUS")
    assert ack(platform, node, cmd, running()).status_code == 200
    tick(platform.app.state.db, platform.settings)
    assert platform.commands(node) == []
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, deployment["id"]).updated_at = offset_time(-60)
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN" and current["status_stale"] is True
    assert current["connection_state"] == "ONLINE" and current["reported_state"] == "RUNNING"
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-3600)
    tick(platform.app.state.db, platform.settings)
    assert platform.commands(node) == []
