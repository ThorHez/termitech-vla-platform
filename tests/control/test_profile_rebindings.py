"""Runtime changes require durable Agent proof and preserve machine identity."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import APIError, now, offset_time, uid
from vla_control.models import Command, Node, Preview, Profile, Run
from vla_control.recipe_writeback import require_no_pending_writeback

from .conftest import ADMIN, ENROLL
from .test_recipe_details import register


def body(**changes):
    return {"profile_id": "default", "expected_code_dir": "/machine/code", "expected_python": "/usr/bin/python3",
            "code_dir": "/machine/new-code", "python": "/opt/conda/envs/training/bin/python", **changes}


def path(node):
    return f"/api/v1/nodes/{node['node_id']}/profile-rebindings"


def create(platform, node, value=None, headers=ADMIN):
    return platform.client.post(path(node), json=body() if value is None else value, headers=headers)


def get(platform, node, operation):
    return platform.client.get(path(node) + "/" + operation["id"], headers=ADMIN)


def result(**changes):
    messages = {"code_directory": "/machine/new-code", "git_repository": "/machine/new-code",
                "python_interpreter": "/opt/conda/envs/training/bin/python",
                "openpi_import": "/machine/new-code/src/openpi/__init__.py",
                "selected_project": "/machine/new-code/src/openpi/training/config.py",
                "selected_project_import": "/machine/new-code/src/openpi/__init__.py",
                "interpreter_probe": "Python 3.11.10 at /opt/conda/envs/training/bin/python", "jax_import": "/opt/jax/__init__.py"}
    runtime = {"profile_id": "default", "status": "READY", "checked_at": now(), "python_version": "3.11.10",
               "executable": "/opt/conda/envs/training/bin/python", "prefix": "/opt/conda/envs/training",
               "module_origin": "/machine/new-code/src/openpi/__init__.py",
               "checks": [{"name": name, "ok": True, "message": messages[name]} for name in (
                   "code_directory", "python_interpreter", "git_repository", "interpreter_probe", "openpi_import", "jax_import",
                   "selected_project", "selected_project_import")]}
    return {"profile_id": "default", "code_dir": "/machine/new-code", "python": "/opt/conda/envs/training/bin/python",
            "runtime_check": runtime, "persisted": True, **changes}


def ack(platform, node, operation, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{operation['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": status, "result": result() if value is None else value})


def profile(platform, node):
    return platform.client.get("/api/v1/profiles", headers=ADMIN, params={"node_id": node["node_id"]}).json()[0]


def heartbeat_profile(platform, node, **changes):
    value = profile(platform, node)
    value = {key: value[key] for key in ("id", "name", "code_dir", "python", "adapter", "recipe", "execution_mode", "docker_image", "metadata")}
    value.update(changes)
    return platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={"node_id": node["node_id"],
        "gpus": [], "metrics": {}, "active_attempts": [], "profiles": [value]})


def add_run(session, node, state):
    run = Run(id=uid(), name="existing", node_id=node["node_id"], profile_id="default", template_id="openpi-sft",
        state=state, created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1, code_expectation={})
    session.add(run)
    return run


def test_agent_persisted_receipt_updates_only_runtime_and_invalidates_old_previews(platform):
    node = register(platform)
    before = profile(platform, node)
    with platform.app.state.db.write() as session:
        historical = add_run(session, node, "SUCCEEDED")
        for status in ("PENDING", "READY"):
            session.add(Preview(id=uid(), node_id=node["node_id"], status=status,
                request={"profile_id": "default"}, created_at=now(), expires_at=offset_time(300)))
    response = create(platform, node)
    assert response.status_code == 202 and response.headers["cache-control"] == "no-store"
    operation = response.json()
    assert operation["status"] == "PENDING" and operation["profile"] is None and operation["error"] is None
    assert operation["requested"] == {key: body()[key] for key in ("code_dir", "python")}
    assert profile(platform, node) == before
    assert create(platform, node).json() == operation
    assert platform.client.get(path(node), headers=ADMIN, params={"profile_id": "default"}).json() == {"operation": operation}
    command = next(command for command in platform.commands(node) if command["id"] == operation["id"])
    assert command["type"] == "PROFILE_REBIND"
    assert {key: value for key, value in command["payload"].items() if key != "expires_at"} == body()
    receipt = result(); receipt["private_output"] = "must-not-be-stored"
    assert ack(platform, node, operation, receipt).status_code == 200
    completed = get(platform, node, operation).json()
    assert completed["status"] == "READY" and completed["runtime_check"]["status"] == "READY"
    assert profile(platform, node) == {**before, "code_dir": body()["code_dir"], "python": body()["python"]}
    assert completed["profile"] == profile(platform, node)
    assert create(platform, node).json() == completed
    assert ack(platform, node, operation, receipt).status_code == 200
    assert ack(platform, node, operation, {**receipt, "python": "/other/python"}).status_code == 409
    with platform.app.state.db.read() as session:
        assert session.get(Run, historical.id).state == "SUCCEEDED"
        assert all(preview.status == "FAILED" for preview in session.scalars(select(Preview)))
        stored = session.get(Command, operation["id"])
        assert "private_output" not in stored.result["agent_result"]
        assert session.get(Node, node["node_id"]).metrics["runtime_checks"] == [completed["runtime_check"]]


def test_heartbeat_cannot_apply_before_ack_or_revert_with_known_stale_paths(platform):
    node = register(platform)
    operation = create(platform, node).json()
    proposed = {key: body()[key] for key in ("code_dir", "python")}
    response = heartbeat_profile(platform, node, **proposed)
    assert response.status_code == 200
    assert response.json()["profile_sync"]["code"] == "PROFILE_REBIND_PENDING"
    assert profile(platform, node)["code_dir"] == "/machine/code"
    value = result()
    assert ack(platform, node, operation, value).status_code == 200
    stale = heartbeat_profile(platform, node, code_dir="/machine/code", python="/usr/bin/python3")
    assert stale.json()["profile_sync"]["status"] == "BLOCKED"
    assert profile(platform, node)["code_dir"] == proposed["code_dir"]
    assert heartbeat_profile(platform, node, **proposed).json()["profile_sync"]["status"] == "APPLIED"
    # Once converged, later authenticated Agent reconfiguration keeps its usual semantics.
    assert heartbeat_profile(platform, node, code_dir="/machine/manual").json()["profile_sync"]["status"] == "APPLIED"
    assert profile(platform, node)["code_dir"] == "/machine/manual"


@pytest.mark.parametrize("state", ["QUEUED", "STARTING", "PREPARING", "RUNNING", "CANCELLING", "STOPPING", "UNKNOWN"])
def test_nonterminal_runs_block_rebind_even_without_a_current_gpu_process(platform, state):
    node = register(platform)
    with platform.app.state.db.write() as session:
        add_run(session, node, state)
    response = create(platform, node)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PROFILE_REBIND_BUSY"


@pytest.mark.parametrize("command_type", ["START", "PREFLIGHT", "RECIPE_WRITE"])
def test_pending_checkout_operations_block_rebind(platform, command_type):
    node = register(platform)
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=node["node_id"], type=command_type, status="PENDING", created_at=now(),
            payload={"profile_id": "default", "definition": {"source": {"code_dir": "/machine/code"}}}))
    response = create(platform, node)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PROFILE_REBIND_BUSY"


def test_pending_rebind_blocks_training_source_writeback_and_machine_management(platform):
    node = register(platform)
    operation = create(platform, node).json()
    request = {"name": "blocked", "node_id": node["node_id"], "profile_id": "default", "template_id": "openpi-sft",
               "recipe": "pi0_base", "inputs": {}, "parameters": {}, "gpu_count": 1}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PROFILE_REBIND_PENDING"
    with platform.app.state.db.read() as session:
        with pytest.raises(APIError) as captured:
            require_no_pending_writeback(session, node["node_id"], "/machine/code")
        assert captured.value.code == "PROFILE_REBIND_PENDING"
    management = platform.client.get(f"/api/v1/nodes/{node['node_id']}/management", headers=ADMIN).json()
    assert management["can_disconnect"] is False
    assert get(platform, node, operation).json()["status"] == "PENDING"


def test_expiry_and_ambiguous_ack_never_unlock_a_potentially_persisted_change(platform):
    node = register(platform)
    operation = create(platform, node).json()
    with platform.app.state.db.write() as session:
        command = session.get(Command, operation["id"])
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    expired = get(platform, node, operation).json()
    assert expired["status"] == "PENDING" and expired["error"]["code"] == "PROFILE_REBIND_CONFIRMATION_PENDING"
    assert ack(platform, node, operation, {"error": {"code": "PROFILE_REBIND_FAILED"}}, status="REJECTED").status_code == 200
    assert get(platform, node, operation).json()["status"] == "PENDING"
    # A late cached persisted receipt reconciles the same operation after TTL.
    assert ack(platform, node, operation).status_code == 200
    assert get(platform, node, operation).json()["status"] == "READY"


def test_proven_unmodified_failure_unlocks_and_can_be_retried(platform):
    node = register(platform)
    operation = create(platform, node).json()
    value = {"error": {"code": "PROFILE_REBIND_RUNTIME_UNAVAILABLE", "message": "untrusted details", "persisted": False}}
    assert ack(platform, node, operation, value, status="REJECTED").status_code == 200
    failed = get(platform, node, operation).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == "PROFILE_REBIND_RUNTIME_UNAVAILABLE"
    assert "untrusted details" not in str(failed)
    assert ack(platform, node, operation, value, status="REJECTED").status_code == 200
    retry = create(platform, node)
    assert retry.status_code == 202 and retry.json()["id"] != operation["id"]


@pytest.mark.parametrize("change", ["profile", "run"])
def test_successful_agent_receipt_remains_locked_if_server_runtime_changed_or_became_busy(platform, change):
    node = register(platform); operation = create(platform, node).json()
    with platform.app.state.db.write() as session:
        if change == "profile":
            session.get(Profile, (node["node_id"], "default")).python = "/usr/bin/changed"
        else:
            add_run(session, node, "UNKNOWN")
    assert ack(platform, node, operation).status_code == 200
    checked = get(platform, node, operation).json()
    assert checked["status"] == "PENDING" and checked["error"]["code"] == "PROFILE_REBIND_CONFIRMATION_PENDING"
    assert profile(platform, node)["code_dir"] == "/machine/code"


def test_old_agent_unsupported_command_is_a_safe_unmodified_failure(platform):
    node = register(platform); operation = create(platform, node).json()
    assert ack(platform, node, operation, {"error": {"code": "COMMAND_UNSUPPORTED"}}, status="REJECTED").status_code == 200
    failed = get(platform, node, operation).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == "AGENT_PROFILE_REBIND_UNSUPPORTED"


@pytest.mark.parametrize("mutate", [lambda value:value.update(profile_id="other"), lambda value:value.update(persisted=False),
    lambda value:value.update(code_dir="/machine/other"), lambda value:value.update(python="/usr/bin/python"),
    lambda value:value["runtime_check"].update(status="FAILED"), lambda value:value["runtime_check"].update(profile_id="other"),
    lambda value:value["runtime_check"].update(module_origin="/outside/openpi/__init__.py"),
    lambda value:value["runtime_check"].update(module_origin="/machine/old-code/src/openpi/__init__.py"),
    lambda value:value["runtime_check"]["checks"][1].update(message="/usr/bin/python3"),
    lambda value:value["runtime_check"]["checks"][6].update(message="/machine/other/src/openpi/training/config.py"),
    lambda value:value["runtime_check"]["checks"][7].update(message="/machine/other/src/openpi/__init__.py"),
    lambda value:value["runtime_check"].update(checks=value["runtime_check"]["checks"][:6]),
    lambda value:value["runtime_check"].update(checks=[]), lambda value:value["runtime_check"].update(checked_at="bad")])
def test_invalid_persisted_receipts_remain_pending_and_do_not_change_paths(platform, mutate):
    node = register(platform); operation = create(platform, node).json(); value = result(); mutate(value)
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, node, operation).json()["status"] == "PENDING"
    assert profile(platform, node)["code_dir"] == "/machine/code"


@pytest.mark.parametrize("changes", [{"code_dir": "relative"}, {"code_dir": "/machine/../other"}, {"code_dir": "/outside/code"},
    {"python": "python"}, {"python": "/opt/../bin/python"}, {"python": "/bin/py\x00thon"},
    {"code_dir": "/machine/code\n"}, {"expected_python": "bad"}, {"profile_id": "other"}, {"extra": "no"}])
def test_path_scope_and_request_shape_are_checked_before_dispatch(platform, changes):
    node = register(platform)
    assert create(platform, node, body(**changes)).status_code == 422
    assert platform.commands(node) == []


def test_expected_runtime_cas_and_distinct_pending_operation_conflicts(platform):
    node = register(platform)
    stale = create(platform, node, body(expected_code_dir="/machine/other"))
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "PROFILE_REBIND_CONFLICT"
    operation = create(platform, node).json()
    conflict = create(platform, node, body(code_dir="/machine/third"))
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "PROFILE_REBIND_PENDING"
    assert get(platform, node, operation).status_code == 200


def test_old_success_cannot_be_replayed_after_a_later_runtime_change(platform):
    node = register(platform); operation = create(platform, node).json()
    assert ack(platform, node, operation).status_code == 200
    assert create(platform, node).json()["status"] == "READY"
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).code_dir = "/machine/third"
    stale = create(platform, node)
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "PROFILE_REBIND_CONFLICT"
    assert profile(platform, node)["code_dir"] == "/machine/third"


@pytest.mark.parametrize("change", ["offline", "stale", "inactive", "adapter", "docker"])
def test_only_fresh_online_active_native_openpi_profiles_can_be_rebound(platform, change):
    node = register(platform)
    with platform.app.state.db.write() as session:
        current = session.get(Profile, (node["node_id"], "default")); machine = session.get(Node, node["node_id"])
        if change == "offline":machine.status = "OFFLINE"
        elif change == "stale":machine.last_seen = offset_time(-1000)
        elif change == "inactive":current.active = False
        elif change == "adapter":current.adapter = "smoke"
        else:current.execution_mode = "docker"
    assert create(platform, node).status_code in {409, 422}
    assert platform.commands(node) == []


def test_operation_reads_require_node_scope_and_human_management_auth(platform):
    node = register(platform); other = register(platform, name="other")
    assert platform.client.get(path(node), headers=ADMIN, params={"profile_id": "default"}).json() == {"operation": None}
    for headers in ({}, node["headers"]):
        assert create(platform, node, headers=headers).status_code == 401
    operation = create(platform, node).json()
    assert platform.client.get(path(other) + "/" + operation["id"], headers=ADMIN).status_code == 404
    assert ack(platform, other, operation).status_code == 403
    assert platform.client.get(path(node) + "/" + operation["id"]).status_code == 401


def test_browser_requires_admin_permission_csrf_and_recent_password_proof(platform):
    node = register(platform); password = "isolated-profile-rebind-password"
    for role in ("admin", "operator", "viewer"):
        platform.app.state.auth.create_user("profile-" + role, password, role)
        client = TestClient(platform.app, base_url="https://testserver")
        try:
            logged = client.post("/api/v1/auth/login", json={"username": "profile-" + role, "password": password}).json()
            client.headers["X-CSRF-Token"] = logged["csrf_token"]
            response = client.post(path(node), json=body())
            assert response.status_code == 403
            assert response.json()["error"]["code"] == ("REAUTH_REQUIRED" if role == "admin" else "FORBIDDEN")
            if role == "admin":
                assert client.post("/api/v1/auth/reauth", json={"password": password}).status_code == 200
                assert client.post(path(node), json=body()).status_code == 202
                audits = client.get("/api/v1/audit").json()
                assert any(row["actor_username"] == "profile-admin" and row["path"] == path(node)
                           and row["action"] == "api.mutation" and row["status"] == 202 for row in audits)
            else:
                assert client.get(path(node), params={"profile_id": "default"}).status_code == 403
        finally:
            client.close()


def test_concurrent_distinct_changes_queue_only_one_command(platform):
    node = register(platform); barrier = Barrier(2)
    def submit(index):
        barrier.wait(timeout=5)
        return create(platform, node, body(code_dir=f"/machine/target-{index}"))
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(submit, range(2)))
    assert sorted(response.status_code for response in results) == [202, 409]
    assert len([command for command in platform.commands(node) if command["type"] == "PROFILE_REBIND"]) == 1


def test_concurrent_identical_requests_reuse_the_same_durable_operation(platform):
    node = register(platform); barrier = Barrier(2)
    def submit(_):
        barrier.wait(timeout=5)
        return create(platform, node)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(submit, range(2)))
    assert all(response.status_code == 202 for response in results)
    assert len({response.json()["id"] for response in results}) == 1


def test_actual_local_agent_probe_and_persistence_receipt_is_accepted_end_to_end(platform, tmp_path, monkeypatch):
    """Real temp files/probes connect both sides; package bodies must stay unloaded."""
    from vla_agent.config import load_config
    from vla_agent.profile_rebinding import rebind_profile
    from vla_agent.state import State

    root = tmp_path.resolve()
    checkout = root / "new-project"; checkout.mkdir()
    subprocess.run(["git", "-C", str(checkout), "init", "-b", "fixture"], check=True, capture_output=True)
    for name in ("openpi", "jax"):
        package = checkout / "src" / name; package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise RuntimeError('The availability probe must not import package code')\n")
    source = checkout / "src/openpi/training/config.py"; source.parent.mkdir(); source.write_text("_CONFIGS = []\n")
    alias = root / "project-link"; alias.symlink_to(checkout, target_is_directory=True)
    interpreter = root / "training-python"; interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(checkout / "src"))
    old = root / "old-missing-project"
    old_profile = {"id": "local-profile", "name": "Local fixture", "adapter": "openpi-sft", "execution_mode": "native",
                   "code_dir": str(old), "python": sys.executable, "recipe": None, "metadata": {}}
    enrollment = {"name": "local-fixture", "hostname": "local-fixture", "roots": [str(root)],
                  "output_root": str(root / "outputs"), "profiles": [old_profile]}
    response = platform.client.post("/agent/v1/register", headers=ENROLL, json=enrollment)
    assert response.status_code == 201, response.text
    node = response.json(); node["headers"] = {"Authorization": "Bearer " + node["token"]}
    platform.heartbeat(node, gpus=[])
    config_path = root / "agent.json"
    config_path.write_text(json.dumps({**enrollment, "enrollment_token": "isolated-fixture-token",
                                      "control_plane_url": "http://testserver", "state_dir": str(root / "agent-state")}))
    config_path.chmod(0o600)
    config = load_config(config_path); state = State(config["state_dir"])
    requested = {"profile_id": old_profile["id"], "expected_code_dir": str(old), "expected_python": sys.executable,
                 "code_dir": str(alias), "python": str(interpreter)}
    response = create(platform, node, requested)
    assert response.status_code == 202, response.text
    operation = response.json()
    command = next(item for item in platform.commands(node) if item["id"] == operation["id"])
    actual = rebind_profile(config, state, command["payload"], operation_id=command["id"])
    assert actual["code_dir"] == str(alias) and actual["runtime_check"]["module_origin"] == str(checkout / "src/openpi/__init__.py")
    assert json.loads(config_path.read_text())["profiles"][0]["python"] == str(interpreter)
    assert ack(platform, node, operation, actual).status_code == 200
    complete = get(platform, node, operation).json()
    assert complete["status"] == "READY", complete
    assert complete["profile"]["code_dir"] == str(alias) and complete["profile"]["python"] == str(interpreter)
    cached = rebind_profile(config, state, command["payload"], operation_id=command["id"])
    assert cached == actual
    assert ack(platform, node, operation, cached).status_code == 200
    assert not list(checkout.rglob("__pycache__"))
