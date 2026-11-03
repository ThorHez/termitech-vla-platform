"""An existing Agent can add another checked environment without losing its profiles."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import subprocess
import sys
from threading import Barrier

import pytest

from vla_control.models import Node

from .conftest import ADMIN, ENROLL
from .test_profile_rebindings import ack, add_run, get, result
from .test_recipe_details import register


def body(**changes):
    return {"profile_id": "second-openpi", "name": "Second OpenPI", "code_dir": "/machine/new-code",
            "python": "/opt/conda/envs/training/bin/python", **changes}


def create(platform, node, changes=None, headers=ADMIN):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/profiles", headers=headers,
                                json=body(**(changes or {})))


def capable(platform, node):
    with platform.app.state.db.write() as session:
        machine = session.get(Node, node["node_id"])
        machine.metrics = {**machine.metrics, "capabilities": ["PROFILE_REBIND", "PROFILE_ADD"]}


def profiles(platform, node):
    response = platform.client.get("/api/v1/profiles", headers=ADMIN, params={"node_id": node["node_id"]})
    return {row["id"]: row for row in response.json()}


def receipt(**changes):
    value = result(profile_id=body()["profile_id"], action="add", name=body()["name"], **changes)
    value["runtime_check"]["profile_id"] = value["profile_id"]
    return value


def heartbeat(platform, node, entries):
    values = [{key: row[key] for key in ("id", "name", "code_dir", "python", "adapter", "recipe", "execution_mode", "docker_image", "metadata")}
              for row in entries]
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={"node_id": node["node_id"],
        "gpus": [], "metrics": {"capabilities": ["PROFILE_ADD"]}, "active_attempts": [], "profiles": values})
    assert response.status_code == 200, response.text
    return response.json()["profile_sync"]


def test_add_requires_persisted_receipt_preserves_existing_profile_and_handles_stale_heartbeats(platform):
    node = register(platform); capable(platform, node)
    original = profiles(platform, node)
    response = create(platform, node)
    assert response.status_code == 202, response.text
    operation = response.json()
    assert operation["action"] == "add" and operation["requested"] == {key: body()[key] for key in ("name", "code_dir", "python")}
    assert profiles(platform, node) == original
    assert create(platform, node).json() == operation
    proposed = {**original["default"], "id": body()["profile_id"], "name": body()["name"],
                "code_dir": body()["code_dir"], "python": body()["python"], "recipe": None}
    assert heartbeat(platform, node, [*original.values(), proposed])["status"] == "BLOCKED"
    assert profiles(platform, node) == original
    persisted = receipt()
    assert ack(platform, node, operation, persisted).status_code == 200
    complete = get(platform, node, operation).json()
    assert complete["status"] == "READY" and complete["profile"]["name"] == body()["name"]
    assert profiles(platform, node)["default"] == original["default"]
    assert create(platform, node).json() == complete
    assert ack(platform, node, operation, persisted).status_code == 200
    assert heartbeat(platform, node, original.values())["status"] == "BLOCKED"
    assert body()["profile_id"] in profiles(platform, node)
    current = profiles(platform, node)
    assert heartbeat(platform, node, current.values())["status"] == "APPLIED"
    assert profiles(platform, node) == current


@pytest.mark.parametrize("changes", [{"profile_id": "bad/id"}, {"profile_id": ""}, {"name": "  "},
    {"name": "bad\nname"}, {"code_dir": "relative"}, {"code_dir": "/outside/project"},
    {"python": "python"}, {"unexpected": "field"}])
def test_invalid_addition_does_not_dispatch(platform, changes):
    node = register(platform); capable(platform, node)
    assert create(platform, node, changes).status_code == 422
    assert platform.commands(node) == []


def test_add_rejects_old_agent_and_duplicate_or_pending_identifiers(platform):
    node = register(platform)
    unsupported = create(platform, node)
    assert unsupported.status_code == 409 and unsupported.json()["error"]["code"] == "AGENT_PROFILE_ADD_UNSUPPORTED"
    capable(platform, node)
    assert create(platform, node, {"profile_id": "default"}).status_code == 409
    operation = create(platform, node).json()
    conflicting = create(platform, node, {"name": "Different environment"})
    assert conflicting.status_code == 409 and conflicting.json()["error"]["code"] == "PROFILE_REBIND_PENDING"
    assert ack(platform, node, operation, {"error": {"code": "PROFILE_REBIND_RUNTIME_UNAVAILABLE", "persisted": False}}, status="REJECTED").status_code == 200
    assert len(profiles(platform, node)) == 1
    retry = create(platform, node)
    assert retry.status_code == 202 and retry.json()["id"] != operation["id"]


@pytest.mark.parametrize("mutation", ["missing-action", "wrong-name", "bad-runtime"])
def test_add_invalid_receipt_never_publishes_environment(platform, mutation):
    node = register(platform); capable(platform, node); operation = create(platform, node).json()
    value = receipt()
    if mutation == "missing-action": value.pop("action")
    elif mutation == "wrong-name": value["name"] = "Different environment"
    else: value["runtime_check"]["profile_id"] = "default"
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, node, operation).json()["status"] == "PENDING"
    assert len(profiles(platform, node)) == 1


def test_add_allows_unrelated_running_profile_and_identical_concurrent_requests_share_one_operation(platform):
    node = register(platform); capable(platform, node)
    with platform.app.state.db.write() as session:
        add_run(session, node, "RUNNING")
    barrier = Barrier(2)
    def submit(_):
        barrier.wait(timeout=5)
        return create(platform, node)
    with ThreadPoolExecutor(max_workers=2) as workers:
        responses = list(workers.map(submit, range(2)))
    assert all(response.status_code == 202 for response in responses)
    assert len({response.json()["id"] for response in responses}) == 1
    assert ack(platform, node, responses[0].json(), receipt()).status_code == 200
    assert len(profiles(platform, node)) == 2


def test_actual_agent_addition_receipt_is_accepted_without_restarting_or_altering_original_profile(platform, tmp_path, monkeypatch):
    from vla_agent.config import load_config
    from vla_agent.profile_rebinding import rebind_profile
    from vla_agent.state import State

    root = tmp_path.resolve(); checkout = root / "new-code"; checkout.mkdir()
    subprocess.run(["git", "-C", str(checkout), "init", "-b", "fixture"], check=True, capture_output=True)
    for name in ("openpi", "jax"):
        package = checkout / "src" / name; package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise RuntimeError('probe must not import package')\n")
    source = checkout / "src/openpi/training/config.py"; source.parent.mkdir(); source.write_text("_CONFIGS = []\n")
    interpreter = root / "training-python"; interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(checkout / "src"))
    old = {"id": "existing", "name": "Keep original", "code_dir": str(root / "original"), "python": sys.executable,
           "adapter": "openpi-sft", "execution_mode": "native", "metadata": {"keep": True}}
    enrollment = {"name": "local-fixture", "hostname": "local-fixture", "roots": [str(root)],
                  "output_root": str(root / "outputs"), "profiles": [old]}
    registered = platform.client.post("/agent/v1/register", headers=ENROLL, json=enrollment)
    assert registered.status_code == 201, registered.text
    node = registered.json(); node["headers"] = {"Authorization": "Bearer " + node["token"]}
    platform.heartbeat(node, gpus=[]); capable(platform, node)
    config_path = root / "agent.json"
    config_path.write_text(json.dumps({**enrollment, "control_plane_url": "http://testserver", "state_dir": str(root / "state")}))
    config = load_config(config_path); state = State(config["state_dir"])
    original = copy.deepcopy(config["profiles"])
    response = create(platform, node, {"code_dir": str(checkout), "python": str(interpreter)})
    assert response.status_code == 202, response.text
    operation = response.json(); command = next(row for row in platform.commands(node) if row["id"] == operation["id"])
    actual = rebind_profile(config, state, command["payload"], operation_id=command["id"])
    assert ack(platform, node, operation, actual).status_code == 200
    assert get(platform, node, operation).json()["status"] == "READY"
    assert config["profiles"][:-1] == original and len(load_config(config_path)["profiles"]) == 2
    assert config["profiles"][-1]["python"] == str(interpreter)
    assert rebind_profile(config, state, command["payload"], operation_id=command["id"]) == actual
