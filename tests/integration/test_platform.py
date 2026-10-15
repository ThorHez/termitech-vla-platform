"""Real HTTP + independent Agent/Runner process acceptance, with a CPU fixture."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vla_bootstrap_fixture", ROOT / "scripts" / "bootstrap_demo.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def eventually(fn, check=lambda value: bool(value), timeout=25):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        try:
            last = fn()
            if check(last):
                return last
        except httpx.TransportError:
            pass
        time.sleep(0.15)
    raise AssertionError(f"Timed out; last observation: {last}")


@pytest.fixture(scope="module")
def platform(tmp_path_factory):
    base = tmp_path_factory.mktemp("platform-http")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    fixture = bootstrap.prepare_demo(base, port)
    env = fixture["env"]
    env["VLA_SCHEDULER_INTERVAL_SECONDS"] = "0.1"
    logs = []
    children = []
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5,
                          headers={"Authorization": "Bearer " + fixture["credentials"]["admin_token"]})
    try:
        for name, argv in [("api", [sys.executable, "-m", "vla_control.main"]),
                           ("agent", [sys.executable, "-m", "vla_agent.main", "--config", str(fixture["config_file"])])]:
            log = (base / f"{name}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            if name == "api":
                eventually(lambda: client.get("/healthz").status_code == 200)
        node = eventually(lambda: client.get("/api/v1/nodes").json(), lambda nodes: len(nodes) == 1)[0]
        inputs = {}
        for role, path in fixture["assets"].items():
            response = client.post("/api/v1/assets", json={"name": role, "kind": role.upper(),
                                   "node_id": node["id"], "path": str(path), "version": "test-v1",
                                   "metadata": {"simulation": True}})
            assert response.is_success, response.text
            inputs[role] = response.json()["id"]
        yield client, fixture, node, inputs
    finally:
        client.close()
        for child in reversed(children):
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


def preview(platform, steps=5):
    client, fixture, node, inputs = platform
    response = client.post("/api/v1/run-previews", json={"name": "CPU 集成验收", "node_id": node["id"],
        "profile_id": "local-smoke", "template_id": "smoke", "inputs": inputs,
        "parameters": {"steps": steps, "batch_size": 4, "learning_rate": 0.01}, "gpu_count": 0})
    assert response.is_success, response.text
    result = eventually(lambda: client.get("/api/v1/run-previews/" + response.json()["id"]).json(),
                        lambda p: p["status"] != "PENDING")
    assert result["status"] == "READY", result
    return result


def submit(client, preflight, key):
    result = client.post("/api/v1/runs", json={"preflight_id": preflight["id"]},
                         headers={"Idempotency-Key": key})
    assert result.is_success, result.text
    return result.json()


def test_real_process_training_lifecycle_and_lineage(platform):
    client, fixture, node, inputs = platform
    preflight = preview(platform)
    assert len(preflight["code_snapshot"]["commit_sha"]) == 40
    run = submit(client, preflight, "integration-success")
    assert submit(client, preflight, "integration-success")["id"] == run["id"]
    result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                        lambda r: r["state"] in {"SUCCEEDED", "FAILED"})
    assert result["state"] == "SUCCEEDED", result
    assert result["attempts"][0]["code_snapshot"]["commit_sha"] == preflight["code_snapshot"]["commit_sha"]
    assert result["outputs"] and result["outputs"][0]["metadata"]["simulation"]
    assert result["outputs"][0]["node_id"] == node["id"]
    events = client.get(f"/api/v1/runs/{run['id']}/events").json()
    assert any(event["type"] == "METRIC" for event in events)
    logs = client.get(f"/api/v1/runs/{run['id']}/logs").json()
    assert logs["items"]


def test_cancel_waits_for_runner_exit(platform):
    client, *_ = platform
    run = submit(client, preview(platform, steps=500), "integration-cancel")
    eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(), lambda r: r["state"] == "RUNNING")
    result = client.post(f"/api/v1/runs/{run['id']}/cancel")
    assert result.is_success, result.text
    final = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                       lambda r: r["state"] in {"CANCELLED", "FAILED"})
    assert final["state"] == "CANCELLED", final


def test_changed_code_is_blocked_before_process_start(platform):
    client, fixture, *_ = platform
    preflight = preview(platform)
    code = Path(fixture["config"]["profiles"][0]["code_dir"]) / "smoke_training.py"
    previous = code.read_text()
    try:
        code.write_text(previous + "\n# changed after preview\n")
        run = submit(client, preflight, "integration-code-drift")
        result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                            lambda r: r["state"] == "FAILED")
        events = client.get(f"/api/v1/runs/{run['id']}/events").json()
        assert not any(event["type"] == "STARTED" for event in events)
        assert "CODE_CHANGED_SINCE_PREFLIGHT" in str(result) + str(events)
    finally:
        code.write_text(previous)


def test_changed_runtime_is_blocked_even_with_same_git_commit(platform):
    client, fixture, *_ = platform
    preflight = preview(platform)
    config_file = fixture["config_file"]
    previous = config_file.read_text()
    changed = json.loads(previous)
    changed["profiles"][0]["recipe"] = "changed-after-preflight"
    try:
        config_file.write_text(json.dumps(changed))
        run = submit(client, preflight, "integration-runtime-drift")
        result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                            lambda r: r["state"] == "FAILED")
        events = client.get(f"/api/v1/runs/{run['id']}/events").json()
        assert not any(event["type"] == "STARTED" for event in events)
        assert "CODE_CHANGED_SINCE_PREFLIGHT" in str(result) + str(events)
    finally:
        config_file.write_text(previous)
