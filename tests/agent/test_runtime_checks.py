import copy
import errno
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_agent.main import Agent
from vla_agent import runtime_checks
from vla_agent.preflight import run_preflight
from vla_agent.runtime_checks import RuntimeMonitor, check_runtime, validate_runtime
from vla_agent.safety import AgentError


def test_smoke_runtime_records_actual_python_without_openpi(agent_fixture):
    config = agent_fixture["config"]
    result = validate_runtime(config["profiles"][0], config["roots"])
    assert result["status"] == "READY"
    assert result["python_version"]
    assert Path(result["executable"]).is_absolute()
    assert result["module_origin"] == str(agent_fixture["repo"] / "smoke_training.py")
    assert not any(check["name"] == "openpi_import" for check in result["checks"])
    assert config["profiles"][0]["metadata"] == {}


def test_interpreter_must_be_executable_and_valid_python(agent_fixture):
    config = agent_fixture["config"]
    profile = copy.deepcopy(config["profiles"][0])
    interpreter = agent_fixture["root"] / "not-executable"
    interpreter.write_text("not python")
    interpreter.chmod(0o600)
    profile["python"] = str(interpreter)
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert "not executable" in failure["error"]
    profile["python"] = "/bin/echo"
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert "valid Python environment evidence" in failure["error"]
    profile["python"] = "python"
    assert "absolute" in check_runtime(profile, config["roots"])["error"]


def service_identity(monkeypatch, account="training-user", uid=1001):
    monkeypatch.setattr(runtime_checks.os, "geteuid", lambda: uid)
    monkeypatch.setattr(runtime_checks.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name=account))


def test_inaccessible_python_symlink_reports_actual_account_and_known_target(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = copy.deepcopy(config["profiles"][0])
    interpreter = agent_fixture["root"] / "python-link"
    target = "/root/.local/share/uv/python/cpython-3.11/bin/python3.11"
    interpreter.symlink_to(target)
    profile["python"] = str(interpreter)
    service_identity(monkeypatch, account="collector", uid=996)
    real_is_file = Path.is_file

    def denied(path):
        if path == interpreter:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", denied)
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert failure["checks"][-1]["name"] == "python_interpreter"
    assert failure["checks"][-1]["ok"] is False
    assert "collector (uid=996)" in failure["error"]
    assert str(interpreter) in failure["error"] and target in failure["error"]
    assert "traversal (execute)" in failure["error"]
    assert "read/execute" in failure["error"]
    assert "same account configured for the Agent service" in failure["error"]
    with pytest.raises(AgentError) as error:
        validate_runtime(profile, config["roots"])
    assert error.value.code == "RUNTIME_UNAVAILABLE"
    assert "collector (uid=996)" in str(error.value)


@pytest.mark.parametrize("account,uid", [("training-user", 1001), ("root", 0)])
def test_execute_access_denied_reports_current_service_identity(agent_fixture, monkeypatch, account, uid):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    service_identity(monkeypatch, account, uid)
    real_access = runtime_checks.os.access
    monkeypatch.setattr(runtime_checks.os, "access", lambda path, mode:
                        False if str(path) == profile["python"] else real_access(path, mode))
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert failure["checks"][-1]["name"] == "python_interpreter"
    assert f"{account} (uid={uid})" in failure["error"]
    assert "not executable" in failure["error"]
    assert profile["python"] in failure["error"]
    assert "SSH check as root" not in failure["error"]


def test_spawn_permission_failure_keeps_probe_stage_and_account(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    service_identity(monkeypatch)
    real_run = subprocess.run

    def denied(args, **kwargs):
        if args[0] == profile["python"]:
            raise PermissionError(errno.EACCES, "Permission denied", args[0])
        return real_run(args, **kwargs)

    monkeypatch.setattr(runtime_checks.subprocess, "run", denied)
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert failure["checks"][-1]["name"] == "interpreter_probe"
    assert "training-user (uid=1001)" in failure["error"]
    assert profile["python"] in failure["error"]
    assert "Permission denied" in failure["error"]


def test_permission_diagnostic_survives_unreadable_link_and_missing_account(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    monkeypatch.setattr(runtime_checks.os, "geteuid", lambda: 12345)

    def unavailable(*args):
        raise KeyError("Unknown account")

    real_readlink = runtime_checks.os.readlink

    def denied_link(path, *args, **kwargs):
        if str(path) == profile["python"]:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(runtime_checks.pwd, "getpwuid", unavailable)
    monkeypatch.setattr(runtime_checks.os, "readlink", denied_link)
    monkeypatch.setattr(runtime_checks.os, "access", lambda *args: False)
    failure = check_runtime(profile, config["roots"])
    assert failure["status"] == "FAILED"
    assert "unknown (uid=12345)" in failure["error"]
    assert "Known Python symlink target" not in failure["error"]


def openpi_profile(agent_fixture, monkeypatch, *, external=False, jax=True):
    profile = copy.deepcopy(agent_fixture["config"]["profiles"][0])
    profile.update(adapter="openpi-sft", recipe="pi0_local")
    package_root = agent_fixture["root"] / "other-repo" if external else agent_fixture["repo"] / "src"
    openpi = package_root / "openpi"
    openpi.mkdir(parents=True)
    # If either package is imported this probe will fail. Only find_spec is safe.
    (openpi / "__init__.py").write_text("raise RuntimeError('openpi was imported')\n")
    if jax:
        (package_root / "jax").mkdir()
        (package_root / "jax" / "__init__.py").write_text("raise RuntimeError('JAX was imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(package_root))
    return profile, package_root


def test_openpi_runtime_locates_packages_without_import_or_gpu_init(agent_fixture, monkeypatch):
    profile, package_root = openpi_profile(agent_fixture, monkeypatch)
    result = check_runtime(profile, agent_fixture["config"]["roots"])
    assert result["status"] == "READY", result
    assert result["module_origin"] == str(package_root / "openpi" / "__init__.py")
    assert result["checks"][-1]["name"] == "jax_import"
    assert profile["metadata"] == {}
    assert not list(package_root.rglob("__pycache__"))


def test_openpi_import_must_resolve_inside_registered_repository(agent_fixture, monkeypatch):
    profile, _ = openpi_profile(agent_fixture, monkeypatch, external=True)
    failure = check_runtime(profile, agent_fixture["config"]["roots"])
    assert failure["status"] == "FAILED"
    assert "outside the configured Git repository" in failure["error"]
    assert failure["checks"][-1]["name"] == "openpi_import"


def test_openpi_runtime_reports_missing_jax(agent_fixture, monkeypatch):
    profile, _ = openpi_profile(agent_fixture, monkeypatch)
    real_run = subprocess.run

    def hide_jax(args, **kwargs):
        completed = real_run(args, **kwargs)
        if args[1:3] == ["-B", "-c"]:
            import json
            evidence = json.loads(completed.stdout)
            evidence["jax_origin"] = None
            completed.stdout = json.dumps(evidence)
        return completed

    monkeypatch.setattr("vla_agent.runtime_checks.subprocess.run", hide_jax)
    failure = check_runtime(profile, agent_fixture["config"]["roots"])
    assert failure["status"] == "FAILED"
    assert "cannot locate jax" in failure["error"]
    assert failure["checks"][-1]["name"] == "jax_import"


def test_runtime_probe_times_out_and_sync_preflight_rejects(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    real_run = subprocess.run

    def timeout_probe(args, **kwargs):
        if args[1:3] == ["-B", "-c"]:
            assert kwargs["timeout"] == 10
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            assert "VLA_ADMIN_TOKEN" not in kwargs["env"]
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return real_run(args, **kwargs)

    monkeypatch.setenv("VLA_ADMIN_TOKEN", "private-test-token")
    monkeypatch.setattr("vla_agent.runtime_checks.subprocess.run", timeout_probe)
    with pytest.raises(AgentError) as error:
        validate_runtime(config["profiles"][0], config["roots"])
    assert error.value.code == "RUNTIME_UNAVAILABLE"
    assert "timed out" in str(error.value)
    result = run_preflight(config, agent_fixture["payload"])
    assert result["error"]["code"] == "RUNTIME_UNAVAILABLE"
    assert "timed out" in result["error"]["message"]


def await_result(monitor, profiles, wanted="READY", timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = monitor.snapshot(profiles)
        if result and result[0]["status"] == wanted:
            return result[0]
        time.sleep(0.01)
    raise AssertionError(f"No {wanted} result: {result}")


def test_runtime_worker_reports_permission_failure_then_recovers(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profiles = config["profiles"]
    interpreter = Path(profiles[0]["python"])
    denied = [True]
    current_time = [0]
    real_is_file = Path.is_file

    def accessible(path):
        if path == interpreter and denied[0]:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", accessible)
    monitor = RuntimeMonitor(config["roots"], clock=lambda: current_time[0])
    try:
        failure = await_result(monitor, profiles, wanted="FAILED")
        assert failure["checks"][-1]["name"] == "python_interpreter"
        assert "Agent service account" in failure["error"]
        denied[0] = False
        current_time[0] = 60
        assert monitor.snapshot(profiles)[0]["status"] == "CHECKING"
        assert await_result(monitor, profiles)["status"] == "READY"
    finally:
        monitor.close()


def test_runtime_worker_does_not_block_heartbeat_or_command_poll(agent_fixture, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_check(profile, roots, timeout):
        started.set()
        release.wait(2)
        return {"profile_id": profile["id"], "status": "READY", "checked_at": "checked", "checks": []}

    class Client:
        def __init__(self):
            self.heartbeats = []
            self.commands = 0

        def request(self, method, path, token, body=None, headers=None):
            if path.endswith("heartbeat"):
                self.heartbeats.append(body)
                return {}
            if path.startswith("/agent/v1/commands?"):
                self.commands += 1
                return []
            raise AssertionError(path)

    client = Client()
    config = agent_fixture["config"]
    agent = Agent(config, client=client)
    agent.credentials = {"node_id": "fixture-node", "token": "fixture-token"}
    agent.runtime_monitor = RuntimeMonitor(config["roots"], checker=slow_check)
    monkeypatch.setattr("vla_agent.main.system_metrics", lambda _: {})
    monkeypatch.setattr("vla_agent.main.gpu_inventory", lambda: ([], {"status": "unavailable"}))
    try:
        before = time.monotonic()
        agent.heartbeat()
        assert time.monotonic() - before < 0.5
        assert started.wait(1)
        before = time.monotonic()
        agent.heartbeat()
        agent.poll_commands()
        assert time.monotonic() - before < 0.5
        assert len(client.heartbeats) == 2 and client.commands == 1
        assert client.heartbeats[0]["metrics"]["runtime_checks"][0]["status"] == "CHECKING"
        release.set()
        await_result(agent.runtime_monitor, config["profiles"])
        agent.heartbeat()
        assert client.heartbeats[-1]["metrics"]["runtime_checks"][0]["status"] == "READY"
    finally:
        release.set()
        agent.runtime_monitor.close()


def test_runtime_worker_retries_after_60_seconds_and_config_change(agent_fixture):
    current_time = [0]
    calls = []

    def checker(profile, roots, timeout):
        calls.append(profile["python"])
        return {"profile_id": profile["id"], "status": "READY", "checked_at": "checked", "checks": []}

    config = agent_fixture["config"]
    profiles = copy.deepcopy(config["profiles"])
    monitor = RuntimeMonitor(config["roots"], checker=checker, clock=lambda: current_time[0])
    try:
        assert monitor.snapshot(profiles)[0]["status"] == "CHECKING"
        await_result(monitor, profiles)
        assert len(calls) == 1
        current_time[0] = 59
        assert monitor.snapshot(profiles)[0]["status"] == "READY"
        assert len(calls) == 1
        current_time[0] = 60
        assert monitor.snapshot(profiles)[0]["status"] == "CHECKING"
        await_result(monitor, profiles)
        assert len(calls) == 2
        profiles[0]["python"] = "/different/python"
        assert monitor.snapshot(profiles)[0]["status"] == "CHECKING"
        await_result(monitor, profiles)
        assert calls[-1] == "/different/python"
        assert monitor.snapshot([]) == []
    finally:
        monitor.close()
