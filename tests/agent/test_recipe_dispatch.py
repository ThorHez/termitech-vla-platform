"""An in-progress source write cannot race a newly delivered START."""

from types import SimpleNamespace
import fcntl
from pathlib import Path
import subprocess
import pytest

from vla_agent.main import Agent, _execute_preflight_command
from vla_agent.recipe_writeback import checkout_lock
from vla_agent.safety import AgentError
from vla_agent.state import State


def test_start_remains_retryable_until_writer_releases_checkout(agent_fixture, monkeypatch):
    payload = agent_fixture["payload"]
    command = {"id": "start-after-write", "type": "START", "payload": payload}
    acknowledgements = []

    class Client:
        def request(self, method, path, token, body=None):
            if method == "GET":
                return [command]
            acknowledgements.append(body)
            return {"ok": True}

    agent = Agent(agent_fixture["config"], client=Client())
    agent.credentials = {"node_id": "test-node", "token": "test-token"}
    launches = []
    real_popen = subprocess.Popen

    def launch(*args, **kwargs):
        if "vla_agent.runner" not in args[0]:
            return real_popen(*args, **kwargs)
        launches.append(args)
        return SimpleNamespace(pid=1234, poll=lambda: None)

    monkeypatch.setattr("vla_agent.main.subprocess.Popen", launch)
    try:
        with checkout_lock(agent.config, str(agent_fixture["repo"])):
            agent.poll_commands()
            assert not acknowledgements and not launches
            assert agent.state.attempt(payload["attempt_id"]) is None
            assert agent.state.command(command["id"]) is None
        agent.poll_commands()
        assert len(launches) == len(acknowledgements) == 1
        assert acknowledgements[0]["status"] == "APPLIED"
        assert agent.state.attempt(payload["attempt_id"])["state"] == "STARTING"
        agent.poll_commands()
        assert len(launches) == 1
    finally:
        agent.close()


def test_upgrade_maintenance_defers_start_and_recovery_without_rejecting_command(agent_fixture, monkeypatch):
    payload = agent_fixture["payload"]
    command = {"id": "start-during-upgrade", "type": "START", "payload": payload}
    acknowledgements, launches = [], []

    class Client:
        def request(self, method, path, token, body=None):
            if method == "GET":
                return [command]
            acknowledgements.append(body)
            return {"ok": True}

    agent = Agent(agent_fixture["config"], client=Client())
    agent.credentials = {"node_id": "node", "token": "token"}
    monkeypatch.setattr(agent, "_launch_runner", lambda attempt: launches.append(attempt["id"]))
    locks = Path(agent.config["state_dir"]) / "locks"
    locks.mkdir(exist_ok=True)
    try:
        with (locks / "maintenance.lock").open("a") as maintenance:
            fcntl.flock(maintenance, fcntl.LOCK_EX | fcntl.LOCK_NB)
            agent.poll_commands()
            assert not launches and not acknowledgements
            assert agent.state.attempt(payload["attempt_id"]) is None
            assert agent.state.command(command["id"]) is None
            # An already durable START must also wait for release switching.
            agent.state.create_attempt(payload, str(Path(agent.config["output_root"]) / payload["attempt_id"]))
            agent._recover_runners()
            assert not launches
        agent.poll_commands()
        assert launches == [payload["attempt_id"]]
        assert acknowledgements[0]["status"] == "APPLIED"
        assert acknowledgements[0]["result"]["duplicate"]
    finally:
        agent.close()


@pytest.mark.parametrize("interruption", [
    AgentError("RECIPE_WRITE_RECOVERY_PENDING", "commit exists; index publication interrupted"),
    OSError("durable receipt temporarily unreadable"),
])
def test_post_commit_recovery_does_not_persist_a_false_rejection(agent_fixture, monkeypatch, interruption):
    config = agent_fixture["config"]
    command = {"id": "recover-published-commit", "type": "RECIPE_WRITE", "payload": {}}
    calls = []
    monkeypatch.setattr("vla_agent.main.os.setsid", lambda: None)
    monkeypatch.setattr("vla_agent.main.signal.signal", lambda *_: None)

    def write(*_):
        calls.append(True)
        if len(calls) == 1:
            raise interruption
        return {"commit_sha": "a" * 40}

    monkeypatch.setattr("vla_agent.recipe_writeback.write_recipe", write)
    _execute_preflight_command(config, command)
    state = State(config["state_dir"])
    assert state.command(command["id"]) is None
    _execute_preflight_command(config, command)
    assert state.command(command["id"]) == {"status": "APPLIED", "result": {"commit_sha": "a" * 40}}
    _execute_preflight_command(config, command)
    assert len(calls) == 2
