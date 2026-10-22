import json
import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from vla_agent.main import Agent


class CommandClient:
    def __init__(self, commands):
        self.commands = commands
        self.acks = {}
        self.heartbeats = []
        self.events = []
        self.fail_preflight_ack = False

    def request(self, method, path, token, body=None, headers=None):
        if path.startswith("/agent/v1/commands?"):
            return [command for command in self.commands if command["id"] not in self.acks]
        if path.endswith("/ack"):
            command_id = path.split("/")[-2]
            if command_id.startswith("preview") and self.fail_preflight_ack:
                raise OSError("ACK network unavailable")
            self.acks[command_id] = body
            return {"ok": True}
        if path.endswith("heartbeat"):
            self.heartbeats.append(body)
            return {}
        if path.endswith("events/batch"):
            self.events.extend(body["events"])
            return {"accepted": len(body["events"])}
        raise AssertionError(path)


def wait_until(callback, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(0.03)
    raise AssertionError("Background preflight did not reach expected state")


def blocked_probe(agent_fixture):
    root, config = agent_fixture["root"], agent_fixture["config"]
    marker, release = root / "probe-pids", root / "release-probe"
    interpreter = root / "training-python"
    interpreter.write_text(
        f"#!{sys.executable}\nimport os,sys,time\nfrom pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as out: out.write(str(os.getpid())+'\\n')\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.02)\n"
        f"os.execv({sys.executable!r},[{sys.executable!r},*sys.argv[1:]])\n")
    interpreter.chmod(0o755)
    config["profiles"][0]["python"] = str(interpreter)
    command = {"id": "preview-one", "type": "PREFLIGHT", "payload": {
        **agent_fixture["payload"], "recipe": "smoke", "gpu_count": 0}}
    return command, marker, release


def connected_agent(agent_fixture, client):
    agent = Agent(agent_fixture["config"], client=client)
    agent.credentials = {"node_id": "node-test", "token": "local-token"}
    agent.state.save_credentials(agent.credentials)
    return agent


def test_slow_preflight_does_not_block_heartbeat_events_cancel_or_duplicate_queue(agent_fixture, monkeypatch):
    command, marker, release = blocked_probe(agent_fixture)
    payload = agent_fixture["payload"]
    cancel = {"id": "cancel-now", "type": "CANCEL", "payload": {
        "attempt_id": payload["attempt_id"], "execution_token": payload["execution_token"]}}
    client = CommandClient([command])
    agent = connected_agent(agent_fixture, client)
    agent.state.create_attempt(payload, str(Path(agent_fixture["config"]["output_root"]) / "attempt"))
    agent.state.emit(payload["attempt_id"], "LOG", {"text": "must flush during preflight"})
    monkeypatch.setattr("vla_agent.main.gpu_inventory", lambda: ([], {"available": False}))
    monkeypatch.setattr("vla_agent.main.system_metrics", lambda _: {})
    monkeypatch.setattr(agent.runtime_monitor, "snapshot", lambda profiles: [])
    try:
        agent.poll_commands()
        wait_until(marker.exists)
        worker_pid = agent._preflight_worker.pid
        # A large burst creates no local backlog or additional worker processes.
        client.commands.extend([{**command, "id": f"preview-queued-{i}"} for i in range(100)])
        client.commands.append(cancel)
        start = time.monotonic()
        agent.heartbeat()
        agent.flush_events()
        for _ in range(3):
            agent.poll_commands()
        assert time.monotonic() - start < 1
        assert len(client.heartbeats) == 1 and len(client.events) == 1
        assert agent.state.attempt(payload["attempt_id"])["cancel_requested"] == 1
        assert "cancel-now" in client.acks and "preview-one" not in client.acks
        assert agent._preflight_worker.pid == worker_pid
        assert len(marker.read_text().splitlines()) == 1
        # Retain only the first preview so releasing it doesn't dispatch queued
        # fixtures unrelated to the concurrency assertion above.
        client.commands = [command, cancel]
        release.touch()

        def acknowledged():
            agent.poll_commands()
            return client.acks.get("preview-one")

        result = wait_until(acknowledged)
        assert result["status"] == "APPLIED"
        assert agent.state.command("preview-one")["status"] == "APPLIED"
        assert len(marker.read_text().splitlines()) == 1
    finally:
        release.touch()
        agent.close()


def test_completed_preflight_ack_retries_after_restart_without_rerunning(agent_fixture):
    command, marker, release = blocked_probe(agent_fixture)
    release.touch()
    client = CommandClient([command])
    client.fail_preflight_ack = True
    agent = connected_agent(agent_fixture, client)
    restarted = None
    try:
        agent.poll_commands()
        result = wait_until(lambda: agent.state.command(command["id"]))
        assert result["status"] == "APPLIED"
        with pytest.raises(OSError, match="ACK network"):
            agent.poll_commands()
        agent.close()
        count = len(marker.read_text().splitlines())
        restarted = Agent(agent_fixture["config"], client=client)
        client.fail_preflight_ack = False
        restarted.poll_commands()
        assert client.acks[command["id"]]["result"] == result["result"]
        assert restarted._preflight_worker is None
        assert len(marker.read_text().splitlines()) == count == 1
    finally:
        agent.close()
        if restarted:
            restarted.close()


def test_shutdown_kills_active_probe_and_restart_retries_unacknowledged_command(agent_fixture):
    command, marker, release = blocked_probe(agent_fixture)
    client = CommandClient([command])
    agent = connected_agent(agent_fixture, client)
    restarted = None
    try:
        agent.poll_commands()
        wait_until(marker.exists)
        child_pid = int(marker.read_text().splitlines()[0])
        agent.close()
        assert agent.state.command(command["id"]) is None
        assert command["id"] not in client.acks
        wait_until(lambda: not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
        release.touch()
        restarted = Agent(agent_fixture["config"], client=client)

        def acknowledged():
            restarted.poll_commands()
            return client.acks.get(command["id"])

        assert wait_until(acknowledged)["status"] == "APPLIED"
        assert len(marker.read_text().splitlines()) == 2
    finally:
        release.touch()
        agent.close()
        if restarted:
            restarted.close()
