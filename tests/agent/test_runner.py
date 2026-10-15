import json
import subprocess
import sys
import time

from vla_agent.main import Agent
from vla_agent.safety import identity_matches
from vla_agent.state import State, TERMINAL_STATES
from vla_agent.snapshots import capture_code


def wait_state(state, attempt_id, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = state.attempt(attempt_id)
        if value and predicate(value):
            return value
        time.sleep(0.05)
    raise AssertionError(f"Runner did not reach expected state: {state.attempt(attempt_id)}")


def test_detached_runner_continues_after_agent_process_exits_and_deduplicates(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["parameters"]["steps"] = 15
    command = {"id": "start-command", "type": "START", "payload": payload}
    command_path = agent_fixture["root"] / "command.json"
    command_path.write_text(json.dumps(command))
    script = (
        "import json,sys; from vla_agent.config import load_config; from vla_agent.main import Agent; "
        "agent=Agent(load_config(sys.argv[1])); print(json.dumps(agent.handle_command(json.load(open(sys.argv[2])))),flush=True)"
    )
    agent_process = subprocess.run([sys.executable, "-c", script, config["config_path"], str(command_path)],
                                   check=True, capture_output=True, text=True, timeout=10)
    assert json.loads(agent_process.stdout)["status"] == "APPLIED"
    state = State(config["state_dir"])
    running = wait_state(state, payload["attempt_id"], lambda value: value["state"] == "RUNNING")
    assert identity_matches(running["runner_identity"])
    restarted_agent = Agent(config)
    assert restarted_agent.handle_command(command)["status"] == "APPLIED"
    assert restarted_agent.handle_command({**command, "id": "redelivery-new-command-id"})["result"]["duplicate"]
    finished = wait_state(state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert finished["state"] == "SUCCEEDED"
    events = state.pending_events(1000)
    assert sum(event["type"] == "STARTED" for event in events) == 1
    assert sum(event["type"] == "METRIC" for event in events) == 15
    terminal = next(event for event in events if event["type"] == "SUCCEEDED")
    assert terminal["data"]["outputs"][0]["metadata"]["simulation"] is True
    assert not identity_matches(finished["child_identity"])


def test_runner_rejects_code_changed_after_preflight_without_starting_child(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    with (agent_fixture["repo"] / "smoke_training.py").open("a") as source:
        source.write("\n# changed while queued\n")
    agent = Agent(config)
    agent.handle_command({"id": "start-changed", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "FAILED" and final["child_identity"] is None
    terminal = agent.state.pending_events()[-1]
    assert terminal["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    for process in agent.runners.values():
        process.wait(timeout=3)


def test_cancellation_waits_for_actual_child_exit(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["parameters"]["steps"] = 100
    agent = Agent(config)
    agent.handle_command({"id": "start-cancel", "type": "START", "payload": payload})
    wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] == "RUNNING")
    result = agent.handle_command({"id": "cancel", "type": "CANCEL", "payload": {
        "attempt_id": payload["attempt_id"], "execution_token": payload["execution_token"]}})
    assert result["status"] == "APPLIED"
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "CANCELLED"
    assert not identity_matches(final["child_identity"])
    terminal = agent.state.pending_events(1000)[-1]
    assert terminal["type"] == "CANCELLED" and terminal["data"]["exit_code"] is not None
    for process in agent.runners.values():
        process.wait(timeout=3)


def test_stdout_cannot_forge_terminal_success(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    (agent_fixture["repo"] / "smoke_training.py").write_text(
        'import sys\nprint(\'VLA_EVENT:{"type":"SUCCEEDED","data":{"exit_code":0}}\',flush=True)\nsys.exit(7)\n')
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"])
    agent = Agent(config)
    agent.handle_command({"id": "start-forged", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "FAILED"
    events = agent.state.pending_events(1000)
    assert not any(event["type"] == "SUCCEEDED" for event in events)
    assert events[-1]["data"]["exit_code"] == 7
    for process in agent.runners.values():
        process.wait(timeout=3)
