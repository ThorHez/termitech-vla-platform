import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vla_agent import output_isolation, runner
from vla_agent.main import Agent
from vla_agent.safety import identity_matches
from vla_agent.state import State, TERMINAL_STATES
from vla_agent.snapshots import capture_code
from vla_protocol import canonical_digest

from tests.agent.conftest import bind_config_confirmation


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


def test_runner_rejects_changed_frozen_recipe_before_launch(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"], recipe="smoke")
    payload["recipe"] = "another_recipe"
    agent = Agent(config)
    agent.handle_command({"id": "start-recipe-tampered", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "FAILED" and final["child_identity"] is None
    assert agent.state.pending_events()[-1]["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    for process in agent.runners.values():
        process.wait(timeout=3)


def test_runner_rejects_changed_requested_entrypoint_before_launch(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    (agent_fixture["repo"] / "other_train.py").write_text("pass\n")
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"], entrypoint="smoke_training.py")
    payload["entrypoint"] = "other_train.py"
    agent = Agent(config)
    agent.handle_command({"id": "start-entrypoint-tampered", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "FAILED" and final["child_identity"] is None
    assert agent.state.pending_events()[-1]["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    for process in agent.runners.values():
        process.wait(timeout=3)


def test_runner_frozen_recipe_still_detects_registered_default_drift(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = config["profiles"][0]
    profile["recipe"] = "smoke"
    payload["recipe"] = "smoke"
    payload["code_expectation"] = capture_code(profile, config["roots"], recipe="smoke")
    saved = json.loads(Path(config["config_path"]).read_text())
    saved["profiles"][0]["recipe"] = "changed_after_preflight"
    Path(config["config_path"]).write_text(json.dumps(saved))
    agent = Agent(config)
    agent.handle_command({"id": "start-default-drift", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "FAILED" and final["child_identity"] is None
    assert agent.state.pending_events()[-1]["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    for process in agent.runners.values():
        process.wait(timeout=3)


def test_runner_records_frozen_recipe_in_start_and_end_evidence(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["recipe"] = "smoke"
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"], recipe="smoke")
    agent = Agent(config)
    agent.handle_command({"id": "start-frozen-recipe", "type": "START", "payload": payload})
    final = wait_state(agent.state, payload["attempt_id"], lambda value: value["state"] in TERMINAL_STATES)
    assert final["state"] == "SUCCEEDED"
    started = next(event for event in agent.state.pending_events(1000) if event["type"] == "STARTED")
    assert started["data"]["code_snapshot"]["recipe"] == "smoke"
    evidence = Path(final["output_dir"]) / "provenance"
    for path in (evidence / "code-snapshot.json", evidence / "end" / "code-snapshot.json"):
        assert json.loads(path.read_text())["recipe"] == "smoke"
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


@pytest.mark.parametrize("exit_code,discovery_error", [(0, False), (7, False), (0, True), (7, True)])
def test_runner_flushes_committed_artifact_before_terminal_without_masking_process_exit(
        agent_fixture, monkeypatch, caplog, exit_code, discovery_error):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {}}
    automatic["plan_digest"] = canonical_digest(automatic["plan"])
    baseline = payload["code_expectation"]
    payload.update(template_id="openpi-sft", recipe="fixture_recipe",
                   code_expectation={**baseline, "automatic_preflight": automatic})
    bind_config_confirmation(payload, automatic)
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(output_isolation, "git", lambda *_: str(agent_fixture["repo"]).encode())

    def capture(*args, **kwargs):
        if len(args) > 2:
            args[2].mkdir(parents=True, exist_ok=True)
        return dict(baseline)

    monkeypatch.setattr(runner, "capture_code", capture)
    monkeypatch.setattr(runner, "prepare_launch", lambda launch, *args, **kwargs: launch)
    monkeypatch.setattr(runner, "validate_training", lambda *args, **kwargs: automatic)
    # Exercise an actual child exit and actual final-pass checkpoint discovery;
    # only CPU preflight/provenance are substituted for this small local fixture.
    code = (
        "import json,sys;from pathlib import Path;"
        f"output=Path({str(output)!r});saved=output/'checkpoints'/'fixture_recipe'/'platform'/'3000';"
        "(saved/'params'/'d').mkdir(parents=True);"
        "(saved/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':1792136926465153081}));"
        "(saved/'params'/'_METADATA').write_text('{}');"
        "(saved/'params'/'manifest.ocdbt').write_bytes(b'manifest');"
        "(saved/'params'/'d'/'shard').write_bytes(b'tensor');"
        "(output/'artifacts.json').write_text(json.dumps([{'kind':'MODEL','name':'fixture_recipe / 3000',"
        "'version':'3000','path':str(saved)}]));"
        f"sys.exit({exit_code})"
    )
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: {
        "cwd": str(agent_fixture["repo"]), "argv": [sys.executable, "-c", code]})
    if discovery_error:
        def failed_discovery(*args, **kwargs):
            raise OSError("temporarily unreadable checkpoint mount")
        monkeypatch.setattr(runner.ArtifactMonitor, "report", failed_discovery)

    assert runner.execute(config, payload["attempt_id"]) == 0
    events = state.pending_events(1000)
    terminal = events[-1]
    assert terminal["type"] == ("SUCCEEDED" if exit_code == 0 else "FAILED")
    assert terminal["data"]["exit_code"] == exit_code
    if exit_code:
        assert terminal["data"]["error"]["code"] == "PROCESS_EXITED"
    artifacts = [event for event in events if event["type"] == "ARTIFACT"]
    if discovery_error:
        assert artifacts == []
        assert "preserving process exit result" in caplog.text
    else:
        assert len(artifacts) == 1
        assert artifacts[0]["seq"] < terminal["seq"]
        assert artifacts[0]["data"]["outputs"][0]["version"] == "3000"
        assert artifacts[0]["data"]["removed_paths"] == []
