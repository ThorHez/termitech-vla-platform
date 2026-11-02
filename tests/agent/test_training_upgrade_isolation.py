"""Real CPU subprocess coverage for preserving training across Agent releases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import venv

import psutil
import vla_adapters
import vla_agent
import vla_protocol

from vla_agent.main import Agent
from vla_agent.runner_runtime import snapshot_runner_config, upgrade_status
from vla_agent.safety import identity_matches
from vla_agent.state import State, TERMINAL_STATES

from tests.agent.conftest import bind_config_confirmation


def runtime(prefix: Path, name: str) -> dict:
    """Install only local package files, without editable paths or downloads."""
    directory = prefix / "runtimes" / name
    venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
    python = directory / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True,
    ).strip())
    for package in (vla_agent, vla_adapters, vla_protocol, psutil):
        shutil.copytree(Path(package.__file__).parent, site / package.__name__,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    build = ("a" if name == "old" else "b") * 64
    (directory / "vla-training-runtime.json").write_text(json.dumps({
        "schema_version": 1, "runtime_path": str(directory), "build_id": build,
    }))
    return {"protocol_version": 1, "runtime_path": str(directory), "build_id": build}


def wait_until(predicate, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Training subprocess did not produce the expected durable evidence")


def stop_runner(agent: Agent, payload: dict):
    """Keep a failed assertion from leaving a detached test workload behind."""
    agent.close()
    attempt = agent.state.attempt(payload["attempt_id"])
    if attempt and attempt["state"] not in TERMINAL_STATES:
        agent.state.request_cancel(payload["attempt_id"], payload["execution_token"])
    for process in agent.runners.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def test_frozen_training_collects_while_agent_is_offline_and_new_release_can_cancel_and_backfill(
        agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["parameters"]["steps"] = 300
    prefix = agent_fixture["root"].resolve() / "agent-install"
    old, new = runtime(prefix, "old"), runtime(prefix, "new")
    alias = prefix / ".venv"
    alias.symlink_to(old["runtime_path"], target_is_directory=True)
    monkeypatch.setattr("vla_agent.main.installed_runtime", lambda: old)
    # A detached runner must ignore inherited editable paths and current cwd.
    poison = agent_fixture["root"] / "mutable-imports" / "vla_agent"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text("raise RuntimeError('mutable PYTHONPATH was imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(poison.parent))
    agent = Agent(config)
    agent.state.save_credentials({"node_id": "isolation-node", "token": "isolation-token"})
    try:
        result = agent.handle_command({"id": "start-isolated", "type": "START", "payload": payload})
        assert result["status"] == "APPLIED", result
        wait_until(lambda: len([event for event in agent.state.pending_events(1000)
                                if event["type"] == "METRIC"]) >= 3)
        running = agent.state.attempt(payload["attempt_id"])
        assert running["state"] == "RUNNING"
        assert upgrade_status(config["state_dir"])["upgrade_safe"]
        snapshot = Path(running["runner_runtime"]["config_path"])
        frozen = snapshot.read_bytes()
        log = Path(running["output_dir"]) / "train.log"
        old_log_size = log.stat().st_size
        first_metrics = len([event for event in agent.state.pending_events(1000)
                             if event["type"] == "METRIC"])
        first_resources = len([event for event in agent.state.pending_events(1000)
                               if event["type"] == "RESOURCE"])
        agent.close()

        # Model the installer's atomic release switch and current-config update.
        staged_alias = prefix / ".venv.next"
        staged_alias.symlink_to(new["runtime_path"], target_is_directory=True)
        os.replace(staged_alias, alias)
        replacement = json.loads(Path(config["config_path"]).read_text())
        replacement["name"] = "upgraded-feature-agent"
        replacement["output_root"] = str(agent_fixture["root"] / "new-outputs")
        replacement["profiles"][0]["code_dir"] = str(agent_fixture["root"] / "future-code")
        Path(config["config_path"]).write_text(json.dumps(replacement))

        wait_until(lambda: len([event for event in agent.state.pending_events(1000)
                                if event["type"] == "METRIC"]) >= first_metrics + 4)
        wait_until(lambda: len([event for event in agent.state.pending_events(1000)
                                if event["type"] == "RESOURCE"]) > first_resources)
        continued = agent.state.attempt(payload["attempt_id"])
        assert continued["runner_identity"] == running["runner_identity"]
        assert continued["child_identity"] == running["child_identity"]
        assert identity_matches(continued["runner_identity"])
        assert identity_matches(continued["child_identity"])
        assert snapshot.read_bytes() == frozen
        assert log.stat().st_size > old_log_size
        assert upgrade_status(config["state_dir"])["upgrade_safe"]

        # A real Agent imported from the new release controls the original
        # child through shared durable state and sends the old runner's outbox.
        receipt = agent_fixture["root"] / "delivered-events.jsonl"
        script = """
import json,sys,time
from vla_agent.config import load_config
from vla_agent.main import Agent
from vla_agent.runner_runtime import installed_runtime
from vla_agent.state import TERMINAL_STATES
class Sink:
    def request(self, method, path, token, body=None):
        assert path == '/agent/v1/events/batch'
        with open(sys.argv[4], 'a') as stream:
            stream.write(json.dumps(body['events']) + '\\n')
        return {'accepted': len(body['events'])}
agent = Agent(load_config(sys.argv[1]), Sink())
result = agent.handle_command({'id':'cancel-from-new-release','type':'CANCEL','payload':{
    'attempt_id':sys.argv[2], 'execution_token':sys.argv[3]}})
deadline = time.monotonic() + 10
while agent.state.attempt(sys.argv[2])['state'] not in TERMINAL_STATES:
    if time.monotonic() >= deadline:
        raise RuntimeError('original training did not stop')
    time.sleep(0.05)
while agent.state.pending_events():
    agent.flush_events()
print(json.dumps({'runtime':installed_runtime(), 'result':result}))
agent.close()
"""
        upgraded = subprocess.run(
            [str(alias / "bin/python"), "-I", "-c", script, config["config_path"],
             payload["attempt_id"], payload["execution_token"], str(receipt)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        proof = json.loads(upgraded.stdout)
        assert proof["runtime"] == new
        assert proof["result"]["status"] == "APPLIED"
        finished = agent.state.attempt(payload["attempt_id"])
        assert finished["state"] == "CANCELLED"
        assert finished["child_identity"] == running["child_identity"]
        assert not identity_matches(finished["child_identity"])
        events = [event for line in receipt.read_text().splitlines() for event in json.loads(line)]
        assert sum(event["type"] == "STARTED" for event in events) == 1
        assert sum(event["type"] == "METRIC" for event in events) >= first_metrics + 4
        assert events[-1]["type"] == "CANCELLED"
        assert agent.state.pending_events() == []
        assert (Path(old["runtime_path"]) / "bin/python").is_file()
        assert snapshot.read_bytes() == frozen
    finally:
        stop_runner(agent, payload)


def test_new_agent_recovers_durable_start_before_runner_spawn_from_original_runtime(
        agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    prefix = agent_fixture["root"].resolve() / "agent-install"
    old = runtime(prefix, "old")
    metadata = snapshot_runner_config(config, payload["attempt_id"], old)
    state = State(config["state_dir"])
    output = str(Path(config["output_root"]) / payload["attempt_id"])
    state.create_attempt(payload, output, runner_runtime=metadata)
    # Simulate a feature Agent restart after durable registration but before
    # Popen: there is no process or output evidence and recovery is permitted.
    monkeypatch.setattr("vla_agent.main.installed_runtime", lambda: None)
    agent = Agent(config)
    try:
        agent._recover_runners()
        wait_until(lambda: state.attempt(payload["attempt_id"])["state"] in TERMINAL_STATES)
        finished = state.attempt(payload["attempt_id"])
        assert finished["state"] == "SUCCEEDED"
        assert finished["runner_runtime"] == metadata
        agent._recover_runners()
        events = state.pending_events(1000)
        assert sum(event["type"] == "STARTED" for event in events) == 1
        assert sum(event["type"] == "METRIC" for event in events) == payload["parameters"]["steps"]
    finally:
        stop_runner(agent, payload)


def test_runner_publishes_periodic_checkpoint_with_feature_agent_closed(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    prefix = agent_fixture["root"].resolve() / "agent-install"
    old = runtime(prefix, "old")
    output = Path(config["output_root"]) / payload["attempt_id"]
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {}}
    payload.update(template_id="openpi-sft", recipe="fixture_recipe")
    bind_config_confirmation(payload, automatic)
    metadata = snapshot_runner_config(config, payload["attempt_id"], old)
    state = State(config["state_dir"])
    state.create_attempt(payload, str(output), runner_runtime=metadata)
    trigger = agent_fixture["root"] / "save-checkpoint"
    child = """
import json,sys,time
from pathlib import Path
output,trigger = map(Path,sys.argv[1:])
while not trigger.exists():
    time.sleep(0.05)
checkpoint = output/'checkpoints'/'fixture_recipe'/'platform'/'3000'
(checkpoint/'params'/'d').mkdir(parents=True)
(checkpoint/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':1792136926465153081}))
(checkpoint/'params'/'_METADATA').write_text('{}')
(checkpoint/'params'/'manifest.ocdbt').write_bytes(b'manifest')
(checkpoint/'params'/'d'/'shard').write_bytes(b'tensor')
print('checkpoint committed while feature Agent is closed',flush=True)
while True:
    time.sleep(0.05)
"""
    # Only ML-specific validation and the launch plan are substituted. The
    # training process, log pipe, disk scan, local outbox and cancellation are real.
    script = """
import json,sys
from functools import partial
from pathlib import Path
from vla_agent import runner
from vla_agent.config import load_config
from vla_agent.state import State
config = load_config(sys.argv[1])
payload = State(config['state_dir']).attempt(sys.argv[2])['payload']
profile = {**config['profiles'][0], 'adapter':'openpi-sft'}
baseline = payload['code_expectation']
runner.validate_local_inputs = lambda *_: (profile,{})
def capture(*args, **kwargs):
    if len(args) > 2:
        args[2].mkdir(parents=True,exist_ok=True)
    return dict(baseline)
runner.capture_code = capture
runner.prepare_launch = lambda launch,*args,**kwargs: launch
runner.validate_training = lambda *args,**kwargs: baseline['automatic_preflight']
runner.ArtifactMonitor = partial(runner.ArtifactMonitor, interval=0.1)
runner.build_launch = lambda *args,**kwargs: {
    'cwd':profile['code_dir'], 'argv':[profile['python'],'-c',sys.argv[3],
        str(Path(config['output_root'])/sys.argv[2]),sys.argv[4]]}
raise SystemExit(runner.execute(config,sys.argv[2]))
"""
    feature_agent = Agent(config)
    log = agent_fixture["root"] / "isolated-checkpoint-runner.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [str(Path(old["runtime_path"]) / "bin/python"), "-I", "-c", script,
             metadata["config_path"], payload["attempt_id"], child, str(trigger)],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_until(lambda: state.attempt(payload["attempt_id"])["state"] == "RUNNING")
        feature_agent.close()
        trigger.touch()
        events = wait_until(lambda: [event for event in state.pending_events(1000)
                                    if event["type"] == "ARTIFACT"])
        assert state.attempt(payload["attempt_id"])["state"] == "RUNNING"
        assert process.poll() is None
        assert events[0]["data"]["outputs"][0]["version"] == "3000"
        assert events[0]["data"]["outputs"][0]["metadata"]["format"] == "openpi-orbax"
        # The checkpoint was published before final exit, with no Agent scan.
        assert not any(event["type"] in TERMINAL_STATES for event in state.pending_events(1000))
        state.request_cancel(payload["attempt_id"], payload["execution_token"])
        process.wait(timeout=5)
        assert process.returncode == 0, log.read_text()
        assert state.attempt(payload["attempt_id"])["state"] == "CANCELLED"
    finally:
        feature_agent.close()
        if process.poll() is None:
            state.request_cancel(payload["attempt_id"], payload["execution_token"])
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
