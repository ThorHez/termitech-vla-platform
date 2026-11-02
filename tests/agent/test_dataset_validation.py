import json
import os
from pathlib import Path

import pytest

from vla_agent import dataset_validation as validation
from vla_agent.main import Agent, _execute_preflight_command
from vla_agent.safety import AgentError
from vla_agent.state import State


@pytest.fixture
def source_request(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    path = agent_fixture["root"] / "dataset"
    path.mkdir()
    payload = {"asset_id": "dataset-one", "path": str(path), "profile_id": "smoke-profile",
               "registered_profile": {"fps": 20}, "expires_at": "2099-01-01T00:00:00+00:00"}
    monkeypatch.setattr(validation, "validate_runtime", lambda *_: None)
    return config, payload


def test_registration_validation_uses_cpu_and_agent_owned_cache(source_request, monkeypatch):
    config, payload = source_request
    observed = {}
    monkeypatch.setenv("VLA_AGENT_TOKEN", "private-node-token")
    before = dict(os.environ)

    def probe(argv, cwd, environment, *, timeout):
        observed.update(argv=argv, cwd=cwd, environment=environment, timeout=timeout)
        response = {"schema_version": 1, "scope": "dataset_source", "path": payload["path"],
                    "profile_id": payload["profile_id"], "checks": [{"name": name, "ok": True} for name in
                        ("source_metadata", "dataset_timestamps", "raw_samples", "dataset_cache")]}
        return 0, validation.dataset_source_validation.PREFIX + json.dumps(response), ""

    monkeypatch.setattr(validation, "_probe_process", probe)
    result = validation.validate_dataset(config, {**payload, "validation_cache_dir": "/tmp/untrusted"})
    assert result["scope"] == "dataset_source" and os.environ == before
    spec = json.loads(observed["argv"][-1])
    assert spec["validation_cache_dir"] == str(Path(config["state_dir"]) / "dataset-validation-cache")
    assert spec["inputs"]["dataset"]["dataset_profile"] == payload["registered_profile"]
    assert "VLA_AGENT_TOKEN" not in observed["environment"]
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["environment"]["JAX_PLATFORMS"] == "cpu"
    assert observed["environment"]["HF_HUB_OFFLINE"] == "1"
    assert observed["timeout"] == 300


def test_active_training_defers_registration_validation_before_reading_data(source_request, agent_fixture, monkeypatch):
    config, payload = source_request
    State(config["state_dir"]).create_attempt(agent_fixture["payload"], config["output_root"])
    monkeypatch.setattr(validation, "_probe_process", lambda *_args, **_kwargs: pytest.fail("training interrupted"))
    with pytest.raises(AgentError) as error:
        validation.validate_dataset(config, payload)
    assert error.value.code == "DATASET_VALIDATION_BUSY"


@pytest.mark.parametrize("change", [{"expires_at": "2000-01-01T00:00:00+00:00"}, {"profile_id": "missing"}, {"registered_profile": []}])
def test_invalid_source_request_never_starts_probe(source_request, monkeypatch, change):
    config, payload = source_request
    monkeypatch.setattr(validation, "_probe_process", lambda *_args, **_kwargs: pytest.fail("invalid probe"))
    with pytest.raises(AgentError):
        validation.validate_dataset(config, {**payload, **change})


def test_mismatched_source_evidence_cannot_pass(source_request, monkeypatch):
    config, payload = source_request
    response = {"schema_version": 1, "scope": "dataset_source", "path": "/another-dataset",
                "profile_id": payload["profile_id"], "checks": [{"ok": True}]}
    monkeypatch.setattr(validation, "_probe_process", lambda *_a, **_k: (0, validation.dataset_source_validation.PREFIX + json.dumps(response), ""))
    with pytest.raises(AgentError) as error:
        validation.validate_dataset(config, payload)
    assert error.value.code == "DATASET_VALIDATION_INVALID_RESPONSE"


def test_partial_source_checks_cannot_become_a_success(source_request, monkeypatch):
    config, payload = source_request
    response = {"schema_version": 1, "scope": "dataset_source", "path": payload["path"],
                "profile_id": payload["profile_id"], "checks": [{"name": "raw_samples", "ok": True}]}
    monkeypatch.setattr(validation, "_probe_process", lambda *_a, **_k: (0, validation.dataset_source_validation.PREFIX + json.dumps(response), ""))
    with pytest.raises(AgentError) as error:
        validation.validate_dataset(config, payload)
    assert error.value.code == "DATASET_VALIDATION_INVALID_RESPONSE"


def test_busy_registration_worker_leaves_command_pending_for_retry(source_request, monkeypatch):
    config, payload = source_request
    monkeypatch.setattr("vla_agent.main.signal.signal", lambda *_: None)
    monkeypatch.setattr("vla_agent.main.os.setsid", lambda: None)

    def busy(*_):
        raise AgentError("DATASET_VALIDATION_BUSY", "training active")

    monkeypatch.setattr(validation, "validate_dataset", busy)
    command = {"id": "deferred-validation", "type": "DATASET_VALIDATE", "payload": payload}
    _execute_preflight_command(config, command)
    assert State(config["state_dir"]).command(command["id"]) is None


def test_agent_does_not_start_registration_worker_while_training(source_request, agent_fixture, monkeypatch):
    config, payload = source_request
    command = {"id": "queued-validation", "type": "DATASET_VALIDATE", "payload": payload}

    class Client:
        def request(self, method, path, token, body=None):
            assert method == "GET" and path.startswith("/agent/v1/commands?")
            return [command]

    agent = Agent(config, client=Client())
    agent.credentials = {"node_id": "node", "token": "token"}
    agent.state.create_attempt(agent_fixture["payload"], config["output_root"])
    monkeypatch.setattr("vla_agent.main.multiprocessing.get_context", lambda *_: pytest.fail("active training has priority"))
    try:
        agent.poll_commands()
        assert agent._preflight_worker is None and agent.state.command(command["id"]) is None
    finally:
        agent.close()
