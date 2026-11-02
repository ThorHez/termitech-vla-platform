"""The Agent preserves and verifies an explicit training normalization choice."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from vla_agent import preflight, runner, snapshots, training_validation
from vla_agent.safety import AgentError
from vla_agent.state import State
from vla_protocol import canonical_digest

from tests.agent.conftest import bind_config_confirmation
from tests.agent.test_automatic_preflight import probe_fixture  # noqa: F401


def test_capture_freezes_method_and_legacy_none_keeps_runtime_identity(agent_fixture, monkeypatch):
    profile = {**agent_fixture["config"]["profiles"][0], "adapter": "openpi-sft"}
    original = snapshots.subprocess.run
    def local_probe(argv, *args, **kwargs):
        if argv[0] == profile["python"] and "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"origin": str(agent_fixture["repo"] / "smoke_training.py")}), "")
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(snapshots.subprocess, "run", local_probe)
    roots = agent_fixture["config"]["roots"]
    legacy = snapshots.capture_code(profile, roots)
    same = snapshots.capture_code(profile, roots, normalization_method=None)
    explicit = snapshots.capture_code(profile, roots, normalization_method="minmax")
    assert "normalization_method" not in legacy and "normalization_method" not in same
    assert legacy["runtime_digest"] == same["runtime_digest"]
    assert explicit["normalization_method"] == "minmax"
    assert explicit["runtime_digest"] != legacy["runtime_digest"]
    with pytest.raises(AgentError, match="normalization_method"):
        snapshots.compare_code(legacy, explicit)


def test_preflight_forwards_method_to_both_code_captures_and_launch(agent_fixture, monkeypatch):
    profile = {**agent_fixture["config"]["profiles"][0], "adapter": "openpi-sft"}
    payload = {**agent_fixture["payload"], "template_id": "openpi-sft", "normalization_method": "quantile", "gpu_count": 1}
    baseline = {**payload["code_expectation"], "normalization_method": "quantile"}
    captured, launched = [], []
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda *_: (profile, {}))
    def capture(*args, **kwargs):
        captured.append(kwargs["normalization_method"])
        return dict(baseline)
    def launch(*args, **kwargs):
        launched.append(kwargs["normalization_method"])
        return {}
    monkeypatch.setattr(preflight, "capture_code", capture)
    monkeypatch.setattr(preflight, "build_launch", launch)
    monkeypatch.setattr(preflight, "prepare_launch", lambda *args: {})
    monkeypatch.setattr(preflight, "validate_training", lambda *args: {"plan": {"normalization_method": "quantile"}})
    result = preflight.run_preflight(agent_fixture["config"], payload)
    assert "error" not in result, result
    assert captured == ["quantile", "quantile"] and launched == ["quantile"]
    assert result["code_snapshot"]["normalization_method"] == "quantile"


@pytest.mark.parametrize("reported", [None, "quantile", "minmax"])
def test_cpu_wrapper_must_confirm_the_selected_method(probe_fixture, reported):
    launch = copy.deepcopy(probe_fixture["launch"])
    index = launch["argv"].index("--spec-json") + 1
    spec = json.loads(launch["argv"][index])
    spec["normalization_method"] = "minmax"
    launch["argv"][index] = json.dumps(spec)
    plan = copy.deepcopy(probe_fixture["plan"])
    if reported is not None:
        plan["normalization_method"] = reported
        source = "FROZEN_REQUEST = " + repr({"normalization_method": reported}) + "\n"
        plan["generated_config"].update(source=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
                "checks": [{"name": name, "ok": True, "message": "verified"} for name in (
                    "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
                    "model_checkpoint", "training_entrypoint", "training_cache")]}
    probe_fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")
    if reported == "minmax":
        assert training_validation._wrapper_validation(probe_fixture["config"], launch, [])["normalization_method"] == "minmax"
    else:
        with pytest.raises(AgentError, match="normalization_method") as failure:
            training_validation._wrapper_validation(probe_fixture["config"], launch, [])
        assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"


def test_runner_passes_frozen_method_to_capture_and_launch_without_starting_training(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload.update(template_id="openpi-sft", normalization_method="zscore")
    payload["code_expectation"]["normalization_method"] = "zscore"
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"normalization_method": "zscore"}}
    automatic["plan_digest"] = canonical_digest(automatic["plan"])
    bind_config_confirmation(payload, automatic)
    baseline = dict(payload["code_expectation"])
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(runner, "validate_output_isolation", lambda *_: None)
    captured, launched = [], []
    def capture(*args, **kwargs):
        captured.append(kwargs["normalization_method"])
        return dict(baseline)
    def launch(*args, **kwargs):
        launched.append(kwargs["normalization_method"])
        raise AgentError("NORMALIZATION_TEST_STOP", "Stop after validating launch propagation")
    monkeypatch.setattr(runner, "capture_code", capture)
    monkeypatch.setattr(runner, "build_launch", launch)
    assert runner.execute(config, payload["attempt_id"]) == 1
    assert captured == launched == ["zscore"]
    attempt = state.attempt(payload["attempt_id"])
    assert attempt["child_identity"] is None
    assert state.pending_events()[-1]["data"]["error"]["code"] == "NORMALIZATION_TEST_STOP"
