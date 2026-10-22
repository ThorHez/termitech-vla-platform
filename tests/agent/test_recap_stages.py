"""Phase choices stay frozen from local preflight through optimizer launch."""

import copy
import json
from pathlib import Path

import pytest

from vla_agent import preflight, runner, training_validation
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code, compare_code
from vla_agent.state import State

from tests.agent.test_automatic_preflight import probe_fixture  # noqa: F401
from tests.agent.test_multi_dataset_agent import explicit_probe


@pytest.mark.parametrize("changed_stage,changed_config", [("offline_rl", {}), ("value", {"val_ratio": .2})])
def test_phase_and_configuration_are_frozen_in_digest_and_evidence(agent_fixture, changed_stage, changed_config):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    evidence_dir = agent_fixture["root"] / "phase-evidence"
    frozen = capture_code(profile, config["roots"], evidence_dir, training_stage="value", stage_config={"val_ratio": .1})
    assert frozen["training_stage"] == "value" and frozen["stage_config"] == {"val_ratio": .1}
    assert json.loads((evidence_dir / "code-snapshot.json").read_text())["stage_config"] == {"val_ratio": .1}
    current = capture_code(profile, config["roots"], training_stage=changed_stage, stage_config=changed_config)
    assert frozen["runtime_digest"] != current["runtime_digest"]
    with pytest.raises(AgentError, match="training_stage|stage_config"):
        compare_code(frozen, current)


def test_preflight_forwards_phase_to_every_snapshot_and_launch(agent_fixture, monkeypatch):
    profile = agent_fixture["config"]["profiles"][0]
    seen = []
    payload = {**agent_fixture["payload"], "template_id": "recap", "training_stage": "value", "stage_config": {"val_ratio": .1}}
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda *_: (profile, {}))

    def capture(*args, **kwargs):
        seen.append(("capture", kwargs["training_stage"], kwargs["stage_config"]))
        return {**payload["code_expectation"], "training_stage": "value", "stage_config": {"val_ratio": .1}}

    def launch(*args, **kwargs):
        seen.append(("launch", kwargs["training_stage"], kwargs["stage_config"]))
        return {}

    monkeypatch.setattr(preflight, "capture_code", capture)
    monkeypatch.setattr(preflight, "build_launch", launch)
    assert "error" not in preflight.run_preflight(agent_fixture["config"], payload)
    assert seen == [("capture", "value", {"val_ratio": .1}), ("launch", "value", {"val_ratio": .1})]


def staged_probe(fixture):
    launch, plan = explicit_probe(fixture)
    argv = list(launch["argv"])
    index = argv.index("--spec-json")
    spec = json.loads(argv[index + 1])
    spec.update(training_stage="offline_rl", stage_config={"acp_dropout": .1})
    argv[index + 1] = json.dumps(spec)
    plan.update(training_stage=spec["training_stage"], stage_config=spec["stage_config"], recap_stage_contract={
        "stage": "offline_rl", "model_type": "Pi0ACPConfig", "loader_type": "LeRobotACPDataset",
        "datasets": [{"path": data["path"], "required_columns": ["is_positive"]} for data in plan["datasets"]]})
    return {**launch, "argv": argv}, plan


def write_response(fixture, plan, *, with_check=True):
    checks = ["recipe_configuration", "data_contract", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache"]
    if with_check:
        checks.append("recap_stage")
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
                "checks": [{"name": name, "ok": True, "message": "checked"} for name in checks]}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")


def test_phase_probe_requires_real_model_loader_and_each_derived_dataset_evidence(probe_fixture):
    launch, plan = staged_probe(probe_fixture)
    write_response(probe_fixture, plan)
    evidence = training_validation.validate_training(probe_fixture["config"], launch, [])
    assert evidence["plan"]["training_stage"] == "offline_rl"


@pytest.mark.parametrize("mutation", ["old_wrapper", "different_stage", "different_config", "missing_check", "missing_contract", "wrong_dataset", "no_columns", "wrong_model", "wrong_supervision"])
def test_phase_probe_rejects_stale_or_incomplete_runtime_evidence(probe_fixture, mutation):
    launch, plan = staged_probe(probe_fixture)
    plan = copy.deepcopy(plan)
    if mutation == "old_wrapper":
        plan.pop("training_stage")
        plan.pop("stage_config")
    elif mutation == "different_stage":
        plan["training_stage"] = "value"
    elif mutation == "different_config":
        plan["stage_config"] = {"acp_dropout": .5}
    elif mutation == "missing_contract":
        plan.pop("recap_stage_contract")
    elif mutation == "wrong_dataset":
        plan["recap_stage_contract"]["datasets"][-1]["path"] = "/unbound/derived-data"
    elif mutation == "wrong_model":
        plan["recap_stage_contract"]["model_type"] = "Pi0ValueConfig"
    elif mutation == "wrong_supervision":
        plan["recap_stage_contract"]["datasets"][-1]["required_columns"] = ["value_target"]
    elif mutation == "no_columns":
        plan["recap_stage_contract"]["datasets"][-1]["required_columns"] = []
    write_response(probe_fixture, plan, with_check=mutation != "missing_check")
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(probe_fixture["config"], launch, [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"


def test_runner_rejects_phase_change_before_building_or_spawning_training(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    payload = {**agent_fixture["payload"], "template_id": "recap", "training_stage": "offline_rl", "stage_config": {}}
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"], training_stage="value", stage_config={})
    state = State(config["state_dir"])
    state.create_attempt(payload, str(Path(config["output_root"]) / payload["attempt_id"]))
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: pytest.fail("changed phase reached launch builder"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    final = state.pending_events()[-1]
    assert final["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    assert final["data"]["process_started"] is False


@pytest.mark.parametrize("template,stage,config", [("recap", None, None), ("recap", "value", {"acp_dropout": .1}),
                                                   ("openpi-sft", "value", {}), ("smoke", None, {})])
def test_local_input_validation_rejects_missing_or_incompatible_phase_configuration(agent_fixture, template, stage, config):
    with pytest.raises(AgentError) as failure:
        preflight.validate_local_inputs(agent_fixture["config"], {**agent_fixture["payload"], "template_id": template,
                                                               "training_stage": stage, "stage_config": config})
    assert failure.value.code == "TRAINING_STAGE_INVALID"
