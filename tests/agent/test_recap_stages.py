"""Phase choices stay frozen from local preflight through optimizer launch."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vla_adapters.recap import (ACP_MULTI_ENTRYPOINT, RECAP_TORCH_ENTRY, RECAP_TORCH_LOADER, STAGE_ANNOTATION,
                                STAGE_PROGRESS_DATASET, VALUE_ANNOTATION, VALUE_ANNOTATION_DATASET, recap_torch_module_paths)
from vla_agent import preflight, runner, training_validation
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code, compare_code
from vla_agent.state import State

from tests.agent.test_automatic_preflight import probe_fixture  # noqa: F401
from tests.agent.test_artifacts import checkpoint as orbax_checkpoint
from tests.agent.test_multi_dataset_agent import explicit_probe


@pytest.mark.parametrize("stage", [STAGE_ANNOTATION, VALUE_ANNOTATION])
def test_annotation_runner_only_accepts_data_linked_to_frozen_model_and_receipt(tmp_path, stage):
    output = tmp_path.resolve()
    value = stage == VALUE_ANNOTATION
    dataset = output / (VALUE_ANNOTATION_DATASET if value else STAGE_PROGRESS_DATASET)
    (dataset / "meta").mkdir(parents=True)
    scorer_path = str(output.parent / "source-checkpoint/20")
    settings = {"n_step": 10, "positive_ratio": 0.5}
    receipt = {"schema_version": 1, "frame_count": 2, "positive_count": 1, **settings}
    if value:
        receipt.update(method="n_step_value", backend="pytorch", checkpoint=scorer_path, thresholds={"0": 0.3})
    else:
        settings["stage_count"] = 1
        receipt.update(method="kai0_stage_progress", score="absolute_advantage", stage_count=1,
                       stages=[{"stage": 0, "frame_count": 2, "positive_count": 1, "threshold": 0.3}])
    receipt_path = dataset / "meta" / ("value_advantage.json" if value else "stage_advantage.json")
    receipt_path.write_text(json.dumps(receipt))
    source_manifest = "a" * 64
    plan = {"datasets": [{"frames": 2}], "model": {"checkpoint_path": scorer_path, "scorer_checkpoint": {"step": 20}},
            "recap_stage_contract": {"annotation": {"source_manifest_digest": source_manifest, **settings}}}
    inputs = {key: {"id": key + "-asset"} for key in ("model", "dataset_0", "norm_0")}
    payload = {"inputs": inputs, "training_stage": stage, "code_expectation": {"automatic_preflight": {"plan": plan}}}
    metadata = {"format": "lerobot", "registration_source": "recap_value_annotation" if value else "recap_stage_annotation",
                "source_dataset_asset_id": "dataset_0-asset",
                "source_norm_asset_id": "norm_0-asset", "scorer_model_asset_id": "model-asset", "scorer_checkpoint_path": scorer_path,
                "source_manifest_digest": source_manifest, "annotation": receipt,
                "annotation_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    artifact = {"kind": "DATASET", "name": "Annotated dataset", "path": str(dataset), "version": "20", "metadata": metadata}
    manifest = output / "artifacts.json"
    manifest.write_text(json.dumps([artifact]))
    assert runner.verified_outputs(output, False, template_id="recap", stage_annotation=payload) == [artifact]
    metadata["scorer_model_asset_id"] = "different-model"
    manifest.write_text(json.dumps([artifact]))
    with pytest.raises(AgentError, match="frozen scorer"):
        runner.verified_outputs(output, False, template_id="recap", stage_annotation=payload)
    metadata["scorer_model_asset_id"] = "model-asset"
    manifest.write_text(json.dumps([artifact]))
    receipt_path.write_text("{}")
    with pytest.raises(AgentError, match="receipt"):
        runner.verified_outputs(output, False, template_id="recap", stage_annotation=payload)


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
    spec.update(template_id="recap", recap_backend="pytorch", training_stage="offline_rl", stage_config={"acp_dropout": .1})
    argv[index + 1] = json.dumps(spec)
    model = plan["inputs"]["model"]
    weight = model + "/model.safetensors"
    plan["model"] = {"path": weight, "checkpoint_path": model, "format": "openpi-pytorch", "parameter_bytes": 1024,
                     "metadata": [{"path": weight, "scope": "safetensors_header", "sha256": "a" * 64}]}
    plan["entrypoint"] = spec["code_dir"] + "/" + RECAP_TORCH_ENTRY
    plan.update(training_stage=spec["training_stage"], stage_config=spec["stage_config"], recap_stage_contract={
        "stage": "offline_rl", "model_type": "Pi0Config", "loader_type": RECAP_TORCH_LOADER, "backend": "pytorch",
        "runtime_versions": {"torch": "2.7.1", "safetensors": "0.5.3", "transformers": "4.53.2"},
        "source_sha256": {"src/openpi/" + path: "b" * 64 for path in recap_torch_module_paths("offline_rl")},
        "datasets": [{"path": data["path"], "required_columns": ["is_positive"]} for data in plan["datasets"]]})
    return {**launch, "argv": argv}, plan


def write_response(fixture, plan, *, with_check=True):
    # Synthetic wrapper attestations only; this fixture does not decode data.
    checks = ["recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples",
              "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache"]
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


@pytest.mark.parametrize("mutation", [None, "torch_backend", "estimator", "torch_entrypoint", "missing_manifest"])
def test_jax_acp_probe_checks_native_policy_and_backend_evidence(probe_fixture, mutation):
    launch, plan = staged_probe(probe_fixture)
    spec = json.loads(launch["argv"][-1])
    spec["recap_backend"] = "jax"
    launch["argv"][-1] = json.dumps(spec)
    contract = plan["recap_stage_contract"]
    contract.update(backend="jax", loader_type="CheckpointWeightLoader")
    path = plan["inputs"]["model"]
    plan["entrypoint"] = spec["code_dir"] + "/" + ACP_MULTI_ENTRYPOINT
    plan["model"].update(format="openpi-orbax", model_role="policy", path=path + "/params",
                         metadata=[{"path": path + "/params/_METADATA", "sha256": "a" * 64}])
    if mutation == "torch_backend":
        contract["backend"] = "pytorch"
    elif mutation == "estimator":
        plan["model"]["model_role"] = "value_estimator"
    elif mutation == "torch_entrypoint":
        plan["entrypoint"] = spec["code_dir"] + "/" + RECAP_TORCH_ENTRY
    elif mutation == "missing_manifest":
        plan["model"]["metadata"] = []
    write_response(probe_fixture, plan)
    if mutation:
        with pytest.raises(AgentError, match="ReCap"):
            training_validation.validate_training(probe_fixture["config"], launch, [])
    else:
        evidence = training_validation.validate_training(probe_fixture["config"], launch, [])
        assert evidence["plan"]["recap_stage_contract"]["backend"] == "jax"


def test_jax_acp_runner_requires_final_committed_orbax_policy(tmp_path):
    checkpoint = orbax_checkpoint(tmp_path, "19")
    artifact = {"kind": "MODEL", "name": "policy", "path": str(checkpoint), "version": "19",
                "metadata": {"format": "openpi-orbax", "backend": "jax", "model_role": "policy", "step": 19}}
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(json.dumps([artifact]))
    plan = {"recipe": "fixture_recipe", "parameters": {"steps": 20}}
    assert runner.verified_outputs(tmp_path, False, template_id="recap", recap_acp=plan) == [artifact]
    with pytest.raises(AgentError, match="final Orbax"):
        runner.verified_outputs(tmp_path, False, template_id="recap", recap_acp={**plan, "parameters": {"steps": 21}})
    (checkpoint / "_CHECKPOINT_METADATA").unlink()
    with pytest.raises(AgentError, match="final Orbax"):
        runner.verified_outputs(tmp_path, False, template_id="recap", recap_acp=plan)


@pytest.mark.parametrize("mutation", ["old_wrapper", "different_stage", "different_config", "missing_check", "missing_contract", "wrong_dataset", "no_columns", "wrong_model", "wrong_supervision", "jax_evidence"])
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
    elif mutation == "jax_evidence":
        plan["model"]["format"] = "openpi-orbax"
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
