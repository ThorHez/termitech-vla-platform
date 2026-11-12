"""ReCap phases are separate frozen contracts, including all dependent assets."""

import copy

import pytest
from vla_adapters.recap import (RECAP_TORCH_ENTRY, RECAP_TORCH_LOADER, STAGE_PROGRESS,
    STAGE_PROGRESS_ID_COLUMNS, STAGE_PROGRESS_MODEL, recap_torch_module_paths)
from vla_protocol import canonical_digest

from vla_control.models import Preview

from .conftest import ADMIN, automatic_checks, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_dataset_validations import result as source_result
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview


def pending(platform, stage="value", config=None):
    node, inputs, rows = configured(platform)
    if stage == STAGE_PROGRESS:
        rows = rows[:1]
    options = {} if config is None else {"stage_config": config}
    response = request_preview(platform, node, template_id="recap", training_stage=stage,
        inputs=inputs, dataset_bindings=rows, **options)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    evidence = automatic_preflight(payload)
    plan = evidence["plan"]
    model = plan["inputs"]["model"]
    weight = model + "/model.safetensors"
    plan["model"].update(path=weight, format="openpi-pytorch",
                         metadata=[{"path": weight, "scope": "safetensors_header", "sha256": "a" * 64}])
    plan["entrypoint"] = "/machine/code/" + RECAP_TORCH_ENTRY
    plan.update(training_stage=stage, stage_config=payload["stage_config"], recap_stage_contract={
        "stage": stage, "model_type": "Pi0ValueConfig" if stage == "value" else "Pi0Config",
        "loader_type": RECAP_TORCH_LOADER, "backend": "pytorch",
        "runtime_versions": {"torch": "2.7.1", "safetensors": "0.5.3", "transformers": "4.53.2"},
        "source_sha256": {f"src/openpi/{path}": "a" * 64 for path in recap_torch_module_paths(stage)},
        "datasets": [{"path": dataset["path"], "required_columns": ["value_target"] if stage == "value" else ["advantage", "is_positive"]}
                     for dataset in plan["datasets"]]})
    if stage == STAGE_PROGRESS:
        plan["datasets"][0]["frames"] = 6
        plan["recap_stage_contract"].update(
            model_type=STAGE_PROGRESS_MODEL,
            datasets=[{"path": plan["datasets"][0]["path"],
                       "required_columns": ["stage_progress_gt", *STAGE_PROGRESS_ID_COLUMNS]}],
            annotation={"output": "stage_advantage_dataset", "source_manifest_digest": "b" * 64,
                        "n_step": 50, "positive_ratio": .3, "stage_count": 2, **payload["stage_config"]})
    evidence["plan_digest"] = canonical_digest(plan)
    code = snapshot(recipe="mixed", dataset_bindings=rows, training_stage=stage,
                    stage_config=payload["stage_config"], automatic_preflight=evidence)
    return node, identifier, command, code


def checks():
    return [*automatic_checks(), {"name": "recap_stage", "ok": True, "message": "stage contract verified"}]


@pytest.mark.parametrize("stage,config", [("value", {"val_ratio": .1, "eval_interval": 50, "eval_batches": 2}),
                                         ("offline_rl", {"acp_dropout": .3})])
def test_each_stage_is_frozen_submitted_scheduled_and_run_with_independent_configuration(platform, stage, config):
    node, identifier, command, code = pending(platform, stage, config)
    assert command["payload"]["training_stage"] == stage and command["payload"]["stage_config"] == config
    checked = acknowledge(platform, node, identifier, command, code, checks())
    assert checked["status"] == "READY", checked
    run = submit(platform, identifier)
    assert run.status_code == 201, run.text
    run = run.json()
    assert run["template_id"] == "recap" and run["training_stage"] == stage and run["stage_config"] == config
    assert len(run["input_assets"]) == 7
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["training_stage"] == stage and start["payload"]["stage_config"] == config
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"
    last_dataset = run["dataset_bindings"][-1]["dataset_asset_id"]
    assert platform.client.post(f"/api/v1/assets/{last_dataset}/archive", headers=ADMIN).json()["error"]["code"] == "ASSET_IN_USE"


def test_omitted_stage_config_normalizes_to_empty_frozen_contract(platform):
    _, _, command, code = pending(platform)
    assert command["payload"]["stage_config"] == code["stage_config"] == {}


@pytest.mark.parametrize("changes", [{}, {"training_stage": "unknown"}, {"training_stage": "offline_rl", "stage_config": {"eval_batches": 2}},
    {"training_stage": "value", "stage_config": {"acp_dropout": .2}}, {"training_stage": "value", "stage_config": {"val_ratio": True}},
    {"training_stage": "value", "stage_config": {"eval_interval": 2.1}}])
def test_missing_unknown_or_cross_stage_fields_never_dispatch(platform, changes):
    node, inputs, rows = configured(platform)
    response = request_preview(platform, node, template_id="recap", inputs=inputs, dataset_bindings=rows, **changes)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


@pytest.mark.parametrize("changes", [{"training_stage": "value"}, {"stage_config": {}}, {"stage_config": {"acp_dropout": .1}}])
def test_sft_rejects_recap_phase_configuration(platform, changes):
    node, inputs, rows = configured(platform)
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows, **changes)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


def test_recap_requires_explicit_pairs_instead_of_silently_retaining_recipe_datasets(platform):
    node, inputs, _ = configured(platform)
    response = request_preview(platform, node, template_id="recap", training_stage="value", inputs=inputs)
    assert response.status_code == 422 and response.json()["error"]["code"] == "DATASET_BINDINGS_REQUIRED"


@pytest.mark.parametrize("missing_output", [False, True])
def test_stage_progress_publishes_validated_dataset_with_frozen_lineage(platform, missing_output):
    node, identifier, command, code = pending(platform, STAGE_PROGRESS, {"n_step": 25})
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    output = "/machine/output/" + attempt["id"]
    step = code["automatic_preflight"]["plan"]["parameters"]["steps"] - 1
    model = {"kind": "MODEL", "name": "stage scorer", "path": output + f"/checkpoints/mixed/platform/{step}", "version": str(step),
             "metadata": {"format": "openpi-pytorch", "backend": "pytorch", "model_role": "stage_progress_estimator", "step": step}}
    dataset = {"kind": "DATASET", "name": "Kai0 labels", "path": output + "/stage_advantage_dataset", "version": str(step)}
    dataset["metadata"] = {
        "format": "lerobot", "registration_source": "recap_stage_progress",
        "source_dataset_asset_id": run["inputs"]["dataset_0"], "source_norm_asset_id": run["inputs"]["norm_0"],
        "scorer_checkpoint_path": model["path"], "annotation_sha256": "c" * 64, "source_manifest_digest": "b" * 64,
        "source_validation": source_result(dataset),
        "annotation": {"schema_version": 1, "method": "kai0_stage_progress", "score": "absolute_advantage",
                       "n_step": 25, "positive_ratio": .3, "stage_count": 2,
                       "frame_count": 6, "positive_count": 2,
                       "stages": [{"stage": stage, "frame_count": 3, "positive_count": 1, "threshold": .5}
                                  for stage in range(2)]}}
    outputs = [model] if missing_output else [model, dataset]
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": outputs})
    assert response.status_code == 200, response.text
    finished = platform.get_run(run)
    if missing_output:
        assert finished["state"] == "FAILED" and finished["error"]["code"] == "RECAP_STAGE_OUTPUT_INVALID"
    else:
        assert finished["state"] == "SUCCEEDED", finished.get("error")
        assets = platform.client.get("/api/v1/assets", headers=ADMIN).json()
        output_asset = next(row for row in assets if row["path"] == dataset["path"])
        assert output_asset["norm_asset_id"] == run["inputs"]["norm_0"]
        assert output_asset["metadata"]["source_dataset_asset_id"] == run["inputs"]["dataset_0"]
        assert output_asset["validation"]["status"] == "SUCCEEDED"


def test_stage_progress_rejects_multi_dataset_before_dispatch(platform):
    node, inputs, rows = configured(platform)
    response = request_preview(platform, node, template_id="recap", training_stage=STAGE_PROGRESS,
                               inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECAP_STAGE_DATASET_COUNT"


@pytest.mark.parametrize("mutation,expected", [("old_agent", "AGENT_TRAINING_STAGE_UNSUPPORTED"),
    ("snapshot_stage", "TRAINING_STAGE_MISMATCH"), ("plan_stage", "TRAINING_STAGE_MISMATCH"),
    ("config", "TRAINING_STAGE_MISMATCH"), ("check", "AGENT_RECAP_STAGE_INVALID"),
    ("contract", "AGENT_RECAP_STAGE_INVALID"), ("dataset", "AGENT_RECAP_STAGE_INVALID"), ("wrong_model", "AGENT_RECAP_STAGE_INVALID"),
    ("wrong_supervision", "AGENT_RECAP_STAGE_INVALID")])
def test_old_agent_and_wrong_stage_or_dependent_data_cannot_authorize_training(platform, mutation, expected):
    node, identifier, command, code = pending(platform)
    code = copy.deepcopy(code)
    plan = code["automatic_preflight"]["plan"]
    active_checks = checks()
    if mutation == "old_agent":
        code.pop("training_stage")
        code.pop("stage_config")
        plan.pop("training_stage")
        plan.pop("stage_config")
    elif mutation == "snapshot_stage":
        code["training_stage"] = "offline_rl"
    elif mutation == "plan_stage":
        plan["training_stage"] = "offline_rl"
    elif mutation == "config":
        plan["stage_config"] = {"val_ratio": .2}
    elif mutation == "check":
        active_checks.pop()
    elif mutation == "contract":
        plan.pop("recap_stage_contract")
    elif mutation == "wrong_model":
        plan["recap_stage_contract"]["model_type"] = "Pi0Config"
    elif mutation == "wrong_supervision":
        plan["recap_stage_contract"]["datasets"][-1]["required_columns"] = ["is_positive"]
    else:
        plan["recap_stage_contract"]["datasets"][-1]["path"] = "/machine/unbound-data"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code, active_checks)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected, checked
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_frozen_request_stage_mutation_is_rejected_on_submission(platform):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        preview.request = {**preview.request, "training_stage": "offline_rl"}
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "TRAINING_STAGE_MISMATCH"


def test_started_phase_drift_cancels_without_early_gpu_release(platform):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.get_run(run)["attempts"][0]
    actual = copy.deepcopy(run["code_expectation"])
    actual["training_stage"] = "offline_rl"
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234,
        "code_snapshot": actual, "output_dir": "/machine/output/" + attempt["id"]})
    assert response.status_code == 200
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == "TRAINING_STAGE_MISMATCH"
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == attempt["id"]
