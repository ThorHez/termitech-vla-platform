"""ReCap phases are separate frozen contracts, including all dependent assets."""

import copy

import pytest
from vla_protocol import canonical_digest

from vla_control.models import Preview

from .conftest import ADMIN, automatic_checks, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview


def pending(platform, stage="value", config=None):
    node, inputs, rows = configured(platform)
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
    plan.update(training_stage=stage, stage_config=payload["stage_config"], recap_stage_contract={
        "stage": stage, "model_type": "Pi0ValueConfig" if stage == "value" else "Pi0ACPConfig",
        "loader_type": "LeRobotValueDataset" if stage == "value" else "LeRobotACPDataset",
        "datasets": [{"path": dataset["path"], "required_columns": ["value_target"] if stage == "value" else ["advantage", "is_positive"]}
                     for dataset in plan["datasets"]]})
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
