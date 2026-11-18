"""ReCap phases are separate frozen contracts, including all dependent assets."""

import copy

import pytest
from vla_adapters.recap import (ACP_MULTI_ENTRYPOINT, ACP_STAGE, RECAP_TORCH_LOADER, STAGE_ADVANTAGE_STAGES, STAGE_ANNOTATION, STAGE_PROGRESS,
    STAGE_PROGRESS_ID_COLUMNS, STAGE_PROGRESS_MODEL, VALUE_ANNOTATION, recap_model_role,
    recap_torch_entrypoint, recap_torch_module_paths)
from vla_protocol import canonical_digest

from vla_control.common import now
from vla_control.models import Asset, Attempt, Preview, Run

from .conftest import ADMIN, automatic_checks, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_dataset_validations import result as source_result
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview


def pending(platform, stage="value", config=None, resources=None):
    node, inputs, rows = resources or configured(platform)
    if stage in (*STAGE_ADVANTAGE_STAGES, VALUE_ANNOTATION):
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
    plan["entrypoint"] = "/machine/code/" + recap_torch_entrypoint(stage)
    plan.update(training_stage=stage, stage_config=payload["stage_config"], recap_stage_contract={
        "stage": stage, "model_type": "Pi0ValueConfig" if stage == "value" else "Pi0Config",
        "loader_type": RECAP_TORCH_LOADER, "backend": "pytorch",
        "runtime_versions": {"torch": "2.7.1", "safetensors": "0.5.3", "transformers": "4.53.2"},
        "source_sha256": {f"src/openpi/{path}": "a" * 64 for path in recap_torch_module_paths(stage)},
        "datasets": [{"path": dataset["path"], "required_columns": ["value_target"] if stage == "value" else ["advantage", "is_positive"]}
                     for dataset in plan["datasets"]]})
    if stage == ACP_STAGE:
        params = model + "/params"
        plan["entrypoint"] = "/machine/code/" + ACP_MULTI_ENTRYPOINT
        plan["model"].update(path=params, format="openpi-orbax", model_role="policy",
                             metadata=[{"path": params + "/_METADATA", "sha256": "a" * 64}])
        contract = plan["recap_stage_contract"]
        contract.update(backend="jax", loader_type="CheckpointWeightLoader")
        contract.pop("runtime_versions")
        contract.pop("source_sha256")
    if stage in STAGE_ADVANTAGE_STAGES:
        plan["datasets"][0]["frames"] = 6
        plan["recap_stage_contract"].update(
            model_type=STAGE_PROGRESS_MODEL,
            datasets=[{"path": plan["datasets"][0]["path"],
                       "required_columns": ["stage_progress_gt", *STAGE_PROGRESS_ID_COLUMNS]}])
        if stage == STAGE_PROGRESS:
            plan["recap_stage_contract"]["operation"] = "train"
        else:
            plan["model"]["scorer_checkpoint"] = {"stage": STAGE_PROGRESS, "step": int(model.rsplit("/", 1)[-1])}
            plan["model"]["metadata"].append({"path": model + "/recap_checkpoint.json", "sha256": "b" * 64})
            plan["recap_stage_contract"]["annotation"] = {"output": "stage_advantage_dataset",
                "source_manifest_digest": "b" * 64, "n_step": 50, "positive_ratio": .3,
                "stage_count": 2, **payload["stage_config"]}
    if stage == VALUE_ANNOTATION:
        plan["datasets"][0]["frames"] = 6
        plan["model"]["scorer_checkpoint"] = {"stage": "value", "step": int(model.rsplit("/", 1)[-1])}
        plan["model"]["metadata"].append({"path": model + "/recap_checkpoint.json", "sha256": "b" * 64})
        plan["recap_stage_contract"].update(
            model_type="Pi0ValueConfig", datasets=[{"path": plan["datasets"][0]["path"],
                "required_columns": ["value_target", *STAGE_PROGRESS_ID_COLUMNS]}],
            annotation={"output": "value_advantage_dataset", "source_manifest_digest": "b" * 64,
                        "n_step": 50, "positive_ratio": .3, **payload["stage_config"]})
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


@pytest.mark.parametrize("format,role,error", [
    ("openpi-pytorch", "policy", "MODEL_FORMAT"),
    ("openpi-orbax", "stage_progress_estimator", "ACP_POLICY_REQUIRED"),
])
def test_acp_rejects_incompatible_model_registration(platform, format, role, error):
    node, inputs, rows = configured(platform)
    with platform.app.state.db.write() as session:
        session.get(Asset, inputs["model"]).metadata_json = {"format": format, "model_role": role}
    response = request_preview(platform, node, template_id="recap", training_stage=ACP_STAGE,
                               inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422 and response.json()["error"]["code"] == error
    assert platform.commands(node) == []


def test_acp_requires_current_jax_receipt(platform):
    node, identifier, command, code = pending(platform, ACP_STAGE)
    evidence = code["automatic_preflight"]
    evidence["plan"]["recap_stage_contract"]["backend"] = "pytorch"
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    response = acknowledge(platform, node, identifier, command, code, checks())
    assert response["status"] == "FAILED" and response["error"]["code"] == "AGENT_RECAP_STAGE_INVALID"


@pytest.mark.parametrize("format", ["openpi-orbax", "openpi-pytorch"])
def test_acp_registers_only_final_orbax_policy(platform, format):
    node, identifier, command, code = pending(platform, ACP_STAGE)
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    step = code["automatic_preflight"]["plan"]["parameters"]["steps"] - 1
    output = {"kind": "MODEL", "name": "ACP", "version": str(step),
              "path": f"/machine/output/{attempt['id']}/checkpoints/mixed/platform/{step}",
              "metadata": {"format": format, "backend": "jax", "model_role": "policy", "step": step}}
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": [output]})
    assert response.status_code == 200, response.text
    finished = platform.get_run(run)
    if format == "openpi-orbax":
        assert finished["state"] == "SUCCEEDED", finished.get("error")
        assert finished["outputs"][0]["metadata"]["format"] == format
    else:
        assert finished["state"] == "FAILED" and not finished["outputs"]


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


def complete_progress_training(platform, *, earlier=False, stage=STAGE_PROGRESS):
    node, identifier, command, code = pending(platform, stage)
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    output = "/machine/output/" + attempt["id"]
    step = code["automatic_preflight"]["plan"]["parameters"]["steps"] - 1
    model = {"kind": "MODEL", "name": "estimator", "path": output + f"/checkpoints/mixed/platform/{step}", "version": str(step),
             "metadata": {"format": "openpi-pytorch", "backend": "pytorch", "model_role": recap_model_role(stage), "step": step}}
    seq = 2
    if earlier:
        intermediate = {**model, "path": output + f"/checkpoints/mixed/platform/{step - 1}", "version": str(step - 1),
                        "metadata": {**model["metadata"], "step": step - 1}}
        artifact = platform.emit(node, attempt["id"], seq, "ARTIFACT", {"outputs": [intermediate]})
        assert artifact.status_code == 200, artifact.text
        in_progress = platform.get_run(run)["outputs"]
        assert len(in_progress) == 1 and "stage_annotation_source" not in in_progress[0]
        seq += 1
    response = platform.emit(node, attempt["id"], seq, "SUCCEEDED", {"exit_code": 0, "outputs": [model]})
    assert response.status_code == 200, response.text
    finished = platform.get_run(run)
    assert finished["state"] == "SUCCEEDED", finished.get("error")
    assert len(finished["outputs"]) == (2 if earlier else 1)
    selected_path = intermediate["path"] if earlier else model["path"]
    checkpoint = next(asset for asset in finished["outputs"] if asset["path"] == selected_path and asset["kind"] == "MODEL")
    return node, finished, checkpoint


@pytest.mark.parametrize("scorer_mismatch", [False, True])
def test_stage_progress_and_annotation_submit_independently_with_frozen_lineage(platform, scorer_mismatch):
    node, upstream, checkpoint = complete_progress_training(platform)
    node, identifier, command, code = pending(platform, STAGE_ANNOTATION, {"n_step": 25},
        (node, {"model": checkpoint["id"]}, upstream["dataset_bindings"]))
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    scoring_asset = next(asset for asset in run["input_assets"] if asset["id"] == checkpoint["id"])
    assert scoring_asset["origin_run_id"] == upstream["id"]
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    output = "/machine/output/" + attempt["id"]
    dataset = {"kind": "DATASET", "name": "Stage Advantage labels", "path": output + "/stage_advantage_dataset",
               "version": checkpoint["version"]}
    dataset["metadata"] = {
        "format": "lerobot", "registration_source": "recap_stage_annotation",
        "source_dataset_asset_id": run["inputs"]["dataset_0"], "source_norm_asset_id": run["inputs"]["norm_0"],
        "scorer_model_asset_id": checkpoint["id"] if not scorer_mismatch else "wrong-model",
        "scorer_checkpoint_path": checkpoint["path"], "annotation_sha256": "c" * 64, "source_manifest_digest": "b" * 64,
        "source_validation": source_result(dataset),
        "annotation": {"schema_version": 1, "method": "kai0_stage_progress", "score": "absolute_advantage",
                       "n_step": 25, "positive_ratio": .3, "stage_count": 2,
                       "frame_count": 6, "positive_count": 2,
                       "stages": [{"stage": stage, "frame_count": 3, "positive_count": 1, "threshold": .5}
                                  for stage in range(2)]}}
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": [dataset]})
    assert response.status_code == 200, response.text
    finished = platform.get_run(run)
    if scorer_mismatch:
        assert finished["state"] == "FAILED" and finished["error"]["code"] == "RECAP_STAGE_OUTPUT_INVALID"
        assert not finished["outputs"]
    else:
        assert finished["state"] == "SUCCEEDED", finished.get("error")
        assets = platform.client.get("/api/v1/assets", headers=ADMIN).json()
        output_asset = next(row for row in assets if row["path"] == dataset["path"])
        assert output_asset["norm_asset_id"] == run["inputs"]["norm_0"]
        assert output_asset["metadata"]["source_dataset_asset_id"] == run["inputs"]["dataset_0"]
        assert output_asset["validation"]["status"] == "SUCCEEDED"


def test_progress_checkpoint_can_start_more_training_without_producing_annotation(platform):
    node, upstream, checkpoint = complete_progress_training(platform)
    _, identifier, command, code = pending(platform, STAGE_PROGRESS,
        resources=(node, {"model": checkpoint["id"]}, upstream["dataset_bindings"]))
    assert command["payload"]["inputs"]["model"]["id"] == checkpoint["id"]
    assert not command["payload"].get("resume")
    checked = acknowledge(platform, node, identifier, command, code, checks())
    assert checked["status"] == "READY"
    run = submit(platform, identifier).json()
    assert run["id"] != upstream["id"] and run["training_stage"] == STAGE_PROGRESS
    assert run["stage_config"] == {} and not run["outputs"]


@pytest.mark.parametrize("wrong_checkpoint", [False, True])
def test_value_annotation_uses_completed_value_checkpoint_and_registers_only_checked_labels(platform, wrong_checkpoint):
    node, upstream, checkpoint = complete_progress_training(platform, stage="value")
    source = checkpoint["value_annotation_source"]
    assert source["run_id"] == upstream["id"] and source["step"] == checkpoint["metadata"]["step"]
    _, identifier, command, code = pending(platform, VALUE_ANNOTATION, {"n_step": 25},
        (node, {"model": checkpoint["id"]}, upstream["dataset_bindings"]))
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    run = submit(platform, identifier).json()
    scorer = next(asset for asset in run["input_assets"] if asset["id"] == checkpoint["id"])
    assert scorer["origin_run_id"] == upstream["id"]
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    output = {"kind": "DATASET", "name": "Value labels", "version": checkpoint["version"],
              "path": f"/machine/output/{attempt['id']}/value_advantage_dataset"}
    output["metadata"] = {
        "format": "lerobot", "registration_source": "recap_value_annotation",
        "scorer_model_asset_id": checkpoint["id"], "scorer_checkpoint_path": checkpoint["path"],
        "source_dataset_asset_id": run["inputs"]["dataset_0"], "source_norm_asset_id": run["inputs"]["norm_0"],
        "source_manifest_digest": "b" * 64, "annotation_sha256": "c" * 64,
        "source_validation": source_result(output),
        "annotation": {"schema_version": 1, "backend": "pytorch", "method": "n_step_value",
            "checkpoint": "/machine/unrelated" if wrong_checkpoint else checkpoint["path"],
            "n_step": 25, "positive_ratio": .3, "frame_count": 6, "positive_count": 2, "thresholds": {"0": .5}},
    }
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": [output]})
    assert response.status_code == 200, response.text
    finished = platform.get_run(run)
    if wrong_checkpoint:
        assert finished["state"] == "FAILED" and not finished["outputs"]
    else:
        assert finished["state"] == "SUCCEEDED", finished.get("error")
        dataset, = finished["outputs"]
        assert dataset["norm_asset_id"] == run["inputs"]["norm_0"]
        assert dataset["validation"]["status"] == "SUCCEEDED"


@pytest.mark.parametrize("legacy", [False, True])
def test_stage_annotation_picker_exposes_available_intermediate_job_checkpoints(platform, legacy):
    node, run, checkpoint = complete_progress_training(platform, earlier=True)
    expected = {"run_id": run["id"], "run_name": run["name"], "attempt_id": checkpoint["origin_attempt_id"],
                "step": checkpoint["metadata"]["step"]}
    assert checkpoint["stage_annotation_source"] == expected
    if legacy:
        with platform.app.state.db.write() as session:
            stored = session.get(Run, run["id"])
            snapshot = copy.deepcopy(stored.code_expectation)
            settings = {"n_step": 25, "positive_ratio": .3, "stage_count": 2}
            snapshot["stage_config"] = settings
            plan = snapshot["automatic_preflight"]["plan"]
            plan["stage_config"] = settings
            contract = plan["recap_stage_contract"]
            contract.pop("operation")
            contract["annotation"] = {"output": "stage_advantage_dataset", "source_manifest_digest": "b" * 64, **settings}
            snapshot["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
            stored.code_expectation = snapshot
    summaries = platform.client.get("/api/v1/assets?view=summary", headers=ADMIN).json()
    listed = next(asset for asset in summaries if asset["id"] == checkpoint["id"])
    assert listed["stage_annotation_source"] == expected
    detail = platform.client.get(f"/api/v1/assets/{checkpoint['id']}", headers=ADMIN).json()
    assert detail["stage_annotation_source"] == expected
    _, identifier, command, code = pending(platform, STAGE_ANNOTATION,
        resources=(node, {"model": checkpoint["id"]}, run["dataset_bindings"]))
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    assert submit(platform, identifier).status_code == 201


def test_stage_annotation_uses_a_registered_pytorch_checkpoint_without_claiming_run_provenance(platform):
    node, run, checkpoint = complete_progress_training(platform)
    registered = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "name": "duplicate registration", "kind": "MODEL", "node_id": node["node_id"], "path": checkpoint["path"],
        "metadata": {**checkpoint["metadata"], "origin_attempt_id": checkpoint["origin_attempt_id"],
                     "stage_annotation_source": checkpoint["stage_annotation_source"]},
    })
    assert registered.status_code == 201, registered.text
    reference = registered.json()
    assert reference["origin_attempt_id"] is None and "stage_annotation_source" not in reference
    listed = platform.client.get(f"/api/v1/assets/{reference['id']}", headers=ADMIN).json()
    assert "stage_annotation_source" not in listed
    _, identifier, command, code = pending(platform, STAGE_ANNOTATION,
        resources=(node, {"model": reference["id"]}, run["dataset_bindings"]))
    assert command["payload"]["inputs"]["model"]["id"] == reference["id"]
    assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    scorer = next(asset for asset in response.json()["input_assets"] if asset["id"] == reference["id"])
    assert scorer.get("origin_run_id") is None


@pytest.mark.parametrize("metadata", [{}, {"format": "openpi-pytorch", "model_role": "value_estimator"},
                                    {"format": "openpi-pytorch", "availability": "removed"}])
def test_stage_annotation_rejects_incompatible_external_model_registrations(platform, metadata):
    node, inputs, rows = configured(platform)
    with platform.app.state.db.write() as session:
        session.get(Asset, inputs["model"]).metadata_json = metadata
    response = request_preview(platform, node, template_id="recap", training_stage=STAGE_ANNOTATION,
                               inputs=inputs, dataset_bindings=rows[:1])
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECAP_ANNOTATION_MODEL_REQUIRED"
    assert platform.commands(node) == []


def test_stage_annotation_requires_agent_checkpoint_evidence_for_external_models(platform):
    node, inputs, rows = configured(platform)
    with platform.app.state.db.write() as session:
        model = session.get(Asset, inputs["model"])
        model.path = "/machine/imported/900"
        model.metadata_json = {"format": "torch"}
    _, identifier, command, code = pending(platform, STAGE_ANNOTATION, resources=(node, inputs, rows))
    evidence = code["automatic_preflight"]
    evidence["plan"]["model"]["metadata"].pop()
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    result = acknowledge(platform, node, identifier, command, code, checks())
    assert result["status"] == "FAILED" and result["error"]["code"] == "AGENT_RECAP_STAGE_INVALID"


@pytest.mark.parametrize("change", ["run_failed", "attempt_failed", "other_stage", "wrong_path", "removed",
                                     "wrong_role", "deleted"])
def test_stage_annotation_rejects_unusable_checkpoint_origin(platform, change):
    node, run, checkpoint = complete_progress_training(platform)
    with platform.app.state.db.write() as session:
        model = session.get(Asset, checkpoint["id"])
        upstream = session.get(Run, run["id"])
        if change == "run_failed":
            upstream.state = "FAILED"
        elif change == "attempt_failed":
            session.get(Attempt, model.origin_attempt_id).state = "FAILED"
        elif change == "other_stage":
            upstream.code_expectation = {**upstream.code_expectation, "training_stage": "value"}
        elif change == "wrong_path":
            model.path = "/machine/other-checkpoint"
        elif change == "removed":
            model.metadata_json = {**model.metadata_json, "availability": "removed"}
        elif change == "wrong_role":
            model.metadata_json = {**model.metadata_json, "model_role": "value_estimator"}
        elif change == "deleted":
            upstream.deleted_at = now()
    detail = platform.client.get(f"/api/v1/assets/{checkpoint['id']}", headers=ADMIN).json()
    assert "stage_annotation_source" not in detail
    response = request_preview(platform, node, template_id="recap", training_stage=STAGE_ANNOTATION,
        inputs={"model": checkpoint["id"]}, dataset_bindings=run["dataset_bindings"])
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECAP_ANNOTATION_MODEL_REQUIRED"
    assert platform.commands(node) == []


@pytest.mark.parametrize("changed", ["actual_step", "deleted_after_preflight"])
def test_stage_annotation_rechecks_source_step_and_origin_before_submission(platform, changed):
    node, run, checkpoint = complete_progress_training(platform)
    _, identifier, command, code = pending(platform, STAGE_ANNOTATION,
        resources=(node, {"model": checkpoint["id"]}, run["dataset_bindings"]))
    if changed == "actual_step":
        evidence = code["automatic_preflight"]
        evidence["plan"]["model"]["scorer_checkpoint"]["step"] += 1
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
        response = acknowledge(platform, node, identifier, command, code, checks())
        assert response["status"] == "FAILED" and response["error"]["code"] == "RECAP_ANNOTATION_MODEL_REQUIRED"
    else:
        assert acknowledge(platform, node, identifier, command, code, checks())["status"] == "READY"
        with platform.app.state.db.write() as session:
            session.get(Run, run["id"]).deleted_at = now()
        response = submit(platform, identifier)
        assert response.status_code == 422 and response.json()["error"]["code"] == "RECAP_ANNOTATION_MODEL_REQUIRED"


@pytest.mark.parametrize("stage", [STAGE_PROGRESS, STAGE_ANNOTATION])
def test_stage_advantage_rejects_multi_dataset_before_dispatch(platform, stage):
    node, inputs, rows = configured(platform)
    response = request_preview(platform, node, template_id="recap", training_stage=stage,
                               inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECAP_STAGE_DATASET_COUNT"


def test_stage_progress_requires_train_only_proof_from_agent(platform):
    node, identifier, command, code = pending(platform, STAGE_PROGRESS)
    evidence = code["automatic_preflight"]
    evidence["plan"]["recap_stage_contract"].pop("operation")
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    response = acknowledge(platform, node, identifier, command, code, checks())
    assert response["status"] == "FAILED" and response["error"]["code"] == "AGENT_RECAP_STAGE_INVALID"


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
