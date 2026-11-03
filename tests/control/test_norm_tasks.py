import copy

import pytest

from vla_protocol import canonical_digest
from vla_protocol.norm_plan import NORM_CHECKS
from .conftest import ADMIN, confirmed_run_request, generated_config, snapshot


def numeric_io(spec=None):
    states = [row["key"] for row in spec["state_fields"]] if spec is not None else ["observation.state"]
    action = spec["action_key"] if spec is not None else "action"
    return {"schema_version": 1, "reader": "pyarrow_parquet_projection", "columns": [*states, action],
            "state_columns": states, "action_column": action, "temporal_columns": [], "action_sequence": "packed",
            "image_columns": [], "video_decode": False,
            "source": {"path": "src/openpi/training/numeric_stats.py", "sha256": "9" * 64}}


def numeric_cache(datasets):
    return {"mode": "numeric_projection", "persistent_cache": False,
            "paths": {"HF_DATASETS_CACHE": "/machine/cache/datasets"},
            "datasets": [{"path": dataset["path"], "columns": list(dataset["numeric_io"]["columns"])}
                         for dataset in datasets]}


def pending_norm(platform):
    node = platform.register(adapter="openpi-sft", gpus=[])
    asset = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "Data", "kind": "DATASET",
        "node_id": node["node_id"], "path": "/machine/data"}).json()
    request = {"name": "Compute statistics", "node_id": node["node_id"], "profile_id": "default",
               "template_id": "openpi-norm", "recipe": "my_recipe", "inputs": {"dataset": asset["id"]},
               "parameters": {"batch_size": 32, "num_workers": 0}, "gpu_count": 0}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 202, response.text
    preview = response.json()
    command = next(command for command in platform.commands(node) if command["type"] == "PREFLIGHT")
    plan = {"template_id": "openpi-norm", "recipe": "my_recipe", "recipe_definition_digest": None,
            "input_authority": "registered_assets", "requested_inputs": {"dataset": "/machine/data"},
            "inputs": {"dataset": "/machine/data"}, "parameters": request["parameters"], "gpu_count": 0,
            "entrypoint": "/machine/code/scripts/compute_norm_stats.py", "entrypoint_sha256": "a" * 64,
            "adapter_sha256": "b" * 64, "generated_config": generated_config(),
            "datasets": [{"path": "/machine/data", "frames": 32, "metadata_sha256": "c" * 64,
                          "manifest_sha256": "d" * 64, "sample_shapes": {"state": [14], "actions": [50, 14]},
                          "actions_shape": [50, 14], "numeric_io": numeric_io()}]}
    plan["training_cache"] = numeric_cache(plan["datasets"])
    code = snapshot(recipe="my_recipe", automatic_preflight={"schema_version": 1,
        "scope": "configuration_and_local_assets", "plan": plan, "plan_digest": canonical_digest(plan)})
    checks = [{"name": name, "ok": True, "message": "checked"} for name in NORM_CHECKS]
    return node, request, preview, command, code, checks


def acknowledge(platform, node, preview, command, code, checks):
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": "APPLIED", "result": {"code_snapshot": code, "checks": checks}})
    assert response.status_code == 200, response.text
    return platform.client.get(f"/api/v1/run-previews/{preview['id']}", headers=ADMIN).json()


def test_norm_template_preview_confirmation_cpu_schedule_and_output_registration(platform):
    templates = platform.client.get("/api/v1/templates", headers=ADMIN).json()
    assert any(row["id"] == "openpi-norm" and row["status"] == "AVAILABLE" for row in templates)
    node, request, preview, command, code, checks = pending_norm(platform)
    ready = acknowledge(platform, node, preview, command, code, checks)
    assert ready["status"] == "READY", ready
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "norm-no-confirm"},
                                    json={"preflight_id": preview["id"]})
    assert response.status_code == 409 and response.json()["error"]["code"] == "CONFIG_CONFIRMATION_REQUIRED"
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "norm-confirmed"},
                                    json=confirmed_run_request(ready))
    assert response.status_code == 201, response.text
    run = response.json()
    platform.schedule()
    started = platform.get_run(run)
    attempt = started["attempts"][-1]
    assert attempt["gpu_uuids"] == []
    start = next(row for row in platform.commands(node) if row["type"] == "START")
    assert start["payload"]["confirmed_config_sha256"] == generated_config()["sha256"]
    output_dir = f"/machine/output/{run['id']}/{attempt['id']}"
    platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 123, "code_snapshot": start["payload"]["code_expectation"],
                  "output_dir": output_dir})
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": [{
        "kind": "NORM", "name": "my_recipe norm_stats", "path": output_dir + "/norm_stats.json", "version": "sha256"}]})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed["state"] == "SUCCEEDED", completed
    assets = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    assert any(row["kind"] == "NORM" and row["origin_attempt_id"] == attempt["id"] for row in assets)
    produced = next(row for row in assets if row["origin_attempt_id"] == attempt["id"])
    source = next(row for row in assets if row["id"] == request["inputs"]["dataset"])
    assert source["norm_asset_id"] == produced["id"]
    assert produced["name"] == "Data norm_stats"
    assert produced["metadata"]["source_dataset_asset_id"] == source["id"]


@pytest.mark.parametrize("change", ["gpu", "model", "steps", "normalization", "bindings", "auto_config"])
def test_norm_rejects_training_only_inputs_and_parameters(platform, change):
    _, request, *_ = pending_norm(platform)
    request = copy.deepcopy(request)
    if change == "gpu":
        request["gpu_count"] = 1
    elif change == "model":
        request["inputs"]["model"] = "unselected-model"
    elif change == "steps":
        request["parameters"]["steps"] = 100
    elif change == "normalization":
        request["normalization_method"] = "quantile"
    elif change == "bindings":
        request["dataset_bindings"] = [{"dataset_asset_id": request["inputs"]["dataset"], "norm_asset_id": "old-norm", "recipe_index": 0, "weight": 1.}]
    else:
        request["auto_config"] = {"model_type": "pi05", "finetune_mode": "full", "action_horizon": 50, "action_dim": 32}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("change", ["missing", "sample", "parameters", "digest", "entrypoint", "checks"])
def test_norm_requires_current_bound_statistics_evidence(platform, change):
    node, _, preview, command, code, checks = pending_norm(platform)
    plan = code["automatic_preflight"]["plan"]
    if change == "missing":
        del code["automatic_preflight"]
    elif change == "sample":
        plan["datasets"][0]["sample_shapes"] = {}
    elif change == "parameters":
        plan["parameters"]["batch_size"] = 8
    elif change == "digest":
        code["automatic_preflight"]["plan_digest"] = "e" * 64
    elif change == "entrypoint":
        plan["entrypoint"] = "/other/code/compute.py"
    else:
        checks = [item for item in checks if item["name"] != "norm_entrypoint"]
    if change not in {"missing", "digest"}:
        code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked["status"] == "FAILED", checked


def pending_batch_norm(platform, *, explicit_processor=False, auto_config=None, composed=False, composition_options=None):
    if auto_config is None:
        node, single_request, _, _, _, _ = pending_norm(platform)
        first_id = single_request["inputs"]["dataset"]
    else:
        # Automatic statistics must work in a profile with no named code recipe.
        node = platform.register(adapter="openpi-sft", gpus=[], recipe=None, metadata={})
        first = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "Data", "kind": "DATASET",
            "node_id": node["node_id"], "path": "/machine/data"}).json()
        first_id = first["id"]
        single_request = {"node_id": node["node_id"], "profile_id": "default", "template_id": "openpi-norm",
                          "parameters": {"batch_size": 32, "num_workers": 0}, "gpu_count": 0,
                          "auto_config": copy.deepcopy(auto_config)}
    second = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "Other data", "kind": "DATASET",
        "node_id": node["node_id"], "path": "/machine/other-data"}).json()
    rows = [{"dataset_asset_id": first_id, "recipe_index": 0},
            {"dataset_asset_id": second["id"], "recipe_index": 1}]
    if explicit_processor:
        rows[1] = {"dataset_asset_id": second["id"],
                   "processor": {"name": "LeRobotAlohaDataConfig", "parameters": {"adapt_to_pi": False}}}
    if auto_config is not None:
        rows = [{"dataset_asset_id": row["dataset_asset_id"], "processor": {
            "name": "LeRobotAlohaDataConfig", "parameters": {"adapt_to_pi": False}}} for row in rows]
    if composed:
        from vla_adapters.composition_spec import composition_choice
        from .test_composed_asset_processing import description
        declared = description(**(composition_options or {}))
        processor = composition_choice(declared, norm_stats=True)
        processor["parameters"]["composition"]["state_fields"].reverse()
        processor["parameters"]["composition"]["normalization"] = "zscore"
        for row in rows:
            response = platform.client.patch(f"/api/v1/assets/{row['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                             json={"dataset_profile": declared})
            assert response.status_code == 200, response.text
            row.pop("recipe_index", None)
            row["processor"] = copy.deepcopy(processor)
    request = {**single_request, "name": "Independent batch statistics", "inputs": {}, "norm_datasets": rows}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 202, response.text
    preview = response.json()
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT"
                   and item["payload"]["preview_id"] == preview["id"])
    paths = {role: value["path"] for role, value in command["payload"]["inputs"].items()}
    payload = command["payload"]
    plan = {"template_id": "openpi-norm", "mode": "per_dataset_batch", "recipe": payload["recipe"], "recipe_definition_digest": None,
            "input_authority": "registered_assets", "requested_inputs": paths, "inputs": paths,
            "parameters": request["parameters"], "gpu_count": 0, "norm_datasets": rows,
            "entrypoint": "/machine/code/scripts/compute_norm_stats.py", "entrypoint_sha256": "a" * 64,
            "adapter_sha256": "b" * 64, "generated_config": generated_config(),
            "datasets": [{"path": paths[f"dataset_{i}"], "input_role": f"dataset_{i}",
                          **row,
                          "frames": 32, "metadata_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                          "sample_shapes": {"state": [14], "actions": [50, 14]},
                          "actions_shape": [50, 14], "numeric_io": numeric_io()}
                         for i, row in enumerate(rows)]}
    for dataset in plan["datasets"]:
        if "processor" in dataset:
            name = dataset["processor"]["name"]
            dataset.update(factory=name, processor_identity={"name": name, "module": "openpi.training.config",
                "source": {"path": "src/openpi/training/config.py", "sha256": "e" * 64}})
        if auto_config is not None:
            dataset["model_configuration"] = {key: payload["auto_config"][key]
                for key in ("model_type", "action_horizon", "action_dim")}
        if composed:
            spec = dataset["processor"]["parameters"]["composition"]
            model = payload["auto_config"]
            raw_state = sum(field["width"] for field in spec["state_fields"])
            raw_action = sum(field["width"] for field in spec["action_layout"])
            dataset["actions_shape"] = [model["action_horizon"], raw_action]
            dataset["sample_shapes"] = {"state": [raw_state], "actions": dataset["actions_shape"]}
            dataset["numeric_io"] = numeric_io(spec)
            dataset["composition"] = {"schema_version": 1, "spec": copy.deepcopy(spec),
                "digest": canonical_digest(spec), "training_stage": "sft", "model_type": "Pi0Config",
                "pi05": model["model_type"] == "pi05", "action_dim": model["action_dim"],
                "action_horizon": model["action_horizon"], "raw_state_dim": raw_state,
                "raw_action_dim": raw_action, "state_history": spec["state_history"],
                "normalization": spec["normalization"], "action_sequence_keys": [],
                "action_metadata_sha256": dataset["metadata_sha256"],
                "action_loss_mask": [1.] * raw_action + [0.] * (model["action_dim"] - raw_action),
                "image_keys": [], "depth_keys": [],
                "source": {"path": "src/openpi/training/composed_processing.py", "sha256": "f" * 64}}
    plan["training_cache"] = numeric_cache(plan["datasets"])
    if auto_config is not None:
        plan["auto_config"] = payload["auto_config"]
    code = snapshot(recipe=payload["recipe"], norm_datasets=rows,
        **({"auto_config": payload["auto_config"]} if auto_config is not None else {}), automatic_preflight={"schema_version": 1,
        "scope": "configuration_and_local_assets", "plan": plan, "plan_digest": canonical_digest(plan)})
    checks = [{"name": name, "ok": True, "message": "checked"} for name in NORM_CHECKS]
    return node, request, preview, command, code, checks


def start_batch_norm(platform, **options):
    node, request, preview, command, code, checks = pending_batch_norm(platform, **options)
    ready = acknowledge(platform, node, preview, command, code, checks)
    assert ready["status"] == "READY", ready
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": preview["id"]},
                                    json=confirmed_run_request(ready))
    assert response.status_code == 201, response.text
    run = response.json()
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][-1]
    start = next(row for row in platform.commands(node) if row["type"] == "START")
    output_dir = f"/machine/output/{run['id']}/{attempt['id']}"
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 123,
        "code_snapshot": start["payload"]["code_expectation"], "output_dir": output_dir})
    assert response.status_code == 200, response.text
    return node, request, run, attempt, start, output_dir


def batch_outputs(request, output_dir):
    return [{"kind": "NORM", "name": f"Statistics {i}", "path": f"{output_dir}/dataset-{i}/norm_stats.json",
             "version": str(i), "metadata": {"source_dataset_asset_id": row["dataset_asset_id"]}}
            for i, row in enumerate(request["norm_datasets"])]


def test_batch_norm_frozen_inputs_start_and_one_registered_output_per_dataset(platform):
    node, request, run, attempt, start, output_dir = start_batch_norm(platform)
    assert run["norm_datasets"] == request["norm_datasets"]
    assert start["payload"]["norm_datasets"] == request["norm_datasets"]
    assert set(start["payload"]["inputs"]) == {"dataset_0", "dataset_1"}
    assert all(value["kind"] == "DATASET" for value in start["payload"]["inputs"].values())
    assert start["payload"]["gpu_uuids"] == []
    outputs = batch_outputs(request, output_dir)
    # A completed first source is still partial batch state, not a usable asset.
    response = platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": outputs[:1]})
    assert response.status_code == 200, response.text
    assert platform.get_run(run)["outputs"] == []
    response = platform.emit(node, attempt["id"], 3, "SUCCEEDED", {"exit_code": 0, "outputs": outputs})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed["state"] == "SUCCEEDED", completed
    assert len(completed["outputs"]) == 2
    assert {row["metadata"]["source_dataset_asset_id"] for row in completed["outputs"]} == {
        row["dataset_asset_id"] for row in request["norm_datasets"]}


def test_batch_norm_freezes_each_independent_processor_without_training_bindings(platform):
    _, request, run, _, start, _ = start_batch_norm(platform, explicit_processor=True)
    expected = {"name": "LeRobotAlohaDataConfig", "parameters": {"adapt_to_pi": False}}
    assert run["norm_datasets"][1]["processor"] == expected
    assert start["payload"]["norm_datasets"][1]["processor"] == expected
    assert run["dataset_bindings"] is None
    assert all("weight" not in row and "norm_asset_id" not in row for row in request["norm_datasets"])


@pytest.mark.parametrize("auto_config", [{}, {"model_type": "pi0", "action_horizon": 16, "action_dim": 14}])
def test_automatic_norm_needs_no_code_recipe_or_training_assets_and_freezes_model_choices(platform, auto_config):
    from vla_adapters.automatic_config import normalize_auto_config
    node, request, run, attempt, start, output_dir = start_batch_norm(platform, auto_config=auto_config)
    expected = normalize_auto_config(auto_config)
    assert "recipe" not in request
    assert run["auto_config"] == expected
    assert start["payload"]["auto_config"] == expected
    assert run["recipe"] == f"auto_{expected['model_type']}_full_h{expected['action_horizon']}_d{expected['action_dim']}_sft"
    assert start["payload"]["recipe"] == run["recipe"]
    assert run["code_expectation"]["auto_config"] == expected
    assert run["norm_datasets"] == request["norm_datasets"]
    assert set(start["payload"]["inputs"]) == {"dataset_0", "dataset_1"}
    assert all(value["kind"] == "DATASET" for value in start["payload"]["inputs"].values())
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0,
        "outputs": batch_outputs(request, output_dir)})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed["state"] == "SUCCEEDED", completed
    assert len(completed["outputs"]) == 2


def test_automatic_norm_accepts_and_freezes_custom_field_composition_for_each_source(platform):
    _, request, run, _, start, _ = start_batch_norm(platform,
        auto_config={"model_type": "pi05", "action_horizon": 16, "action_dim": 32}, composed=True)
    for index, row in enumerate(request["norm_datasets"]):
        chosen = row["processor"]
        assert chosen["name"] == "ComposableDataConfig"
        assert chosen["parameters"]["composition"]["normalization"] == "zscore"
        assert chosen["parameters"]["composition"]["state_fields"][0]["key"] == "observation.gripper"
        assert chosen["parameters"]["composition"]["cameras"] == []
        assert run["norm_datasets"][index]["processor"] == chosen
        assert start["payload"]["norm_datasets"][index]["processor"] == chosen
        profile = start["payload"]["inputs"][f"dataset_{index}"]["dataset_profile"]
        assert profile["processing"]["normalization"] == "quantile"
    assert run["dataset_bindings"] is None


def test_automatic_norm_freezes_numeric_composition_with_no_visual_configuration(platform):
    _, request, run, _, start, _ = start_batch_norm(platform, auto_config={}, composed=True)
    assert run["norm_datasets"] == request["norm_datasets"] == start["payload"]["norm_datasets"]
    for dataset in run["code_expectation"]["automatic_preflight"]["plan"]["datasets"]:
        composition = dataset["composition"]
        assert composition["spec"]["cameras"] == []
        assert composition["image_keys"] == []
        assert composition["depth_keys"] == []
        assert composition["raw_state_dim"] == 7
        assert composition["raw_action_dim"] == 10


def test_norm_request_discards_legacy_visual_configuration_before_freezing(platform):
    node, request, *_ = pending_batch_norm(platform, auto_config={}, composed=True)
    request = copy.deepcopy(request)
    for row in request["norm_datasets"]:
        # Historical visual settings are inert for a statistics task, even if
        # their old slots or image layout are no longer supported by training.
        row["processor"]["parameters"]["composition"]["cameras"] = [
            {"key": "video.legacy", "slot": "obsolete", "layout": "invalid"}]
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 202, response.text
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT"
                   and item["payload"]["preview_id"] == response.json()["id"])
    assert all(row["processor"]["parameters"]["composition"]["cameras"] == []
               for row in command["payload"]["norm_datasets"])


@pytest.mark.parametrize("change", ["missing_projection", "video_decode", "image_column", "extra_column",
    "unselected_state", "wrong_reader", "cache_outside_roots", "old_builder"])
def test_norm_preflight_requires_numeric_only_reader_and_managed_source_environment(platform, change):
    node, _, preview, command, code, checks = pending_batch_norm(platform, auto_config={}, composed=True)
    plan = code["automatic_preflight"]["plan"]
    source = plan["datasets"][1]
    projection = source["numeric_io"]
    if change == "missing_projection":
        source.pop("numeric_io")
    elif change == "video_decode":
        projection["video_decode"] = True
    elif change == "image_column":
        projection["image_columns"] = ["observation.images.wrist"]
    elif change == "extra_column":
        projection["columns"].append("video.legacy")
    elif change == "unselected_state":
        projection["state_columns"][0] = "unselected.state"
        projection["columns"][0] = "unselected.state"
    elif change == "wrong_reader":
        projection["reader"] = "lerobot_decoder"
    elif change == "cache_outside_roots":
        plan["training_cache"]["paths"]["TMPDIR"] = "/outside/cache"
    else:
        plan["training_cache"]["builders"] = []
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"


@pytest.mark.parametrize("change", ["missing", "spec", "source", "horizon", "mask", "visuals"])
def test_automatic_norm_requires_actual_custom_composition_model_and_source_evidence(platform, change):
    node, _, preview, command, code, checks = pending_batch_norm(platform,
        auto_config={"model_type": "pi05", "action_horizon": 16, "action_dim": 32}, composed=True)
    dataset = code["automatic_preflight"]["plan"]["datasets"][1]
    evidence = dataset["composition"]
    if change == "missing":
        dataset.pop("composition")
    elif change == "spec":
        evidence["spec"]["normalization"] = "quantile"
    elif change == "source":
        evidence["source"]["sha256"] = "unknown"
    elif change == "horizon":
        evidence["action_horizon"] = 50
    elif change == "mask":
        evidence["action_loss_mask"] = [1.] * 32
    else:
        evidence["image_keys"] = ["base_0_rgb"]
    code["automatic_preflight"]["plan_digest"] = canonical_digest(code["automatic_preflight"]["plan"])
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"


@pytest.mark.parametrize("change", ["lora", "recipe", "saved_recipe", "recipe_index", "legacy_input", "entrypoint", "stage"])
def test_automatic_norm_rejects_training_modes_and_ambiguous_processing(platform, change):
    node, request, *_ = pending_batch_norm(platform, auto_config={})
    request = copy.deepcopy(request)
    if change == "lora":
        request["auto_config"]["finetune_mode"] = "lora"
    elif change == "recipe":
        request["recipe"] = "my_recipe"
    elif change == "saved_recipe":
        request["recipe_revision_id"] = "some-saved-revision"
    elif change == "recipe_index":
        request["norm_datasets"][1].pop("processor")
        request["norm_datasets"][1]["recipe_index"] = 0
    elif change == "legacy_input":
        request["inputs"] = {"dataset": request.pop("norm_datasets")[0]["dataset_asset_id"]}
    elif change == "entrypoint":
        request["entrypoint"] = "scripts/train.py"
    else:
        request["training_stage"] = "value"
    dispatched = len(platform.commands(node))
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 422, response.text
    assert len(platform.commands(node)) == dispatched


@pytest.mark.parametrize("change", ["snapshot_missing", "snapshot_changed", "plan_missing", "plan_changed", "model_missing", "model_changed"])
def test_automatic_norm_preflight_requires_frozen_model_configuration_in_every_evidence_layer(platform, change):
    node, _, preview, command, code, checks = pending_batch_norm(platform, auto_config={})
    plan = code["automatic_preflight"]["plan"]
    if change == "snapshot_missing":
        code.pop("auto_config")
    elif change == "snapshot_changed":
        code["auto_config"] = {**code["auto_config"], "action_horizon": 25}
    elif change == "plan_missing":
        plan.pop("auto_config")
    elif change == "plan_changed":
        plan["auto_config"] = {**plan["auto_config"], "action_dim": 14}
    elif change == "model_missing":
        plan["datasets"][1].pop("model_configuration")
    else:
        plan["datasets"][1]["model_configuration"]["action_dim"] = 14
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] in {"AGENT_AUTO_CONFIG_UNSUPPORTED", "AUTO_CONFIG_MISMATCH", "AGENT_PREFLIGHT_INVALID"}


@pytest.mark.parametrize("change", ["missing_output", "duplicate_source", "wrong_source", "shared_path", "model_output", "missing_source"])
def test_batch_norm_rejects_partial_shared_or_unattributed_outputs(platform, change):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    outputs = batch_outputs(request, output_dir)
    if change == "missing_output":
        outputs.pop()
    elif change == "duplicate_source":
        outputs[1]["metadata"] = dict(outputs[0]["metadata"])
    elif change == "wrong_source":
        outputs[1]["metadata"]["source_dataset_asset_id"] = "unselected-data"
    elif change == "shared_path":
        outputs[1]["path"] = outputs[0]["path"]
    elif change == "model_output":
        outputs[1]["kind"] = "MODEL"
    else:
        outputs[1]["metadata"] = {}
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0, "outputs": outputs})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed["state"] == "FAILED", completed
    assert completed["error"]["code"] in {"OUTPUT_INVALID", "OUTPUT_INCOMPLETE"}
    assert completed["outputs"] == []


@pytest.mark.parametrize("change", ["weight", "norm", "duplicate", "model", "legacy_input", "unknown_role", "other_template", "both_processing", "no_processing"])
def test_batch_norm_request_rejects_training_fields_duplicate_sources_and_mixed_inputs(platform, change):
    _, request, *_ = pending_batch_norm(platform)
    request = copy.deepcopy(request)
    if change == "weight":
        request["norm_datasets"][0]["weight"] = 1.
    elif change == "norm":
        request["norm_datasets"][0]["norm_asset_id"] = "old-norm"
    elif change == "duplicate":
        request["norm_datasets"][1]["dataset_asset_id"] = request["norm_datasets"][0]["dataset_asset_id"]
    elif change == "model":
        request["inputs"] = {"model": "unused-model"}
    elif change == "legacy_input":
        request["inputs"] = {"dataset": request["norm_datasets"][0]["dataset_asset_id"]}
    elif change == "unknown_role":
        request["inputs"] = {"dataset_7": request["norm_datasets"][0]["dataset_asset_id"]}
    elif change == "other_template":
        request["template_id"] = "openpi-sft"
    elif change == "both_processing":
        request["norm_datasets"][0]["processor"] = {"name": "LeRobotAlohaDataConfig", "parameters": {}}
    else:
        del request["norm_datasets"][0]["recipe_index"]
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("change", ["snapshot_ignored", "plan_ignored", "row_processing", "wrong_data_path", "wrong_order", "missing_cache_projection"])
def test_batch_norm_preflight_requires_every_requested_source_and_processing_choice(platform, change):
    node, _, preview, command, code, checks = pending_batch_norm(platform)
    evidence = code["automatic_preflight"]
    plan = evidence["plan"]
    if change == "snapshot_ignored":
        del code["norm_datasets"]
    elif change == "plan_ignored":
        del plan["norm_datasets"]
    elif change == "row_processing":
        plan["datasets"][1]["recipe_index"] = 0
    elif change == "wrong_data_path":
        plan["datasets"][1]["path"] = "/machine/data"
    elif change == "wrong_order":
        plan["datasets"].reverse()
    else:
        plan["training_cache"]["datasets"].pop()
    evidence["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked["status"] == "FAILED", checked
