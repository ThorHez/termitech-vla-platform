"""State transitions shared by HTTP handlers and the scheduler."""

from datetime import datetime, timezone
from pathlib import PurePosixPath

from sqlalchemy import delete, select
from vla_adapters import get_templates, resolve_recipe, validate_parameters
from vla_adapters.dataset_bindings import binding_inputs, normalize_dataset_bindings
from vla_adapters.entrypoints import validate_entrypoint
from vla_adapters.recipe_definitions import recipe_parameters, resolve_definition
from vla_adapters.recap import runtime_adapter_for, validate_training_stage
from vla_protocol import canonical_digest
from vla_protocol.config_preview import validate_generated_config

from .common import APIError, INPUT_KINDS, TERMINAL, absolute_path, digest, now, require_path, snapshot_identity, uid, under_root, valid_snapshot
from .models import Allocation, Asset, Attempt, Command, Event, Node, Preview, Profile, Run
from .scheduler import enqueue, resolve_inputs


def get_required(session, model, identifier, label="Record"):
    value = session.get(model, identifier)
    if value is None:
        raise APIError(404, "NOT_FOUND", f"{label} not found")
    return value


def validate_request(session, data: dict, require_online=True, *, frozen=False) -> tuple[Node, Profile, dict]:
    node = session.scalar(select(Node).where(Node.id == data["node_id"]).with_for_update()
                          .execution_options(populate_existing=True))
    if node is None:
        raise APIError(404, "NOT_FOUND", "Node not found")
    if require_online and node.status != "ONLINE":
        raise APIError(409, "NODE_OFFLINE", "Node must be online before preflight or submission")
    profile = session.get(Profile, (node.id, data["profile_id"]))
    if profile is None or not profile.active:
        raise APIError(422, "PROFILE_NODE_MISMATCH", "Runtime profile does not belong to this node")
    from .profile_rebindings import require_not_pending
    require_not_pending(session, node.id, profile.id, profile.code_dir)
    template = next((t for t in get_templates() if t["id"] == data["template_id"]), None)
    if not template or template["status"] != "AVAILABLE":
        raise APIError(422, "TEMPLATE_UNAVAILABLE", "Template has not been made available")
    try:
        stage, stage_config = validate_training_stage(template["id"], data.get("training_stage"), data.get("stage_config"))
    except ValueError as exc:
        raise APIError(422, str(exc).split(":", 1)[0], str(exc)) from exc
    if stage is not None:
        data["training_stage"], data["stage_config"] = stage, stage_config
    if profile.adapter != runtime_adapter_for(template["id"]):
        raise APIError(422, "PROFILE_TEMPLATE_MISMATCH", "Runtime profile does not support this template")
    if profile.execution_mode != "native":
        raise APIError(422, "EXECUTION_MODE_UNSUPPORTED", "This release supports registered native environments")
    try:
        validate_entrypoint(data.get("entrypoint"))
    except ValueError as exc:
        raise APIError(422, "INVALID_ENTRYPOINT", "训练入口必须是所选项目内的相对 Python 文件路径") from exc
    if data.get("entrypoint") is not None and template["id"] not in {"openpi-sft", "recap"}:
        raise APIError(422, "ENTRYPOINT_UNSUPPORTED", "该训练模板不支持指定 Python 训练入口")
    try:
        if data.get("auto_config") is not None:
            from vla_adapters.automatic_config import automatic_recipe_name, normalize_auto_config
            if template["id"] not in {"openpi-sft", "recap"}:
                raise APIError(422, "AUTO_CONFIG_UNSUPPORTED", "该训练模板不支持自动组装配置")
            if data.get("recipe_revision_id") or data.get("recipe_definition") is not None:
                raise APIError(422, "AUTO_CONFIG_CONFLICT", "自动组装与高级代码配方不能同时使用")
            data["auto_config"] = normalize_auto_config(data["auto_config"])
            recipe = automatic_recipe_name(data["auto_config"], data.get("training_stage"))
            if data.get("recipe") not in {None, recipe}:
                raise APIError(422, "AUTO_CONFIG_CONFLICT", "自动组装不能同时指定其他代码配方")
            data["recipe"] = recipe
        elif data.get("recipe_revision_id"):
            from .recipe_library import get_recipe_definition
            definition = get_recipe_definition(session, data["recipe_revision_id"], node.id, profile.id)
            if data.get("recipe_definition") is not None and data["recipe_definition"] != definition:
                raise APIError(409, "RECIPE_DEFINITION_CHANGED", "配方内容与冻结版本不一致，请重新预检")
            data["recipe_definition"] = definition
            data["recipe"] = resolve_definition({"id": profile.id, "adapter": profile.adapter,
                "metadata": profile.metadata_json}, definition, data.get("recipe"), require_validated=False)
        else:
            if data.get("recipe_definition") is not None:
                raise APIError(422, "RECIPE_DEFINITION_INVALID", "自定义配方必须使用中台保存的版本")
            data["recipe"] = resolve_recipe({"adapter": profile.adapter, "recipe": profile.recipe,
                "metadata": profile.metadata_json}, data.get("recipe"), require_validated=False)
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        messages = {"RECIPE_REQUIRED": "请在创建训练任务时选择训练配方",
                    "RECIPE_INVALID": "训练配方必须是代码中已有的命名配置，仅可包含字母、数字、下划线或连字符",
                    "INVALID_RECIPE": "训练配方必须是代码中已有的命名配置，仅可包含字母、数字、下划线或连字符",
                    "RECIPE_DEFINITION_INVALID": "保存的配方版本无效，请重新选择",
                    "RECIPE_DEFINITION_MISMATCH": "保存的配方不属于所选训练环境",
                    "AUTO_CONFIG_INVALID": "自动组装的模型、微调方式或模型维度配置无效",
                    "PROFILE_TEMPLATE_MISMATCH": "所选训练环境不支持该训练模板"}
        if code not in messages:
            raise
        raise APIError(422, code, messages[code]) from exc
    if data["gpu_count"] < template.get("min_gpus", 1):
        raise APIError(422, "GPU_REQUIRED", "The selected template requires a GPU")
    if data["gpu_count"] > len(node.gpus):
        raise APIError(422, "GPU_CAPACITY_EXCEEDED", "Requested GPU count exceeds this node's reported devices")
    try:
        parameters = validate_parameters(template["id"], {**recipe_parameters(data.get("recipe_definition")), **data["parameters"]})
    except ValueError as exc:
        raise APIError(422, "INVALID_PARAMETERS", str(exc)) from exc
    if data["gpu_count"] and parameters["batch_size"] % data["gpu_count"]:
        raise APIError(422, "INVALID_BATCH_SIZE", "Batch size must be divisible by GPU count")
    try:
        bindings = normalize_dataset_bindings(data.get("dataset_bindings"))
    except ValueError as exc:
        raise APIError(422, "DATASET_BINDINGS_INVALID", "数据集、归一化参数、配方位置和混采权重配置无效") from exc
    input_kinds = INPUT_KINDS
    required_inputs = set(template["required_inputs"])
    if template["id"] == "recap" and bindings is None:
        raise APIError(422, "DATASET_BINDINGS_REQUIRED", "ReCap 训练需要逐组配置数据集与归一化参数")
    if data.get("auto_config") is not None and (not bindings or any(row.get("processor") is None for row in bindings)):
        raise APIError(422, "AUTO_CONFIG_PROCESSOR_REQUIRED", "自动组装需要每组数据明确绑定已登记处理配置或独立数据处理器")
    if bindings is not None:
        if template["id"] not in {"openpi-sft", "recap"}:
            raise APIError(422, "DATASET_BINDINGS_UNSUPPORTED", "该训练模板不支持多数据集绑定")
        if {"dataset", "norm"} & data["inputs"].keys():
            raise APIError(422, "DATASET_BINDINGS_CONFLICT", "多数据集绑定不能与单数据集或归一化参数配置同时使用")
        bound_inputs = binding_inputs(bindings)
        provided = {role: asset_id for role, asset_id in data["inputs"].items() if role != "model"}
        # Initial requests contain only the model; a frozen preview contains the
        # expanded roles, which must still exactly identify the same row assets.
        if (frozen or provided) and provided != bound_inputs:
            raise APIError(422, "DATASET_BINDINGS_MISMATCH", "数据集绑定与资产输入不一致，请重新配置并预检")
        data["dataset_bindings"] = bindings
        data["inputs"] = {**data["inputs"], **bound_inputs}
        input_kinds = {"model": "MODEL", **{role: "DATASET" if role.startswith("dataset_") else "NORM" for role in bound_inputs}}
        required_inputs = {"model", *bound_inputs}
    if set(data["inputs"]) - input_kinds.keys():
        raise APIError(422, "UNKNOWN_INPUT_ROLE", "Allowed input roles are dataset, model and norm")
    if required_inputs - data["inputs"].keys():
        raise APIError(422, "INPUT_REQUIRED", "A required local training asset is missing")
    for role, asset_id in data["inputs"].items():
        asset = get_required(session, Asset, asset_id, "Asset")
        if asset.node_id != node.id:
            raise APIError(422, "ASSET_NODE_MISMATCH", "All assets and the runtime must be on the selected node")
        if asset.kind != input_kinds[role]:
            raise APIError(422, "ASSET_KIND_MISMATCH", "Asset type does not match the input role")
        if asset.state not in {"REGISTERED", "READY"}:
            raise APIError(409, "INPUT_UNAVAILABLE", "An input asset is archived or unavailable")
        require_path(asset.path, node.roots)
    return node, profile, parameters


AUTOMATIC_PREFLIGHT_CHECKS = {"recipe_configuration", "data_contract", "norm_contract", "model_checkpoint",
                              "training_entrypoint", "training_cache", "output_storage", "gpu_capacity", "runtime", "local_paths",
                              "code_version", "adapter"}
CACHE_PATH_KEYS = {"HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
                   "TRANSFORMERS_CACHE", "TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME",
                   "JAX_COMPILATION_CACHE_DIR", "OPENPI_DATA_HOME"}


def training_cache_evidence_valid(cache, datasets, roots: list[str]) -> bool:
    """Older metadata-only checks cannot authorize a new training attempt."""
    def local(value):
        return isinstance(value, str) and 0 < len(value) <= 4096 and under_root(value, roots)

    if (not isinstance(cache, dict) or not local(cache.get("datasets_cache"))
            or not isinstance(datasets, list) or not 1 <= len(datasets) <= 64):
        return False
    paths, builders = cache.get("paths"), cache.get("builders")
    if (not isinstance(paths, dict) or not 1 <= len(paths) <= len(CACHE_PATH_KEYS)
            or not set(paths) <= CACHE_PATH_KEYS or paths.get("HF_DATASETS_CACHE") != cache["datasets_cache"]
            or any(not local(value) for value in paths.values())
            or not isinstance(builders, list) or len(builders) != len(datasets)):
        return False
    return all(isinstance(dataset, dict) and isinstance(builder, dict)
               and builder.get("path") == dataset.get("path") and local(builder.get("path")) and local(builder.get("cache_dir"))
               and under_root(builder["cache_dir"], [cache["datasets_cache"]])
               for dataset, builder in zip(datasets, builders))


def entrypoint_snapshot_error(requested_entrypoint: str | None, snapshot) -> dict | None:
    """Explicit scripts need evidence from a current Agent at each boundary."""
    evidence = snapshot.get("automatic_preflight") if isinstance(snapshot, dict) else None
    plan = evidence.get("plan") if isinstance(evidence, dict) else None
    if requested_entrypoint is None:
        if ((isinstance(snapshot, dict) and snapshot.get("requested_entrypoint") is not None)
                or (isinstance(plan, dict) and plan.get("requested_entrypoint") is not None)):
            return {"code": "ENTRYPOINT_MISMATCH", "message": "Agent 返回了本次任务未指定的训练入口，请重新预检"}
        return None
    if (not isinstance(snapshot, dict) or "requested_entrypoint" not in snapshot
            or not isinstance(plan, dict) or "requested_entrypoint" not in plan):
        return {"code": "AGENT_ENTRYPOINT_UNSUPPORTED", "message": "Agent 未确认所选训练入口，请更新 Agent 后重新预检"}
    if (snapshot["requested_entrypoint"] != requested_entrypoint or plan["requested_entrypoint"] != requested_entrypoint
            or not isinstance(plan.get("entrypoint"), str) or not isinstance(snapshot.get("code_dir"), str)
            or not under_root(plan["entrypoint"], [snapshot["code_dir"]])):
        return {"code": "ENTRYPOINT_MISMATCH", "message": "Agent 实际训练入口与所选项目或入口不一致，请重新预检"}
    return None


def dataset_bindings_snapshot_error(requested_bindings, snapshot) -> dict | None:
    evidence = snapshot.get("automatic_preflight") if isinstance(snapshot, dict) else None
    plan = evidence.get("plan") if isinstance(evidence, dict) else None
    if requested_bindings is None:
        if ((isinstance(snapshot, dict) and snapshot.get("dataset_bindings") is not None)
                or (isinstance(plan, dict) and plan.get("dataset_bindings") is not None)):
            return {"code": "DATASET_BINDINGS_MISMATCH", "message": "Agent 返回了本次任务未指定的数据集绑定，请重新预检"}
        return None
    if (not isinstance(snapshot, dict) or "dataset_bindings" not in snapshot
            or not isinstance(plan, dict) or "dataset_bindings" not in plan):
        return {"code": "AGENT_DATASET_BINDINGS_UNSUPPORTED", "message": "Agent 未确认全部数据集与归一化参数绑定，请更新 Agent 后重新预检"}
    if (canonical_digest(snapshot["dataset_bindings"]) != canonical_digest(requested_bindings)
            or canonical_digest(plan["dataset_bindings"]) != canonical_digest(requested_bindings)):
        return {"code": "DATASET_BINDINGS_MISMATCH", "message": "Agent 数据集绑定与预检冻结的配置不一致，请重新预检"}
    return None


def training_stage_snapshot_error(requested_stage, requested_config, snapshot) -> dict | None:
    """A staged task cannot be accepted from an Agent that ignored its phase."""
    evidence = snapshot.get("automatic_preflight") if isinstance(snapshot, dict) else None
    plan = evidence.get("plan") if isinstance(evidence, dict) else None
    if requested_stage is None:
        if (isinstance(snapshot, dict) and (snapshot.get("training_stage") is not None or snapshot.get("stage_config") is not None)
                or isinstance(plan, dict) and (plan.get("training_stage") is not None or plan.get("stage_config") is not None)):
            return {"code": "TRAINING_STAGE_MISMATCH", "message": "Agent 返回了本次任务未指定的训练阶段，请重新预检"}
        return None
    if (not isinstance(snapshot, dict) or not isinstance(plan, dict)
            or any(field not in value for value in (snapshot, plan) for field in ("training_stage", "stage_config"))):
        return {"code": "AGENT_TRAINING_STAGE_UNSUPPORTED", "message": "Agent 未确认训练阶段和阶段配置，请更新 Agent 后重新预检"}
    if any(canonical_digest(value["training_stage"]) != canonical_digest(requested_stage)
           or canonical_digest(value["stage_config"]) != canonical_digest(requested_config) for value in (snapshot, plan)):
        return {"code": "TRAINING_STAGE_MISMATCH", "message": "Agent 实际训练阶段或阶段配置与本次任务不一致，请重新预检"}
    return None


def automatic_preflight_error(session, request: dict, snapshot, checks) -> dict | None:
    """Require Agent evidence for this exact task; discovery is not execution proof."""
    if request["template_id"] not in {"openpi-sft", "recap"}:
        return None
    from .storage_configurations import snapshot_error as storage_snapshot_error
    storage_error = storage_snapshot_error(session, request["node_id"],
        snapshot.get("automatic_preflight") if isinstance(snapshot, dict) else None)
    if storage_error:
        return storage_error
    requested_auto = request.get("auto_config")
    if requested_auto is not None and (not isinstance(snapshot, dict) or snapshot.get("auto_config") is None):
        return {"code": "AGENT_AUTO_CONFIG_UNSUPPORTED", "message": "Agent 未确认自动组装配置，请更新 Agent 后重新预检"}
    if isinstance(snapshot, dict) and canonical_digest(snapshot.get("auto_config")) != canonical_digest(requested_auto):
        return {"code": "AUTO_CONFIG_MISMATCH", "message": "Agent 自动组装配置与本次任务不一致，请重新预检"}
    stage_error = training_stage_snapshot_error(request.get("training_stage"), request.get("stage_config"), snapshot)
    if stage_error:
        return stage_error
    entrypoint_error = entrypoint_snapshot_error(request.get("entrypoint"), snapshot)
    if entrypoint_error:
        return entrypoint_error
    binding_error = dataset_bindings_snapshot_error(request.get("dataset_bindings"), snapshot)
    if binding_error:
        return binding_error
    evidence = snapshot.get("automatic_preflight") if isinstance(snapshot, dict) else None
    if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1 or evidence.get("scope") != "configuration_and_local_assets":
        return {"code": "AGENT_PREFLIGHT_UNSUPPORTED", "message": "Agent 未返回自动预检证据，请更新 Agent 后重新预检"}
    if (not isinstance(checks, list) or any(not isinstance(check, dict) or check.get("ok") is not True for check in checks)
            or not AUTOMATIC_PREFLIGHT_CHECKS <= {check.get("name") for check in checks if isinstance(check.get("name"), str)}):
        return {"code": "AGENT_PREFLIGHT_INCOMPLETE", "message": "Agent 自动预检未完成全部必需检查，请更新 Agent 后重新预检"}
    plan = evidence.get("plan")
    if not isinstance(plan, dict) or evidence.get("plan_digest") != canonical_digest(plan):
        return {"code": "AGENT_PREFLIGHT_INVALID", "message": "Agent 自动预检计划不完整或摘要不一致，请重新预检"}
    definition = request.get("recipe_definition")
    resolved_inputs = resolve_inputs(session, request)
    expected = {"recipe": request["recipe"], "recipe_definition_digest": definition["digest"] if definition else None,
                "requested_inputs": {role: value["path"] for role, value in resolved_inputs.items()},
                "parameters": request["parameters"], "gpu_count": request["gpu_count"]}
    if request.get("entrypoint") is not None or "requested_entrypoint" in plan:
        expected["requested_entrypoint"] = request.get("entrypoint")
    if request.get("dataset_bindings") is not None or "dataset_bindings" in plan:
        expected["dataset_bindings"] = request.get("dataset_bindings")
    if requested_auto is not None or "auto_config" in plan:
        expected["auto_config"] = requested_auto
    if request.get("training_stage") is not None:
        expected.update(training_stage=request["training_stage"], stage_config=request["stage_config"])
    if any(key not in plan or canonical_digest(plan[key]) != canonical_digest(value) for key, value in expected.items()):
        return {"code": "AGENT_PREFLIGHT_MISMATCH", "message": "Agent 自动预检的配方、资产或参数与本次任务不一致，请重新预检"}
    def local_path(value):
        return isinstance(value, str) and absolute_path(value)

    inputs = plan.get("inputs")
    datasets, model, storage = plan.get("datasets"), plan.get("model"), plan.get("storage")
    if (not isinstance(inputs, dict) or set(inputs) != set(expected["requested_inputs"])
            or any(not local_path(path) for path in inputs.values())
            or not local_path(plan.get("entrypoint")) or not isinstance(datasets, list) or not datasets
            or any(not isinstance(dataset, dict) or not local_path(dataset.get("path"))
                   or not local_path(dataset.get("norm_path")) for dataset in datasets)
            or not isinstance(model, dict) or not local_path(model.get("path"))
            or model.get("checkpoint_path") != inputs.get("model") or not isinstance(model.get("metadata"), list)
            or type(model.get("parameter_bytes")) is not int or model["parameter_bytes"] <= 0
            or not isinstance(storage, dict) or any(type(storage.get(key)) is not int or storage[key] < 0
                   for key in ("required_bytes", "reference_checkpoint_bytes", "model_params_bytes"))):
        return {"code": "AGENT_PREFLIGHT_INVALID", "message": "Agent 自动预检未记录完整训练入口和数据集计划，请重新预检"}
    bindings = request.get("dataset_bindings")
    if bindings is not None:
        for dataset, binding in zip(datasets, bindings):
            processor = binding.get("processor")
            if processor is None:
                continue
            if dataset.get("processor") is None:
                return {"code": "AGENT_DATA_PROCESSOR_UNSUPPORTED", "message": "Agent 未记录独立数据处理器的实际配置，请更新 Agent 并重新预检"}
            from .data_processors import _source as processor_source
            identity = dataset.get("processor_identity")
            if (canonical_digest(dataset.get("processor")) != canonical_digest(processor)
                    or dataset.get("factory") != processor["name"] or not isinstance(identity, dict)
                    or identity.get("name") != processor["name"] or not isinstance(identity.get("module"), str)
                    or not 1 <= len(identity["module"]) <= 512 or processor_source(identity.get("source")) is None):
                return {"code": "DATA_PROCESSOR_MISMATCH", "message": "实际数据处理器或参数与本次选择不一致，请重新预检"}
    if request["template_id"] == "recap":
        contract = plan.get("recap_stage_contract")
        contract_datasets = contract.get("datasets") if isinstance(contract, dict) else None
        if (not any(check.get("name") == "recap_stage" and check.get("ok") is True for check in checks)
                or not isinstance(contract, dict) or contract.get("stage") != request["training_stage"]
                or not isinstance(contract.get("model_type"), str) or not contract["model_type"]
                or (contract["model_type"] == "Pi0ValueConfig") != (request["training_stage"] == "value")
                or not isinstance(contract.get("loader_type"), str) or not contract["loader_type"]
                or not isinstance(contract_datasets, list) or len(contract_datasets) != len(datasets)
                or any(not isinstance(item, dict) or item.get("path") != data.get("path")
                       or not isinstance(item.get("required_columns"), list) or not item["required_columns"]
                       or ("value_target" if request["training_stage"] == "value" else "is_positive") not in item["required_columns"]
                       or any(not isinstance(column, str) or not column for column in item["required_columns"])
                       for item, data in zip(contract_datasets, datasets))):
            return {"code": "AGENT_RECAP_STAGE_INVALID", "message": "Agent 未完成所选 ReCap 阶段的模型、数据加载器和数据字段检查，请更新 Agent 后重新预检"}
    def binding_paths_match(index, dataset):
        norm_input = inputs[f"norm_{index}"]
        norm_matches = (dataset["norm_asset_path"] == norm_input if "norm_asset_path" in dataset
                        else dataset["norm_path"] in {norm_input, norm_input.rstrip("/") + "/norm_stats.json"})
        return dataset["path"] == inputs[f"dataset_{index}"] and norm_matches

    if bindings is not None and (len(datasets) != len(bindings)
            or any(not binding_paths_match(index, dataset) for index, dataset in enumerate(datasets))
            or canonical_digest(plan.get("weights")) != canonical_digest([row["weight"] for row in bindings])):
        return {"code": "DATASET_BINDINGS_MISMATCH", "message": "Agent 实际训练数据集、归一化参数或混采权重与逐行绑定不一致，请重新预检"}
    if plan.get("input_authority") != "registered_assets":
        return {"code": "AGENT_INPUT_AUTHORITY_UNSUPPORTED", "message": "Agent 未证明实际输入来自所选资产，请更新 Agent 并重新预检"}
    pairs = ([(f"dataset_{index}", f"norm_{index}") for index in range(len(bindings))]
             if bindings is not None else [("dataset", "norm")])
    if len(datasets) != len(pairs):
        return {"code": "INPUT_BINDING_MISMATCH", "message": "实际训练包含未登记或未选择的数据集，请逐组绑定资产后重新预检"}
    for index, (dataset, (dataset_role, norm_role)) in enumerate(zip(datasets, pairs)):
        norm_input = inputs.get(norm_role)
        norm_matches = (dataset.get("norm_asset_path") == norm_input if "norm_asset_path" in dataset
                        else isinstance(norm_input, str) and dataset.get("norm_path") in {norm_input, norm_input.rstrip("/") + "/norm_stats.json"})
        if (dataset.get("path") != inputs.get(dataset_role) or not norm_matches
                or dataset.get("dataset_asset_id") != resolved_inputs[dataset_role]["id"]
                or dataset.get("norm_asset_id") != resolved_inputs[norm_role]["id"]):
            return {"code": "INPUT_BINDING_MISMATCH", "message": "实际数据集、归一化参数与所选资产身份不一致，请重新预检"}
        declared = request.get("dataset_profiles", {}).get(dataset_role)
        if declared is not None:
            if dataset.get("registered_profile") is None:
                return {"code": "AGENT_DATASET_PROFILE_UNSUPPORTED", "message": "Agent 未核验登记的数据集描述，请更新 Agent 并重新预检"}
            if canonical_digest(dataset["registered_profile"]) != canonical_digest(declared):
                return {"code": "DATASET_PROFILE_MISMATCH", "message": "Agent 实际核验的数据集描述与本次冻结登记不一致，请重新预检"}
            registered_fps = declared.get("fps")
            if type(registered_fps) in (int, float) and 0 < registered_fps <= 1000:
                if "effective_fps" not in dataset or dataset.get("fps_source") != "registered":
                    return {"code": "AGENT_DATASET_FPS_UNSUPPORTED", "message": "Agent 未证明训练按登记的采样频率加载数据，请更新 Agent 与训练代码后重新预检"}
                if type(dataset["effective_fps"]) not in (int, float) or dataset["effective_fps"] != registered_fps:
                    return {"code": "DATASET_FPS_MISMATCH", "message": "实际训练采样频率与本次冻结登记不一致，请重新预检"}
        processor = bindings[index].get("processor") if bindings is not None else None
        expected_spec = (processor.get("parameters", {}).get("composition")
                         if isinstance(processor, dict) and processor.get("name") == "ComposableDataConfig" else None)
        if expected_spec is not None:
            from vla_adapters.composition_spec import composition_digest, validate_composition_profile_input
            from .data_processors import _source as processor_source
            try:
                validate_composition_profile_input(declared, expected_spec)
            except ValueError as exc:
                return {"code": str(exc).split(":", 1)[0], "message": str(exc)}
            composed = dataset.get("composition")
            if (not isinstance(composed, dict) or type(composed.get("schema_version")) is not int
                    or composed["schema_version"] != 1 or composed.get("digest") != composition_digest(expected_spec)
                    or canonical_digest(composed.get("spec")) != canonical_digest(expected_spec)
                    or composed.get("training_stage") != (request.get("training_stage") or "sft")
                    or composed.get("model_type") not in {"Pi0Config", "Pi0GripperConfig", "Pi0ValueConfig"}
                    or (composed.get("model_type") == "Pi0ValueConfig") != (request.get("training_stage") == "value")
                    or type(composed.get("action_horizon")) is not int or composed["action_horizon"] <= 0
                    or type(composed.get("action_dim")) is not int or not 1 <= composed["action_dim"] <= 4096
                    or processor_source(composed.get("source")) is None
                    or composed["source"]["path"] not in {"src/openpi/training/composed_processing.py", "openpi/training/composed_processing.py"}
                    or dataset.get("factory") != "ComposableDataConfig"):
                return {"code": "AGENT_COMPOSITION_UNSUPPORTED", "message": "Agent 未完成组合处理的配置、模型和源码核验，请更新 Agent 与训练代码后重新预检"}
            raw_state = sum(field["width"] for field in expected_spec["state_fields"])
            raw_action = sum(field["width"] for field in expected_spec["action_layout"])
            dimensions = {"raw_state_dim": raw_state, "raw_action_dim": raw_action,
                          "state_history": expected_spec["state_history"], "normalization": expected_spec["normalization"]}
            shape = dataset.get("actions_shape")
            sequence = [expected_spec["action_key"]] if shape == [raw_action] else []
            mask = [1.] * raw_action + [0.] * (composed["action_dim"] - raw_action)
            if (raw_action > composed["action_dim"] or type(composed.get("pi05")) is not bool
                    or any(canonical_digest(composed.get(key)) != canonical_digest(value) for key, value in dimensions.items())
                    or not isinstance(composed.get("action_metadata_sha256"), str)
                    or len(composed["action_metadata_sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in composed["action_metadata_sha256"])
                    or composed.get("action_metadata_sha256") != dataset.get("metadata_sha256")
                    or shape not in ([raw_action], [composed["action_horizon"], raw_action])
                    or canonical_digest(composed.get("action_sequence_keys")) != canonical_digest(sequence)
                    or canonical_digest(composed.get("action_loss_mask")) != canonical_digest(mask)):
                return {"code": "AGENT_COMPOSITION_UNSUPPORTED", "message": "Agent 未证明组合字段宽度、动作采样和真实动作损失掩码与本次训练数据绑定一致，请更新后重新预检"}
            image_keys, depth_keys = composed.get("image_keys"), composed.get("depth_keys")
            slots = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
            if (not isinstance(image_keys, list) or not isinstance(depth_keys, list)
                    or not all(isinstance(key, str) and key in slots for key in image_keys + depth_keys)
                    or len(set(image_keys + depth_keys)) != len(image_keys + depth_keys)
                    or any(camera["slot"] not in (depth_keys if camera["kind"] == "depth" else image_keys)
                           for camera in expected_spec["cameras"])):
                return {"code": "AGENT_COMPOSITION_UNSUPPORTED", "message": "Agent 未证明所选模型使用本次选择的全部视觉，或深度/RGB 视图分类不一致，请重新预检"}
        elif dataset.get("composition") is not None or dataset.get("factory") == "ComposableDataConfig":
            return {"code": "AGENT_COMPOSITION_UNSUPPORTED", "message": "Agent 返回了本次训练数据绑定未选择的组合处理配置，请重新预检"}
    if model.get("asset_id") != resolved_inputs["model"]["id"]:
        return {"code": "INPUT_BINDING_MISMATCH", "message": "实际初始化模型与所选模型资产身份不一致，请重新预检"}
    node = session.get(Node, request["node_id"])
    if node is None or not training_cache_evidence_valid(plan.get("training_cache"), datasets, node.roots):
        return {"code": "AGENT_PREFLIGHT_INVALID", "message": "Agent 自动预检未记录完整训练缓存与逐个数据集构建检查，请更新 Agent 后重新预检"}
    generated = plan.get("generated_config")
    if generated is None:
        return {"code": "AGENT_CONFIG_PREVIEW_UNSUPPORTED", "message": "Agent 未生成可核对的 Python config，请更新 Agent 后重新预检"}
    try:
        validate_generated_config(generated)
    except ValueError:
        return {"code": "AGENT_CONFIG_PREVIEW_INVALID", "message": "生成的 Python config 内容或摘要无效，请重新预检"}
    return None


def config_confirmation_error(request: dict, snapshot: dict, confirmed_sha256: str | None) -> dict | None:
    """Submission must acknowledge the exact code the successful preflight produced."""
    if request["template_id"] not in {"openpi-sft", "recap"}:
        return None
    generated = snapshot.get("automatic_preflight", {}).get("plan", {}).get("generated_config")
    if confirmed_sha256 is None:
        return {"code": "CONFIG_CONFIRMATION_REQUIRED", "message": "请先检查本次生成的 Python config 并确认，再启动训练"}
    if not isinstance(generated, dict) or confirmed_sha256 != generated.get("sha256"):
        return {"code": "CONFIG_CONFIRMATION_MISMATCH", "message": "确认的 config 与本次预检不一致，请重新查看并确认"}
    return None


def sync_profiles(session, node: Node, entries: list[dict]) -> dict:
    """Update node-owned environments without mutating a live run's runtime."""
    if len({entry["id"] for entry in entries}) != len(entries):
        return {"status": "REJECTED", "code": "DUPLICATE_PROFILE", "message": "Profile IDs must be unique"}
    for entry in entries:
        try:
            require_path(entry["code_dir"], node.roots)
            require_path(entry["python"])
        except APIError as exc:
            return {"status": "REJECTED", "code": exc.code, "message": exc.message}
    existing = {p.id: p for p in session.scalars(select(Profile).where(Profile.node_id == node.id).with_for_update())}
    incoming = {entry["id"]: entry for entry in entries}
    from .profile_rebindings import sync_blocked_profiles
    rebinding = sync_blocked_profiles(session, node.id, incoming)
    changed = set()
    for profile_id in existing.keys() | incoming.keys():
        current, proposed = existing.get(profile_id), incoming.get(profile_id)
        if current is None:
            changed.add(profile_id)
        elif proposed is None:
            if current.active:
                changed.add(profile_id)
        elif not current.active or any(getattr(current, "metadata_json" if key == "metadata" else key) != value for key, value in proposed.items()):
            changed.add(profile_id)
    used = set(session.scalars(select(Run.profile_id).where(Run.node_id == node.id, Run.state.not_in(TERMINAL), Run.profile_id.in_(changed)))) if changed else set()
    applied = changed - used - rebinding
    for profile_id in applied:
        current, proposed = existing.get(profile_id), incoming.get(profile_id)
        if proposed is None:
            current.active = False  # Historical runs still reference this row.
        else:
            values = {("metadata_json" if key == "metadata" else key): value for key, value in proposed.items()}
            if current is None:
                session.add(Profile(node_id=node.id, active=True, **values))
            else:
                for key, value in values.items():
                    setattr(current, key, value)
                current.active = True
    if applied:
        for preview in session.scalars(select(Preview).where(Preview.node_id == node.id, Preview.status.in_(["PENDING", "READY"])).with_for_update()):
            if preview.request["profile_id"] in applied:
                preview.status = "FAILED"
                preview.error = {"code": "PROFILE_CHANGED_SINCE_PREFLIGHT", "message": "Runtime configuration changed; run preflight again"}
    if rebinding & changed:
        return {"status": "BLOCKED", "code": "PROFILE_REBIND_PENDING", "message": "Runtime profile change awaits a persisted Agent acknowledgement",
                "blocked_profile_ids": sorted(used | rebinding & changed), "updated_profile_ids": sorted(applied)}
    if used:
        return {"status": "BLOCKED", "code": "PROFILE_IN_USE", "message": "Active runs retain their registered runtime configuration",
                "blocked_profile_ids": sorted(used), "updated_profile_ids": sorted(applied)}
    return {"status": "APPLIED", "updated_profile_ids": sorted(applied)}


def apply_ack(session, command: Command, status: str, result: dict) -> None:
    if command.status != "PENDING":
        if command.result == {"terminal_event_received": True}:
            return
        if command.status != status or digest(command.result) != digest(result):
            raise APIError(409, "ACK_CONFLICT", "This command already has a different acknowledgement")
        return
    command.status, command.result, command.acknowledged_at = status, result, now()
    if command.type != "PREFLIGHT":
        # Acknowledgement proves delivery, never process exit. Reservations are
        # released exclusively by a terminal execution event.
        return
    preview = session.scalar(select(Preview).where(Preview.id == command.payload["preview_id"]).with_for_update())
    if preview.status != "PENDING":
        return
    if preview.expires_at < now():
        preview.status = "EXPIRED"
        preview.error = {"code": "PREFLIGHT_EXPIRED", "message": "Preflight expired"}
        return
    checks = result.get("checks", [])
    snapshot = result.get("code_snapshot")
    error = result.get("error")
    if not isinstance(checks, list) or not checks or any(not isinstance(c, dict) or c.get("ok") is not True for c in checks):
        error = error or {"code": "PREFLIGHT_FAILED", "message": "Node validation failed or returned no checks"}
    if not valid_snapshot(snapshot):
        error = error or {"code": "CODE_VERSION_UNAVAILABLE", "message": "Complete actual code version could not be established"}
    if isinstance(snapshot, dict) and "recipe" in preview.request:
        if "recipe" not in snapshot:
            error = error or {"code": "AGENT_RECIPE_UNSUPPORTED", "message": "Agent 未返回预检使用的训练配方，请更新 Agent 后重新预检"}
        elif snapshot["recipe"] != preview.request["recipe"]:
            error = error or {"code": "RECIPE_MISMATCH", "message": "Agent 预检的训练配方与所选配方不一致，请更新 Agent 后重新预检"}
        elif preview.request.get("recipe_definition") != snapshot.get("recipe_definition"):
            error = error or {"code": "RECIPE_DEFINITION_MISMATCH", "message": "Agent 未确认本次自定义配方版本，请更新 Agent 并重新预检"}
    error = error or automatic_preflight_error(session, preview.request, snapshot, checks)
    if status == "REJECTED":
        error = error or {"code": "PREFLIGHT_REJECTED", "message": "Agent rejected the preflight command"}
    preview.checks = checks if isinstance(checks, list) else []
    preview.code_snapshot = snapshot if isinstance(snapshot, dict) else None
    preview.error = error
    preview.status = "FAILED" if error else "READY"
    if not error:
        for asset_id in preview.request["inputs"].values():
            asset = session.get(Asset, asset_id)
            if asset.state == "REGISTERED":
                asset.state = "READY"


def canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError) as exc:
        raise APIError(422, "INVALID_TIMESTAMP", "Event timestamps must include a timezone") from exc


def event_failure(session, attempt, run, code: str, message: str) -> None:
    run.error = attempt.error = {"code": code, "message": message}
    if run.state != "CANCELLING":
        enqueue(session, run.node_id, "CANCEL", {"attempt_id": attempt.id, "execution_token": attempt.execution_token})
    run.state = attempt.state = "CANCELLING"


def output_path(node: Node, attempt: Attempt, path) -> str | None:
    if (not isinstance(path, str) or len(path) > 4096
            or not isinstance(attempt.output_dir, str) or not under_root(attempt.output_dir, [node.output_root])
            or not under_root(path, [attempt.output_dir]) or not under_root(path, [node.output_root])):
        return None
    return str(PurePosixPath(path))


def validated_output(node: Node, attempt: Attempt, run: Run, output) -> dict | None:
    """Bind a reported artifact to this execution, never another node or attempt."""
    if not isinstance(output, dict):
        return None
    kind, name, path = output.get("kind"), output.get("name"), output_path(node, attempt, output.get("path"))
    version, metadata = output.get("version", "1"), output.get("metadata", {})
    if (not isinstance(kind, str) or kind not in set(INPUT_KINDS.values())
            or not isinstance(name, str) or not name.strip() or len(name) > 200 or "\x00" in name
            or not isinstance(version, str) or not version.strip() or len(version) > 200 or "\x00" in version
            or path is None
            or not isinstance(metadata, dict)):
        return None
    expected = {"node_id": node.id, "origin_attempt_id": attempt.id,
                "source_attempt_id": attempt.id, "source_run_id": run.id}
    if any(key in output and output[key] != value for key, value in expected.items()):
        return None
    return {"kind": kind, "name": name, "path": path, "version": version, "metadata": metadata}


def register_outputs(session, node: Node, attempt: Attempt, outputs: list[dict]) -> None:
    """Reuse an execution's checkpoint asset across scans, restarts and completion."""
    existing = {(asset.kind, str(PurePosixPath(asset.path)), asset.version): asset
                for asset in session.scalars(select(Asset).where(Asset.origin_attempt_id == attempt.id))}
    for output in outputs:
        identity = (output["kind"], output["path"], output["version"])
        asset = existing.get(identity)
        if asset is None:
            asset = Asset(id=uid(), node_id=node.id, name=output["name"], kind=output["kind"],
                          path=output["path"], version=output["version"], state="READY",
                          metadata_json=output["metadata"], created_at=now(), origin_attempt_id=attempt.id)
            session.add(asset)
            existing[identity] = asset
        else:
            asset.name = output["name"]
            asset.metadata_json = {**asset.metadata_json, **output["metadata"]}
            if asset.state == "UNAVAILABLE":
                asset.state = "READY"
                asset.metadata_json = {**asset.metadata_json, "availability": "available"}


def mark_removed_outputs(session, node: Node, attempt: Attempt, paths, timestamp: str) -> bool:
    """Keep checkpoint history while marking confirmed retention-policy removals."""
    if not isinstance(paths, list) or len(paths) > 1000:
        return False
    valid_paths = {value for path in paths if (value := output_path(node, attempt, path)) is not None}
    changed = False
    if valid_paths:
        for asset in session.scalars(select(Asset).where(Asset.origin_attempt_id == attempt.id,
                Asset.node_id == node.id, Asset.kind == "MODEL", Asset.path.in_(valid_paths))):
            asset.state = "UNAVAILABLE"
            asset.metadata_json = {**asset.metadata_json, "availability": "removed", "removed_at": timestamp}
            changed = True
    return changed


def reduce_event(session, node: Node, attempt: Attempt, event: Event) -> None:
    run = session.scalar(select(Run).where(Run.id == attempt.run_id).with_for_update())
    data = event.data
    if event.type == "LOG":
        if not isinstance(data.get("text"), str) or len(data["text"]) > 100000:
            raise APIError(422, "INVALID_EVENT", "Log events need bounded text")
        return
    if event.seq < attempt.last_event_seq:
        return
    if event.type == "METRIC":
        attempt.latest_metrics = {**attempt.latest_metrics, **data, "timestamp": event.timestamp}
        return
    if event.type == "RESOURCE":
        attempt.resources = {**data, "timestamp": event.timestamp}
        return
    if attempt.state in TERMINAL:
        if event.type in TERMINAL and event.type != attempt.state:
            raise APIError(409, "TERMINAL_STATE_CONFLICT", "Attempt already has a different terminal state")
        return
    if event.type == "ARTIFACT":
        # Discovery reports are informational. Bad entries must not reject the
        # event batch, finish an execution or release its device reservations.
        latest = session.scalar(select(Attempt.id).where(Attempt.run_id == run.id)
                                .order_by(Attempt.number.desc()).limit(1))
        if (attempt.id != latest or attempt.state not in {"RUNNING", "CANCELLING", "UNKNOWN"}
                or run.state not in {"RUNNING", "CANCELLING", "UNKNOWN"}
                or attempt.started_at is None or attempt.error or not valid_snapshot(attempt.code_snapshot)
                or snapshot_identity(attempt.code_snapshot) != snapshot_identity(run.code_expectation)):
            return
        outputs = data.get("outputs", [])
        rows = ([row for output in outputs if (row := validated_output(node, attempt, run, output)) is not None]
                if isinstance(outputs, list) and len(outputs) <= 1000 else [])
        if rows:
            register_outputs(session, node, attempt, rows)
        # A checkpoint can be rotated after discovery in the same scan. The
        # later confirmed absence wins over that scan's earlier output entry.
        removed = mark_removed_outputs(session, node, attempt, data.get("removed_paths"), event.timestamp)
        if rows or removed:
            run.updated_at = now()
        return
    run.updated_at = now()
    if event.type == "STARTED":
        snapshot = data.get("code_snapshot")
        if attempt.code_snapshot is not None and snapshot != attempt.code_snapshot:
            raise APIError(409, "CODE_SNAPSHOT_IMMUTABLE", "Actual execution code snapshot is immutable")
        attempt.code_snapshot = snapshot
        attempt.started_at = attempt.started_at or event.timestamp
        pid = data.get("pid")
        attempt.pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None
        attempt.output_dir = data.get("output_dir")
        if isinstance(snapshot, dict) and "recipe" in run.code_expectation and snapshot.get("recipe") != run.code_expectation["recipe"]:
            event_failure(session, attempt, run, "RECIPE_MISMATCH", "Runner 实际训练配方与预检冻结的配方不一致，请更新 Agent 后重新预检")
        elif isinstance(snapshot, dict) and snapshot.get("recipe_definition") != run.code_expectation.get("recipe_definition"):
            event_failure(session, attempt, run, "RECIPE_DEFINITION_MISMATCH", "Runner 实际自定义配方版本与预检不一致")
        elif error := entrypoint_snapshot_error(run.code_expectation.get("requested_entrypoint"), snapshot):
            event_failure(session, attempt, run, error["code"], error["message"])
        elif error := dataset_bindings_snapshot_error(run.code_expectation.get("dataset_bindings"), snapshot):
            event_failure(session, attempt, run, error["code"], error["message"])
        elif error := training_stage_snapshot_error(run.code_expectation.get("training_stage"), run.code_expectation.get("stage_config"), snapshot):
            event_failure(session, attempt, run, error["code"], error["message"])
        elif not valid_snapshot(snapshot) or snapshot_identity(snapshot) != snapshot_identity(run.code_expectation):
            event_failure(session, attempt, run, "CODE_CHANGED_SINCE_PREFLIGHT", "Reported execution code differs from the frozen preflight")
        elif not isinstance(attempt.output_dir, str) or not under_root(attempt.output_dir, [node.output_root]):
            event_failure(session, attempt, run, "OUTPUT_PATH_INVALID", "Runner output directory is outside the registered output root")
        elif run.state == "CANCELLING" or run.previous_state == "CANCELLING":
            run.state = attempt.state = "CANCELLING"
        else:
            run.state = attempt.state = "RUNNING"
        return
    if event.type not in TERMINAL:
        return
    exit_code = data.get("exit_code")
    no_process_started = (data.get("process_started") is False and attempt.pid is None and attempt.started_at is None)
    if (not isinstance(exit_code, int) or isinstance(exit_code, bool)) and not (exit_code is None and no_process_started and event.type in {"FAILED", "CANCELLED"}):
        raise APIError(422, "EXIT_NOT_CONFIRMED", "A terminal event must confirm the process exit code")
    if attempt.code_snapshot is None and isinstance(data.get("code_snapshot"), dict):
        attempt.code_snapshot = data["code_snapshot"]
    attempt.code_changed_during_run = data.get("code_changed_during_run") if isinstance(data.get("code_changed_during_run"), bool) else None
    final_state = event.type
    failure = data.get("error") if event.type == "FAILED" else None
    output_rows = []
    if event.type == "SUCCEEDED":
        outputs = data.get("outputs", [])
        if exit_code != 0 or not isinstance(outputs, list):
            failure = {"code": "OUTPUT_INCOMPLETE", "message": "Successful execution needs exit code zero and an output manifest"}
        elif run.template_id != "smoke" and not any(isinstance(o, dict) and o.get("kind") == "MODEL" for o in outputs):
            failure = {"code": "OUTPUT_INCOMPLETE", "message": "Training completed without a model artifact"}
        else:
            for output in outputs:
                row = validated_output(node, attempt, run, output)
                if row is None:
                    failure = {"code": "OUTPUT_INVALID", "message": "Output manifest has an invalid kind or local path"}
                    break
                output_rows.append(row)
        if attempt.code_snapshot is None:
            failure = {"code": "CODE_VERSION_UNAVAILABLE", "message": "No execution code snapshot was recorded"}
    if attempt.error:
        failure = attempt.error
    if failure or event.type == "FAILED":
        final_state = "FAILED"
        if not isinstance(failure, dict):
            failure = {"code": "TRAINING_FAILED", "message": str(failure or "Runner reported a failed execution")}
    if final_state == "SUCCEEDED":
        register_outputs(session, node, attempt, output_rows)
    attempt.state = run.state = final_state
    attempt.exit_code = exit_code
    attempt.error = run.error = failure
    attempt.finished_at = run.finished_at = event.timestamp
    run.queue_reason = None
    run.previous_state = attempt.previous_state = None
    session.execute(delete(Allocation).where(Allocation.attempt_id == attempt.id))
    # Pending deliveries are no longer useful after confirmed exit. Keep the
    # command history, but prevent a late START from being redelivered forever.
    for command in session.scalars(select(Command).where(Command.node_id == node.id, Command.status == "PENDING")):
        if command.payload.get("attempt_id") == attempt.id:
            command.status = "APPLIED"
            command.result = {"terminal_event_received": True}
            command.acknowledged_at = now()


def ingest_events(session, node: Node, entries: list[dict]) -> int:
    accepted = 0
    for item in sorted(entries, key=lambda e: (e["attempt_id"], e["seq"])):
        attempt = session.scalar(select(Attempt).where(Attempt.id == item["attempt_id"]).with_for_update())
        if attempt is None or attempt.node_id != node.id:
            raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Attempt is not assigned to this node")
        item = {**item, "timestamp": canonical_timestamp(item["timestamp"])}
        event_digest = digest(item)
        existing = session.get(Event, (item["attempt_id"], item["seq"]))
        if existing:
            if existing.digest != event_digest:
                raise APIError(409, "EVENT_SEQUENCE_CONFLICT", "Sequence number was reused with different event data")
            accepted += 1
            continue
        event = Event(**item, digest=event_digest)
        session.add(event)
        reduce_event(session, node, attempt, event)
        attempt.last_event_seq = max(attempt.last_event_seq, item["seq"])
        session.flush()
        accepted += 1
    return accepted
