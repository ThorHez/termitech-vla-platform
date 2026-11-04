from __future__ import annotations

from pathlib import Path

from vla_adapters import build_launch
from vla_adapters.dataset_bindings import validate_binding_assets
from vla_adapters.norm_datasets import validate_norm_assets, validate_norm_configuration
from vla_adapters.recap import validate_training_stage
from vla_adapters.runtime_options import normalize_normalization_method

from .config import get_profile
from .runtime_checks import validate_runtime
from .safety import AgentError, allowed_path
from .snapshots import capture_code, compare_code
from .training_validation import prepare_launch, validate_training
from .metrics import check_training_gpus
from .training_gpu import requested_training_gpus, training_gpu_evidence, validate_training_gpu_policy


def validate_local_inputs(config: dict, payload: dict) -> tuple[dict, dict]:
    if payload["template_id"] == "openpi-norm":
        try:
            validate_norm_assets(payload.get("norm_datasets"), payload.get("inputs", {}))
            validate_norm_configuration(payload.get("auto_config"), payload.get("norm_datasets"))
        except ValueError as exc:
            raise AgentError("NORM_DATASETS_INVALID", str(exc)) from exc
    elif payload.get("norm_datasets") is not None:
        raise AgentError("NORM_DATASETS_UNSUPPORTED", "当前模板不支持独立统计数据集列表")
    try:
        normalize_normalization_method(payload.get("normalization_method"), template_id=payload["template_id"])
    except ValueError as exc:
        raise AgentError(str(exc).split(":", 1)[0], str(exc)) from exc
    try:
        validate_training_stage(payload["template_id"], payload.get("training_stage"), payload.get("stage_config"))
    except ValueError as exc:
        raise AgentError("TRAINING_STAGE_INVALID", str(exc)) from exc
    try:
        bindings = validate_binding_assets(payload.get("dataset_bindings"), payload.get("inputs", {}))
    except (ValueError, TypeError) as exc:
        raise AgentError("DATASET_BINDINGS_INVALID", str(exc)) from exc
    if bindings is not None:
        for role, asset in payload["inputs"].items():
            kind = "MODEL" if role == "model" else "DATASET" if role.startswith("dataset_") else "NORM"
            if not isinstance(asset, dict) or asset.get("kind") != kind:
                raise AgentError("DATASET_BINDINGS_INVALID", f"输入 {role} 的资产类型应为 {kind}")
    if payload.get("recipe_definition") is not None:
        from vla_adapters.recipe_definitions import validate_definition
        definition = validate_definition(payload["recipe_definition"])
        if definition["node_id"] != payload.get("node_id"):
            raise AgentError("RECIPE_DEFINITION_MISMATCH", "自定义配方不属于本次命令的目标机器")
    try:
        profile = get_profile(config, payload["profile_id"])
    except ValueError as exc:
        raise AgentError("RUNTIME_UNAVAILABLE", str(exc)) from exc
    validate_runtime(profile, config["roots"])
    validate_training_gpu_policy(payload, profile)
    inputs = {}
    for role, asset in payload.get("inputs", {}).items():
        path = allowed_path(asset["path"], config["roots"])
        inputs[role] = {**asset, "path": str(path)}
    return profile, inputs


def run_preflight(config: dict, payload: dict) -> dict:
    from .profile_rebinding import profile_configuration_lock
    try:
        with profile_configuration_lock(config, busy_code="STORAGE_CONFIGURE_BUSY"):
            return _run_preflight(config, payload)
    except AgentError as exc:
        error = exc.as_dict()
        return {"code_snapshot": None, "checks": [{"name": "validation", "ok": False, "message": error["message"]}],
                "error": error}


def _run_preflight(config: dict, payload: dict) -> dict:
    checks = []
    snapshot = None
    try:
        if config.get("resource_guard") is not None:
            from .resource_guard import protected_gpus
            known, blocked = protected_gpus(config)
            if payload.get("gpu_count", 0) > len(known - blocked):
                raise AgentError("GPU_RESERVED_BY_PRODUCTION", "正式环境的预约或分配使测试可用显卡不足，请调整 GPU 数量后重试")
        profile, inputs = validate_local_inputs(config, payload)
        policy = validate_training_gpu_policy(payload, profile)
        selected = requested_training_gpus(payload) if policy is not None else None
        if policy is not None:
            from .resource_guard import require_gpu_access
            require_gpu_access(config, selected)
            check_training_gpus(selected, policy)
        checks.append({"name": "runtime", "ok": True, "message": "Training Python executable and adapter environment are available"})
        checks.append({"name": "local_paths", "ok": True, "message": "Input assets and code exist inside configured roots"})
        snapshot = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
        if not snapshot["complete"]:
            raise AgentError("CODE_VERSION_UNAVAILABLE", "Incomplete code snapshot: " + "; ".join(snapshot["missing"]))
        checks.append({"name": "code_version", "ok": True, "message": snapshot["commit_sha"]})
        output = str(Path(config["output_root"]) / "preflight-placeholder")
        count = payload.get("gpu_count", 0)
        launch = build_launch(profile, payload["template_id"], inputs, payload.get("parameters", {}), output,
                              [f"preflight-gpu-{i}" for i in range(count)], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
        checks.append({"name": "adapter", "ok": True, "message": "Template and trusted runtime configuration validated"})
        if profile.get("adapter") != "smoke":
            launch = prepare_launch(launch, config, payload.get("inputs", {}))
            automatic = validate_training(config, launch, checks)
            # A CPU probe imports registered code. Capture again so concurrent
            # edits cannot turn its result into authorization for different code.
            current = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
            compare_code(snapshot, current)
            snapshot["automatic_preflight"] = automatic
        if policy is not None:
            from .resource_guard import require_gpu_access
            require_gpu_access(config, selected)
            devices = check_training_gpus(selected, policy)
            snapshot["training_gpu_check"] = training_gpu_evidence(policy, selected, devices)
            snapshot["requested_gpu_uuids"] = list(selected)
            checks.append({"name": "training_gpu", "ok": True,
                           "message": "已核对所选 GPU 的实际剩余显存、每卡训练预算及余量；启动时会重新检查"})
        return {"code_snapshot": snapshot, "checks": checks}
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "PREFLIGHT_FAILED", "message": str(exc)}
        failure = {"name": "validation", "ok": False, "message": error["message"]}
        if failure not in checks:
            checks.append(failure)
        return {"code_snapshot": snapshot, "checks": checks, "error": error}

