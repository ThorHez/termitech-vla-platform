from __future__ import annotations

from pathlib import Path

from vla_adapters import build_launch
from vla_adapters.dataset_bindings import validate_binding_assets
from vla_adapters.recap import validate_training_stage

from .config import get_profile
from .runtime_checks import validate_runtime
from .safety import AgentError, allowed_path
from .snapshots import capture_code, compare_code
from .training_validation import prepare_launch, validate_training


def validate_local_inputs(config: dict, payload: dict) -> tuple[dict, dict]:
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
    inputs = {}
    for role, asset in payload.get("inputs", {}).items():
        path = allowed_path(asset["path"], config["roots"])
        inputs[role] = {**asset, "path": str(path)}
    return profile, inputs


def run_preflight(config: dict, payload: dict) -> dict:
    checks = []
    snapshot = None
    try:
        profile, inputs = validate_local_inputs(config, payload)
        checks.append({"name": "runtime", "ok": True, "message": "Training Python executable and adapter environment are available"})
        checks.append({"name": "local_paths", "ok": True, "message": "Input assets and code exist inside configured roots"})
        snapshot = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
        if not snapshot["complete"]:
            raise AgentError("CODE_VERSION_UNAVAILABLE", "Incomplete code snapshot: " + "; ".join(snapshot["missing"]))
        checks.append({"name": "code_version", "ok": True, "message": snapshot["commit_sha"]})
        output = str(Path(config["output_root"]) / "preflight-placeholder")
        count = payload.get("gpu_count", 0)
        launch = build_launch(profile, payload["template_id"], inputs, payload.get("parameters", {}), output,
                              [f"preflight-gpu-{i}" for i in range(count)], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
        checks.append({"name": "adapter", "ok": True, "message": "Template and trusted runtime configuration validated"})
        if profile.get("adapter") != "smoke":
            launch = prepare_launch(launch, config, payload.get("inputs", {}))
            automatic = validate_training(config, launch, checks)
            # A CPU probe imports registered code. Capture again so concurrent
            # edits cannot turn its result into authorization for different code.
            current = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
            compare_code(snapshot, current)
            snapshot["automatic_preflight"] = automatic
        return {"code_snapshot": snapshot, "checks": checks}
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "PREFLIGHT_FAILED", "message": str(exc)}
        checks.append({"name": "validation", "ok": False, "message": error["message"]})
        return {"code_snapshot": snapshot, "checks": checks, "error": error}

