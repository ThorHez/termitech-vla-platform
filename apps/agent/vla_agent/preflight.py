from __future__ import annotations

from pathlib import Path

from vla_adapters import build_launch

from .config import get_profile
from .safety import AgentError, allowed_path
from .snapshots import capture_code


def validate_local_inputs(config: dict, payload: dict) -> tuple[dict, dict]:
    try:
        profile = get_profile(config, payload["profile_id"])
    except ValueError as exc:
        raise AgentError("RUNTIME_UNAVAILABLE", str(exc)) from exc
    allowed_path(profile["code_dir"], config["roots"])
    interpreter = Path(profile["python"])
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise AgentError("RUNTIME_UNAVAILABLE", "Training interpreter must be an existing absolute path")
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
        checks.append({"name": "local_paths", "ok": True, "message": "Input assets and code exist inside configured roots"})
        snapshot = capture_code(profile, config["roots"])
        if not snapshot["complete"]:
            raise AgentError("CODE_VERSION_UNAVAILABLE", "Incomplete code snapshot: " + "; ".join(snapshot["missing"]))
        checks.append({"name": "code_version", "ok": True, "message": snapshot["commit_sha"]})
        output = str(Path(config["output_root"]) / "preflight-placeholder")
        count = payload.get("gpu_count", 0)
        build_launch(profile, payload["template_id"], inputs, payload.get("parameters", {}), output,
                     [f"preflight-gpu-{i}" for i in range(count)])
        checks.append({"name": "adapter", "ok": True, "message": "Template and trusted runtime configuration validated"})
        return {"code_snapshot": snapshot, "checks": checks}
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "PREFLIGHT_FAILED", "message": str(exc)}
        checks.append({"name": "validation", "ok": False, "message": error["message"]})
        return {"code_snapshot": snapshot, "checks": checks, "error": error}

