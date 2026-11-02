"""Asynchronous dataset-registration validation in a CPU-only child."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from vla_adapters import dataset_source_validation

from .config import get_profile
from .runtime_checks import PLATFORM_CREDENTIALS, validate_runtime
from .safety import AgentError, allowed_path
from .state import State
from .storage_configuration import cache_environment
from .training_validation import _probe_process


def validate_dataset(config, payload):
    if State(config["state_dir"]).active_attempts():
        raise AgentError("DATASET_VALIDATION_BUSY", "机器有活动训练，基础数据校验等待空闲后执行")
    try:
        expires = datetime.fromisoformat(payload["expires_at"])
        if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
            raise ValueError("expired")
        path = allowed_path(payload["path"], config["roots"])
        profile = get_profile(config, payload["profile_id"])
        if profile.get("adapter") != "openpi-sft":
            raise ValueError("unsupported profile")
        validate_runtime(profile, config["roots"])
    except (ValueError, KeyError, TypeError) as error:
        raise AgentError("DATASET_VALIDATION_INVALID", "数据集基础校验的环境、路径或有效期无效") from error
    registered = payload.get("registered_profile")
    if registered is not None and not isinstance(registered, dict):
        raise AgentError("DATASET_VALIDATION_INVALID", "登记数据描述无效")
    spec = {"path": str(path), "code_dir": profile["code_dir"], "profile_id": profile["id"],
            "allowed_roots": list(config["roots"]), "registered_profile": registered,
            "inputs": {"dataset": {"id": payload["asset_id"], "kind": "DATASET", "path": str(path),
                                    **({"dataset_profile": registered} if registered is not None else {})}},
            "validation_cache_dir": str(Path(config["state_dir"]) / "dataset-validation-cache")}
    environment = {key: value for key, value in os.environ.items() if key not in PLATFORM_CREDENTIALS}
    environment.update(cache_environment(config))
    environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", WANDB_MODE="disabled", WANDB_DISABLED="true",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    argv = [profile["python"], "-u", str(Path(dataset_source_validation.__file__).resolve()),
            "--spec-json", json.dumps(spec, ensure_ascii=False, allow_nan=False)]
    status, stdout, _ = _probe_process(argv, profile["code_dir"], environment, timeout=300)
    lines = [line.removeprefix(dataset_source_validation.PREFIX) for line in stdout.splitlines()
             if line.startswith(dataset_source_validation.PREFIX)]
    try:
        if len(lines) != 1:
            raise ValueError("missing response")
        result = json.loads(lines[0])
        if not isinstance(result, dict):
            raise ValueError("invalid response")
        if result.get("error"):
            return result
        if (status or result.get("schema_version") != 1 or result.get("scope") != "dataset_source"
                or result.get("path") != str(path) or result.get("profile_id") != profile["id"]
                or not isinstance(result.get("checks"), list) or not result["checks"]
                or any(not isinstance(check, dict) or check.get("ok") is not True for check in result["checks"])
                or not {"source_metadata", "dataset_timestamps", "raw_samples", "dataset_cache"}.issubset(
                    {check.get("name") for check in result["checks"]})):
            raise ValueError("mismatched evidence")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise AgentError("DATASET_VALIDATION_INVALID_RESPONSE", "数据集基础校验未返回有效、完整且匹配的证据") from error
    return result
