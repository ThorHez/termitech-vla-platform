"""Bounded, CPU-only validation through the registered training interpreter.

This checks configuration and local assets. It never runs optimizer steps and
does not turn a successful preflight into a claim of real training convergence.
"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from vla_adapters.dataset_bindings import validate_binding_assets
from vla_protocol import canonical_digest

from .metrics import gpu_inventory
from .runtime_checks import PLATFORM_CREDENTIALS
from .safety import AgentError, allowed_path


VALIDATION_SCOPE = "configuration_and_local_assets"
PROBE_PREFIX = "VLA_PREFLIGHT:"
PROBE_TIMEOUT_SECONDS = 90
PROBE_OUTPUT_BYTES = 1024 * 1024
CACHE_PATH_KEYS = {"HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
                   "TRANSFORMERS_CACHE", "TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME",
                   "JAX_COMPILATION_CACHE_DIR", "OPENPI_DATA_HOME"}


def _validate_training_cache(plan: dict, roots: list[str]):
    """Bind cache evidence to every dataset and to Agent-managed local paths."""
    cache, datasets = plan.get("training_cache"), plan.get("datasets")
    if not isinstance(cache, dict) or not isinstance(datasets, list) or not 1 <= len(datasets) <= 64:
        raise ValueError("missing training cache evidence")

    def local(value):
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise ValueError("invalid training cache path")
        try:
            path = allowed_path(value, roots, must_exist=False)
        except AgentError as exc:
            raise ValueError("training cache path is outside managed roots or unavailable") from exc
        if str(path) != value:
            raise ValueError("training cache path is not canonical")
        return path

    dataset_cache = local(cache.get("datasets_cache"))
    paths, builders = cache.get("paths"), cache.get("builders")
    if (not isinstance(paths, dict) or not 1 <= len(paths) <= len(CACHE_PATH_KEYS)
            or not set(paths) <= CACHE_PATH_KEYS or paths.get("HF_DATASETS_CACHE") != str(dataset_cache)
            or not isinstance(builders, list) or len(builders) != len(datasets)):
        raise ValueError("incomplete training cache paths or dataset builders")
    for path in paths.values():
        local(path)
    for dataset, builder in zip(datasets, builders):
        if (not isinstance(dataset, dict) or not isinstance(builder, dict)
                or builder.get("path") != dataset.get("path")):
            raise ValueError("training cache builder does not match dataset order")
        local(builder.get("path"))
        if not local(builder.get("cache_dir")).is_relative_to(dataset_cache):
            raise ValueError("dataset builder cache is outside the configured datasets cache")


def _probe_process(argv: list[str], cwd: str, environment: dict[str, str], *,
                   timeout: float = PROBE_TIMEOUT_SECONDS, output_limit: int = PROBE_OUTPUT_BYTES) -> tuple[int, str, str]:
    """Drain both pipes without permitting unbounded memory or orphan workers."""
    process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    failed = True
    try:
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentError("PREFLIGHT_TIMEOUT", f"训练配方自动预检超过 {timeout:g} 秒，请检查本机环境和数据读取")
            for key, _ in selector.select(timeout=min(remaining, 0.1)):
                try:
                    chunk = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                streams[key.data].extend(chunk)
                if sum(len(value) for value in streams.values()) > output_limit:
                    raise AgentError("PREFLIGHT_OUTPUT_LIMIT", "训练配方自动预检输出超过安全上限，请检查训练环境导入日志")
        returncode = process.wait(timeout=max(deadline - time.monotonic(), 0.001))
        failed = False
        return returncode, streams["stdout"].decode("utf-8", errors="replace"), streams["stderr"].decode("utf-8", errors="replace")
    finally:
        if failed:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        # Imports may have started workers that outlive a completed probe. They
        # belong to this dedicated process group and must not survive preflight.
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        selector.close()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def prepare_launch(launch: dict, config: dict, requested_inputs: dict, *, expected_plan_digest: str | None = None) -> dict:
    """Bind execution to Agent roots and preserve original registered paths."""
    argv = list(launch["argv"])
    try:
        index = argv.index("--spec-json")
        spec = json.loads(argv[index + 1])
        if not isinstance(spec, dict):
            raise ValueError("invalid launch spec")
    except (ValueError, IndexError, TypeError) as exc:
        raise AgentError("PREFLIGHT_UNSUPPORTED", "训练适配器没有提供可自动预检的执行配置") from exc
    spec["allowed_roots"] = list(config["roots"])
    spec["requested_inputs"] = {role: asset["path"] for role, asset in requested_inputs.items()}
    if expected_plan_digest is not None:
        spec["expected_plan_digest"] = expected_plan_digest
    argv[index + 1] = json.dumps(spec, ensure_ascii=False, allow_nan=False)
    return {**launch, "argv": argv}


def _probe_spec(launch: dict, config: dict) -> tuple[list[str], dict]:
    argv = list(launch["argv"])
    try:
        index = argv.index("--spec-json")
        spec = json.loads(argv[index + 1])
        if not isinstance(spec, dict):
            raise ValueError("invalid launch spec")
    except (ValueError, IndexError, TypeError) as exc:
        raise AgentError("PREFLIGHT_UNSUPPORTED", "训练适配器没有提供可自动预检的执行配置") from exc
    spec["allowed_roots"] = list(config["roots"])
    spec.pop("expected_plan_digest", None)
    argv[index + 1] = json.dumps(spec, ensure_ascii=False, allow_nan=False)
    argv.append("--validate-only")
    return argv, spec


def _wrapper_validation(config: dict, launch: dict, checks: list[dict]) -> dict:
    argv, spec = _probe_spec(launch, config)
    environment = {key: value for key, value in os.environ.items() if key not in PLATFORM_CREDENTIALS}
    environment.update({key: value for key, value in launch.get("env", {}).items() if key not in PLATFORM_CREDENTIALS})
    environment.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1",
                        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    returncode, stdout, stderr = _probe_process(argv, launch["cwd"], environment)
    responses = [line[len(PROBE_PREFIX):] for line in stdout.splitlines() if line.startswith(PROBE_PREFIX)]
    try:
        if len(responses) != 1:
            raise ValueError("expected one structured response")
        response = json.loads(responses[0])
        if not isinstance(response, dict) or response.get("schema_version") != 1 or response.get("scope") != VALIDATION_SCOPE:
            raise ValueError("unsupported response schema")
        detailed = response.get("checks")
        if not isinstance(detailed, list) or not detailed or any(
            not isinstance(check, dict) or not isinstance(check.get("name"), str)
            or not isinstance(check.get("ok"), bool) or not isinstance(check.get("message"), str)
            for check in detailed
        ):
            raise ValueError("invalid detailed checks")
        checks.extend(detailed)
        error = response.get("error")
        if error is not None:
            if not isinstance(error, dict) or not all(isinstance(error.get(key), str) for key in ("code", "message")):
                raise ValueError("invalid validation error")
            raise AgentError(error["code"], error["message"])
        if returncode or any(not check["ok"] for check in detailed):
            raise AgentError("RECIPE_VALIDATION_FAILED", "训练配方自动预检未通过，请查看具体检查项")
        required = {"recipe_configuration", "data_contract", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache"}
        if spec.get("training_stage") is not None:
            required.add("recap_stage")
        if not required.issubset({check["name"] for check in detailed if check["ok"]}):
            raise ValueError("missing required validation checks")
        plan = response.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("missing resolved execution plan")
        _validate_training_cache(plan, config["roots"])
        for field in ("recipe", "parameters", "gpu_count"):
            if plan.get(field) != spec.get(field):
                raise ValueError("resolved execution plan does not match " + field)
        for field in ("training_stage", "stage_config"):
            if canonical_digest(plan.get(field)) != canonical_digest(spec.get(field)):
                raise ValueError("resolved execution plan does not match " + field)
        if spec.get("training_stage") is not None:
            contract = plan.get("recap_stage_contract")
            datasets = plan.get("datasets")
            contract_datasets = contract.get("datasets") if isinstance(contract, dict) else None
            if (not isinstance(contract, dict) or contract.get("stage") != spec["training_stage"]
                    or not isinstance(contract.get("model_type"), str) or not contract["model_type"]
                    or (contract["model_type"] == "Pi0ValueConfig") != (spec["training_stage"] == "value")
                    or not isinstance(contract.get("loader_type"), str) or not contract["loader_type"]
                    or not isinstance(datasets, list) or not isinstance(contract_datasets, list)
                    or len(contract_datasets) != len(datasets)
                    or any(not isinstance(item, dict) or item.get("path") != data.get("path")
                           or not isinstance(item.get("required_columns"), list) or not item["required_columns"]
                           or ("value_target" if spec["training_stage"] == "value" else "is_positive") not in item["required_columns"]
                           or any(not isinstance(column, str) or not column for column in item["required_columns"])
                           for item, data in zip(contract_datasets, datasets))):
                raise ValueError("missing ReCap phase model, loader or dataset field evidence")
        if plan.get("requested_entrypoint") != spec.get("entrypoint"):
            raise ValueError("resolved execution plan does not match requested training entrypoint")
        bindings = validate_binding_assets(spec.get("dataset_bindings"), spec.get("inputs", {}))
        if plan.get("dataset_bindings") != bindings:
            raise ValueError("resolved execution plan does not match dataset/norm bindings")
        requested = spec.get("requested_inputs", {role: asset["path"] for role, asset in spec.get("inputs", {}).items()})
        if plan.get("requested_inputs") != requested:
            raise ValueError("resolved execution plan does not match requested assets")
        if not isinstance(plan.get("inputs"), dict) or set(plan["inputs"]) != set(requested):
            raise ValueError("resolved execution plan has incomplete input identities")
        for role, value in plan["inputs"].items():
            if str(allowed_path(value, config["roots"])) != value or str(allowed_path(requested[role], config["roots"])) != value:
                raise ValueError("resolved execution plan has a different input path")
        if bindings is not None:
            datasets = plan.get("datasets")
            if (not isinstance(datasets, list) or len(datasets) != len(bindings)
                    or plan.get("weights") != [row["weight"] for row in bindings]):
                raise ValueError("resolved execution plan has incomplete dataset/norm bindings or weights")
            for i, data in enumerate(datasets):
                norm = allowed_path(plan["inputs"][f"norm_{i}"], config["roots"])
                norm = allowed_path(str(norm / "norm_stats.json"), config["roots"]) if norm.is_dir() else norm
                if (not isinstance(data, dict) or data.get("path") != plan["inputs"][f"dataset_{i}"]
                        or data.get("norm_path") != str(norm)):
                    raise ValueError("resolved execution plan has a different dataset/norm pairing")
        definition = spec.get("recipe_definition")
        if plan.get("recipe_definition_digest") != (definition.get("digest") if definition else None):
            raise ValueError("resolved execution plan does not match recipe definition")
        canonical_digest(plan)  # Reject non-finite or non-JSON plan identities.
        return plan
    except (ValueError, TypeError, KeyError) as exc:
        detail = (stderr or stdout).strip()[-2000:]
        message = f"训练环境未返回有效的自动预检结果: {exc}"
        if detail and not responses:
            message += ": " + detail
        raise AgentError("PREFLIGHT_INVALID_RESPONSE", message) from exc


def _check_gpu_capacity(count: int, checks: list[dict], *, assigned: list[str] | None = None):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise AgentError("GPU_REQUIRED", "正式训练至少需要申请 1 张 GPU")
    rows, availability = gpu_inventory()
    if not availability.get("available"):
        reason = availability.get("reason", "unknown")
        raise AgentError("GPU_UNAVAILABLE", f"Agent 无法通过 NVML 检查 GPU，请检查驱动及 nvidia-ml-py: {reason}")
    detected = {row["uuid"] for row in rows if row.get("uuid")}
    if len(detected) < count:
        raise AgentError("GPU_UNAVAILABLE", f"申请 {count} 张 GPU，本机 NVML 仅检测到 {len(detected)} 张")
    if assigned is not None and (len(assigned) != count or len(set(assigned)) != count or not set(assigned).issubset(detected)):
        raise AgentError("GPU_UNAVAILABLE", "调度分配的 GPU 与当前本机设备不一致，请重新预检和调度")
    checks.append({"name": "gpu_capacity", "ok": True,
                   "message": f"本机检测到 {len(detected)} 张 GPU，申请 {count} 张；空闲设备由调度器在启动时分配"})


def _check_output_storage(config: dict, plan: dict, checks: list[dict]):
    root = allowed_path(config["output_root"], config["roots"])
    if not root.is_dir():
        raise AgentError("OUTPUT_UNAVAILABLE", "训练输出根目录不存在或不是目录")
    storage = plan.get("storage")
    if not isinstance(storage, dict) or any(
        isinstance(storage.get(key), bool) or not isinstance(storage.get(key), int) or storage[key] < 0
        for key in ("required_bytes", "reference_checkpoint_bytes", "model_params_bytes")
    ) or storage["required_bytes"] < 1:
        raise AgentError("STORAGE_ESTIMATE_UNAVAILABLE", "训练配方未提供有效的 checkpoint 空间估算")
    try:
        # Test effective service-user permissions rather than os.access or root
        # mode bits. NamedTemporaryFile removes the tiny probe on every path.
        with tempfile.NamedTemporaryFile(prefix=".vla-preflight-", dir=root) as probe:
            probe.write(b"vla\n")
            probe.flush()
            os.fsync(probe.fileno())
        free = shutil.disk_usage(root).free
    except OSError as exc:
        raise AgentError("OUTPUT_NOT_WRITABLE", f"Agent 无法写入训练输出目录 {root}: {exc}") from exc
    required = storage["required_bytes"]
    if free < required:
        raise AgentError("OUTPUT_SPACE_INSUFFICIENT", f"训练输出空间不足：可用 {free / 2**30:.2f} GiB，预计至少需要 {required / 2**30:.2f} GiB（含 checkpoint 临时写入余量）")
    checks.append({"name": "output_storage", "ok": True,
                   "message": f"输出目录可写，可用 {free / 2**30:.2f} GiB，预计至少需要 {required / 2**30:.2f} GiB"})


def validate_training(config: dict, launch: dict, checks: list[dict], *, assigned: list[str] | None = None) -> dict:
    """Return a deterministic evidence identity after all automatic checks pass."""
    plan = _wrapper_validation(config, launch, checks)
    for name, callback in (("gpu_capacity", lambda: _check_gpu_capacity(plan["gpu_count"], checks, assigned=assigned)),
                           ("output_storage", lambda: _check_output_storage(config, plan, checks))):
        try:
            callback()
        except AgentError as exc:
            checks.append({"name": name, "ok": False, "message": str(exc)})
            raise
    return {"schema_version": 1, "scope": VALIDATION_SCOPE, "plan": plan, "plan_digest": canonical_digest(plan)}
