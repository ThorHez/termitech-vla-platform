"""Executed with the registered policy interpreter; no Agent dependencies."""

from __future__ import annotations

import fcntl
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time


def record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def configuration_receipt(directory):
    """Read the witness for the exact generated config used by this service."""
    path = Path(directory) / "inference-config.json"
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("POLICY_CONFIG_INVALID: inference configuration artifact exceeds size limit")
    artifact = json.loads(path.read_text())
    digest = hashlib.sha256(json.dumps(artifact["config"], sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if artifact.get("schema_version") != 1 or digest != artifact.get("digest"):
        raise ValueError("POLICY_CONFIG_INVALID: inference configuration artifact digest differs")
    return {**artifact["source"], "digest": digest, "path": str(path)}


def local_inference_description(train_config, data, manifest, reason):
    """Describe a local trusted resolver without pretending it is portable code."""
    from vla_adapters.portable_policy import _adapter_files, _code_files, _versions

    if (not isinstance(data.asset_id, str) or not data.asset_id or PurePosixPath(data.asset_id).is_absolute()
            or ".." in PurePosixPath(data.asset_id).parts):
        raise ValueError("POLICY_CONFIG_INVALID: checkpoint normalization asset id must be a relative directory")
    nodes = 0
    checkout = Path(manifest["code_dir"]).resolve() / "src" / "openpi"
    def project(value, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > 20000 or depth > 30:
            raise ValueError("POLICY_CONFIG_INVALID: local inference configuration exceeds projection limits")
        if value is None or type(value) in (bool, int, str):
            if isinstance(value, str) and len(value) > 16384:
                raise ValueError("POLICY_CONFIG_INVALID: local inference configuration text is too long")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("POLICY_CONFIG_INVALID: local inference configuration numbers must be finite")
            return value
        if isinstance(value, Path):
            return str(value)
        identity = type(value).__module__ + "." + type(value).__qualname__
        module = getattr(value, "__module__", type(value).__module__)
        if module.startswith("openpi."):
            imported = importlib.import_module(module)
            if not Path(imported.__file__).resolve().is_relative_to(checkout):
                raise ValueError("MODULE_OUTSIDE_CODE_DIR: inference constructor resolves outside the registered checkout")
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"python_type": identity,
                    "fields": {field.name: project(getattr(value, field.name), depth + 1)
                               for field in dataclasses.fields(value) if field.init}}
        if type(value) in (tuple, list):
            return {"python_type": identity, "items": [project(item, depth + 1) for item in value]}
        if type(value) is dict:
            if not all(isinstance(key, str) for key in value):
                raise ValueError("POLICY_CONFIG_INVALID: local inference configuration requires string map keys")
            return {key: project(item, depth + 1) for key, item in value.items()}
        if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
            return {"python_type": identity, "values": project(value.tolist(), depth + 1)}
        # This is an audit projection only. Opaque callables/constructors stay
        # in the checked same-machine resolver and are never reconstructed here.
        return {"python_type": identity, "resolver_dependency": True,
                "callable": getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", "")}

    result = {"schema_version": 1, "portable": False, "resolver": "checked-frozen-training-request",
              "representation_reason": str(reason), "frozen_request": manifest["frozen_request"],
              "model": project(train_config.model),
              "data": {name: project(getattr(data, name, None)) for name in
                       ("asset_id", "data_transforms", "model_transforms", "use_quantile_norm", "normalize_masks", "robot_type")},
              "assets_dirs": project(train_config.assets_dirs), "policy_metadata": project(train_config.policy_metadata),
              "code_files": _code_files(manifest["code_dir"]), "adapter_files": _adapter_files(), "runtime_versions": _versions()}
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > 768 * 1024:
        raise ValueError("POLICY_CONFIG_INVALID: local inference configuration exceeds artifact size limit")
    return result


def resolve_inference_config(manifest, directory, registry):
    """Generate a serving config without loading weights or querying a GPU.

    Training and transferred configurations are served from their saved safe
    JSON representation. A manual recipe keeps the checkout's own registry and
    does not acquire the serialization restrictions of portable configurations.
    """
    from vla_adapters.portable_policy import (
        PortableRepresentationUnsupported, ResolvedDataFactory, export_portable_policy, load_portable_policy, portable_digest,
    )

    portable = manifest.get("portable_policy")
    if portable is not None:
        description, digest = portable["config"], portable["digest"]
        train_config = load_portable_policy(description, digest, manifest["code_dir"])
        source = {"source": "portable", "config_name": manifest["config_name"], "format": "portable-json"}
        for key in ("source_run_id", "source_node_id", "transfer_id"):
            if manifest.get(key) is not None:
                source[key] = manifest[key]
    else:
        frozen = manifest.get("frozen_request")
        if frozen is None:
            train_config = registry.get_config(manifest["config_name"])
        else:
            from vla_adapters.openpi_entry import prepare_config
            from vla_adapters.generated_config import generated_training_config
            spec = dict(frozen, output_dir=str(Path(directory) / "configuration"))
            train_config, _ = prepare_config(spec)
            if generated_training_config(train_config, spec)["sha256"] != manifest["generated_sha256"]:
                raise ValueError("CONFIG_CHANGED: resolved deployment configuration differs from training")
        if (type(getattr(train_config, "model", None)).__name__ == "Pi0ValueConfig"
                or getattr(train_config.data, "training_stage", None) == "value"):
            raise ValueError("POLICY_VALUE_UNSUPPORTED: a value-only model cannot be served as an action policy")
        if hasattr(train_config.data, "datasets") and len(train_config.data.datasets) > 1:
            raise ValueError("MULTI_DATASET_DEPLOY_UNSUPPORTED: choose a single inference data contract first")
        if frozen is None:
            from vla_adapters.policy_inference import compile_inference_config
            return compile_inference_config(train_config), None
        # Export performs the observation-only adaptation, preserving resolved
        # model settings, masks, tokenizers and both transform chains. Load the
        # saved representation itself so the artifact describes what is served.
        from vla_adapters.policy_inference import compile_inference_config
        compiled = compile_inference_config(train_config)
        resolved_data = compiled.data.create(train_config.assets_dirs, train_config.model)
        try:
            description = export_portable_policy(train_config, manifest["code_dir"], resolved_data=resolved_data)
        except PortableRepresentationUnsupported as exc:
            # Same-machine recipes may legitimately use custom constructors
            # (for example FAST tokenizers). They retain the checked frozen
            # resolver and the already resolved data, never the base registry.
            description = local_inference_description(compiled, resolved_data, manifest, exc)
            train_config = dataclasses.replace(compiled, data=ResolvedDataFactory(resolved_data))
            configuration_format = "local-frozen-resolver"
        else:
            train_config = load_portable_policy(description, portable_digest(description), manifest["code_dir"])
            configuration_format = "portable-json"
        digest = portable_digest(description)
        source = {"source": "training", "config_name": frozen.get("recipe", manifest["config_name"]),
                  "generated_sha256": manifest["generated_sha256"], "format": configuration_format}
        if manifest.get("source_run_id") is not None:
            source["source_run_id"] = manifest["source_run_id"]
    record(Path(directory) / "inference-config.json",
           {"schema_version": 1, "source": source, "digest": digest, "config": description})
    return train_config, configuration_receipt(directory)


def probe(manifest):
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    with websockets.sync.client.connect(
        f"ws://127.0.0.1:{manifest['port']}/_termitech_health", compression=None,
        open_timeout=0.7, close_timeout=0.1, max_size=1024 * 1024,
    ) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=0.7))
        if not isinstance(metadata, dict) or metadata.get("_termitech_deployment") != manifest["nonce"]:
            raise ValueError("Policy service metadata does not match this deployment")
        if (manifest.get("gpu_policy", {}).get("mode") == "shared"
                and metadata.get("_termitech_gpu_policy") != manifest["gpu_policy"]):
            raise ValueError("Policy service GPU sharing configuration does not match this deployment")
        if (manifest.get("resolved_backend") is not None
                and metadata.get("_termitech_inference_backend") != manifest["resolved_backend"]):
            raise ValueError("POLICY_BACKEND_MISMATCH: Policy service framework does not match this deployment")
        if manifest.get("inference_config_schema") == 1:
            directory = Path(manifest["state_dir"]) / "policy-deployments" / (manifest.get("execution_id") or manifest["deployment_id"])
            if metadata.get("_termitech_inference_config") != configuration_receipt(directory):
                raise ValueError("Policy service generated configuration does not match this deployment")
        return {"ok": True}


def backend_runtime_options(manifest):
    """Recheck the frozen format and selected device before importing OpenPI.

    Legacy running services keep their health/stop contract. New executions
    attest the detected backend and never silently fall back from GPU to CPU.
    """
    expected = manifest.get("resolved_backend")
    if expected is None:
        return {}
    # The selected OpenPI interpreter does not need the Agent installed.
    sys.path.insert(0, manifest["protocol_root"])
    from vla_protocol.policy_backend import resolve_checkpoint_backend
    actual = resolve_checkpoint_backend(manifest["checkpoint_path"], manifest["inference_backend"])
    if actual != expected:
        raise ValueError("POLICY_BACKEND_MISMATCH: 权重格式与启动时识别的推理框架不一致")
    try:
        runtime = importlib.import_module("torch" if actual == "pytorch" else "jax")
    except (ImportError, OSError) as exc:
        raise ValueError(f"POLICY_RUNTIME_UNAVAILABLE: 当前运行环境无法导入 {actual}：{exc}") from exc
    if manifest.get("gpu_uuid"):
        try:
            available = (runtime.cuda.is_available() and runtime.cuda.device_count() > 0 if actual == "pytorch"
                         else bool(runtime.devices("gpu")))
        except Exception as exc:
            raise ValueError(f"POLICY_BACKEND_GPU_UNAVAILABLE: {actual} 无法访问所选 GPU：{exc}") from exc
        if not available:
            raise ValueError(f"POLICY_BACKEND_GPU_UNAVAILABLE: {actual} 无法访问所选 GPU，请检查对应框架的 CUDA 环境")
    return {"pytorch_device": "cuda:0" if manifest.get("gpu_uuid") else "cpu"} if actual == "pytorch" else {}


def validate_backend_model(manifest, train_config):
    if manifest.get("resolved_backend") != "pytorch":
        return
    model = train_config.model
    # The registered OpenPI loader constructs PI0Pytorch for all torch files;
    # FAST/value/memory families must not silently become a different policy.
    if (type(model).__module__ != "openpi.models.pi0_config" or type(model).__name__ != "Pi0Config"
            or getattr(model, "action_dim", None) != 32):
        raise ValueError("POLICY_BACKEND_MODEL_UNSUPPORTED: 当前 PyTorch 加载器仅支持动作维度为 32 的 Pi0/Pi05 配置，请使用匹配的权重与配置")


def verify_loaded_backend(manifest, policy):
    expected = manifest.get("resolved_backend")
    if expected is None:
        return
    actual = getattr(policy, "_is_pytorch_model", None)
    if type(actual) is not bool or actual != (expected == "pytorch"):
        raise ValueError("POLICY_BACKEND_MISMATCH: 已加载策略的实际框架与部署请求不一致")
    if expected == "pytorch":
        device = "cuda:0" if manifest.get("gpu_uuid") else "cpu"
        if getattr(policy, "_pytorch_device", None) != device:
            raise ValueError("POLICY_BACKEND_GPU_UNAVAILABLE: 已加载策略未使用指定的 PyTorch 设备")


def check_gpu_before_load(manifest):
    """Recheck after acquiring the GPU lock using the Agent's NVML environment.

    The selected training interpreter does not need Agent dependencies installed.
    This subprocess is bounded and does not import the model or allocate CUDA.
    """
    if not manifest.get("gpu_uuid") or "gpu_policy" not in manifest:
        return  # Legacy manifests retain their previous process contract.
    policy = manifest["gpu_policy"]
    source = (
        "import json,sys;sys.path[:0]=json.loads(sys.argv[1]);"
        "from vla_agent.metrics import check_deployment_gpu;"
        "from vla_agent.safety import AgentError;"
        "policy=json.loads(sys.argv[3]);"
        "\ntry:\n actual=check_deployment_gpu(sys.argv[2],policy['mode'],policy['memory_mb'])"
        "\nexcept AgentError as exc:\n print(json.dumps({'error':exc.as_dict()}));sys.exit(1)"
        "\nprint(json.dumps({'ok':True,'gpu_policy':actual}))"
    )
    result = subprocess.run(
        [manifest["agent_python"], "-I", "-c", source,
         json.dumps([manifest["agent_root"], manifest["protocol_root"]]),
         manifest["gpu_uuid"], json.dumps(policy)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20,
    )
    try:
        receipt = json.loads(result.stdout.strip())
    except (ValueError, TypeError):
        raise ValueError("GPU_TELEMETRY_UNAVAILABLE: 无法读取加载模型前的 GPU 检查结果") from None
    if result.returncode or receipt.get("ok") is not True:
        error = receipt.get("error") or {}
        code = error.get("code", "GPU_TELEMETRY_UNAVAILABLE")
        message = error.get("message", "加载模型前无法确认 GPU 可用性")
        raise ValueError(f"{code}: {message}")
    if receipt.get("gpu_policy") != policy:
        raise ValueError("GPU_POLICY_INVALID: GPU 检查回执与冻结的共享模式或显存预算不一致")


def serve(manifest, directory):
    locks = []
    if manifest.get("gpu_uuid"):
        lock_dir = Path(manifest["state_dir"]) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        handle = (lock_dir / ("gpu-" + hashlib.sha256(manifest["gpu_uuid"].encode()).hexdigest() + ".lock")).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("GPU_BUSY: GPU reserved by a local training Runner") from exc
        locks.append(handle)
    check_gpu_before_load(manifest)
    runtime_options = backend_runtime_options(manifest)
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi_client import msgpack_numpy
    if not Path(config.__file__).resolve().is_relative_to(Path(manifest["code_dir"]).resolve()):
        raise ValueError("MODULE_OUTSIDE_CODE_DIR: OpenPI configuration resolves outside the registered checkout")

    class DeploymentPolicyServer(WebsocketPolicyServer):
        async def _handler(self, websocket):
            # OpenPI's normal connection handler resets the shared policy both
            # on connect and disconnect. Health checks must leave RTC/memory
            # state and action chunks of an existing robot client untouched.
            if websocket.request.path == "/_termitech_health":
                await websocket.send(msgpack_numpy.Packer().pack(self._metadata))
                return
            if getattr(self, "_active_policy_client", False):
                await websocket.send("POLICY_CLIENT_BUSY: another client is using this stateful policy")
                await websocket.close(code=1013, reason="Policy client already connected")
                return
            self._active_policy_client = True
            try:
                await super()._handler(websocket)
            finally:
                self._active_policy_client = False

    sys.path.insert(0, manifest["adapter_root"])
    train_config, configuration = resolve_inference_config(manifest, directory, config)
    validate_backend_model(manifest, train_config)
    policy = policy_config.create_trained_policy(
        train_config, manifest["checkpoint_path"], default_prompt=manifest.get("default_prompt") or None,
        **runtime_options,
    )
    verify_loaded_backend(manifest, policy)
    metadata = dict(policy.metadata or {}, _termitech_deployment=manifest["nonce"])
    if manifest.get("resolved_backend") is not None:
        metadata["_termitech_inference_backend"] = manifest["resolved_backend"]
    if manifest.get("gpu_policy") is not None:
        metadata["_termitech_gpu_policy"] = manifest["gpu_policy"]
    if configuration is not None:
        metadata["_termitech_inference_config"] = configuration
    DeploymentPolicyServer(policy, host="127.0.0.1", port=manifest["port"], metadata=metadata).serve_forever()


def infer(manifest, directory):
    def timeout(_sig, _frame):
        raise TimeoutError("Single inference exceeded 120 seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    test = json.loads((directory / "test-request.json").read_text())
    result = {"status": "FAILED", "observation_path": test["observation_path"]}
    try:
        import numpy as np
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        # The probe has bounded connection and receive deadlines. The client
        # itself retries forever, so the overall alarm also covers its init.
        probe(manifest)
        client = WebsocketClientPolicy("127.0.0.1", manifest["port"])
        observation = json.loads(Path(test["snapshot_path"]).read_text())

        def convert(value, key=""):
            if isinstance(value, dict):
                if set(value) == {"__array__", "dtype"}:
                    if value["dtype"] not in {"uint8", "uint16", "float32", "float64", "int32", "int64", "bool"}:
                        raise ValueError("Unsupported explicit observation array dtype")
                    return np.asarray(value["__array__"], dtype=value["dtype"])
                return {name: convert(item, name) for name, item in value.items()}
            if isinstance(value, list):
                # Composition camera contracts perform the final conversion;
                # names alone cannot distinguish RGB from depth observations.
                return np.asarray(value)
            return value

        started = time.monotonic()
        output = client.infer(convert(observation))
        latency = (time.monotonic() - started) * 1000
        actions = np.asarray(output["actions"])
        if actions.size == 0 or actions.size > 1024 * 1024 or not np.isfinite(actions).all():
            raise ValueError("Policy returned empty, oversized or non-finite actions")

        def serializable(value):
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (np.ndarray, np.generic)):
                return value.tolist()
            return value

        output_path = directory / "test-output.json"
        record(output_path, serializable(output))
        result.update(status="SUCCEEDED", output_path=str(output_path), action_shape=list(actions.shape),
                      latency_ms=round(latency, 3), finite=True)
    except Exception as exc:
        result["error"] = {"code": "POLICY_TEST_FAILED", "message": str(exc)[:2000]}
    finally:
        signal.alarm(0)
        record(directory / "test-result.json", result)


def main():
    manifest_path = Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent
    mode = sys.argv[1]
    if mode == "probe":
        print(json.dumps(probe(manifest)), flush=True)
        return
    record(directory / ("pid.json" if mode == "serve" else "test-pid.json"), {"pid": os.getpid()})
    try:
        if mode == "serve":
            serve(manifest, directory)
        elif mode == "test":
            infer(manifest, directory)
        else:
            raise ValueError("Unknown policy worker mode")
    except Exception as exc:
        if mode == "serve":
            code = str(exc).split(":", 1)[0]
            if len(code) > 100 or not code.replace("_", "").isalnum() or not code.isupper():
                code = "POLICY_DEPLOY_FAILED"
            record(directory / "failure.json", {"code": code, "message": str(exc)[:2000]})
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
