"""Local-only OpenPI policy services, owned and recovered by process identity."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time

import psutil

from .config import get_profile
from .runtime_checks import PLATFORM_CREDENTIALS
from .safety import AgentError, allowed_path, identity_matches, process_identity, safe_identifier, signal_owned_group


WORKER = Path(__file__).with_name("policy_deployment_worker.py").resolve()
_CHILDREN = {}


def _requested_gpu_policy(payload):
    from vla_protocol.gpu_policy import normalize_gpu_policy
    try:
        return normalize_gpu_policy(payload.get("gpu_mode", "exclusive"), payload.get("gpu_memory_mb"))
    except (TypeError, ValueError) as exc:
        raise AgentError("GPU_POLICY_INVALID", str(exc)) from exc


def _requested_backend(payload):
    from vla_protocol.policy_backend import normalize_inference_backend
    try:
        return normalize_inference_backend(payload.get("inference_backend", "auto"))
    except ValueError as exc:
        raise AgentError("POLICY_BACKEND_INVALID", str(exc)) from exc


def _checkpoint_backend(checkpoint, requested):
    from vla_protocol.policy_backend import resolve_checkpoint_backend
    try:
        return resolve_checkpoint_backend(checkpoint, requested)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        raise AgentError(code, message or str(exc)) from exc


def _verify_gpu_policy_manifest(manifest, payload):
    requested = _requested_gpu_policy(payload)
    saved = manifest.get("gpu_policy", {"mode": "exclusive", "memory_mb": None, "reserve_mb": 0})
    if saved != requested or manifest.get("gpu_uuid") != payload.get("gpu_uuid"):
        raise AgentError("POLICY_GPU_POLICY_CONFLICT", "同一部署标识的 GPU、模式或显存预算已变化，请重新创建部署")
    if manifest.get("inference_backend", "auto") != _requested_backend(payload):
        raise AgentError("POLICY_BACKEND_CONFLICT", "同一部署标识的推理框架已变化，请重新创建部署")


def _record(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".policy-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _execution_identifiers(payload):
    deployment_id = safe_identifier(payload.get("deployment_id"))
    value = payload.get("execution_id")
    return deployment_id, deployment_id if value is None else safe_identifier(value)


def _execution_receipt(payload):
    return {"execution_id": safe_identifier(payload["execution_id"])} if payload.get("execution_id") is not None else {}


def _verify_execution(directory, deployment_id, execution_id):
    """Bind even a pre-start STOP tombstone to exactly one logical deployment."""
    expected = {"deployment_id": deployment_id, "execution_id": execution_id}
    saved = _read(directory / "execution.json")
    manifest = _read(directory / "manifest.json")
    if saved is None and manifest:
        saved = {"deployment_id": manifest.get("deployment_id"),
                 "execution_id": manifest.get("execution_id") or manifest.get("deployment_id")}
    if saved is None and (directory / "stopped.json").exists():
        # Older Agents stored STOP-before-START by logical ID alone.
        saved = {"deployment_id": directory.name, "execution_id": directory.name}
    if saved is not None and saved != expected:
        raise AgentError("POLICY_DEPLOY_EXECUTION_CONFLICT", "Execution identifier belongs to another policy deployment")
    if manifest and (manifest.get("deployment_id") != deployment_id
                     or (manifest.get("execution_id") or manifest.get("deployment_id")) != execution_id):
        raise AgentError("POLICY_DEPLOY_EXECUTION_CONFLICT", "Execution manifest does not match this policy deployment")
    if not (directory / "execution.json").exists():
        _record(directory / "execution.json", expected)


@contextmanager
def _directory(config, payload, *, wait_seconds=0):
    deployment_id, execution_id = _execution_identifiers(payload)
    directory = Path(config["state_dir"]) / "policy-deployments" / execution_id
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (directory / "operation.lock").open("a") as lock:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise AgentError("POLICY_DEPLOY_BUSY", "Policy deployment operation is in progress") from exc
                time.sleep(.02)
        try:
            _verify_execution(directory, deployment_id, execution_id)
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _environment(manifest):
    environment = {key: value for key, value in os.environ.items() if key not in PLATFORM_CREDENTIALS}
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # Empty CUDA visibility explicitly selects CPU, never an arbitrary shared GPU.
    environment["CUDA_VISIBLE_DEVICES"] = manifest.get("gpu_uuid") or ""
    return environment


def _identity(directory, mode="serve"):
    runtime = directory / ("identity.json" if mode == "serve" else "test-identity.json")
    identity = _read(runtime)
    if identity is not None:
        return identity
    pid = _read(directory / ("pid.json" if mode == "serve" else "test-pid.json"))
    if not isinstance(pid, dict) or type(pid.get("pid")) is not int:
        return None
    try:
        process = psutil.Process(pid["pid"])
        arguments = process.cmdline()
        if str(WORKER) not in arguments or str(directory / "manifest.json") not in arguments or mode not in arguments:
            return None
        identity = process_identity(pid["pid"])
        if identity is not None:
            _record(runtime, identity)
        return identity
    except psutil.Error:
        return None


def _launch(directory, manifest, mode):
    log = directory / ("policy.log" if mode == "serve" else "test.log")
    with log.open("ab") as output:
        child = subprocess.Popen([manifest["python"], "-u", str(WORKER), mode, str(directory / "manifest.json")],
                                 cwd=manifest["code_dir"], env=_environment(manifest), stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    identity = process_identity(child.pid)
    _CHILDREN[child.pid] = child
    if identity:
        _record(directory / ("identity.json" if mode == "serve" else "test-identity.json"), identity)
    return identity


def _tail(path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 24000))
            return handle.read(24000).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _healthy(directory, manifest):
    try:
        completed = subprocess.run([manifest["python"], "-u", str(WORKER), "probe", str(directory / "manifest.json")],
                                   cwd=manifest["code_dir"], env=_environment(manifest), stdin=subprocess.DEVNULL,
                                   capture_output=True, timeout=2.5, text=True)
        return completed.returncode == 0 and json.loads(completed.stdout.strip()).get("ok") is True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _status(directory, *, health=True):
    for pid, child in list(_CHILDREN.items()):
        if child.poll() is not None:
            del _CHILDREN[pid]
    manifest = _read(directory / "manifest.json")
    if not manifest:
        if (directory / "stopped.json").exists():
            stopped = _read(directory / "stopped.json")
            binding = _read(directory / "execution.json", {})
            if _other_execution_running(directory, binding.get("deployment_id", directory.name)):
                return {"error": {"code": "POLICY_STOP_UNCONFIRMED",
                    "message": "Another execution of this logical deployment is still running; this execution was cancelled without stopping that process"},
                    **_execution_receipt(stopped)}
            return {"state": "STOPPED", "pid": None, "logs": "", "checks": [],
                    **_execution_receipt(stopped)}
        raise AgentError("POLICY_DEPLOY_NOT_FOUND", "Policy deployment was not found on this Agent")
    identity = _identity(directory)
    alive = identity_matches(identity)
    stopped = (directory / "stopped.json").exists()
    error = _read(directory / "failure.json")
    healthy = alive and not stopped and health and _healthy(directory, manifest)
    state = "STOPPED" if stopped and not alive else "RUNNING" if healthy else "STARTING" if alive else "FAILED"
    result = {"state": state, "pid": identity["pid"] if alive else None,
              "endpoint": f"ws://127.0.0.1:{manifest['port']}", "logs": _tail(directory / "policy.log"),
              "checks": [{"name": "websocket_metadata", "ok": bool(healthy),
                          "message": "Policy WebSocket metadata verified" if healthy else "Policy metadata is not currently reachable"}]}
    result.update(_execution_receipt(manifest))
    if manifest.get("resolved_backend") is not None:
        result["resolved_backend"] = manifest["resolved_backend"]
    if manifest.get("gpu_policy") is not None:
        result["gpu_policy"] = manifest["gpu_policy"]
    if manifest.get("source_warnings"):
        result["warnings"] = manifest["source_warnings"]
    if (directory / "inference-config.json").is_file():
        from .policy_deployment_worker import configuration_receipt
        try:
            result["inference_config"] = configuration_receipt(directory)
        except (KeyError, TypeError, ValueError, OSError):
            # The health probe fails closed if its artifact is missing or has
            # changed. Keep status and stop available to inspect that failure.
            pass
    portable = manifest.get("portable_policy")
    if portable is not None:
        # The controller matches both witnesses to its successful transfer.
        # A service that merely knows POLICY_DEPLOY_START cannot substitute a
        # named base recipe and still claim that transferred policy is ready.
        result["portable_digest"] = portable["digest"]
        result["checkpoint_manifest_digest"] = manifest["checkpoint_manifest_digest"]
    if state == "FAILED":
        result["error"] = error or {"code": "POLICY_PROCESS_EXITED", "message": "Policy service process is no longer running; inspect its log"}
    test = _read(directory / "test-request.json")
    if test:
        test_result = _read(directory / "test-result.json")
        if test_result is None:
            test_result = {"status": "RUNNING", "observation_path": test["observation_path"]}
            if not identity_matches(_identity(directory, "test")):
                test_result.update(status="FAILED", error={"code": "POLICY_TEST_EXITED", "message": "Inference test process exited without a result"})
        result["test_result"] = test_result
    return result


def _frozen(config, profile, payload):
    source_training = payload.get("source_training")
    if source_training is None:
        return {}
    from vla_protocol import canonical_digest
    from .snapshots import capture_code, compare_code
    try:
        expected = source_training["code_expectation"]
        # Attribution to one training run is advisory. Machine managed roots
        # remain the independent access boundary, including resolved symlinks.
        checkpoint = allowed_path(payload["checkpoint_path"], config["roots"])
        checkpoint_roots = source_training["checkpoint_roots"]
        if not isinstance(checkpoint_roots, list) or not all(isinstance(root, str) and root.startswith("/") for root in checkpoint_roots):
            raise ValueError("Training checkpoint ownership roots must be a list of absolute paths")
        source_warnings = []
        try:
            allowed_path(str(checkpoint), checkpoint_roots)
        except AgentError as exc:
            if exc.code != "PATH_OUTSIDE_ROOTS":
                raise
            source_warnings = [{"code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH",
                "message": "权重目录不属于所选训练的已记录产物；将继续使用该训练配置，请确认模型结构和归一化参数一致。"}]
        automatic = expected["automatic_preflight"]
        plan = automatic["plan"]
        if automatic["plan_digest"] != canonical_digest(plan):
            raise ValueError("Training preflight digest is invalid")
        if len(plan.get("datasets", [])) > 1 or expected.get("training_stage") == "value":
            raise AgentError("POLICY_CONFIG_UNSUPPORTED", "Multi-dataset or value-only training cannot be deployed as an action policy")
        generated = plan["generated_config"]
        source = generated["source"]
        if len(source.encode()) > 384 * 1024 or hashlib.sha256(source.encode()).hexdigest() != generated["sha256"]:
            raise ValueError("Training generated configuration digest is invalid")
        tree = ast.parse(source)
        values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST" for target in node.targets)]
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("Training configuration has no unique frozen request")
        frozen = values[0]
        if frozen.get("training_stage") == "value":
            raise AgentError("POLICY_CONFIG_UNSUPPORTED", "Value-only training cannot produce an action policy")
        current = capture_code(profile, config["roots"], recipe=frozen.get("recipe"),
                               recipe_definition=frozen.get("recipe_definition"), entrypoint=frozen.get("entrypoint"),
                               dataset_bindings=frozen.get("dataset_bindings"), training_stage=frozen.get("training_stage"),
                               stage_config=frozen.get("stage_config"), auto_config=frozen.get("auto_config"))
        compare_code(expected, current, compare_automatic=False)
        if str(allowed_path(frozen["code_dir"], config["roots"])) != profile["code_dir"]:
            raise ValueError("Training configuration belongs to another checkout")
        result = {"frozen_request": frozen, "generated_sha256": generated["sha256"]}
        if source_training.get("run_id") is not None:
            result["source_run_id"] = source_training["run_id"]
        if source_warnings:
            result["source_warnings"] = source_warnings
        return result
    except AgentError:
        raise
    except (KeyError, TypeError, ValueError, SyntaxError, RecursionError) as exc:
        raise AgentError("POLICY_TRAINING_CONFIG_INVALID", str(exc)) from exc


def _portable(config, checkpoint, payload):
    """Confirm the controller's transfer receipt against every local file."""
    portable = payload.get("portable_policy")
    if portable is None:
        return {}
    from vla_adapters.portable_policy import portable_digest, validate_portable_policy
    try:
        if not isinstance(portable, dict):
            raise ValueError("Portable policy must be an object")
        description = validate_portable_policy(portable["config"], portable["digest"])
        manifest = portable["manifest"]
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files", "bytes_total", "files_total"}
                or manifest["schema_version"] != 1 or not isinstance(manifest["files"], list)
                or len(manifest["files"]) > 200000
                or portable_digest(manifest) != portable["checkpoint_manifest_digest"]):
            raise ValueError("Transfer integrity manifest is invalid")
        expected, total = set(), 0
        for item in manifest["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise ValueError("Invalid checkpoint file receipt")
            name = item["path"]
            if (not isinstance(name, str) or not name or name.startswith("/") or ".." in Path(name).parts
                    or name in expected or str(Path(name)) != name or type(item["size"]) is not int or item["size"] < 0
                    or not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
                raise ValueError("Invalid checkpoint file identity")
            expected.add(name)
            path = allowed_path(str(checkpoint / name), config["roots"])
            if path != checkpoint / name or not path.is_relative_to(checkpoint):
                raise ValueError("Checkpoint file escapes the transferred directory")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != item["size"]:
                    raise ValueError("Checkpoint file differs from the verified transfer")
                checksum = hashlib.sha256()
                while block := stream.read(4 * 1024 * 1024):
                    checksum.update(block)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or checksum.hexdigest() != item["sha256"]:
                    raise ValueError("Checkpoint file checksum differs from the verified transfer")
            total += item["size"]
        actual = set()
        # New transfer manifests omit optimizer state. Existing published
        # receipts that explicitly include it retain their full verification.
        omit_train_state = not any(name.startswith("train_state/") for name in expected)
        for root, directories, files in os.walk(checkpoint, followlinks=False):
            for name in directories + files:
                path = Path(root) / name
                if path.is_symlink():
                    raise ValueError("Transferred checkpoint contains a symbolic link")
            if omit_train_state and Path(root) == checkpoint and "train_state" in directories:
                directories.remove("train_state")
            for name in files:
                actual.add(str((Path(root) / name).relative_to(checkpoint)))
        if actual != expected or total != manifest["bytes_total"] or len(expected) != manifest["files_total"]:
            raise ValueError("Checkpoint directory inventory differs from the transfer receipt")
        sidecar = checkpoint / "termitech-policy.json"
        if sidecar.stat().st_size > 1024 * 1024:
            raise ValueError("Portable policy sidecar exceeds size limit")
        if json.loads(sidecar.read_text()) != {"config": description, "digest": portable["digest"]}:
            raise ValueError("Portable policy sidecar differs from the transfer receipt")
        result = {"portable_policy": {"config": description, "digest": portable["digest"]},
                  "checkpoint_manifest_digest": portable["checkpoint_manifest_digest"],
                  "transfer_id": portable.get("transfer_id")}
        for key in ("source_run_id", "source_node_id"):
            if portable.get(key) is not None:
                result[key] = portable[key]
        return result
    except AgentError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RecursionError) as exc:
        raise AgentError("POLICY_TRANSFER_INTEGRITY_INVALID", str(exc)) from exc


def _start_deadline(payload):
    expires = payload.get("expires_at")
    if expires is not None:
        try:
            deadline = datetime.fromisoformat(expires)
            if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
                raise ValueError("expired")
        except (TypeError, ValueError) as exc:
            raise AgentError("POLICY_DEPLOY_EXPIRED", "Policy start request expired; create a fresh deployment") from exc


def _other_execution_running(directory, deployment_id):
    for path in directory.parent.glob("*/manifest.json"):
        if path.parent == directory:
            continue
        manifest = _read(path)
        if not manifest or manifest.get("deployment_id") != deployment_id:
            continue
        if any(identity_matches(_identity(path.parent, mode)) for mode in ("serve", "test")):
            return True
    return False


def _require_previous_execution_stopped(config, deployment_id, directory):
    if _other_execution_running(directory, deployment_id):
        raise AgentError("POLICY_DEPLOY_EXECUTION_BUSY", "A previous execution of this deployment is still running; confirm it stopped before restarting")


def _handle_policy_command(config, command_type, payload, *, prepared=None):
    with _directory(config, payload, wait_seconds=3 if command_type == "POLICY_DEPLOY_STOP" else 0) as directory:
        if command_type == "POLICY_DEPLOY_START":
            if (directory / "stopped.json").exists():
                return _status(directory, health=False)
            existing = _read(directory / "manifest.json")
            if existing:
                # An identifier is one deployment lifecycle; retries recover it
                # rather than launching another service or reusing its port.
                _verify_gpu_policy_manifest(existing, payload)
                return _status(directory)
            if prepared is None:
                _start_deadline(payload)
            profile = get_profile(config, payload["profile_id"])
            if profile.get("adapter") != "openpi-sft":
                raise AgentError("POLICY_PROFILE_UNSUPPORTED", "Policy deployment requires a native OpenPI environment")
            code = allowed_path(profile["code_dir"], config["roots"])
            interpreter = Path(profile["python"])
            if not code.is_dir() or not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                raise AgentError("POLICY_RUNTIME_UNAVAILABLE", "Policy code directory or Python interpreter is unavailable")
            checkpoint = allowed_path(payload["checkpoint_path"], config["roots"])
            if not checkpoint.is_dir() or not ((checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").is_file()):
                raise AgentError("POLICY_CHECKPOINT_INVALID", "Checkpoint requires params/ or model.safetensors")
            assets = allowed_path(str(checkpoint / "assets"), config["roots"])
            norm_paths = [*assets.glob("*/norm_stats.json"), *assets.glob("norm_stats.json")]
            if not assets.is_dir() or not any(allowed_path(str(path), config["roots"]).is_file() for path in norm_paths):
                raise AgentError("POLICY_NORM_MISSING", "Checkpoint assets must contain norm_stats.json")
            for path in (checkpoint / "params", checkpoint / "model.safetensors"):
                if path.exists():
                    allowed_path(str(path), config["roots"])
            inference_backend = _requested_backend(payload)
            resolved_backend = _checkpoint_backend(checkpoint, inference_backend)
            port = payload.get("port")
            if type(port) is not int or not 1024 <= port <= 65535:
                raise AgentError("POLICY_PORT_INVALID", "Policy port must be between 1024 and 65535")
            name = payload.get("config_name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}", name):
                raise AgentError("POLICY_CONFIG_INVALID", "Policy config name is invalid")
            prompt = payload.get("default_prompt")
            if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 4096):
                raise AgentError("POLICY_PROMPT_INVALID", "Default prompt must contain at most 4096 characters")
            gpu = payload.get("gpu_uuid")
            if gpu is not None and (not isinstance(gpu, str) or not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9/-]{1,196}", gpu)):
                raise AgentError("POLICY_GPU_INVALID", "GPU must be selected using its UUID")
            gpu_policy = _requested_gpu_policy(payload)
            if gpu_policy["mode"] == "shared" and not gpu:
                raise AgentError("POLICY_GPU_INVALID", "共享部署必须明确选择 GPU")
            import vla_adapters
            import vla_protocol
            if payload.get("portable_policy") is not None and payload.get("source_training") is not None:
                raise AgentError("POLICY_CONFIG_AMBIGUOUS", "Use either transferred inference configuration or local training configuration")
            inference = prepared if prepared is not None else _portable(config, checkpoint, payload) if payload.get("portable_policy") is not None else _frozen(config, profile, payload)
            manifest = {"deployment_id": payload["deployment_id"], "profile_id": profile["id"],
                        "python": str(interpreter), "code_dir": str(code), "checkpoint_path": str(checkpoint),
                        "config_name": name, "port": port, "default_prompt": prompt, "gpu_uuid": gpu,
                        "inference_backend": inference_backend, "resolved_backend": resolved_backend,
                        "state_dir": config["state_dir"], "nonce": secrets.token_hex(24),
                        "adapter_root": str(Path(vla_adapters.__file__).resolve().parent.parent),
                        "gpu_policy": gpu_policy, "agent_python": sys.executable,
                        "agent_root": str(Path(__file__).resolve().parent.parent),
                        "protocol_root": str(Path(vla_protocol.__file__).resolve().parent.parent), **inference}
            if inference.get("frozen_request") is not None or inference.get("portable_policy") is not None:
                manifest["inference_config_schema"] = 1
            manifest.update(_execution_receipt(payload))
            _require_previous_execution_stopped(config, payload["deployment_id"], directory)
            if gpu:
                from .metrics import check_deployment_gpu
                check_deployment_gpu(gpu, gpu_policy["mode"], gpu_policy["memory_mb"])
            _record(directory / "manifest.json", manifest)
            _launch(directory, manifest, "serve")
            return _status(directory, health=False)
        if command_type == "POLICY_DEPLOY_STATUS":
            return _status(directory)
        if command_type == "POLICY_DEPLOY_STOP":
            manifest = _read(directory / "manifest.json")
            if not manifest:
                # The manifest is durable before any Popen, so absence proves
                # this identifier never launched a service on this Agent.
                # Preserve a tombstone so an older START received after STOP
                # cannot resurrect a cancelled deployment.
                _record(directory / "stopped.json", {"stopped": True, **_execution_receipt(payload)})
                return _status(directory, health=False)
            unconfirmed = False
            for mode in ("test", "serve"):
                identity = _identity(directory, mode)
                if identity_matches(identity):
                    signal_owned_group(identity, signal.SIGTERM)
                    deadline = time.monotonic() + 0.4
                    while identity_matches(identity) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    if identity_matches(identity):
                        signal_owned_group(identity, signal.SIGKILL)
                        deadline = time.monotonic() + 0.2
                        while identity_matches(identity) and time.monotonic() < deadline:
                            time.sleep(0.02)
                    unconfirmed = unconfirmed or identity_matches(identity)
            if unconfirmed:
                result = _status(directory, health=False)
                result["error"] = {"code": "POLICY_STOP_UNCONFIRMED", "message": "A deployment process could not be confirmed stopped; retry stop and inspect its process permissions"}
                return result
            _record(directory / "stopped.json", {"stopped": True, **_execution_receipt(payload)})
            return _status(directory, health=False)
        if command_type == "POLICY_DEPLOY_TEST":
            result = _status(directory)
            if result["state"] != "RUNNING":
                raise AgentError("POLICY_SERVICE_NOT_READY", "Policy WebSocket metadata must be healthy before testing")
            if identity_matches(_identity(directory, "test")):
                raise AgentError("POLICY_TEST_BUSY", "An inference test is already running")
            observation = allowed_path(payload["observation_path"], config["roots"])
            descriptor = os.open(observation, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
                    raise AgentError("POLICY_OBSERVATION_INVALID", "Observation must be a regular JSON file no larger than 16 MiB")
                contents = handle.read(16 * 1024 * 1024 + 1)
            if len(contents) > 16 * 1024 * 1024 or not isinstance(json.loads(contents), dict):
                raise AgentError("POLICY_OBSERVATION_INVALID", "Observation must be a JSON object no larger than 16 MiB")
            snapshot = directory / "test-observation.json"
            snapshot.write_bytes(contents)
            snapshot.chmod(0o600)
            for name in ("test-result.json", "test-identity.json", "test-pid.json"):
                (directory / name).unlink(missing_ok=True)
            _record(directory / "test-request.json", {"observation_path": str(observation), "snapshot_path": str(snapshot)})
            manifest = _read(directory / "manifest.json")
            _launch(directory, manifest, "test")
            result["test_result"] = {"status": "RUNNING", "observation_path": str(observation)}
            return result
        raise AgentError("COMMAND_UNSUPPORTED", "Unknown policy deployment command")


def handle_policy_command(config, command_type, payload):
    if command_type == "POLICY_DEPLOY_START":
        from .profile_rebinding import profile_configuration_lock
        from .recipe_writeback import checkout_lock
        with profile_configuration_lock(config, busy_code="POLICY_DEPLOY_BUSY"):
            profile = get_profile(config, payload["profile_id"])
            with checkout_lock(config, profile["code_dir"], require_git=True):
                # Full checkpoint hashing may take minutes. Never hold the
                # per-deployment operation lock while doing it: STOP must be
                # able to persist its tombstone before any process can launch.
                with _directory(config, payload) as directory:
                    if (directory / "stopped.json").exists():
                        return _status(directory)
                    existing = _read(directory / "manifest.json")
                    if existing is not None:
                        _verify_gpu_policy_manifest(existing, payload)
                        return _status(directory)
                _start_deadline(payload)
                if payload.get("portable_policy") is not None and payload.get("source_training") is not None:
                    raise AgentError("POLICY_CONFIG_AMBIGUOUS", "Use either transferred inference configuration or local training configuration")
                checkpoint = allowed_path(payload["checkpoint_path"], config["roots"])
                prepared = _portable(config, checkpoint, payload) if payload.get("portable_policy") is not None else _frozen(config, profile, payload)
                # Re-entering the lock rechecks the STOP tombstone. There is no
                # yielding work between its final check and owned process launch.
                return _handle_policy_command(config, command_type, payload, prepared=prepared)
    return _handle_policy_command(config, command_type, payload)


def execute_policy_command(config, command_type, payload):
    """Include terminal proof for a launch rejected before creating any process."""
    try:
        return handle_policy_command(config, command_type, payload)
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "POLICY_DEPLOY_FAILED", "message": str(exc)[:2000]}
        result = {"error": error}
        try:
            result.update(_execution_receipt(payload))
        except AgentError:
            pass
        if command_type == "POLICY_DEPLOY_START":
            try:
                deployment_id, execution_id = _execution_identifiers(payload)
                directory = Path(config["state_dir"]) / "policy-deployments" / execution_id
                if (error["code"] not in {"POLICY_DEPLOY_EXECUTION_BUSY", "POLICY_DEPLOY_EXECUTION_CONFLICT"}
                        and not (directory / "manifest.json").exists()
                        and not _other_execution_running(directory, deployment_id)):
                    result.update(state="FAILED", pid=None, logs="", checks=[{"name": "deployment_preflight", "ok": False, "message": error["message"]}])
            except AgentError:
                pass
        return result


def active_policy_checkouts(config):
    """Used by local source mutations as well as control-plane guards."""
    root = Path(config["state_dir"]) / "policy-deployments"
    if not root.exists():
        return []
    active = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = _read(manifest_path)
        if manifest and identity_matches(_identity(manifest_path.parent)):
            active.append(Path(manifest["code_dir"]).resolve())
    return active
