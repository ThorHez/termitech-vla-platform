"""Detached supervisor; agent restarts and network outages do not stop training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path

import psutil
from vla_adapters import build_launch
from vla_protocol.config_preview import validate_generated_config

from .artifacts import ArtifactMonitor
from .config import load_config
from .metrics import check_gpus, check_training_gpus, process_metrics
from .output_isolation import validate_output_isolation
from .preflight import validate_local_inputs
from .safety import AgentError, allowed_path, identity_matches, process_identity, safe_identifier, signal_owned_group
from .snapshots import capture_code, compare_code
from .state import State, TERMINAL_STATES
from .training_validation import prepare_launch, validate_training
from .training_gpu import shared_training_environment, training_gpu_evidence, validate_training_gpu_policy


def verified_outputs(output_dir: Path, smoke: bool, *, template_id: str | None = None,
                     norm_datasets: list[dict] | None = None) -> list[dict]:
    manifest = output_dir / "artifacts.json"
    if not manifest.exists():
        if not smoke:
            raise AgentError("ARTIFACT_INVALID", "Training exited without an artifact manifest")
        return []
    manifest = allowed_path(str(manifest), [str(output_dir)])
    try:
        value = json.loads(manifest.read_text())
        outputs = value.get("outputs") if isinstance(value, dict) else value
        if not isinstance(outputs, list):
            raise ValueError("artifacts.json must contain an output list")
        result = []
        for output in outputs:
            if not isinstance(output, dict) or not all(key in output for key in ("kind", "name", "path", "version")):
                raise ValueError("Each artifact requires kind, name, path, and version")
            if output["kind"] not in {"DATASET", "MODEL", "NORM"}:
                raise ValueError("Unknown artifact kind")
            if not isinstance(output["name"], str) or not isinstance(output["version"], str):
                raise ValueError("Artifact names and versions must be strings")
            path = allowed_path(output["path"], [str(output_dir)])
            metadata = {**output.get("metadata", {})}
            if smoke:
                metadata["simulation"] = True
            result.append({**output, "path": str(path), "metadata": metadata})
        required_kind = "NORM" if template_id == "openpi-norm" else "MODEL"
        if not smoke and not any(artifact["kind"] == required_kind for artifact in result):
            raise ValueError(f"Task exited without a {required_kind} artifact")
        if template_id == "openpi-norm":
            from vla_adapters.norm_entry import verify_statistics
            from vla_adapters.norm_datasets import normalize_norm_datasets
            rows = normalize_norm_datasets(norm_datasets)
            if rows is not None and (len(result) != len(rows) or any(
                artifact["kind"] != "NORM" or artifact.get("metadata", {}).get("source_dataset_asset_id") != row["dataset_asset_id"]
                for artifact, row in zip(result, rows)
            ) or len({artifact["path"] for artifact in result}) != len(rows)):
                raise ValueError("Statistics batch output does not match every independent input dataset")
            for artifact in result:
                if artifact["kind"] == "NORM":
                    verify_statistics(Path(artifact["path"]))
        return result
    except (OSError, ValueError, TypeError, KeyError, AgentError) as exc:
        raise AgentError("ARTIFACT_INVALID", f"Cannot verify output manifest: {exc}") from exc


def _descendants(identity: dict | None) -> list[dict]:
    if not identity_matches(identity):
        return []
    try:
        return [value for child in psutil.Process(identity["pid"]).children(recursive=True)
                if (value := process_identity(child.pid)) is not None]
    except (psutil.Error, OSError):
        return []


def _kill_owned(identity: dict | None, descendants: list[dict], sig: int):
    if identity:
        signal_owned_group(identity, sig)
    for descendant in descendants:
        if identity_matches(descendant):
            try:
                os.kill(descendant["pid"], sig)
            except ProcessLookupError:
                pass


def execute(config: dict, attempt_id: str) -> int:
    safe_identifier(attempt_id)
    state = State(config["state_dir"])
    attempt = state.attempt(attempt_id)
    if not attempt or attempt["state"] in TERMINAL_STATES:
        return 0
    lock_dir = Path(config["state_dir"]) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / f"attempt-{attempt_id}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    attempt = state.attempt(attempt_id)
    if not attempt or attempt["state"] in TERMINAL_STATES:
        lock.close()
        return 0
    # An existing child means this is recovery, not authorization to launch a replacement.
    if attempt["child_identity"]:
        lock.close()
        return 0
    state.set_identity(attempt_id, "runner", process_identity(os.getpid()))
    payload = attempt["payload"]
    child = None
    identity = None
    descendants: list[dict] = []
    gpu_locks = []
    selector = None
    artifact_monitor = None
    output = None
    profile = None
    start_snapshot = None
    launch_checks = []
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    old_signals = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        runtime = attempt.get("runner_runtime")
        if runtime:
            from .runner_runtime import installed_runtime
            installed = installed_runtime()
            if (not installed or installed["runtime_path"] != runtime["runtime_path"]
                    or installed.get("build_id") != runtime.get("build_id")
                    or config["config_path"] != runtime["config_path"]
                    or hashlib.sha256(Path(config["config_path"]).read_bytes()).hexdigest() != runtime["config_sha256"]):
                raise AgentError("RUNNER_RUNTIME_CHANGED", "训练监控环境或配置已变化，拒绝启动训练")
        if attempt["cancel_requested"]:
            state.emit(attempt_id, "CANCELLED", {"exit_code": None, "process_started": False})
            return 0
        output = allowed_path(attempt["output_dir"], [config["output_root"]], must_exist=False)
        profile, inputs = validate_local_inputs(config, payload)
        policy = validate_training_gpu_policy(payload, profile, require_frozen=True)
        if profile.get("adapter") != "smoke":
            expectation = payload.get("code_expectation", {})
            frozen_plan = expectation.get("automatic_preflight", {}).get("plan", {})
            if frozen_plan.get("resume_source") != payload.get("resume_source"):
                raise AgentError("RESUME_SOURCE_MISMATCH", "启动命令与预检冻结的续训来源不一致，请重新预检")
            generated = frozen_plan.get("generated_config")
            try:
                validate_generated_config(generated)
            except ValueError as exc:
                raise AgentError("CONFIG_CONFIRMATION_REQUIRED", "训练缺少可核对的 config，请重新预检并确认") from exc
            confirmed = expectation.get("confirmed_config_sha256")
            if confirmed is None:
                raise AgentError("CONFIG_CONFIRMATION_REQUIRED", "训练缺少用户 config 确认，请重新预检并确认")
            if confirmed != generated["sha256"] or payload.get("confirmed_config_sha256") != confirmed:
                raise AgentError("CONFIG_CONFIRMATION_MISMATCH", "启动命令与用户确认的 config 不一致，请重新预检并确认")
        validate_output_isolation(config, profile["code_dir"], inputs, str(output))
        output.mkdir(parents=True, exist_ok=False)
        start_snapshot = capture_code(profile, config["roots"], output / "provenance", recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
        compare_code(payload["code_expectation"], start_snapshot, compare_automatic=False)
        for uuid in sorted(payload.get("gpu_uuids", [])):
            # GPU UUIDs never become path components directly.
            from hashlib import sha256
            gpu_lock = (lock_dir / f"gpu-{sha256(uuid.encode()).hexdigest()}.lock").open("a")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                gpu_lock.close()
                raise AgentError("GPU_BUSY", f"GPU reserved by another local Runner: {uuid}") from exc
            gpu_locks.append(gpu_lock)
        from .resource_guard import require_gpu_access
        require_gpu_access(config, payload.get("gpu_uuids", []))
        if policy is not None:
            check_training_gpus(payload["gpu_uuids"], policy)
        launch = build_launch(profile, payload["template_id"], inputs, payload.get("parameters", {}), str(output),
                              payload.get("gpu_uuids", []), recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}), **({"resume_source": payload["resume_source"]} if payload.get("resume_source") is not None else {}))
        allowed_path(launch["cwd"], config["roots"])
        if profile.get("adapter") != "smoke":
            if not isinstance(payload["code_expectation"].get("automatic_preflight"), dict):
                raise AgentError("AUTOMATIC_PREFLIGHT_REQUIRED", "正式训练缺少 Agent 自动预检记录，请重新执行预检")
            launch = prepare_launch(launch, config, payload.get("inputs", {}))
            automatic = validate_training(config, launch, launch_checks, assigned=payload.get("gpu_uuids", []))
            current = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
            compare_code(start_snapshot, current, compare_automatic=False)
            start_snapshot["automatic_preflight"] = automatic
            compare_code(payload["code_expectation"], start_snapshot)
            launch = prepare_launch(launch, config, payload.get("inputs", {}), expected_plan_digest=automatic["plan_digest"])
            (output / "provenance" / "code-snapshot.json").write_text(json.dumps(start_snapshot, indent=2))
        # Recheck assigned devices after the CPU probe, just before spawning.
        devices = None
        if policy is None:
            check_gpus(payload.get("gpu_uuids", []))
        else:
            devices = check_training_gpus(payload["gpu_uuids"], policy)
            start_snapshot["training_gpu_check"] = training_gpu_evidence(policy, payload["gpu_uuids"], devices)
            (output / "provenance" / "code-snapshot.json").write_text(json.dumps(start_snapshot, indent=2))
        require_gpu_access(config, payload.get("gpu_uuids", []))
        if state.attempt(attempt_id)["cancel_requested"] or interrupted:
            state.emit(attempt_id, "CANCELLED", {"exit_code": None, "process_started": False})
            return 0
        platform_credentials = {"VLA_ADMIN_TOKEN", "VLA_ENROLLMENT_TOKEN", "VLA_DATABASE_URL", "VLA_AGENT_TOKEN"}
        environment = {key: value for key, value in os.environ.items() if key not in platform_credentials}
        environment.update({**launch.get("env", {}), "PYTHONUNBUFFERED": "1"})
        from .storage_configuration import cache_environment
        environment.update(cache_environment(config))
        environment = shared_training_environment(environment, policy, devices)
        child = subprocess.Popen(launch["argv"], cwd=launch["cwd"], env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        # Even a very short command must be reaped; psutil may observe a zombie immediately.
        if identity is None:
            identity = {"pid": child.pid, "created_at": psutil.Process(child.pid).create_time()}
        state.set_identity(attempt_id, "child", identity)
        state.emit(attempt_id, "STARTED", {"pid": child.pid, "code_snapshot": start_snapshot, "output_dir": str(output)})
        if profile.get("adapter") == "openpi-sft":
            artifact_monitor = ArtifactMonitor(state, config["output_root"],
                                               output_roots=config.get("_output_root_history", []))
        selector = selectors.DefaultSelector()
        os.set_blocking(child.stdout.fileno(), False)
        selector.register(child.stdout, selectors.EVENT_READ)
        buffer = b""
        last_metrics = 0.0
        cancelling = False
        cancel_started = 0.0
        drain_deadline = None

        def record_line(raw: bytes):
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if text.startswith("VLA_EVENT:"):
                try:
                    event = json.loads(text[len("VLA_EVENT:"):].strip())
                    # Training code cannot manufacture a terminal lifecycle event.
                    if event.get("type") == "METRIC" and isinstance(event.get("data"), dict):
                        state.emit(attempt_id, "METRIC", event["data"])
                        return
                except (ValueError, AttributeError):
                    pass
            state.emit(attempt_id, "LOG", {"text": text})

        with (output / "train.log").open("ab", buffering=0) as log:
            while True:
                now = time.monotonic()
                if identity_matches(identity):
                    new_descendants = _descendants(identity)
                    by_identity = {(p["pid"], p["created_at"]): p for p in descendants + new_descendants}
                    descendants = list(by_identity.values())
                current = state.attempt(attempt_id)
                if artifact_monitor:
                    artifact_monitor.schedule([current])
                if (current["cancel_requested"] or interrupted) and not cancelling:
                    cancelling, cancel_started = True, now
                    _kill_owned(identity, descendants, signal.SIGTERM)
                if cancelling and now - cancel_started > config.get("cancel_grace_seconds", 10):
                    _kill_owned(identity, descendants, signal.SIGKILL)
                if now - last_metrics >= config.get("heartbeat_interval_seconds", 10):
                    sampled = process_metrics(identity)
                    if sampled:
                        state.emit(attempt_id, "RESOURCE", sampled)
                    last_metrics = now
                for key, _ in selector.select(timeout=0.1):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        for offset in range(0, max(len(line), 1), 16384):
                            record_line(line[offset:offset + 16384])
                    while len(buffer) > 16384:
                        record_line(buffer[:16384])
                        buffer = buffer[16384:]
                if child.poll() is not None:
                    alive = [p for p in descendants if identity_matches(p)]
                    if alive:
                        _kill_owned(None, alive, signal.SIGTERM if drain_deadline is None else signal.SIGKILL)
                    if drain_deadline is None:
                        drain_deadline = now + 3
                    if not alive and (not selector.get_map() or now >= drain_deadline):
                        break
            if buffer:
                record_line(buffer)
        if artifact_monitor:
            artifact_monitor.close()
        code_changed = False
        try:
            end_snapshot = capture_code(profile, config["roots"], output / "provenance" / "end", recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"), **({"auto_config": payload["auto_config"]} if payload.get("auto_config") is not None else {}), **({"normalization_method": payload["normalization_method"]} if payload.get("normalization_method") is not None else {}), **({"norm_datasets": payload["norm_datasets"]} if payload.get("norm_datasets") is not None else {}), **({"training_gpu_policy": policy} if policy is not None else {}))
            if "automatic_preflight" in start_snapshot:
                end_snapshot["automatic_preflight"] = start_snapshot["automatic_preflight"]
                (output / "provenance" / "end" / "code-snapshot.json").write_text(json.dumps(end_snapshot, indent=2))
            try:
                compare_code(start_snapshot, end_snapshot, compare_automatic=False)
            except AgentError:
                code_changed = True
        except AgentError:
            code_changed = True
        common = {"exit_code": child.returncode, "code_changed_during_run": code_changed}
        if profile.get("adapter") == "openpi-sft":
            try:
                # The last save can finish between periodic scans. Publish its
                # checkpoint and confirmed retention removals before lifecycle
                # completion closes the attempt's transactional outbox.
                ArtifactMonitor(state, config["output_root"], output_roots=config.get("_output_root_history", [])).report(state.attempt(attempt_id))
            except Exception:
                logging.getLogger("vla_agent").warning(
                    "Final checkpoint discovery failed for %s; preserving process exit result", attempt_id, exc_info=True)
        if cancelling:
            state.emit(attempt_id, "CANCELLED", common)
        elif child.returncode == 0:
            state.emit(attempt_id, "SUCCEEDED", {**common, "outputs": verified_outputs(output, profile.get("adapter") == "smoke", template_id=payload["template_id"], norm_datasets=payload.get("norm_datasets"))})
        else:
            state.emit(attempt_id, "FAILED", {**common, "error": {"code": "PROCESS_EXITED", "message": f"Training process exited with code {child.returncode}"}})
        return 0
    except Exception as exc:
        if child is not None:
            _kill_owned(identity, descendants, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_owned(identity, descendants, signal.SIGKILL)
                child.wait(timeout=5)
            alive = [p for p in descendants if identity_matches(p)]
            if alive:
                # No terminal event (and no resource release) until live workers are gone.
                return 2
        error = exc.as_dict() if isinstance(exc, AgentError) else {"code": "RUNNER_ERROR", "message": str(exc)}
        state.emit(attempt_id, "FAILED", {"exit_code": child.returncode if child else None, "error": error,
                                           "process_started": child is not None, "code_snapshot": start_snapshot,
                                           "checks": launch_checks})
        return 1
    finally:
        if artifact_monitor:
            artifact_monitor.close()
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        if selector:
            selector.close()
        if child and child.stdout:
            child.stdout.close()
        for gpu_lock in gpu_locks:
            gpu_lock.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(load_config(args.config), args.attempt_id))


if __name__ == "__main__":
    main()
