"""Detached supervisor; agent restarts and network outages do not stop training."""
from __future__ import annotations

import argparse
import fcntl
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

from .artifacts import ArtifactMonitor
from .config import load_config
from .metrics import check_gpus, process_metrics
from .preflight import validate_local_inputs
from .safety import AgentError, allowed_path, identity_matches, process_identity, safe_identifier, signal_owned_group
from .snapshots import capture_code, compare_code, git
from .state import State, TERMINAL_STATES
from .training_validation import prepare_launch, validate_training


def verified_outputs(output_dir: Path, smoke: bool) -> list[dict]:
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
        if not smoke and not any(artifact["kind"] == "MODEL" for artifact in result):
            raise ValueError("Training exited without a model artifact")
        return result
    except (OSError, ValueError, TypeError, AgentError) as exc:
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
        output = allowed_path(attempt["output_dir"], [config["output_root"]], must_exist=False)
        profile, inputs = validate_local_inputs(config, payload)
        code_dir = Path(git(Path(profile["code_dir"]), "rev-parse", "--show-toplevel").decode().strip()).resolve()
        if output == code_dir or code_dir in output.parents:
            raise AgentError("INVALID_PATH", "Attempt output must be outside the training code directory")
        for asset in inputs.values():
            asset_path = Path(asset["path"])
            if asset_path == output or asset_path in output.parents or output in asset_path.parents:
                raise AgentError("INVALID_PATH", "Attempt output must be separate from input assets")
        output.mkdir(parents=True, exist_ok=False)
        start_snapshot = capture_code(profile, config["roots"], output / "provenance", recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
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
        launch = build_launch(profile, payload["template_id"], inputs, payload.get("parameters", {}), str(output),
                              payload.get("gpu_uuids", []), recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
        allowed_path(launch["cwd"], config["roots"])
        if profile.get("adapter") != "smoke":
            if not isinstance(payload["code_expectation"].get("automatic_preflight"), dict):
                raise AgentError("AUTOMATIC_PREFLIGHT_REQUIRED", "正式训练缺少 Agent 自动预检记录，请重新执行预检")
            launch = prepare_launch(launch, config, payload.get("inputs", {}))
            automatic = validate_training(config, launch, launch_checks, assigned=payload.get("gpu_uuids", []))
            current = capture_code(profile, config["roots"], recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
            compare_code(start_snapshot, current, compare_automatic=False)
            start_snapshot["automatic_preflight"] = automatic
            compare_code(payload["code_expectation"], start_snapshot)
            launch = prepare_launch(launch, config, payload.get("inputs", {}), expected_plan_digest=automatic["plan_digest"])
            (output / "provenance" / "code-snapshot.json").write_text(json.dumps(start_snapshot, indent=2))
        # Recheck assigned devices after the CPU probe, just before spawning.
        check_gpus(payload.get("gpu_uuids", []))
        if state.attempt(attempt_id)["cancel_requested"] or interrupted:
            state.emit(attempt_id, "CANCELLED", {"exit_code": None, "process_started": False})
            return 0
        platform_credentials = {"VLA_ADMIN_TOKEN", "VLA_ENROLLMENT_TOKEN", "VLA_DATABASE_URL", "VLA_AGENT_TOKEN"}
        environment = {key: value for key, value in os.environ.items() if key not in platform_credentials}
        environment.update({**launch.get("env", {}), "PYTHONUNBUFFERED": "1"})
        child = subprocess.Popen(launch["argv"], cwd=launch["cwd"], env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        # Even a very short command must be reaped; psutil may observe a zombie immediately.
        if identity is None:
            identity = {"pid": child.pid, "created_at": psutil.Process(child.pid).create_time()}
        state.set_identity(attempt_id, "child", identity)
        state.emit(attempt_id, "STARTED", {"pid": child.pid, "code_snapshot": start_snapshot, "output_dir": str(output)})
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
        code_changed = False
        try:
            end_snapshot = capture_code(profile, config["roots"], output / "provenance" / "end", recipe=payload.get("recipe"), recipe_definition=payload.get("recipe_definition"), entrypoint=payload.get("entrypoint"), dataset_bindings=payload.get("dataset_bindings"), training_stage=payload.get("training_stage"), stage_config=payload.get("stage_config"))
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
                ArtifactMonitor(state, config["output_root"]).report(state.attempt(attempt_id))
            except Exception:
                logging.getLogger("vla_agent").warning(
                    "Final checkpoint discovery failed for %s; preserving process exit result", attempt_id, exc_info=True)
        if cancelling:
            state.emit(attempt_id, "CANCELLED", common)
        elif child.returncode == 0:
            state.emit(attempt_id, "SUCCEEDED", {**common, "outputs": verified_outputs(output, profile.get("adapter") == "smoke")})
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
