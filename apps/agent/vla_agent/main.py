from __future__ import annotations

import argparse
import fcntl
import json
import logging
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import psutil

from .artifacts import ArtifactMonitor
from .config import load_config
from .metrics import gpu_inventory, process_metrics, system_metrics
from .path_suggestions import suggest_paths
from .preflight import run_preflight
from .recipe_details import recipe_details
from .recipe_suggestions import suggest_recipes
from .runtime_checks import RuntimeMonitor
from .safety import AgentError, identity_matches, process_identity, safe_identifier, signal_owned_group
from .state import State, TERMINAL_STATES, utc_now


LOG = logging.getLogger("vla_agent")


def _execute_preflight_command(config: dict, command: dict):
    """One isolated worker; only the main Agent communicates with the server.

    Unacknowledged commands remain on the control plane. An interrupted worker
    writes no completion, so restart safely retries the same command. A local
    lock also prevents a worker orphaned by SIGKILL from duplicating probes after
    the Agent restarts.
    """
    os.setsid()

    def stop(_signum, _frame):
        # BaseException passes through preflight's error reporting and executes
        # subprocess cleanup, without persisting shutdown as a rejected command.
        # Discover children here as well: a signal can arrive just after Popen
        # forks, before the normal probe cleanup context has been established.
        try:
            descendants = [identity for child in psutil.Process().children(recursive=True)
                           if (identity := process_identity(child.pid)) is not None]
        except (psutil.Error, OSError):
            descendants = []
        for identity in descendants:
            signal_owned_group(identity, signal.SIGKILL)
            if identity_matches(identity):
                try:
                    os.kill(identity["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (Path(config["state_dir"]) / "preflight-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = State(config["state_dir"])
        if state.command(command["id"]):
            return
        try:
            if command["type"] == "RECIPE_WRITE":
                from .recipe_writeback import write_recipe
                result = write_recipe(config, command["payload"], state)
            elif command["type"] == "DATA_PROCESSORS":
                from .data_processors import data_processors
                result = data_processors(config, command["payload"])
            else:
                result = run_preflight(config, command["payload"])
            status = "REJECTED" if result.get("error") else "APPLIED"
        except Exception as exc:
            if command["type"] == "RECIPE_WRITE" and (isinstance(exc, OSError)
                    or isinstance(exc, AgentError) and exc.code == "RECIPE_WRITE_RECOVERY_PENDING"):
                LOG.warning("Recipe commit %s needs recovery; leaving its command unacknowledged", command["id"])
                return
            result = {"error": exc.as_dict() if isinstance(exc, AgentError) else {
                "code": "RECIPE_WRITE_FAILED" if command["type"] == "RECIPE_WRITE" else "PREFLIGHT_FAILED", "message": str(exc)}}
            status = "REJECTED"
        state.save_command(command["id"], status, result)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Every Agent endpoint has a stable address. urllib's default redirect
        # handler forwards Authorization to a different host and even from
        # HTTPS to HTTP; reject instead of transferring a node credential.
        return None


class Client:
    def __init__(self, config: dict):
        self.url = config["control_plane_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("control_plane_url must be an HTTP(S) URL without embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote control planes require HTTPS to protect node credentials")
        self.timeout = config["http_timeout_seconds"]
        self._opener = urllib.request.build_opener(_RejectRedirects())

    def request(self, method: str, path: str, token: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", **(headers or {})})
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read())


class Agent:
    def __init__(self, config: dict, client=None):
        self.config = config
        self.state = State(config["state_dir"])
        self.client = client if client is not None else Client(config)
        self.credentials = self.state.credentials()
        self.runners: dict[str, subprocess.Popen] = {}
        self.last_profile_sync = None
        self.runtime_monitor = RuntimeMonitor(config["roots"])
        self.artifact_monitor = ArtifactMonitor(self.state, config["output_root"])
        self._preflight_worker = None
        self._preflight_command_id: str | None = None
        self._closed = False

    def register(self):
        if self.credentials:
            return
        instance_path = Path(self.config["state_dir"]) / "instance-id"
        if not instance_path.exists():
            instance_path.write_text(str(uuid.uuid4()))
            instance_path.chmod(0o600)
        registration = {key: self.config[key] for key in ("name", "roots", "output_root", "profiles")}
        registration["hostname"] = socket.gethostname()
        credentials = self.client.request("POST", "/agent/v1/register", self.config.get("enrollment_token", ""),
                                          registration, headers={"Idempotency-Key": instance_path.read_text().strip()})
        if not credentials.get("node_id") or not credentials.get("token"):
            raise ValueError("Invalid node registration response")
        self.state.save_credentials(credentials)
        self.credentials = credentials

    def api(self, method: str, path: str, body: dict | None = None):
        if not self.credentials:
            raise ValueError("Agent is not registered")
        return self.client.request(method, path, self.credentials["token"], body)

    def handle_command(self, command: dict) -> dict:
        previous = self.state.command(command["id"])
        if previous:
            return previous
        payload = command["payload"]
        try:
            if command["type"] == "PATH_SUGGEST":
                result, status = suggest_paths(self.config, payload), "APPLIED"
            elif command["type"] == "RECIPE_SUGGEST":
                result, status = suggest_recipes(self.config, payload), "APPLIED"
            elif command["type"] == "RECIPE_DETAIL":
                result, status = recipe_details(self.config, payload), "APPLIED"
            elif command["type"] == "DATA_PROCESSORS":
                from .data_processors import data_processors
                result, status = data_processors(self.config, payload), "APPLIED"
            elif command["type"] == "RECIPE_WRITE":
                from .recipe_writeback import write_recipe
                result, status = write_recipe(self.config, payload, self.state), "APPLIED"
            elif command["type"] == "PREFLIGHT":
                result = run_preflight(self.config, payload)
                status = "REJECTED" if result.get("error") else "APPLIED"
            elif command["type"] == "START":
                attempt_id = safe_identifier(payload["attempt_id"])
                if not payload.get("execution_token"):
                    raise AgentError("INVALID_COMMAND", "START requires an execution token")
                output = str(Path(self.config["output_root"]) / attempt_id)
                profile = next((item for item in self.config["profiles"] if item["id"] == payload.get("profile_id")), None)
                if profile is None:
                    raise AgentError("PROFILE_UNAVAILABLE", "START requires a configured training environment")
                from .recipe_writeback import checkout_lock
                # Serialize the durable attempt registration with source writes.
                # A writer then sees this STARTING attempt before it can edit.
                with checkout_lock(self.config, profile["code_dir"], require_git=profile.get("adapter") == "openpi-sft"):
                    created = self.state.create_attempt(payload, output)
                if created:
                    logs = Path(self.config["state_dir"]) / "runner-logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    try:
                        with (logs / f"{attempt_id}.log").open("ab") as runner_log:
                            process = subprocess.Popen([sys.executable, "-m", "vla_agent.runner", "--config", self.config["config_path"],
                                                        "--attempt-id", attempt_id], stdin=subprocess.DEVNULL,
                                                       stdout=runner_log, stderr=subprocess.STDOUT,
                                                       start_new_session=True, close_fds=True)
                        self.runners[attempt_id] = process
                    except OSError as exc:
                        self.state.emit(attempt_id, "FAILED", {"exit_code": None, "process_started": False,
                                                                "error": {"code": "RUNNER_LAUNCH_FAILED", "message": str(exc)}})
                result, status = {"attempt_id": attempt_id, "accepted": True, "duplicate": not created}, "APPLIED"
            elif command["type"] == "CANCEL":
                self.state.request_cancel(payload["attempt_id"], payload["execution_token"])
                result, status = {"attempt_id": payload["attempt_id"], "cancel_requested": True}, "APPLIED"
            else:
                raise AgentError("COMMAND_UNSUPPORTED", f"Unknown command type: {command['type']}")
        except Exception as exc:
            if command["type"] == "RECIPE_WRITE" and (isinstance(exc, OSError)
                    or isinstance(exc, AgentError) and exc.code == "RECIPE_WRITE_RECOVERY_PENDING"):
                raise
            if command["type"] == "START" and isinstance(exc, AgentError) and exc.code == "RECIPE_WRITE_BUSY":
                raise  # Keep START unacknowledged until its checkout is available.
            result = {"error": exc.as_dict() if isinstance(exc, AgentError) else {"code": "COMMAND_REJECTED", "message": str(exc)}}
            status = "REJECTED"
        self.state.save_command(command["id"], status, result)
        return {"status": status, "result": result}

    def poll_commands(self):
        self._collect_preflight_worker()
        commands = self.api("GET", "/agent/v1/commands?" + urllib.parse.urlencode({"node_id": self.credentials["node_id"]}))
        for command in commands:
            acknowledgement = self.state.command(command["id"])
            if acknowledgement is None and command["type"] in {"PREFLIGHT", "RECIPE_WRITE", "DATA_PROCESSORS"}:
                if self._preflight_worker is None and not self._closed:
                    worker = multiprocessing.get_context("spawn").Process(
                        target=_execute_preflight_command, args=(self.config, command),
                        name="vla-background-command", daemon=True)
                    worker.start()
                    self._preflight_worker = worker
                    self._preflight_command_id = command["id"]
                # No ACK until a durable terminal result exists. Continue this
                # response so START/CANCEL never wait for a slow training import
                # or a source write and Git commit.
                continue
            if acknowledgement is None:
                try:
                    acknowledgement = self.handle_command(command)
                except AgentError as exc:
                    if ((command["type"] == "START" and exc.code == "RECIPE_WRITE_BUSY")
                            or command["type"] == "RECIPE_WRITE" and exc.code == "RECIPE_WRITE_RECOVERY_PENDING"):
                        continue
                    raise
            self.api("POST", f"/agent/v1/commands/{urllib.parse.quote(command['id'], safe='')}/ack",
                     {"node_id": self.credentials["node_id"], **acknowledgement})

    def _collect_preflight_worker(self):
        worker = self._preflight_worker
        if worker is not None and not worker.is_alive():
            worker.join(timeout=0)
            if worker.exitcode:
                LOG.warning("Preflight worker exited with code %s; unacknowledged command %s remains retryable",
                            worker.exitcode, self._preflight_command_id)
            worker.close()
            self._preflight_worker = None
            self._preflight_command_id = None

    def close(self):
        """Stop background workers while detached training Runners keep running."""
        self._closed = True
        self.runtime_monitor.close()
        self.artifact_monitor.close()
        worker = self._preflight_worker
        if worker is None:
            return
        descendants = []
        if worker.is_alive():
            try:
                descendants = [identity for child in psutil.Process(worker.pid).children(recursive=True)
                               if (identity := process_identity(child.pid)) is not None]
            except (psutil.Error, OSError):
                pass
            worker.terminate()
            worker.join(timeout=2)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=3)
        # Includes the wrapper's separate process group, even after its worker
        # has exited. Identity checks exclude reused process IDs.
        for identity in descendants:
            signal_owned_group(identity, signal.SIGKILL)
            if identity_matches(identity):
                try:
                    os.kill(identity["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._collect_preflight_worker()

    def flush_events(self):
        # Exact persisted payloads are retried; successful delivery followed by a crash is safe.
        events = self.state.pending_events(self.config["event_batch_size"])
        if not events:
            return
        response = self.api("POST", "/agent/v1/events/batch", {"node_id": self.credentials["node_id"], "events": events})
        if response.get("accepted") != len(events):
            raise ValueError("Control plane did not acknowledge the entire event batch")
        self.state.acknowledge_events(events)

    def heartbeat(self):
        for attempt_id, process in list(self.runners.items()):
            if process.poll() is not None:
                del self.runners[attempt_id]
        active = []
        owned_pids = {}
        process_samples = []
        for attempt in self.state.active_attempts():
            runner_alive = identity_matches(attempt["runner_identity"])
            state = attempt["state"] if runner_alive else "UNKNOWN"
            child = attempt["child_identity"]
            active.append({"attempt_id": attempt["id"], "pid": child["pid"] if child and identity_matches(child) else None, "state": state})
            sampled = process_metrics(child) if child else None
            if sampled:
                process_samples.append({"attempt_id": attempt["id"], **sampled})
                owned_pids.update({pid: attempt["id"] for pid in sampled["process_ids"]})
        gpus, availability = gpu_inventory()
        for gpu in gpus:
            if gpu["processes"] is not None:
                for process in gpu["processes"]:
                    process["attempt_id"] = owned_pids.get(process["pid"])
                    process["ownership"] = "platform" if process["attempt_id"] else "external"
        metrics = {**system_metrics(self.config["output_root"]), "gpu_monitoring": availability,
                   "attempt_processes": process_samples, "sampled_at": utc_now(),
                   "runtime_checks": self.runtime_monitor.snapshot(self.config["profiles"])}
        response = self.api("POST", "/agent/v1/heartbeat", {"node_id": self.credentials["node_id"], "gpus": gpus,
                                                            "metrics": metrics, "active_attempts": active,
                                                            "profiles": self.config["profiles"]})
        sync = response.get("profile_sync")
        if sync != self.last_profile_sync:
            if sync and sync.get("status") == "BLOCKED":
                LOG.warning("Runtime profile synchronization deferred: %s", sync.get("message", sync.get("code")))
            self.last_profile_sync = sync

    def run(self, once: bool = False):
        stopping = False

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        last_heartbeat = 0.0
        try:
            while not stopping:
                self.artifact_monitor.schedule(self.state.active_attempts())
                try:
                    self.register()
                except (OSError, ValueError, urllib.error.URLError):
                    LOG.exception("Registration failed; persisted local Runner state remains available")
                if self.credentials:
                    if time.monotonic() - last_heartbeat >= self.config["heartbeat_interval_seconds"]:
                        try:
                            self.heartbeat()
                            last_heartbeat = time.monotonic()
                        except (OSError, ValueError, urllib.error.URLError):
                            LOG.exception("Heartbeat failed; local training continues")
                    for operation in (self.flush_events, self.poll_commands):
                        try:
                            operation()
                        except (OSError, ValueError, urllib.error.URLError):
                            LOG.exception("%s failed; other channels remain independently retryable", operation.__name__)
                if once:
                    break
                deadline = time.monotonic() + self.config["poll_interval_seconds"]
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        finally:
            self.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description="VLA bare-metal agent (native execution)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true", help="Run one control-plane cycle for diagnostics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    # Keep the credentials, SQLite WAL and local process evidence private to the service user.
    os.umask(0o077)
    with (Path(config["state_dir"]) / "agent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another agent already owns this state directory")
        Agent(config).run(once=args.once)


if __name__ == "__main__":
    main()
