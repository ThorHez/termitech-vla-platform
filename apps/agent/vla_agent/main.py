from __future__ import annotations

import argparse
import fcntl
import json
import logging
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

from .config import load_config
from .metrics import gpu_inventory, process_metrics, system_metrics
from .preflight import run_preflight
from .safety import AgentError, identity_matches, safe_identifier
from .state import State, TERMINAL_STATES, utc_now


LOG = logging.getLogger("vla_agent")


class Client:
    def __init__(self, config: dict):
        self.url = config["control_plane_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("control_plane_url must be an HTTP(S) URL without embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote control planes require HTTPS to protect node credentials")
        self.timeout = config["http_timeout_seconds"]

    def request(self, method: str, path: str, token: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


class Agent:
    def __init__(self, config: dict, client=None):
        self.config = config
        self.state = State(config["state_dir"])
        self.client = client if client is not None else Client(config)
        self.credentials = self.state.credentials()
        self.runners: dict[str, subprocess.Popen] = {}
        self.last_profile_sync = None

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
            if command["type"] == "PREFLIGHT":
                result = run_preflight(self.config, payload)
                status = "REJECTED" if result.get("error") else "APPLIED"
            elif command["type"] == "START":
                attempt_id = safe_identifier(payload["attempt_id"])
                if not payload.get("execution_token"):
                    raise AgentError("INVALID_COMMAND", "START requires an execution token")
                output = str(Path(self.config["output_root"]) / attempt_id)
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
            result = {"error": exc.as_dict() if isinstance(exc, AgentError) else {"code": "COMMAND_REJECTED", "message": str(exc)}}
            status = "REJECTED"
        self.state.save_command(command["id"], status, result)
        return {"status": status, "result": result}

    def poll_commands(self):
        commands = self.api("GET", "/agent/v1/commands?" + urllib.parse.urlencode({"node_id": self.credentials["node_id"]}))
        for command in commands:
            acknowledgement = self.handle_command(command)
            self.api("POST", f"/agent/v1/commands/{urllib.parse.quote(command['id'], safe='')}/ack",
                     {"node_id": self.credentials["node_id"], **acknowledgement})

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
                   "attempt_processes": process_samples, "sampled_at": utc_now()}
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
