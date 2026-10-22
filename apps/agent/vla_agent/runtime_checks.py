"""Read-only training environment probes, independent of the control loop."""

from __future__ import annotations

import copy
import errno
import json
import os
import pwd
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .safety import AgentError, allowed_path
from .state import utc_now


PLATFORM_CREDENTIALS = {"VLA_ADMIN_TOKEN", "VLA_ENROLLMENT_TOKEN", "VLA_DATABASE_URL", "VLA_AGENT_TOKEN"}


def _known_symlink_target(interpreter: Path) -> str | None:
    """Read accessible links without requiring access to their final target."""
    target = interpreter
    known = None
    visited = set()
    for _ in range(8):
        if target in visited:
            break
        visited.add(target)
        try:
            linked = Path(os.readlink(target))
        except (OSError, ValueError):
            break
        target = linked if linked.is_absolute() else target.parent / linked
        target = Path(os.path.normpath(target))
        known = str(target)
    return known


def _permission_diagnostic(profile: dict, current_check: str, exc: PermissionError) -> str:
    uid = os.geteuid()
    try:
        account = pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        account = "unknown"
    configured = str(profile.get("python", "<unset>"))
    detail = exc.strerror or str(exc) or "Permission denied"
    message = (f"Training environment check '{current_check}' denied for Agent service account "
               f"{account} (uid={uid}): {detail}. Configured training Python: {configured}.")
    if exc.filename and str(exc.filename) != configured:
        message += f" Denied path: {exc.filename}."
    target = _known_symlink_target(Path(configured))
    if target:
        message += f" Known Python symlink target: {target}."
    return (message + " This account needs traversal (execute) permission on every parent directory "
            "of the training code, Python and its symlink target, read/execute access to the interpreter, "
            "and read access to required runtime files. Verify access using the same account "
            "configured for the Agent service.")


def check_runtime(profile: dict, roots: list[str], *, timeout: float = 10) -> dict:
    """Locate packages with the training interpreter; never import JAX or Torch.

    The import search path matches the actual launch wrapper, so an editable
    installation in the registered environment is required for OpenPI profiles.
    Availability does not mark a recipe as administrator validated.
    """
    result = {"profile_id": profile["id"], "status": "CHECKING", "checked_at": utc_now(), "checks": []}
    current_check = "code_directory"

    def passed(name: str, message: str):
        result["checks"].append({"name": name, "ok": True, "message": message})

    try:
        root = allowed_path(profile["code_dir"], roots)
        if not root.is_dir():
            raise ValueError("Training code path must be an existing directory")
        passed(current_check, str(root))
        current_check = "python_interpreter"
        interpreter = Path(profile["python"])
        if not interpreter.is_absolute() or not interpreter.is_file():
            raise ValueError("Training Python must be an existing absolute executable path")
        if not os.access(interpreter, os.X_OK):
            raise PermissionError(errno.EACCES, "Training Python is not executable by the Agent service account",
                                  str(interpreter))
        passed(current_check, str(interpreter))
        if profile.get("execution_mode", "native") != "native":
            current_check = "execution_mode"
            raise ValueError("This Agent supports native training environments only")
        adapter = profile.get("adapter")
        repository = None
        if adapter == "smoke":
            current_check = "smoke_entrypoint"
            entrypoint = allowed_path(str(root / "smoke_training.py"), [str(root)])
            if not entrypoint.is_file():
                raise ValueError("smoke_training.py must be a file inside the configured code directory")
            result["module_origin"] = str(entrypoint)
            passed(current_check, str(entrypoint))
        elif adapter == "openpi-sft":
            current_check = "git_repository"
            completed = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                                       check=True, capture_output=True, text=True, timeout=timeout)
            repository = allowed_path(completed.stdout.strip(), roots)
            if not repository.is_dir() or not root.is_relative_to(repository):
                raise ValueError("Configured OpenPI code directory must be inside its Git repository")
            passed(current_check, str(repository))
        else:
            current_check = "adapter"
            raise ValueError(f"Runtime checks do not support adapter: {adapter}")

        current_check = "interpreter_probe"
        # This only imports Python standard-library modules. Top-level find_spec
        # locates packages without executing their __init__.py or loading CUDA.
        import vla_adapters
        wrapper_dir = str(Path(vla_adapters.__file__).resolve().parent)
        probe = (
            "import importlib.util,json,sys; "
            f"sys.path[0]={wrapper_dir!r}; "
            "locate=lambda name: getattr(importlib.util.find_spec(name),'origin',None); "
            "print(json.dumps({'probe':'vla-runtime-v1','python_version':sys.version.split()[0],"
            "'executable':sys.executable,'prefix':sys.prefix,"
            f"'module_origin':locate('openpi') if {adapter == 'openpi-sft'!r} else None,"
            f"'jax_origin':locate('jax') if {adapter == 'openpi-sft'!r} else None}}))"
        )
        environment = {key: value for key, value in os.environ.items() if key not in PLATFORM_CREDENTIALS}
        environment.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"})
        completed = subprocess.run([str(interpreter), "-B", "-c", probe], cwd=root, env=environment,
                                   check=True, capture_output=True, text=True, timeout=timeout)
        try:
            evidence = json.loads(completed.stdout.strip())
            if not isinstance(evidence, dict) or evidence.get("probe") != "vla-runtime-v1":
                raise ValueError("Unexpected interpreter probe response")
            for field in ("python_version", "executable", "prefix"):
                if not isinstance(evidence.get(field), str) or not evidence[field]:
                    raise ValueError(f"Missing interpreter evidence: {field}")
        except (ValueError, TypeError) as exc:
            raise ValueError("Configured executable did not return valid Python environment evidence") from exc
        result.update({key: evidence[key] for key in ("python_version", "executable", "prefix")})
        passed(current_check, f"Python {evidence['python_version']} at {evidence['executable']}")
        if adapter == "openpi-sft":
            current_check = "openpi_import"
            located = evidence.get("module_origin")
            if not isinstance(located, str) or not located:
                raise ValueError("The training Python environment cannot locate openpi")
            try:
                origin = allowed_path(located, [str(repository)])
            except AgentError as exc:
                raise ValueError("Training Python resolves openpi outside the configured Git repository: " + located) from exc
            if not origin.is_file():
                raise ValueError("The resolved openpi module is not a readable local file")
            result["module_origin"] = str(origin)
            passed(current_check, str(origin))
            current_check = "jax_import"
            if not isinstance(evidence.get("jax_origin"), str) or not evidence["jax_origin"]:
                raise ValueError("The training Python environment cannot locate jax")
            passed(current_check, evidence["jax_origin"] + " (located without importing JAX)")
        result["status"] = "READY"
    except subprocess.TimeoutExpired:
        message = f"Training environment check '{current_check}' timed out after {timeout:g} seconds"
        result.update(status="FAILED", error=message)
        result["checks"].append({"name": current_check, "ok": False, "message": message})
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()[-1000:]
        message = f"Training environment check '{current_check}' failed (exit {exc.returncode})"
        if detail:
            message += ": " + detail
        result.update(status="FAILED", error=message)
        result["checks"].append({"name": current_check, "ok": False, "message": message})
    except PermissionError as exc:
        message = _permission_diagnostic(profile, current_check, exc)
        result.update(status="FAILED", error=message)
        result["checks"].append({"name": current_check, "ok": False, "message": message})
    except (KeyError, TypeError, ValueError, OSError, AgentError) as exc:
        message = str(exc)
        result.update(status="FAILED", error=message)
        result["checks"].append({"name": current_check, "ok": False, "message": message})
    result["checked_at"] = utc_now()
    return result


def validate_runtime(profile: dict, roots: list[str]) -> dict:
    evidence = check_runtime(profile, roots)
    if evidence["status"] != "READY":
        raise AgentError("RUNTIME_UNAVAILABLE", evidence["error"])
    return evidence


class RuntimeMonitor:
    """One daemon probe worker; taking a heartbeat snapshot never waits on it."""

    def __init__(self, roots: list[str], *, interval: float = 60, checker=check_runtime, clock=time.monotonic):
        self.roots = list(roots)
        self.interval = interval
        self.checker = checker
        self.clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._signatures: dict[str, str] = {}
        self._results: dict[str, dict] = {}
        self._next_due: dict[str, float] = {}
        self._pending: OrderedDict[str, tuple[str, dict]] = OrderedDict()
        self._inflight: set[str] = set()

    def snapshot(self, profiles: list[dict]) -> list[dict]:
        copied = copy.deepcopy(profiles)
        ids = {profile["id"] for profile in copied}
        with self._lock:
            for removed in set(self._signatures) - ids:
                self._signatures.pop(removed, None)
                self._results.pop(removed, None)
                self._next_due.pop(removed, None)
                self._pending.pop(removed, None)
            for profile in copied:
                profile_id = profile["id"]
                signature = json.dumps(profile, sort_keys=True, separators=(",", ":"))
                changed = self._signatures.get(profile_id) != signature
                due = self.clock() >= self._next_due.get(profile_id, 0)
                if changed or (due and profile_id not in self._pending and profile_id not in self._inflight):
                    self._signatures[profile_id] = signature
                    self._results[profile_id] = {"profile_id": profile_id, "status": "CHECKING", "checked_at": utc_now(), "checks": []}
                    self._pending[profile_id] = (signature, profile)
                    self._wake.set()
            if self._pending and self._thread is None and not self._stopping.is_set():
                self._thread = threading.Thread(target=self._work, name="vla-runtime-checks", daemon=True)
                self._thread.start()
            return copy.deepcopy([self._results[profile["id"]] for profile in copied])

    def _work(self):
        while not self._stopping.is_set():
            with self._lock:
                if self._pending:
                    profile_id, (signature, profile) = self._pending.popitem(last=False)
                    self._inflight.add(profile_id)
                else:
                    profile_id = None
                    self._wake.clear()
            if profile_id is None:
                self._wake.wait(1)
                continue
            try:
                result = self.checker(profile, self.roots, timeout=10)
            except Exception as exc:
                # An unexpected package/path failure must never kill the worker
                # or the Agent's independent heartbeat/command channels.
                message = f"Training environment probe failed: {exc}"
                result = {"profile_id": profile_id, "status": "FAILED", "checked_at": utc_now(), "error": message,
                          "checks": [{"name": "runtime_probe", "ok": False, "message": message}]}
            with self._lock:
                self._inflight.discard(profile_id)
                if self._signatures.get(profile_id) == signature:
                    self._results[profile_id] = result
                    self._next_due[profile_id] = self.clock() + self.interval

    def close(self):
        self._stopping.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=0.1)
