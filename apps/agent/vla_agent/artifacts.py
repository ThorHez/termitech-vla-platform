"""Read-only discovery of committed OpenPI checkpoints during native training.

The Agent owns this worker, so deploying it also observes detached Runners that
were started with an older package. No training interpreter or tensor is loaded.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import stat
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from .safety import allowed_path, safe_identifier
from .state import State


LOG = logging.getLogger("vla_agent")


class _ScanLimit(Exception):
    pass


class _Scan:
    def __init__(self, root: Path, max_entries: int, timeout: float):
        self.root = root
        self.remaining = max_entries
        self.deadline = time.monotonic() + timeout

    def check(self):
        self.remaining -= 1
        if self.remaining < 0 or time.monotonic() > self.deadline:
            raise _ScanLimit

    def local(self, path: Path) -> Path:
        self.check()
        resolved = path.resolve(strict=True)
        if resolved != path or not resolved.is_relative_to(self.root):
            raise ValueError("Checkpoint paths must stay inside their output directory without symlinks")
        return path

    def directory(self, path: Path) -> bool:
        return stat.S_ISDIR(self.local(path).lstat().st_mode)

    def regular(self, path: Path) -> os.stat_result:
        info = self.local(path).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError("Checkpoint metadata and data must be nonempty regular files")
        return info

    def json(self, path: Path, maximum: int = 8 * 1024 * 1024) -> dict:
        self.local(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise ValueError("Expected bounded checkpoint JSON metadata")
            content = stream.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError("Checkpoint JSON metadata exceeds its size limit")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Checkpoint JSON metadata must be an object")
        return value

    def children(self, path: Path):
        self.local(path)
        with os.scandir(path) as entries:
            for entry in entries:
                self.check()
                if not entry.is_symlink():
                    yield entry


def completed_openpi_checkpoint(checkpoint: Path, *, scan: _Scan | None = None) -> dict | None:
    """Require Orbax's committed timestamp and an actual params payload.

    Temporary directories and metadata without a commit are still saving. This
    checks completion metadata and nonempty payload files, not tensor validity
    or model quality. Unavailable or partial checkpoints are simply retried.
    """
    scan = scan or _Scan(checkpoint, 4096, 2)
    try:
        if not checkpoint.name.isascii() or not checkpoint.name.isdigit() or not scan.directory(checkpoint):
            return None
        committed = scan.json(checkpoint / "_CHECKPOINT_METADATA", 1024 * 1024).get("commit_timestamp_nsecs")
        if type(committed) is not int or committed <= 0:
            return None
        saved_at = datetime.fromtimestamp(committed / 1_000_000_000, timezone.utc).isoformat()
        params = checkpoint / "params"
        if not scan.directory(params):
            return None
        scan.json(params / "_METADATA")
        scan.regular(params / "manifest.ocdbt")
        # TensorStore stores committed tensor shards in d/ (or process-local d/
        # on distributed writers). Only inspect file metadata; never read shards.
        candidates = [params / "d"]
        for entry in scan.children(params):
            if entry.name.startswith("ocdbt.process_") and entry.is_dir(follow_symlinks=False):
                candidates.append(Path(entry.path) / "d")
        for directory in candidates:
            try:
                if not scan.directory(directory):
                    continue
                for entry in scan.children(directory):
                    if entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_size > 0:
                        scan.regular(Path(entry.path))
                        return {"saved_at": saved_at}
            except (OSError, ValueError):
                continue
    except (OSError, ValueError, TypeError, OverflowError):
        pass
    return None


def discover_openpi_outputs(output_dir: Path, recipe: str, *, max_entries: int = 4096,
                            timeout: float = 2) -> list[dict]:
    """Bound traversal to the attempt's checkpoints tree and three levels.

    Standard OpenPI layout is checkpoints/<config>/platform/<step>. Searching
    below only this owned root supports existing frozen plans without importing
    the recipe or consulting a mutable training checkout.
    """
    results = []
    scan = _Scan(output_dir, max_entries, timeout)
    pending = [(output_dir / "checkpoints", 0)]
    try:
        while pending and len(results) < 256:
            current, depth = pending.pop()
            try:
                if not scan.directory(current):
                    continue
                for entry in scan.children(current):
                    if not entry.is_dir(follow_symlinks=False) or ".orbax-checkpoint-tmp" in entry.name:
                        continue
                    path = Path(entry.path)
                    if entry.name.isascii() and entry.name.isdigit():
                        completed = completed_openpi_checkpoint(path, scan=scan)
                        if completed:
                            results.append({"kind": "MODEL", "name": recipe + " / " + entry.name,
                                            "path": str(path), "version": entry.name,
                                            "metadata": {"format": "openpi-orbax", "recipe": recipe,
                                                         "step": int(entry.name), "evaluated": False, **completed}})
                    elif depth < 2 and not entry.name.startswith("."):
                        pending.append((path, depth + 1))
                    if len(results) >= 256:
                        break
            except (OSError, ValueError):
                continue
    except _ScanLimit:
        pass
    return sorted(results, key=lambda output: (int(output["version"]), output["path"]))


class ArtifactMonitor:
    """One daemon worker; scheduling never waits for disk scanning or weights."""

    def __init__(self, state: State, output_root: str, *, interval: float = 10,
                 scanner=discover_openpi_outputs, clock=time.monotonic):
        self.state, self.output_root = state, output_root
        self.interval, self.scanner, self.clock = interval, scanner, clock
        self._lock = threading.Lock()
        self._wake, self._stopping = threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: OrderedDict[str, dict] = OrderedDict()
        self._inflight: set[str] = set()
        self._next_due: dict[str, float] = {}

    def schedule(self, attempts: list[dict]):
        eligible = {attempt["id"]: attempt for attempt in attempts
                    if attempt["payload"].get("template_id") == "openpi-sft"}
        with self._lock:
            for attempt_id in set(self._next_due) - eligible.keys():
                self._next_due.pop(attempt_id, None)
                self._pending.pop(attempt_id, None)
            for attempt_id, attempt in eligible.items():
                if (self.clock() >= self._next_due.get(attempt_id, 0)
                        and attempt_id not in self._inflight and attempt_id not in self._pending):
                    self._pending[attempt_id] = copy.deepcopy(attempt)
                    self._wake.set()
            if self._pending and self._thread is None and not self._stopping.is_set():
                self._thread = threading.Thread(target=self._work, name="vla-artifacts", daemon=True)
                self._thread.start()

    def _work(self):
        while not self._stopping.is_set():
            with self._lock:
                if self._pending:
                    attempt_id, attempt = self._pending.popitem(last=False)
                    self._inflight.add(attempt_id)
                else:
                    attempt_id = None
                    self._wake.clear()
            if attempt_id is None:
                self._wake.wait(1)
                continue
            try:
                self.report(attempt)
            except FileNotFoundError:
                pass  # STARTING attempts may not have created their output yet.
            except Exception:
                LOG.warning("Checkpoint discovery failed for attempt %s; training continues", attempt_id, exc_info=True)
            finally:
                with self._lock:
                    self._inflight.discard(attempt_id)
                    self._next_due[attempt_id] = self.clock() + self.interval

    def report(self, attempt: dict) -> int:
        """One bounded discovery, also used by Runners before their final event.

        Calling this directly starts no worker. The final pass closes the gap
        between checkpoint retention and the next periodic Agent scan.
        """
        if self._stopping.is_set():
            return 0
        attempt_id = safe_identifier(attempt["id"])
        # A newly accepted START can wait on preflight before its Runner creates
        # the owned output directory. The background worker retries this later.
        Path(attempt["output_dir"]).lstat()
        output = allowed_path(attempt["output_dir"], [self.output_root])
        expected = Path(self.output_root).resolve() / attempt_id
        if output != expected:
            raise ValueError("Attempt output directory does not match its identity")
        payload = attempt["payload"]
        recipe = payload.get("recipe") or payload.get("code_expectation", {}).get("recipe")
        if not isinstance(recipe, str) or not recipe:
            raise ValueError("Attempt is missing its frozen recipe")
        outputs = self.scanner(output, recipe)
        removed = self._removed_paths(attempt_id, output)
        if self._stopping.is_set():
            return 0
        return self.state.emit_artifacts(attempt_id, outputs, removed)

    def _removed_paths(self, attempt_id: str, output: Path) -> list[str]:
        """Retention can delete an older checkpoint without invalidating its run.

        A bounded/truncated discovery, unreadable mount, missing parent or other
        scan failure is never evidence of deletion. Require an explicit ENOENT
        for a previously reported checkpoint below an accessible parent.
        """
        removed = []
        scan = _Scan(output, 1024, 2)
        try:
            for artifact in self.state.available_artifacts(attempt_id):
                path = Path(artifact["path"])
                if not path.is_relative_to(output):
                    continue
                try:
                    if not scan.directory(path.parent):
                        continue
                    with os.scandir(path.parent):
                        pass
                    scan.check()
                    path.lstat()
                except FileNotFoundError:
                    # Recheck the parent after the target, so a missing entire
                    # mount/tree cannot turn every checkpoint unavailable.
                    try:
                        if scan.directory(path.parent):
                            removed.append(str(path))
                    except (OSError, ValueError):
                        pass
                except (OSError, ValueError):
                    continue
        except _ScanLimit:
            pass
        return removed

    def close(self):
        self._stopping.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=0.1)
