import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from vla_agent.config import load_config
from vla_agent import runner_runtime
from vla_agent.safety import process_identity
from vla_agent.state import State, TERMINAL_STATES


def make_runtime(root):
    runtime = root / "agent-install" / "runtimes" / "release-one"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").symlink_to(sys.executable)
    marker = {"schema_version": 1, "runtime_path": str(runtime), "build_id": "a" * 64}
    (runtime / runner_runtime.MARKER).write_text(json.dumps(marker))
    return {"protocol_version": 1, "runtime_path": str(runtime), "build_id": marker["build_id"]}


def test_installed_runtime_requires_matching_concrete_release_marker(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(sys, "prefix", runtime["runtime_path"])
    assert runner_runtime.installed_runtime() == runtime
    alias = Path(runtime["runtime_path"]).parent.parent / ".venv"
    alias.symlink_to(runtime["runtime_path"], target_is_directory=True)
    monkeypatch.setattr(sys, "prefix", str(alias))
    assert runner_runtime.installed_runtime() == runtime  # Return the concrete directory.
    marker = Path(runtime["runtime_path"]) / runner_runtime.MARKER
    marker.write_text(json.dumps({"schema_version": 1, "runtime_path": str(alias), "build_id": "a" * 64}))
    assert runner_runtime.installed_runtime() is None


@pytest.mark.parametrize("change", ["marker_permissions", "marker_symlink", "runtime_permissions", "missing_python"])
def test_installed_runtime_fails_closed_for_untrusted_or_incomplete_release(tmp_path, monkeypatch, change):
    runtime = make_runtime(tmp_path)
    path = Path(runtime["runtime_path"])
    monkeypatch.setattr(sys, "prefix", str(path))
    if change == "marker_permissions":
        (path / runner_runtime.MARKER).chmod(0o666)
    elif change == "marker_symlink":
        marker = path / runner_runtime.MARKER
        marker.rename(path / "other.json")
        marker.symlink_to(path / "other.json")
    elif change == "runtime_permissions":
        path.chmod(0o777)
    else:
        (path / "bin/python").unlink()
    assert runner_runtime.installed_runtime() is None


def test_snapshot_is_private_reusable_and_never_replaces_an_existing_attempt(agent_fixture):
    config = agent_fixture["config"]
    runtime = make_runtime(agent_fixture["root"])
    metadata = runner_runtime.snapshot_runner_config(config, "snapshot-attempt", runtime)
    path = Path(metadata["config_path"])
    original = path.read_bytes()
    loaded = load_config(path)
    assert "enrollment_token" not in loaded
    assert loaded["config_path"] == str(path)
    assert loaded["profiles"] == config["profiles"]
    assert metadata["config_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert runner_runtime.snapshot_runner_config(config, "snapshot-attempt", runtime) == metadata
    with pytest.raises(ValueError, match="different runner config"):
        runner_runtime.snapshot_runner_config({**config, "name": "changed"}, "snapshot-attempt", runtime)
    assert path.read_bytes() == original


def test_snapshot_rejects_training_python_through_mutable_agent_alias(agent_fixture):
    runtime = make_runtime(agent_fixture["root"])
    alias = Path(runtime["runtime_path"]).parent.parent / ".venv"
    alias.symlink_to(runtime["runtime_path"], target_is_directory=True)
    config = {**agent_fixture["config"], "profiles": [
        {**agent_fixture["config"]["profiles"][0], "python": str(alias / "bin/python")}]}
    with pytest.raises(ValueError, match="shares the Agent"):
        runner_runtime.snapshot_runner_config(config, "unsafe-attempt", runtime)


def test_legacy_and_unstarted_attempts_are_not_upgrade_safe(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    state = State(config["state_dir"])
    assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is True
    state.create_attempt(payload, str(Path(config["output_root"]) / payload["attempt_id"]))
    result = runner_runtime.upgrade_status(config["state_dir"])
    assert result["upgrade_safe"] is False
    assert result["active_attempts"] == 1
    assert result["active_attempt_ids"] == [payload["attempt_id"]]
    state.emit(payload["attempt_id"], "FAILED", {"process_started": False})
    assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is True


def _enable_test_packages(runtime):
    """Small real venv layout; dependencies remain in the local test install."""
    path = Path(runtime["runtime_path"])
    (path / "pyvenv.cfg").write_text(
        f"home = {Path(sys.base_prefix) / 'bin'}\ninclude-system-site-packages = false\n")
    packages = path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    packages.mkdir(parents=True)
    (packages / "test-runtime.pth").write_text("\n".join(
        str(Path(entry).resolve()) for entry in sys.path if entry and Path(entry).is_dir()) + "\n")


def test_upgrade_proof_checks_real_supervisor_and_snapshot_not_just_metadata(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    payload["parameters"]["steps"] = 100
    runtime = make_runtime(agent_fixture["root"])
    _enable_test_packages(runtime)
    metadata = runner_runtime.snapshot_runner_config(config, payload["attempt_id"], runtime)
    state = State(config["state_dir"])
    state.create_attempt(payload, str(Path(config["output_root"]) / payload["attempt_id"]), metadata)
    assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is False
    command = [str(Path(runtime["runtime_path"]) / "bin/python"), "-I", "-m", "vla_agent.runner",
               "--config", metadata["config_path"], "--attempt-id", payload["attempt_id"]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            attempt = state.attempt(payload["attempt_id"])
            if attempt["state"] == "RUNNING":
                break
            if process.poll() is not None:
                raise AssertionError(process.stdout.read().decode())
            time.sleep(0.05)
        assert attempt["state"] == "RUNNING"
        assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is True
        # A live, same-user process with a matching birth time is insufficient:
        # it must actually execute this runtime and this attempt's supervisor.
        actual = attempt["runner_identity"]
        state.set_identity(payload["attempt_id"], "runner", process_identity(os.getpid()))
        assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is False
        state.set_identity(payload["attempt_id"], "runner", actual)
        snapshot = Path(metadata["config_path"])
        original = snapshot.read_bytes()
        snapshot.write_bytes(original + b" ")
        assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is False
        snapshot.write_bytes(original)
        assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is True
    finally:
        state.request_cancel(payload["attempt_id"], payload["execution_token"])
        process.wait(timeout=10)
        process.stdout.close()
    assert state.attempt(payload["attempt_id"])["state"] in TERMINAL_STATES
    assert runner_runtime.upgrade_status(config["state_dir"])["upgrade_safe"] is True
