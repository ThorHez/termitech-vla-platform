import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vla_agent.artifacts import ArtifactMonitor, completed_openpi_checkpoint, discover_openpi_outputs
from vla_agent.main import Agent
from vla_agent.state import State


def checkpoint(output, name="3000", *, committed=1792136926465153081):
    path = output / "checkpoints" / "fixture_recipe" / "platform" / name
    (path / "params" / "d").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": committed}))
    (path / "params" / "_METADATA").write_text('{"tree_metadata": {}}')
    (path / "params" / "manifest.ocdbt").write_bytes(b"manifest")
    (path / "params" / "d" / "shard").write_bytes(b"tensor payload")
    return path


def outputs(output):
    return discover_openpi_outputs(output, "fixture_recipe")


def wait_until(callback, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Artifact worker did not reach the expected state")


def running_attempt(agent_fixture):
    config = agent_fixture["config"]
    payload = {**agent_fixture["payload"], "template_id": "openpi-sft", "recipe": "fixture_recipe"}
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    state.emit(payload["attempt_id"], "STARTED", {"pid": 1})
    state.acknowledge_events(state.pending_events())
    return state, output, payload


def test_discovery_reports_only_committed_openpi_checkpoints_with_save_time(tmp_path):
    saved = checkpoint(tmp_path)
    checkpoint(tmp_path, "4000.orbax-checkpoint-tmp-1")
    checkpoint(tmp_path, "5000", committed=None)
    found = outputs(tmp_path)
    assert len(found) == 1
    assert found[0] == {"kind": "MODEL", "name": "fixture_recipe / 3000", "path": str(saved), "version": "3000",
                        "metadata": {"format": "openpi-orbax", "recipe": "fixture_recipe", "step": 3000,
                                     "evaluated": False, "saved_at": "2026-10-16T07:48:46.465153+00:00"}}


@pytest.mark.parametrize("partial", ["root_metadata", "uncommitted", "corrupt_metadata", "params_metadata",
                                      "manifest", "empty_shard", "metadata_symlink", "params_symlink"])
def test_partial_and_unreadable_checkpoints_are_pending_instead_of_outputs(tmp_path, partial):
    saved = checkpoint(tmp_path)
    if partial == "root_metadata":
        (saved / "_CHECKPOINT_METADATA").unlink()
    elif partial == "uncommitted":
        (saved / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 0}')
    elif partial == "corrupt_metadata":
        (saved / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs":')
    elif partial == "params_metadata":
        (saved / "params" / "_METADATA").write_text("not JSON")
    elif partial == "manifest":
        (saved / "params" / "manifest.ocdbt").unlink()
    elif partial == "empty_shard":
        (saved / "params" / "d" / "shard").write_bytes(b"")
    elif partial == "metadata_symlink":
        (saved / "params" / "_METADATA").unlink()
        (saved / "params" / "_METADATA").symlink_to(saved / "_CHECKPOINT_METADATA")
    else:
        (saved / "params").rename(saved / "original_params")
        (saved / "params").symlink_to(saved / "original_params")
    assert outputs(tmp_path) == []


def test_discovery_ignores_escaping_symlinks_and_limits_tree_depth_and_work(tmp_path):
    output, elsewhere = tmp_path / "output", tmp_path / "outside"
    saved = checkpoint(elsewhere)
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints" / "3000").symlink_to(saved)
    checkpoint(output / "checkpoints" / "too" / "deep")
    assert outputs(output) == []
    checkpoint(output)
    assert discover_openpi_outputs(output, "fixture_recipe", max_entries=1) == []
    assert discover_openpi_outputs(output, "fixture_recipe", timeout=0) == []


def test_checkpoint_params_payload_can_be_process_local(tmp_path):
    saved = checkpoint(tmp_path)
    (saved / "params" / "d").rename(saved / "params" / "old_d")
    (saved / "params" / "ocdbt.process_0").mkdir()
    (saved / "params" / "old_d").rename(saved / "params" / "ocdbt.process_0" / "d")
    assert completed_openpi_checkpoint(saved)


def test_artifact_receipts_and_outbox_are_atomic_durable_and_keep_training_running(agent_fixture):
    state, output, payload = running_attempt(agent_fixture)
    checkpoint(output)
    found = outputs(output)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(lambda _: state.emit_artifacts(payload["attempt_id"], found), range(12))) == 1
    events = state.pending_events()
    assert [event["type"] for event in events] == ["ARTIFACT"]
    assert events[0]["data"] == {"outputs": found, "removed_paths": []}
    assert state.attempt(payload["attempt_id"])["state"] == "RUNNING"
    restarted = State(agent_fixture["config"]["state_dir"])
    assert restarted.pending_events() == events  # Offline retries preserve exact payload and sequence.
    restarted.acknowledge_events(events)
    assert restarted.emit_artifacts(payload["attempt_id"], found) == 0
    assert restarted.pending_events() == []  # ACK does not erase discovery receipts.


@pytest.mark.parametrize("terminal", ["SUCCEEDED", "FAILED", "CANCELLED"])
def test_terminal_event_wins_a_race_with_checkpoint_discovery(agent_fixture, terminal):
    state, output, payload = running_attempt(agent_fixture)
    checkpoint(output)
    state.emit(payload["attempt_id"], terminal, {})
    assert state.emit_artifacts(payload["attempt_id"], outputs(output)) == 0
    assert state.available_artifacts(payload["attempt_id"]) == []
    assert [event["type"] for event in state.pending_events()] == [terminal]


def test_monitor_observes_existing_detached_attempt_and_retention_then_restoration(agent_fixture):
    state, output, payload = running_attempt(agent_fixture)
    saved = checkpoint(output)
    clock = [0]
    monitor = ArtifactMonitor(state, agent_fixture["config"]["output_root"], interval=1, clock=lambda: clock[0])
    try:
        monitor.schedule(state.active_attempts())
        first = wait_until(lambda: state.pending_events())
        assert first[0]["data"]["outputs"][0]["path"] == str(saved)
        state.acknowledge_events(first)
        assert state.attempt(payload["attempt_id"])["child_identity"] is None  # No launch or Runner mutation.
        wait_until(lambda: not monitor._inflight)
        saved.rename(saved.with_name("retained-elsewhere"))
        clock[0] = 2
        monitor.schedule(state.active_attempts())
        removal = wait_until(lambda: state.pending_events())
        assert removal[0]["data"] == {"outputs": [], "removed_paths": [str(saved)]}
        state.acknowledge_events(removal)
        wait_until(lambda: not monitor._inflight)
        saved.with_name("retained-elsewhere").rename(saved)
        clock[0] = 4
        monitor.schedule(state.active_attempts())
        restored = wait_until(lambda: state.pending_events())
        assert restored[0]["data"]["outputs"][0]["path"] == str(saved)
        assert state.attempt(payload["attempt_id"])["state"] == "RUNNING"
    finally:
        monitor.close()


def test_missing_or_unreadable_parent_and_truncated_scans_do_not_mean_checkpoint_removed(agent_fixture, monkeypatch):
    state, output, payload = running_attempt(agent_fixture)
    saved = checkpoint(output)
    state.emit_artifacts(payload["attempt_id"], outputs(output))
    monitor = ArtifactMonitor(state, agent_fixture["config"]["output_root"])
    saved.parent.rename(saved.parent.with_name("temporarily-unmounted"))
    assert monitor._removed_paths(payload["attempt_id"], output) == []
    saved.parent.with_name("temporarily-unmounted").rename(saved.parent)
    original = Path.lstat

    def denied(path, *args, **kwargs):
        if path == saved:
            raise PermissionError("mount unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    assert monitor._removed_paths(payload["attempt_id"], output) == []
    assert discover_openpi_outputs(output, "fixture_recipe", max_entries=1) == []


def test_blocked_scanner_does_not_block_agent_control_cycle_or_shutdown(agent_fixture, monkeypatch):
    state, output, payload = running_attempt(agent_fixture)
    checkpoint(output)
    entered, release = threading.Event(), threading.Event()

    def slow_scan(*args):
        entered.set()
        assert release.wait(3)
        return outputs(output)

    agent = Agent(agent_fixture["config"], client=object())
    agent.artifact_monitor = ArtifactMonitor(state, agent_fixture["config"]["output_root"], scanner=slow_scan)
    agent.credentials = {"node_id": "fixture", "token": "fixture"}
    calls = []
    monkeypatch.setattr(agent, "register", lambda: None)
    monkeypatch.setattr(agent, "heartbeat", lambda: calls.append("heartbeat"))
    monkeypatch.setattr(agent, "flush_events", lambda: calls.append("events"))
    monkeypatch.setattr(agent, "poll_commands", lambda: calls.append("commands"))
    agent.artifact_monitor.schedule(state.active_attempts())
    assert entered.wait(1)
    started = time.monotonic()
    try:
        agent.run(once=True)
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert calls == ["heartbeat", "events", "commands"]
    finally:
        release.set()
        agent.close()
    assert state.pending_events() == []  # Shutdown never publishes stale worker results.


def test_worker_scan_failure_is_retried_without_failing_training(agent_fixture):
    state, output, payload = running_attempt(agent_fixture)
    checkpoint(output)
    clock, calls = [0], []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("unavailable filesystem")
        return outputs(output)

    monitor = ArtifactMonitor(state, agent_fixture["config"]["output_root"], interval=1, scanner=flaky,
                              clock=lambda: clock[0])
    try:
        monitor.schedule(state.active_attempts())
        wait_until(lambda: calls and not monitor._inflight)
        assert state.pending_events() == []
        clock[0] = 2
        monitor.schedule(state.active_attempts())
        assert wait_until(lambda: state.pending_events())[0]["type"] == "ARTIFACT"
        assert state.attempt(payload["attempt_id"])["state"] == "RUNNING"
    finally:
        monitor.close()


def test_direct_report_reconciles_last_checkpoint_and_retention_before_terminal_event(agent_fixture):
    state, output, payload = running_attempt(agent_fixture)
    previous = checkpoint(output, "3000")
    monitor = ArtifactMonitor(state, agent_fixture["config"]["output_root"])
    attempt = state.attempt(payload["attempt_id"])
    assert monitor.report(attempt) == 1
    state.acknowledge_events(state.pending_events())
    previous.rename(previous.with_name("rotated-3000"))
    latest = checkpoint(output, "4000")
    assert monitor.report(attempt) == 2
    event = state.pending_events()[0]
    assert [artifact["path"] for artifact in event["data"]["outputs"]] == [str(latest)]
    assert event["data"]["removed_paths"] == [str(previous)]
    assert monitor._thread is None
    assert state.attempt(payload["attempt_id"])["state"] == "RUNNING"
    state.emit(payload["attempt_id"], "SUCCEEDED", {"outputs": outputs(output)})
    assert [event["type"] for event in state.pending_events()] == ["ARTIFACT", "SUCCEEDED"]
    assert monitor.report(attempt) == 0
