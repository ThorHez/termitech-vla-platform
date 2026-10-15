import subprocess
import json

import pytest

from vla_agent.safety import AgentError, allowed_path, signal_owned_group
from vla_agent.snapshots import capture_code, compare_code, sanitize_remote
from vla_agent.metrics import check_gpus
from vla_agent.runner import verified_outputs


def test_snapshot_tracks_actual_commit_dirty_contents_and_evidence(agent_fixture):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    repo = agent_fixture["repo"]
    clean = agent_fixture["payload"]["code_expectation"]
    assert clean["commit_sha"] == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert clean["branch"] == "fixture" and clean["dirty"] is False
    assert clean["module_origin"] == str(repo / "smoke_training.py")
    original = (repo / "smoke_training.py").read_text()
    (repo / "smoke_training.py").write_text(original + "\n# first local edit\n")
    dirty = capture_code(profile, config["roots"])
    (repo / "smoke_training.py").write_text(original + "\n# second local edit\n")
    dirtier = capture_code(profile, config["roots"])
    assert dirty["dirty"] and dirtier["dirty"]
    with pytest.raises(AgentError, match="changes_digest"):
        compare_code(dirty, dirtier)
    (repo / "new_source.py").write_text("value = 1\n")
    evidence = agent_fixture["root"] / "evidence"
    new = capture_code(profile, config["roots"], evidence)
    assert (evidence / "untracked" / "new_source.py").read_text() == "value = 1\n"
    (repo / "new_source.py").write_text("value = 2\n")
    with pytest.raises(AgentError, match="untracked_digest"):
        compare_code(new, capture_code(profile, config["roots"]))


def test_path_realpath_blocks_symlink_escape(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AgentError, match="outside configured roots"):
        allowed_path(str(root / "linked"), [str(root)])


def test_stale_pid_is_never_signalled(monkeypatch):
    from vla_agent import safety
    monkeypatch.setattr(safety, "process_identity", lambda pid: {"pid": pid, "created_at": 20.0})
    monkeypatch.setattr(safety.os, "killpg", lambda *args: pytest.fail("reused PID was signalled"))
    assert signal_owned_group({"pid": 777, "created_at": 10.0}) is False


def test_gpu_unknown_and_external_process_block_start():
    with pytest.raises(AgentError, match="NVML"):
        check_gpus(["GPU-1"], inventory=([], {"available": False}))
    with pytest.raises(AgentError, match="existing process"):
        check_gpus(["GPU-1"], inventory=([{"uuid": "GPU-1", "processes": [{"pid": 123}]}], {"available": True}))
    with pytest.raises(AgentError, match="Cannot verify processes"):
        check_gpus(["GPU-1"], inventory=([{"uuid": "GPU-1", "processes": None}], {"available": True}))
    check_gpus([], inventory=([], {"available": False}))


def test_remote_credentials_are_not_recorded():
    assert sanitize_remote("https://user:secret@github.com/owner/repo.git?token=private") == "https://github.com/owner/repo.git"
    assert sanitize_remote("secret@github.com:owner/repo.git") == "github.com:owner/repo.git"


def test_output_artifacts_cannot_escape_attempt_directory(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    outside = tmp_path / "unrelated-model"
    outside.mkdir()
    (output / "model-link").symlink_to(outside, target_is_directory=True)
    (output / "artifacts.json").write_text(json.dumps([{
        "kind": "MODEL", "name": "model", "version": "1", "path": str(output / "model-link")}]))
    with pytest.raises(AgentError, match="outside configured roots"):
        verified_outputs(output, smoke=True)
