import subprocess
import json

import pytest

from vla_agent.safety import AgentError, allowed_path, signal_owned_group
from vla_agent.snapshots import capture_code, compare_code, sanitize_remote
from vla_agent.metrics import check_gpus
from vla_agent.runner import verified_outputs
from vla_agent.preflight import run_preflight


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


def test_task_recipe_is_echoed_and_bound_alongside_original_profile(agent_fixture):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    profile["recipe"] = "smoke"
    frozen = capture_code(profile, config["roots"], recipe="smoke")
    assert frozen["recipe"] == "smoke"
    # Changing the registered default cannot be hidden by retaining a frozen task recipe.
    profile["recipe"] = "changed_after_preflight"
    current = capture_code(profile, config["roots"], recipe="smoke")
    assert current["recipe"] == "smoke"
    with pytest.raises(AgentError, match="runtime_digest"):
        compare_code(frozen, current)
    # The selection itself is also part of the authorization snapshot.
    profile["recipe"] = "smoke"
    with pytest.raises(AgentError, match="runtime_digest"):
        compare_code(frozen, capture_code(profile, config["roots"], recipe="different_recipe"))


def test_preflight_uses_task_recipe_without_changing_profile(agent_fixture):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    assert "recipe" not in profile
    result = run_preflight(config, {**agent_fixture["payload"], "recipe": "smoke"})
    assert "error" not in result
    assert result["code_snapshot"]["recipe"] == "smoke"
    assert "recipe" not in profile


def test_openpi_preflight_accepts_task_recipe_with_no_registered_default(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    profile.update(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["first_config", "second_config"]})
    package_root = agent_fixture["repo"] / "src"
    for name in ("openpi", "jax"):
        package = package_root / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise RuntimeError('training dependencies must not be imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(package_root))
    # Recipe selection/provenance is tested here; the registered interpreter's
    # configuration and asset probe has separate contract tests.
    from vla_agent import preflight
    monkeypatch.setattr(preflight, "validate_training", lambda *args, **kwargs: {
        "schema_version": 1, "scope": "configuration_and_local_assets", "plan": {}, "plan_digest": "test"})
    inputs = {}
    for role in ("dataset", "model", "norm"):
        path = agent_fixture["root"] / role
        path.mkdir()
        inputs[role] = {"path": str(path)}
    payload = {**agent_fixture["payload"], "template_id": "openpi-sft", "recipe": "second_config",
               "inputs": inputs, "gpu_count": 1}
    result = run_preflight(config, payload)
    assert "error" not in result, result
    assert result["code_snapshot"]["recipe"] == "second_config"
    assert profile["recipe"] is None
    assert not list(package_root.rglob("__pycache__"))


def test_legacy_snapshot_digest_remains_compatible_when_recipe_not_frozen(agent_fixture):
    import hashlib

    config = agent_fixture["config"]
    profile = config["profiles"][0]
    snapshot = capture_code(profile, config["roots"])
    registered = {key: profile.get(key) for key in ("python", "recipe", "adapter", "execution_mode", "docker_image", "metadata")}
    assert snapshot["runtime_digest"] == hashlib.sha256(json.dumps(registered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_requested_entrypoint_is_frozen_and_ignored_script_changes_are_detected(agent_fixture):
    config, repo = agent_fixture["config"], agent_fixture["repo"]
    profile = config["profiles"][0]
    with (repo / ".gitignore").open("a") as ignored:
        ignored.write("ignored_train.py\n")
    selected = repo / "ignored_train.py"
    selected.write_text("first = True\n")
    frozen = capture_code(profile, config["roots"], entrypoint="ignored_train.py")
    assert frozen["requested_entrypoint"] == "ignored_train.py"
    assert len(frozen["requested_entrypoint_sha256"]) == 64
    selected.write_text("first = False\n")
    changed = capture_code(profile, config["roots"], entrypoint="ignored_train.py")
    assert frozen["changes_digest"] == changed["changes_digest"]
    assert frozen["untracked_digest"] == changed["untracked_digest"]
    with pytest.raises(AgentError, match="requested_entrypoint_sha256"):
        compare_code(frozen, changed)
    with pytest.raises(AgentError, match="requested_entrypoint"):
        compare_code(frozen, capture_code(profile, config["roots"], entrypoint="smoke_training.py"))


@pytest.mark.parametrize("entrypoint", ["/tmp/train.py", "../train.py", "scripts/../train.py", "train.sh"])
def test_snapshot_rejects_non_project_entrypoint(agent_fixture, entrypoint):
    with pytest.raises(AgentError) as failure:
        capture_code(agent_fixture["config"]["profiles"][0], agent_fixture["config"]["roots"], entrypoint=entrypoint)
    assert failure.value.code == "INVALID_ENTRYPOINT"


def test_snapshot_rejects_entrypoint_symlink_into_other_allowed_project(agent_fixture):
    sibling = agent_fixture["root"] / "other-project"
    sibling.mkdir()
    (sibling / "train.py").write_text("pass\n")
    (agent_fixture["repo"] / "escaped.py").symlink_to(sibling / "train.py")
    with pytest.raises(AgentError) as failure:
        capture_code(agent_fixture["config"]["profiles"][0], agent_fixture["config"]["roots"], entrypoint="escaped.py")
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"


def test_snapshot_rejects_oversized_entrypoint_before_git_capture(agent_fixture):
    (agent_fixture["repo"] / "large.py").write_bytes(b"#" * (2 * 1024 * 1024 + 1))
    with pytest.raises(AgentError) as failure:
        capture_code(agent_fixture["config"]["profiles"][0], agent_fixture["config"]["roots"], entrypoint="large.py")
    assert failure.value.code == "INVALID_ENTRYPOINT"


def test_snapshot_marks_incomplete_when_ignored_entry_changes_during_capture(agent_fixture, monkeypatch):
    from vla_agent import snapshots

    repo = agent_fixture["repo"]
    with (repo / ".gitignore").open("a") as ignored:
        ignored.write("ignored_train.py\n")
    selected = repo / "ignored_train.py"
    selected.write_text("first = True\n")
    real_git = snapshots.git

    def edit_during_capture(root, *args):
        if args == ("rev-parse", "--verify", "HEAD"):
            selected.write_text("first = False\n")
        return real_git(root, *args)

    monkeypatch.setattr(snapshots, "git", edit_during_capture)
    snapshot = capture_code(agent_fixture["config"]["profiles"][0], agent_fixture["config"]["roots"], entrypoint="ignored_train.py")
    assert snapshot["complete"] is False
    assert "code changed during provenance capture" in snapshot["missing"]


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
