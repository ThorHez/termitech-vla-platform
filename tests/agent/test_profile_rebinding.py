import copy
import json
import os
import stat
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vla_agent import profile_rebinding as rebinder
from vla_agent.config import load_config
from vla_agent.main import Agent
from vla_agent.recipe_writeback import checkout_lock
from vla_agent.safety import AgentError
from vla_agent.state import State


@pytest.fixture
def fixture(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    path = Path(config["config_path"])
    disk = json.loads(path.read_text())
    disk["profiles"][0].update(adapter="openpi-sft", metadata={"keep": ["metadata"]})
    disk["profiles"].append({**disk["profiles"][0], "id": "other-profile", "name": "Keep other"})
    disk["unknown_extra"] = {"must": "survive", "credential": "fixture-private-value"}
    path.write_text(json.dumps(disk));path.chmod(0o600)
    config = load_config(path)
    state = State(config["state_dir"])
    state.save_credentials({"node_id": "test-node", "token": "keep-node-token"})
    checkout = agent_fixture["root"] / "new-code"
    checkout.mkdir()
    subprocess.run(["git", "-C", str(checkout), "init", "-b", "fixture"], check=True, capture_output=True)
    for name in ("openpi", "jax"):
        package = checkout / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise RuntimeError('probe must not import package')\n")
    source = checkout / "src/openpi/training/config.py"
    source.parent.mkdir();source.write_text("_CONFIGS = []\n")
    monkeypatch.setenv("PYTHONPATH", str(checkout / "src"))
    python = agent_fixture["root"] / "training-python"
    python.symlink_to(sys.executable)
    profile = config["profiles"][0]
    request = {"profile_id": profile["id"], "expected_code_dir": profile["code_dir"],
               "expected_python": profile["python"], "code_dir": str(checkout), "python": str(python),
               "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}
    return config, state, path, checkout, request


def rebind(fixture, **changes):
    config, state, _, _, request = fixture
    return rebinder.rebind_profile(config, state, {**request, **changes}, operation_id="rebind-1")


def assert_unchanged(fixture, before, memory):
    config, _, path, _, _ = fixture
    assert path.read_bytes() == before
    assert config["profiles"] == memory


def test_live_rebind_probes_actual_origin_and_preserves_private_configuration(fixture):
    config, state, path, checkout, request = fixture
    old = path.read_bytes();original = json.loads(old)
    result = rebind(fixture)
    assert result["persisted"] is True and result["runtime_check"]["status"] == "READY"
    assert result["runtime_check"]["module_origin"] == str(checkout / "src/openpi/__init__.py")
    assert result["python"] == request["python"]  # Preserve the venv/interpreter symlink, not its target.
    disk = json.loads(path.read_text())
    assert disk["profiles"][0] == {**original["profiles"][0], "code_dir": request["code_dir"], "python": request["python"]}
    assert disk["profiles"][1] == original["profiles"][1]
    assert disk["unknown_extra"] == original["unknown_extra"]
    assert disk["enrollment_token"] == original["enrollment_token"]
    assert "config_path" not in disk and disk["profiles"] == config["profiles"]
    assert state.credentials() == {"node_id": "test-node", "token": "keep-node-token"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    backups = list((Path(config["state_dir"]) / "profile-rebind").glob("before-*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == old
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(backups[0].parent.stat().st_mode) == 0o700
    assert not list(checkout.rglob("__pycache__"))


def test_rebind_needs_no_root_and_preserves_lexical_checkout_symlink(fixture, monkeypatch):
    _, _, _, checkout, _ = fixture
    alias = checkout.parent / "checkout-link";alias.symlink_to(checkout, target_is_directory=True)
    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    result = rebind(fixture, code_dir=str(alias))
    assert result["code_dir"] == str(alias)
    assert result["runtime_check"]["status"] == "READY"


@pytest.mark.parametrize("suffix", ["/", "/./"])
def test_legacy_old_directory_paths_compare_after_lexical_normalization(fixture, suffix):
    config, _, path, _, request = fixture
    legacy = request["expected_code_dir"] + suffix
    disk = json.loads(path.read_text());disk["profiles"][0]["code_dir"] = legacy
    path.write_text(json.dumps(disk));config["profiles"][0]["code_dir"] = legacy
    assert rebind(fixture)["persisted"] is True


def test_manual_unicode_profile_identifier_is_supported_without_using_it_as_a_file_name(fixture):
    config, _, path, _, request = fixture
    identifier = "自定义训练环境-" + "x" * 190
    disk = json.loads(path.read_text());disk["profiles"][0]["id"] = identifier
    path.write_text(json.dumps(disk));config["profiles"][0]["id"] = identifier
    request["profile_id"] = identifier
    result = rebind(fixture)
    assert result["profile_id"] == identifier
    assert not any(identifier in item.name for item in (Path(config["state_dir"]) / "profile-rebind").iterdir())


@pytest.mark.parametrize("field,value", [("code_dir", "relative"), ("code_dir", "/tmp/../code"),
                                         ("code_dir", "/tmp/bad\x00path"), ("python", "python3"),
                                         ("python", "/tmp/bad\npython"), ("expected_python", ""),
                                         ("expires_at", "2026-01-01T00:00:00"),
                                         ("expires_at", "2026-01-01T00:00:00+08:00")])
def test_invalid_paths_or_expiry_leave_configuration_intact(fixture, field, value):
    config, _, path, _, _ = fixture
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture, **{field: value})
    assert failure.value.code == "PROFILE_REBIND_INVALID"
    assert failure.value.as_dict()["persisted"] is False
    assert_unchanged(fixture, old, profiles)


def test_expired_request_rejected_before_probe(fixture, monkeypatch):
    monkeypatch.setattr(rebinder, "check_runtime", lambda *a, **k: pytest.fail("expired request must not probe"))
    with pytest.raises(AgentError) as failure:
        rebind(fixture, expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
    assert failure.value.code == "PROFILE_REBIND_EXPIRED"


def test_escaping_directory_symlink_is_rejected_even_when_interpreter_can_be_external(fixture, tmp_path):
    config, _, path, checkout, _ = fixture
    config["roots"] = [str(checkout)]
    escape = checkout / "outside";escape.symlink_to(tmp_path, target_is_directory=True)
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture, code_dir=str(escape))
    assert failure.value.code == "PROFILE_REBIND_INVALID"
    assert_unchanged(fixture, old, profiles)


@pytest.mark.parametrize("failure_kind", ["wrong-origin", "missing-jax", "python-denied", "probe-timeout"])
def test_runtime_failure_does_not_publish_profile(fixture, monkeypatch, failure_kind):
    config, _, path, _, request = fixture
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    real_run = subprocess.run
    def probe(args, **kwargs):
        if args[0] != request["python"]:return real_run(args, **kwargs)
        if failure_kind == "python-denied":raise PermissionError(13, "Permission denied", args[0])
        if failure_kind == "probe-timeout":raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        result = real_run(args, **kwargs);evidence = json.loads(result.stdout)
        evidence["module_origin" if failure_kind == "wrong-origin" else "jax_origin"] = (
            str(Path(request["expected_code_dir"]) / "openpi/__init__.py") if failure_kind == "wrong-origin" else None)
        result.stdout = json.dumps(evidence)
        return result
    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RUNTIME_UNAVAILABLE"
    assert failure.value.as_dict()["runtime_check"]["status"] == "FAILED"
    assert_unchanged(fixture, old, profiles)


def test_interpreter_importing_a_sibling_project_in_the_same_git_repo_is_rejected(fixture):
    config, _, path, checkout, _ = fixture
    sibling = checkout / "other-project"
    source = sibling / "src/openpi/training/config.py"
    source.parent.mkdir(parents=True);source.write_text("_CONFIGS = []\n")
    (sibling / "src/openpi/__init__.py").write_text("raise RuntimeError('must not import')\n")
    before = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    # PYTHONPATH still points to the checkout-root project; Git permits both.
    with pytest.raises(AgentError) as failure:rebind(fixture, code_dir=str(sibling))
    assert failure.value.code == "PROFILE_REBIND_RUNTIME_UNAVAILABLE"
    assert failure.value.as_dict()["runtime_check"]["checks"][-1]["name"] == "selected_project"
    assert_unchanged(fixture, before, profiles)


def test_missing_selected_project_configuration_is_rejected(fixture):
    config, _, path, checkout, _ = fixture
    (checkout / "src/openpi/training/config.py").unlink()
    before = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RUNTIME_UNAVAILABLE"
    assert_unchanged(fixture, before, profiles)


@pytest.mark.parametrize("linked", ["package", "config-source"])
def test_nonstandard_symlinked_package_or_config_source_rejects_before_publish(fixture, linked):
    config, _, path, checkout, _ = fixture
    package = checkout / "src/openpi";source = package / "training/config.py"
    if linked == "package":
        target = checkout / "shared-openpi";package.rename(target)
        package.symlink_to(target, target_is_directory=True)
    else:
        target = package / "actual_config.py";source.rename(target);source.symlink_to(target)
    before = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RUNTIME_UNAVAILABLE"
    assert_unchanged(fixture, before, profiles)


@pytest.mark.parametrize("attempt_state", ["STARTING", "RUNNING", "UNKNOWN"])
@pytest.mark.parametrize("source", ["matching-profile", "old-alias", "new-checkout"])
def test_active_profile_or_checkout_alias_blocks_rebind(fixture, attempt_state, source):
    config, state, path, checkout, request = fixture
    payload = {"attempt_id": "active", "profile_id": "different-profile"}
    if source == "matching-profile":payload["profile_id"] = request["profile_id"]
    else:
        location = Path(request["expected_code_dir"]) if source == "old-alias" else checkout
        location = location / "subdirectory";location.mkdir()
        payload["code_expectation"] = {"code_dir": str(location)}
    state.create_attempt(payload, str(path.parent / "outputs"))
    with state.connect() as db:db.execute("UPDATE attempts SET state=?", (attempt_state,))
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_BUSY"
    assert_unchanged(fixture, old, profiles)


@pytest.mark.parametrize("non_git_directory", [False, True])
def test_deleted_or_non_git_old_checkout_can_be_repaired_only_when_no_attempts_exist(fixture, non_git_directory):
    config, state, path, _, request = fixture
    missing = str(path.parent / "deleted-checkout")
    if non_git_directory:Path(missing).mkdir()
    disk = json.loads(path.read_text());disk["profiles"][0]["code_dir"] = missing
    path.write_text(json.dumps(disk));config["profiles"][0]["code_dir"] = missing
    request["expected_code_dir"] = missing
    state.create_attempt({"attempt_id": "unknown", "profile_id": "unresolved"}, str(path.parent / "outputs"))
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_BUSY"
    with state.connect() as db:db.execute("UPDATE attempts SET state='FAILED'")
    assert rebind(fixture)["persisted"] is True


@pytest.mark.parametrize("where", ["memory", "disk"])
def test_expected_old_paths_must_match_both_live_and_persisted_profile(fixture, where):
    config, _, path, _, _ = fixture
    if where == "memory":config["profiles"][0]["python"] = "/other/python"
    else:
        disk = json.loads(path.read_text());disk["profiles"][0]["python"] = "/other/python"
        path.write_text(json.dumps(disk))
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_CONFLICT"


def test_disk_compare_and_swap_preserves_concurrent_external_change(fixture, monkeypatch):
    config, _, path, _, _ = fixture
    profiles = copy.deepcopy(config["profiles"])
    real_check = rebinder.check_runtime
    def changed(*args, **kwargs):
        result = real_check(*args, **kwargs)
        disk = json.loads(path.read_text());disk["new_external_key"] = "keep"
        path.write_text(json.dumps(disk))
        return result
    monkeypatch.setattr(rebinder, "check_runtime", changed)
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_CONFLICT"
    assert json.loads(path.read_text())["new_external_key"] == "keep"
    assert config["profiles"] == profiles


@pytest.mark.parametrize("stage", ["temporary", "replace", "directory-fsync"])
def test_file_write_failure_keeps_old_profile_and_bytes(fixture, monkeypatch, stage):
    config, _, path, _, _ = fixture
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    if stage == "temporary":
        def failed(*a, **k):raise OSError("disk full")
        monkeypatch.setattr(rebinder, "_temporary", failed)
    elif stage == "replace":
        real_replace = os.replace
        def failed(source, target):
            if Path(target) == path:raise PermissionError("read-only configuration")
            return real_replace(source, target)
        monkeypatch.setattr(os, "replace", failed)
    else:
        real_sync = rebinder._sync_parent;failed_once = False
        def failed(target):
            nonlocal failed_once
            if target == path and not failed_once:
                failed_once = True;raise OSError("directory fsync failed")
            return real_sync(target)
        monkeypatch.setattr(rebinder, "_sync_parent", failed)
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_FAILED"
    assert failure.value.as_dict()["persisted"] is False
    assert_unchanged(fixture, old, profiles)


def test_durable_marker_recovers_after_publish_and_missing_command_ack_even_when_expired(fixture, monkeypatch):
    config, state, path, _, request = fixture
    result = rebind(fixture)
    assert state.command("rebind-1") is None
    # Simulate the running Agent dying before replacing its in-memory profile.
    config["profiles"][0].update(code_dir=request["expected_code_dir"], python=request["expected_python"])
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):return datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(rebinder, "datetime", Later)
    monkeypatch.setattr(rebinder, "check_runtime", lambda *a, **k: pytest.fail("persisted operation must not reprobe"))
    assert rebind(fixture) == result
    assert config["profiles"] == json.loads(path.read_text())["profiles"]
    with pytest.raises(AgentError) as failure:rebind(fixture, python=sys.executable)
    assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
    assert "persisted" not in failure.value.as_dict()


@pytest.mark.parametrize("deleted", [False, True])
def test_published_ledger_with_changed_or_missing_target_never_claims_no_persistence(fixture, deleted):
    _, _, path, _, request = fixture
    rebind(fixture)
    disk = json.loads(path.read_text())
    if deleted:disk["profiles"] = disk["profiles"][1:]
    else:disk["profiles"][0]["code_dir"] = request["expected_code_dir"]
    path.write_text(json.dumps(disk))
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
    assert "persisted" not in failure.value.as_dict()


def test_replace_that_publishes_then_raises_stays_retryable_and_recovers_marker(fixture, monkeypatch):
    config, state, path, _, request = fixture
    profiles = copy.deepcopy(config["profiles"])
    agent = Agent(config)
    command = {"id": "ambiguous-publish", "type": "PROFILE_REBIND", "payload": request}
    real_replace = os.replace
    published_once = False
    def ambiguous(source, target):
        nonlocal published_once
        real_replace(source, target)
        if Path(target) == path and not published_once:
            published_once = True
            raise OSError("replace completed, confirmation lost")
    monkeypatch.setattr(os, "replace", ambiguous)
    try:
        with pytest.raises(AgentError) as failure:agent.handle_command(command)
        assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
        assert "persisted" not in failure.value.as_dict()
        assert state.command(command["id"]) is None and config["profiles"] == profiles
        assert json.loads(path.read_text())["profiles"][0]["code_dir"] == request["code_dir"]
        monkeypatch.setattr(rebinder, "check_runtime", lambda *a, **k: pytest.fail("recovery must use durable marker"))
        assert agent.handle_command(command)["status"] == "APPLIED"
        assert config["profiles"] == json.loads(path.read_text())["profiles"]
    finally:agent.close()


def test_failure_saving_command_ack_recovers_without_a_second_probe(fixture, monkeypatch):
    config, state, _, _, request = fixture
    agent = Agent(config)
    command = {"id": "lost-ack", "type": "PROFILE_REBIND", "payload": request}
    real_save = agent.state.save_command
    def failed(*a, **k):raise OSError("command record unavailable")
    monkeypatch.setattr(agent.state, "save_command", failed)
    try:
        with pytest.raises(OSError):agent.handle_command(command)
        assert state.command(command["id"]) is None
        monkeypatch.setattr(agent.state, "save_command", real_save)
        monkeypatch.setattr(rebinder, "check_runtime", lambda *a, **k: pytest.fail("saved config must not reprobe"))
        assert agent.handle_command(command)["status"] == "APPLIED"
    finally:agent.close()


def test_config_symlink_is_never_followed_for_writing(fixture):
    config, _, path, _, _ = fixture
    original = path.with_suffix(".original")
    path.rename(original);path.symlink_to(original)
    before = original.read_bytes();profiles = copy.deepcopy(config["profiles"])
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
    assert "persisted" not in failure.value.as_dict()
    assert original.read_bytes() == before and path.is_symlink() and config["profiles"] == profiles


def test_rollback_with_failed_directory_fsync_is_uncertain_even_when_old_bytes_are_visible(fixture, monkeypatch):
    config, _, path, _, _ = fixture
    old = path.read_bytes();profiles = copy.deepcopy(config["profiles"])
    real_sync = rebinder._sync_parent
    def unavailable(target):
        if target == path:raise OSError("configuration directory fsync unavailable")
        return real_sync(target)
    monkeypatch.setattr(rebinder, "_sync_parent", unavailable)
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
    assert "persisted" not in failure.value.as_dict()
    assert_unchanged(fixture, old, profiles)


def test_unreadable_config_on_redelivery_never_claims_the_operation_was_unapplied(fixture, monkeypatch):
    _, _, path, _, _ = fixture
    rebind(fixture)
    original = rebinder._read
    def unavailable(target):
        if target == path:raise PermissionError("configuration temporarily inaccessible")
        return original(target)
    monkeypatch.setattr(rebinder, "_read", unavailable)
    with pytest.raises(AgentError) as failure:rebind(fixture)
    assert failure.value.code == "PROFILE_REBIND_RECOVERY_PENDING"
    assert "persisted" not in failure.value.as_dict()


def test_durable_marker_history_is_bounded_and_preserves_other_config_keys(fixture, monkeypatch):
    config, state, path, _, request = fixture
    monkeypatch.setattr(rebinder, "MAX_OPERATIONS", 2)
    for number in range(3):
        request.update(expected_code_dir=config["profiles"][0]["code_dir"], expected_python=config["profiles"][0]["python"])
        rebinder.rebind_profile(config, state, request, operation_id="operation-" + str(number))
    disk = json.loads(path.read_text())
    assert [item["operation_id"] for item in disk[rebinder.OPERATIONS_KEY]] == ["operation-1", "operation-2"]
    assert disk["unknown_extra"]["must"] == "survive"


def test_main_dispatch_caches_result_and_does_not_restart_agent(fixture, monkeypatch):
    config, _, _, _, request = fixture
    agent = Agent(config)
    command = {"id": "rebind-command", "type": "PROFILE_REBIND", "payload": request}
    try:
        result = agent.handle_command(command)
        assert result["status"] == "APPLIED" and result["result"]["persisted"] is True
        monkeypatch.setattr(rebinder, "rebind_profile", lambda *a, **k: pytest.fail("cached command must not rebind"))
        assert agent.handle_command(command) == result
    finally:agent.close()


def test_poll_does_not_acknowledge_ambiguous_publish_and_replays_saved_marker(fixture, monkeypatch):
    config, state, _, _, request = fixture
    agent = Agent(config)
    command = {"id": "recover-through-poll", "type": "PROFILE_REBIND", "payload": request}
    acknowledgements = [];first = True
    real_rebind = rebinder.rebind_profile
    def interrupted(*args, **kwargs):
        nonlocal first
        result = real_rebind(*args, **kwargs)
        if first:
            first = False
            raise rebinder.ProfileRebindError("PROFILE_REBIND_RECOVERY_PENDING", "confirmation interrupted", persisted=None)
        return result
    def api(method, path, body=None):
        if method == "GET":return [command]
        acknowledgements.append(body)
        return {}
    monkeypatch.setattr(rebinder, "rebind_profile", interrupted)
    monkeypatch.setattr(agent, "api", api)
    try:
        agent.poll_commands()
        assert not acknowledgements and state.command(command["id"]) is None
        monkeypatch.setattr(rebinder, "check_runtime", lambda *a, **k: pytest.fail("poll recovery must not reprobe"))
        agent.poll_commands()
        assert len(acknowledgements) == 1 and acknowledgements[0]["status"] == "APPLIED"
        assert acknowledgements[0]["result"]["persisted"] is True
    finally:agent.close()


def test_start_and_both_checkouts_remain_locked_through_probe(fixture, monkeypatch):
    config, state, _, checkout, request = fixture
    entered = threading.Event();release = threading.Event();outcome = []
    real_check = rebinder.check_runtime
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_check(*args, **kwargs)
    monkeypatch.setattr(rebinder, "check_runtime", blocked)
    thread = threading.Thread(target=lambda: outcome.append(rebind(fixture)))
    thread.start();assert entered.wait(10)
    agent = Agent(config)
    try:
        for directory in (request["expected_code_dir"], str(checkout)):
            with pytest.raises(AgentError) as failure:
                with checkout_lock(config, directory, require_git=True):pytest.fail("checkout must remain locked")
            assert failure.value.code == "RECIPE_WRITE_BUSY"
        with pytest.raises(AgentError) as failure:
            agent.handle_command({"id": "start-during-rebind", "type": "START", "payload": {
                "attempt_id": "cannot-start", "execution_token": "token", "profile_id": request["profile_id"]}})
        assert failure.value.code == "RECIPE_WRITE_BUSY" and state.attempt("cannot-start") is None
        assert state.command("start-during-rebind") is None
    finally:
        release.set();thread.join(timeout=15);agent.close()
    assert not thread.is_alive() and outcome[0]["persisted"] is True


def test_runtime_monitor_fingerprint_invalidates_after_success(fixture):
    config, _, _, _, _ = fixture
    agent = Agent(config)
    try:
        agent.runtime_monitor.snapshot(config["profiles"])
        signature = agent.runtime_monitor._signatures[config["profiles"][0]["id"]]
        rebind(fixture)
        agent.runtime_monitor.snapshot(config["profiles"])
        assert agent.runtime_monitor._signatures[config["profiles"][0]["id"]] != signature
    finally:agent.close()
