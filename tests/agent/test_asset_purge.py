import os
import subprocess

import pytest

from vla_agent import asset_purge
from vla_agent.main import Agent, CAPABILITIES, _execute_preflight_command
from vla_agent.state import State


@pytest.fixture
def storage(tmp_path):
    root = tmp_path.resolve()
    for name in ("outputs", "state", "code", "cache", "data"):
        (root / name).mkdir()
    config = {"roots": [str(root)], "output_root": str(root / "outputs"), "state_dir": str(root / "state"),
              "cache_root": str(root / "cache"), "config_path": str(root / "agent.json"),
              "profiles": [{"code_dir": str(root / "code"), "python": "/usr/bin/python3"}]}
    state = State(config["state_dir"])
    return root, config, state


def inspect(storage, path, *, asset_id="first", **extra):
    _, config, state = storage
    return asset_purge.execute_asset_purge_command(config, state, "ASSET_PURGE_INSPECT",
                                                   {"asset_id": asset_id, "path": str(path), **extra})


def purge(storage, receipt, **extra):
    _, config, state = storage
    return asset_purge.execute_asset_purge_command(config, state, "ASSET_PURGE",
                                                   {**{key: receipt[key] for key in ("asset_id", "path", "identity")}, **extra})


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_purge_pins_inspection_and_never_deletes_reused_path(storage, kind):
    root, _, state = storage
    path = root / "data" / "target"
    if kind == "file":
        path.write_bytes(b"authorized asset")
    else:
        path.mkdir()
        (path / "part").write_bytes(b"authorized asset")
    receipt = inspect(storage, path)
    assert receipt["status"] == "READY" and len(receipt["identity"]) == 64
    assert state.asset_purge("first") is None
    result = purge(storage, receipt)
    assert result == {"asset_id": "first", "path": str(path), "status": "DELETED"}
    assert list(path.parent.iterdir()) == []

    path.write_bytes(b"replacement at the same path")
    assert purge(storage, receipt) == result
    assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_ALREADY_ATTEMPTED"
    assert path.read_bytes() == b"replacement at the same path"


def test_missing_receipt_never_authorizes_a_new_file(storage):
    root, _, state = storage
    path = root / "data" / "new" / "target"
    receipt = inspect(storage, path)
    assert receipt == {"asset_id": "first", "path": str(path), "status": "MISSING", "identity": None}
    path.parent.mkdir()
    path.write_text("created after inspection")
    assert purge(storage, receipt)["error"]["code"] == "ASSET_PURGE_IDENTITY_CHANGED"
    assert state.asset_purge("first") is None
    assert path.read_text() == "created after inspection"

    path.unlink()
    result = purge(storage, receipt)
    assert result["status"] == "MISSING"
    path.write_text("created after recorded absence")
    assert purge(storage, receipt) == result
    assert path.read_text() == "created after recorded absence"


def test_current_output_root_children_are_managed_even_outside_storage_roots(storage):
    root, config, _ = storage
    config["roots"] = [str(root / "data")]
    historical = root / "previous-outputs"
    historical.mkdir()
    config["_output_root_history"] = [str(historical)]
    path = root / "outputs" / "target"
    path.write_text("previous run file")
    excluded = historical / "target"
    excluded.write_text("earlier run file")
    for candidate in (root, path.parent, historical, excluded):
        assert inspect(storage, candidate)["error"]["code"] == "ASSET_PURGE_PROTECTED_PATH"
    assert purge(storage, inspect(storage, path))["status"] == "DELETED"
    assert excluded.read_text() == "earlier run file"


@pytest.mark.parametrize("relation", ["same", "parent", "missing_child"])
def test_registered_symlink_alias_blocks_deletion_at_inspection_and_confirmation(storage, relation):
    root, _, state = storage
    path = root / "data" / "target"
    path.mkdir()
    (path / "sample").write_text("asset in use")
    alias = root / "alias"
    receipt = inspect(storage, path)
    if relation == "parent":
        alias.symlink_to(path.parent, target_is_directory=True)
        protected = alias
    else:
        alias.symlink_to(path, target_is_directory=True)
        protected = alias / "missing" if relation == "missing_child" else alias
    paths = [str(protected)]
    assert inspect(storage, path, protected_paths=paths)["error"]["code"] == "ASSET_PURGE_PATH_IN_USE"
    assert purge(storage, receipt, protected_paths=paths)["error"]["code"] == "ASSET_PURGE_PATH_IN_USE"
    assert state.asset_purge("first") is None and (path / "sample").read_text() == "asset in use"


def test_missing_unrelated_protected_path_and_duplicate_are_safe(storage):
    root, _, _ = storage
    path = root / "data" / "target"
    path.write_text("disposable")
    sibling = root / "data" / "unrelated" / "missing"
    paths = [str(sibling), str(sibling)]
    receipt = inspect(storage, path, protected_paths=paths)
    assert purge(storage, receipt, protected_paths=paths)["status"] == "DELETED"


def test_protected_path_loop_is_rejected_without_leaking_path(storage):
    root, _, _ = storage
    path = root / "data" / "target"
    path.write_text("retain")
    alias = root / "data" / "other-registration"
    alias.symlink_to(alias)
    error = inspect(storage, path, protected_paths=[str(alias)])["error"]
    assert error["code"] == "ASSET_PURGE_PATH_IN_USE" and str(alias) not in error["message"]
    assert path.read_text() == "retain"


def test_protected_path_inode_alias_blocks_without_lexical_overlap(storage, monkeypatch):
    root, _, _ = storage
    path = root / "data" / "target"
    path.mkdir()
    other_mount = root / "other-mount"
    other_mount.mkdir()
    filename = other_mount / "child"
    filename.write_text("registered")
    original = type(path).stat

    def stat_path(candidate, *args, **kwargs):
        # A bind mount presents a second name for the same inode.
        if candidate == other_mount:
            return original(path, *args, **kwargs)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(type(path), "stat", stat_path)
    assert inspect(storage, path, protected_paths=[str(filename)])["error"]["code"] == "ASSET_PURGE_PATH_IN_USE"
    assert path.is_dir() and filename.is_file()


@pytest.mark.parametrize("paths", [None, "not a list", ["relative"], ["/data/../other"], ["/control\u0000path"],
                                   ["/control\udcffpath"], ["/data"] * 10_001, ["/" + "x" * 4095] * 257])
def test_protected_paths_have_bounded_and_valid_payload(storage, paths):
    assert inspect(storage, storage[0] / "data" / "target",
                   protected_paths=paths)["error"]["code"] == "ASSET_PURGE_INVALID"


@pytest.mark.parametrize("target", ["outside", "root", "output", "state", "cache", "code", "source",
                                    "symlink", "linked-parent", "nested-link", "hardlink", "fifo"])
def test_protected_and_unusual_targets_are_rejected_without_touching_data(storage, target):
    root, _, state = storage
    sentinel = root / "data" / "sentinel"
    sentinel.write_text("retain")
    path = root / "data" / "target"
    if target == "outside":
        path = root.parent / "unmanaged-target"
    elif target == "root":
        path = root
    elif target in {"output", "state", "cache", "code"}:
        path = root / ("outputs" if target == "output" else target)
    elif target == "source":
        path = root / "code" / "src" / "library.py"
    elif target == "symlink":
        path.symlink_to(sentinel)
    elif target == "linked-parent":
        path.symlink_to(sentinel.parent, target_is_directory=True)
        path = path / sentinel.name
    elif target == "nested-link":
        path.mkdir()
        (path / "link").symlink_to(sentinel)
        (path / "file").write_text("also retain")
    elif target == "hardlink":
        path.hardlink_to(sentinel)
    else:
        os.mkfifo(path)
    assert "error" in inspect(storage, path)
    assert "error" in purge(storage, {"asset_id": "first", "path": str(path), "identity": "0" * 64})
    assert state.asset_purge("first") is None
    assert sentinel.read_text() == "retain"
    if target == "nested-link":
        assert (path / "file").read_text() == "also retain"


def test_changed_contents_or_parent_refuse_the_inspected_identity(storage):
    root, _, state = storage
    path = root / "data" / "target"
    path.mkdir()
    (path / "sample").write_text("inspected data")
    receipt = inspect(storage, path)
    (path / "sample").write_text("different data")
    assert purge(storage, receipt)["error"]["code"] == "ASSET_PURGE_IDENTITY_CHANGED"
    parent = path.parent
    parent.rename(root / "moved")
    parent.mkdir()
    path.mkdir()
    assert purge(storage, receipt)["error"]["code"] == "ASSET_PURGE_IDENTITY_CHANGED"
    assert state.asset_purge("first") is None and path.is_dir()
    assert (root / "moved" / "target" / "sample").read_text() == "different data"


def test_tracked_training_files_are_protected_but_nested_data_can_be_deleted(storage):
    root, _, _ = storage
    project = root / "code"
    path = project / "tracked.txt"
    path.write_text("training source")
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(project), "add", "--", path.name], check=True)
    assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_PROTECTED_PATH"
    assert path.read_text() == "training source"
    nested = project / "datasets"
    nested.mkdir()
    (nested / "data").write_text("disposable registered data")
    assert purge(storage, inspect(storage, nested))["status"] == "DELETED"


def test_unknown_nested_git_repository_blocks_whole_tree_and_individual_children(storage):
    root, _, _ = storage
    asset = root / "data" / "asset"
    repository = asset / "child-repository"
    repository.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    file = repository / "important"
    file.write_text("unregistered source")
    for target in (asset, repository, file):
        assert "error" in inspect(storage, target)
    assert file.read_text() == "unregistered source"


def test_resource_guard_database_sidecars_and_gpu_locks_are_protected(storage):
    root, config, _ = storage
    directory = root / "data" / "controller"
    locks = directory / "locks"
    locks.mkdir(parents=True)
    lock = locks / "gpu-device.lock"
    lock.touch()
    database = directory / "control.sqlite3"
    sidecar = database.with_name(database.name + "-wal")
    database.touch()
    sidecar.touch()
    config["resource_guard"] = {"database": str(database), "lock_directory": str(locks)}
    for path in (directory, database, sidecar, locks, lock):
        assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_PROTECTED_PATH"
    assert all(path.is_file() for path in (database, sidecar, lock))


def test_purge_serializes_with_start_lock_used_for_attempt_admission(storage):
    root, config, _ = storage
    path = root / "data" / "target"
    path.write_text("assets")
    receipt = inspect(storage, path)
    with asset_purge.profile_configuration_lock(config, busy_code="RECIPE_WRITE_BUSY"):
        assert purge(storage, receipt)["error"]["code"] == "ASSET_PURGE_BUSY"
    assert path.read_text() == "assets"
    assert purge(storage, receipt)["status"] == "DELETED"


def test_mount_policy_and_local_training_reference_block_purge(storage, monkeypatch):
    root, _, state = storage
    path = root / "data" / "target"
    path.mkdir()
    opened = inspect(storage, path)
    source = path / "sample"
    source.write_text("training data")
    state.create_attempt({"attempt_id": "running", "inputs": {"dataset": {"path": str(source)}}},
                         str(root / "outputs" / "running"))
    assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_IN_USE"
    assert purge(storage, opened)["error"]["code"] == "ASSET_PURGE_IN_USE"
    assert source.read_text() == "training data"
    if asset_purge.sys.platform.startswith("linux"):
        monkeypatch.setattr(asset_purge, "LOCAL_FILESYSTEMS", set())
        state.emit("running", "SUCCEEDED", {})
        assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_UNSAFE_MOUNT"


def test_active_resume_source_protects_checkpoint_and_original_run_metadata(storage):
    root, _, state = storage
    output = root / "outputs" / "original-run"
    checkpoint = output / "checkpoints" / "100"
    checkpoint.mkdir(parents=True)
    config = output / "effective_training.json"
    config.write_text("{}")
    state.create_attempt({"attempt_id": "running", "resume_source": {
        "path": str(checkpoint), "output_dir": str(output),
    }}, str(root / "outputs" / "running"))
    for path in (checkpoint, config):
        assert inspect(storage, path)["error"]["code"] == "ASSET_PURGE_IN_USE"


@pytest.mark.parametrize("point", ["after_rename", "after_remove", "partial"])
def test_durable_quarantine_crash_replay_never_touches_replacement(storage, monkeypatch, point):
    root, _, state = storage
    path = root / "data" / "target"
    path.mkdir()
    (path / "sample").write_text("original")
    (path / "first-child").write_text("removed in the partial case")
    receipt = inspect(storage, path)

    with monkeypatch.context() as patch:
        if point == "after_rename":
            original = asset_purge.os.rename

            def interrupted(*args, **kwargs):
                original(*args, **kwargs)
                raise KeyboardInterrupt()

            patch.setattr(asset_purge.os, "rename", interrupted)
        else:
            original = asset_purge._remove

            def interrupted(parent, name, *args, **kwargs):
                if not kwargs.get("depth"):
                    if point == "after_remove":
                        original(parent, name, *args, **kwargs)
                    else:
                        with asset_purge._directory_at(parent, name) as descriptor:
                            os.unlink("first-child", dir_fd=descriptor)
                    raise KeyboardInterrupt()
                return original(parent, name, *args, **kwargs)

            patch.setattr(asset_purge, "_remove", interrupted)
        with pytest.raises(KeyboardInterrupt):
            purge(storage, receipt)
    record = state.asset_purge("first")
    assert record["phase"] == ("PREPARED" if point == "after_rename" else "DELETING")
    path.write_text("new registration contents")
    reopened = (root, storage[1], State(storage[1]["state_dir"]))
    replay = purge(reopened, receipt)
    if point == "partial":
        assert replay["staged"] is True
        assert replay["error"]["code"] == "ASSET_PURGE_INDETERMINATE"
        assert (path.parent / record["name"] / "asset" / "sample").read_text() == "original"
    else:
        assert replay["status"] == "DELETED"
    assert path.read_text() == "new registration contents"


def test_replacement_during_final_rename_is_preserved(storage, monkeypatch):
    root, _, state = storage
    path = root / "data" / "target"
    path.write_text("original")
    receipt = inspect(storage, path)
    original = os.rename

    def replace_before_rename(*args, **kwargs):
        original(path, path.with_name("moved"))
        path.write_text("replacement")
        original(*args, **kwargs)

    monkeypatch.setattr(os, "rename", replace_before_rename)
    result = purge(storage, receipt)
    assert result["staged"] is True
    assert result["error"]["code"] == "ASSET_PURGE_INDETERMINATE"
    assert (path.parent / state.asset_purge("first")["name"] / "asset").read_text() == "replacement"
    assert path.with_name("moved").read_text() == "original"


def test_completed_operation_replays_after_path_symlink_and_root_change(storage):
    root, config, _ = storage
    path = root / "data" / "target"
    path.write_text("original")
    receipt = inspect(storage, path)
    assert purge(storage, receipt)["status"] == "DELETED"
    path.parent.rmdir()
    path.parent.symlink_to(root / "outputs")
    config["roots"] = [str(root / "outputs")]
    assert purge(storage, receipt)["status"] == "DELETED"
    assert path.parent.is_symlink()


def test_agent_dispatch_persists_completed_command_for_replay(storage, monkeypatch):
    root, config, state = storage
    path = root / "data" / "target"
    path.write_text("original")
    assert {"ASSET_PURGE_INSPECT", "ASSET_PURGE"} <= set(CAPABILITIES)
    agent = Agent(config, client=object())
    command = {"id": "purge-1", "type": "ASSET_PURGE_INSPECT",
               "payload": {"asset_id": "first", "path": str(path)}}
    monkeypatch.setattr(asset_purge.os, "setsid", lambda: None)
    monkeypatch.setattr("vla_agent.main.signal.signal", lambda *args: None)
    _execute_preflight_command(config, command)
    inspected = state.command(command["id"])
    assert inspected["status"] == "APPLIED" and inspected["result"]["status"] == "READY"
    command = {"id": "purge-2", "type": "ASSET_PURGE",
               "payload": {key: inspected["result"][key] for key in ("asset_id", "path", "identity")}}
    result = agent.handle_command(command)
    assert result["status"] == "APPLIED" and result["result"]["status"] == "DELETED"
    path.write_text("replacement")
    assert agent.handle_command(command) == result
    assert path.read_text() == "replacement"
    agent.close()
