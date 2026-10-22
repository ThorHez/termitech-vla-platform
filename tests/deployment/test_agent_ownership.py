import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "vla_agent_ownership", Path(__file__).resolve().parents[2] / "deploy/agent_ownership.py"
)
ownership = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ownership)


@pytest.fixture
def migration(tmp_path, monkeypatch):
    old_uid = tmp_path.stat().st_uid
    old_gid = tmp_path.stat().st_gid
    target = SimpleNamespace(pw_uid=old_uid + 1000, pw_gid=old_gid + 1000)
    previous = SimpleNamespace(pw_uid=old_uid, pw_gid=old_gid)
    changed, modes = [], []
    monkeypatch.setattr(ownership.os, "chown", lambda path, uid, gid, **kw: changed.append((Path(path), uid, gid, kw)))
    monkeypatch.setattr(ownership.os, "chmod", lambda path, mode, **kw: modes.append((Path(path), mode, kw)))
    return ownership.OwnershipMigration(target, previous), target, previous, changed, modes


def test_private_state_migrates_db_wal_and_locks_with_rollback(tmp_path, migration):
    transaction, target, previous, changed, modes = migration
    state = tmp_path / "state"
    (state / "runner-logs").mkdir(parents=True)
    for name in ("agent.sqlite3", "agent.sqlite3-wal", "agent.sqlite3-shm", "preflight-worker.lock", "instance-id"):
        (state / name).write_text("preserved")
    (state / "runner-logs/run.log").write_text("training log")
    transaction.migrate(state, recursive=True)
    expected = {state, *state.rglob("*")}
    assert {entry[0] for entry in changed} == expected
    assert all(entry[1:3] == (target.pw_uid, target.pw_gid) for entry in changed)
    assert all(entry[3] == {"follow_symlinks": False} for entry in changed)
    changed.clear()
    transaction.rollback()
    assert {entry[0] for entry in changed} == expected
    assert all(entry[1:3] == (previous.pw_uid, previous.pw_gid) for entry in changed)
    assert len(modes) == len(expected)
    assert (state / "agent.sqlite3").read_text() == "preserved"


def test_output_changes_only_top_directory_preserving_historical_artifacts(tmp_path, migration):
    transaction, _, _, changed, _ = migration
    output = tmp_path / "outputs"
    (output / "old-run").mkdir(parents=True)
    (output / "old-run/checkpoint").write_text("weights")
    transaction.migrate(output, recursive=False)
    assert [entry[0] for entry in changed] == [output]
    assert (output / "old-run/checkpoint").read_text() == "weights"


def test_rollback_restores_new_wal_and_lock_owners(tmp_path, migration):
    transaction, _, previous, changed, _ = migration
    private = tmp_path / "state"
    private.mkdir()
    (private / "agent.sqlite3").write_text("identity")
    transaction.migrate(private, recursive=True)
    new_file = private / "agent.sqlite3-wal"
    new_file.write_text("new WAL")
    new_lock = private / "preflight-worker.lock"
    new_lock.write_text("lock")
    changed.clear()
    transaction.rollback()
    for path in (new_file, new_lock):
        assert any(entry[:3] == (path, previous.pw_uid, previous.pw_gid) for entry in changed)


@pytest.mark.parametrize("link_root", [False, True])
def test_symlink_is_rejected_before_any_ownership_change(tmp_path, migration, link_root):
    transaction, _, _, changed, _ = migration
    training = tmp_path / "training"
    training.mkdir()
    private = tmp_path / "state"
    if link_root:
        private.symlink_to(training, target_is_directory=True)
    else:
        private.mkdir()
        (private / "agent.sqlite3").write_text("identity")
        (private / "link").symlink_to(training, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        transaction.migrate(private, recursive=True)
    assert not changed


def test_shared_hardlink_is_rejected_before_ownership_change(tmp_path, migration):
    transaction, _, _, changed, _ = migration
    training = tmp_path / "training-file"
    training.write_text("outside")
    private = tmp_path / "state"
    private.mkdir()
    (private / "shared").hardlink_to(training)
    with pytest.raises(ValueError, match="硬链接"):
        transaction.migrate(private, recursive=True)
    assert not changed


def test_symlink_in_parent_is_rejected_without_touching_external_tree(tmp_path, migration):
    transaction, _, _, changed, _ = migration
    outside = tmp_path / "outside"
    (outside / "state").mkdir(parents=True)
    (outside / "state/agent.sqlite3").write_text("identity")
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        transaction.migrate(alias / "state", recursive=True)
    assert not changed


def test_foreign_owner_is_rejected_before_any_ownership_change(tmp_path, migration, monkeypatch):
    transaction, target, _, changed, _ = migration
    private = tmp_path / "state"
    private.mkdir()
    metadata = private.lstat()
    monkeypatch.setattr(ownership.Path, "lstat", lambda self: SimpleNamespace(
        st_mode=metadata.st_mode, st_uid=target.pw_uid + 1, st_gid=metadata.st_gid, st_nlink=metadata.st_nlink))
    with pytest.raises(ValueError, match="其他账户"):
        transaction.migrate(private)
    assert not changed


def test_missing_directory_and_repeated_rollback_are_noops(tmp_path, migration):
    transaction, _, _, changed, modes = migration
    transaction.migrate(tmp_path / "missing", recursive=True)
    transaction.rollback()
    transaction.rollback()
    assert not changed and not modes


def test_rollback_gives_recreated_private_directory_to_previous_account(tmp_path, migration):
    transaction, _, previous, changed, _ = migration
    home = tmp_path / "home"
    transaction.migrate(home, recursive=True)
    home.mkdir(mode=0o700)
    config = home / ".gitconfig"
    config.write_text("[safe]\n")
    transaction.rollback()
    assert {entry[0] for entry in changed} == {home, config}
    assert all(entry[1:3] == (previous.pw_uid, previous.pw_gid) for entry in changed)
