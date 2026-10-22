import ast
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vla_adapters.recipe_definitions import definition_digest
from vla_agent import recipe_writeback as writer
from vla_agent.recipe_details import recipe_details
from vla_agent.safety import AgentError
from vla_agent.state import State


SOURCE = '''# Existing module and comments must remain byte-for-byte.
raise RuntimeError("training code must never be imported")
class TrainConfig:
    num_train_steps: int = 60000
    seed: int = 42
_CONFIGS: list[TrainConfig] = [
    TrainConfig(
        name="base_config",  # Keep this original comment.
        batch_size=72,
        lr_schedule=Cosine(peak_lr=0.00003),
        data=MultiData(state_pad_dim=128, weights=[1.0, 1.0],
                       datasets=[UmiData(repo_id="/old/a", image_horizon=1), UmiData(repo_id="/old/b", image_horizon=1)]),
    ),
]
'''


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True).stdout


@pytest.fixture
def fixture(tmp_path):
    checkout = tmp_path / "checkout"
    path = checkout / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    (checkout / "other.txt").write_text("original\n")
    for arguments in (("init", "-b", "test"), ("config", "user.name", "Recipe Tests"),
                      ("config", "user.email", "recipe@example.invalid"), ("add", "."), ("commit", "-m", "Fixture")):
        git(checkout, *arguments)
    config = {"roots": [str(tmp_path)], "state_dir": str(tmp_path / "state"), "profiles": [
        {"id": "profile-1", "adapter": "openpi-sft", "code_dir": str(checkout), "execution_mode": "native"}]}
    state = State(config["state_dir"])
    return config, checkout, path, state


def payload(config, **changes):
    expires = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    detail = recipe_details(config, {"profile_id": "profile-1", "recipe": "base_config", "expires_at": expires})
    definition = {"schema_version": 1, "id": "revision-1", "recipe_id": "saved-1", "name": "My recipe", "revision": 1,
                  "base_recipe": "base_config", "node_id": "node-1", "profile_id": "profile-1", "source": detail["source"],
                  "overrides": {"num_train_steps": 100, "batch_size": 8, "lr_schedule.peak_lr": 0.00002,
                                "data.state_pad_dim": 64, "data.datasets[1].image_horizon": 2}}
    definition["digest"] = definition_digest(definition)
    return {"operation_id": "operation-1", "profile_id": "profile-1", "definition": definition,
            "config_name": "new_config", "expires_at": expires, **changes}


def fields(result):
    return {field["path"]: field.get("value") for field in result["detail"]["fields"]}


def test_append_and_commit_nested_recipe_without_executing_source(fixture):
    config, checkout, path, state = fixture
    request = payload(config)
    previous = git(checkout, "rev-parse", "HEAD").decode().strip()
    result = writer.write_recipe(config, request, state)
    assert result["previous_commit_sha"] == previous
    assert result["commit_sha"] == git(checkout, "rev-parse", "HEAD").decode().strip()
    assert result["changed_paths"] == ["src/openpi/training/config.py"]
    assert git(checkout, "status", "--porcelain") == b""
    assert git(checkout, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines() == result["changed_paths"]
    assert git(checkout, "show", "HEAD:src/openpi/training/config.py") == path.read_bytes()
    values = fields(result)
    assert values["name"] == "new_config" and values["num_train_steps"] == 100
    assert values["data.datasets[1].image_horizon"] == 2
    assert values["data.datasets[1].repo_id"] == "/old/b"
    assert values["data.state_pad_dim"] == 64 and values["data.weights"] == [1.0, 1.0]
    assert result["detail"]["source"]["commit_sha"] == result["commit_sha"]
    assert result["detail"]["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert SOURCE[:-2] in path.read_text() and "# Keep this original comment." in path.read_text()
    assert "VLA-Recipe-Operation: operation-1" in git(checkout, "show", "-s", "--format=%B", "HEAD").decode()


def test_commit_preserves_unrelated_staged_and_unstaged_changes(fixture):
    config, checkout, path, state = fixture
    other = checkout / "other.txt"
    other.write_text("staged\n")
    git(checkout, "add", "other.txt")
    other.write_text("unstaged\n")
    before_index = git(checkout, "diff", "--cached", "--binary")
    before_unstaged = git(checkout, "diff", "--binary")
    result = writer.write_recipe(config, payload(config), state)
    assert git(checkout, "diff", "--cached", "--binary") == before_index
    assert git(checkout, "diff", "--binary") == before_unstaged
    assert git(checkout, "show", "HEAD:other.txt") == b"original\n"
    assert result["detail"]["source"]["dirty"] is True
    assert other.read_text() == "unstaged\n"


@pytest.mark.parametrize("stage", [False, True])
def test_dirty_config_is_rejected_even_if_saved_definition_matches(fixture, stage):
    config, checkout, path, state = fixture
    path.write_text(SOURCE.replace("batch_size=72", "batch_size=36"))
    if stage:
        git(checkout, "add", str(path))
    request = payload(config)
    before = path.read_bytes()
    with pytest.raises(AgentError, match="config.py") as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_SOURCE_DIRTY"
    assert path.read_bytes() == before and git(checkout, "rev-list", "--count", "HEAD").strip() == b"1"


def test_assume_unchanged_does_not_hide_existing_source_edits(fixture):
    config, checkout, path, state = fixture
    git(checkout, "update-index", "--assume-unchanged", "src/openpi/training/config.py")
    path.write_text(SOURCE + "# existing hidden edit\n")
    assert git(checkout, "status", "--porcelain") == b""
    before = path.read_bytes()
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, payload(config), state)
    assert failure.value.code == "RECIPE_SOURCE_DIRTY"
    assert path.read_bytes() == before


@pytest.mark.parametrize("attempt_state", ["STARTING", "RUNNING", "UNKNOWN"])
def test_busy_checkout_rejected_with_frozen_path_even_if_profile_was_moved(fixture, attempt_state, tmp_path):
    config, checkout, path, state = fixture
    request = payload(config)
    state.create_attempt({"attempt_id": "live-1", "profile_id": "old-profile", "code_expectation": {"code_dir": str(checkout)}}, str(tmp_path / "out"))
    with state.connect() as db:
        db.execute("UPDATE attempts SET state=?", (attempt_state,))
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_CHECKOUT_BUSY"
    assert path.read_text() == SOURCE


def test_saved_source_hash_and_git_pin_must_match(fixture):
    config, checkout, path, state = fixture
    request = payload(config)
    (checkout / "other.txt").write_text("changed after reading\n")
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_SOURCE_CHANGED" and path.read_text() == SOURCE


def test_duplicate_config_name_is_rejected(fixture):
    config, checkout, path, state = fixture
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, payload(config, config_name="base_config"), state)
    assert failure.value.code == "RECIPE_NAME_CONFLICT"
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"1"


def test_invalid_generic_literal_does_not_modify_code(fixture):
    config, checkout, path, state = fixture
    request = payload(config)
    request["definition"]["overrides"] = {"data.missing": "arbitrary"}
    request["definition"]["digest"] = definition_digest(request["definition"])
    with pytest.raises(ValueError, match="RECIPE_FIELD_UNSUPPORTED"):
        writer.write_recipe(config, request, state)
    assert path.read_text() == SOURCE


@pytest.mark.parametrize("overrides", [{"data.datasets[1].repo_id": "/hidden/unregistered"},
                                       {"data.weights": [2.0, 3.0]},
                                       {"data.datasets[0].assets.assets_dir": "/hidden/unregistered"}])
def test_asset_resource_override_cannot_be_written_or_committed(fixture, overrides):
    config, checkout, path, state = fixture
    request = payload(config)
    request["definition"]["overrides"] = overrides
    request["definition"]["digest"] = definition_digest(request["definition"])
    before = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="RECIPE_ASSET_FIELDS_MANAGED"):
        writer.write_recipe(config, request, state)
    assert path.read_text() == SOURCE
    assert git(checkout, "rev-parse", "HEAD") == before and git(checkout, "status", "--porcelain") == b""


def test_failure_before_ref_publish_rolls_back_only_introduced_source(fixture, monkeypatch):
    config, checkout, path, state = fixture
    (checkout / "other.txt").write_text("keep staged\n")
    git(checkout, "add", "other.txt")
    staged = git(checkout, "diff", "--cached", "--binary")
    request = payload(config)
    original = writer._git_write
    def fail(root, *args, **kwargs):
        if args[0] == "update-ref":
            raise AgentError("RECIPE_COMMIT_FAILED", "test ref failure")
        return original(root, *args, **kwargs)
    monkeypatch.setattr(writer, "_git_write", fail)
    with pytest.raises(AgentError, match="test ref failure"):
        writer.write_recipe(config, request, state)
    assert path.read_text() == SOURCE and git(checkout, "diff", "--cached", "--binary") == staged
    assert not (checkout / ".git/index.lock").exists()
    monkeypatch.setattr(writer, "_git_write", original)
    result = writer.write_recipe(config, request, state)
    assert result["detail"] and git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"


def test_crash_after_commit_ref_recovers_index_without_second_commit(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original_replace = writer.os.replace
    def fail_once(source, target):
        if Path(target) == checkout / ".git/index":
            raise SystemExit("simulated worker crash")
        return original_replace(source, target)
    monkeypatch.setattr(writer.os, "replace", fail_once)
    with pytest.raises(SystemExit, match="simulated"):
        writer.write_recipe(config, request, state)
    committed = git(checkout, "rev-parse", "HEAD")
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"
    monkeypatch.setattr(writer.os, "replace", original_replace)
    # Durable operation intent authorizes recovery even after its dispatch TTL.
    request["expires_at"] = "2000-01-01T00:00:00Z"
    result = writer.write_recipe(config, request, State(config["state_dir"]))
    assert result["detail"] and git(checkout, "rev-parse", "HEAD") == committed
    assert git(checkout, "status", "--porcelain") == b""
    assert writer.write_recipe(config, request, State(config["state_dir"])) == result


def test_ordinary_index_publish_error_requests_retry_instead_of_terminal_rejection(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original = writer.os.replace
    def fail(source, target):
        if Path(target) == checkout / ".git/index":
            raise OSError("temporary index write failure")
        return original(source, target)
    monkeypatch.setattr(writer.os, "replace", fail)
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_WRITE_RECOVERY_PENDING"
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"
    monkeypatch.setattr(writer.os, "replace", original)
    assert writer.write_recipe(config, request, state)["detail"]
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"
    assert git(checkout, "status", "--porcelain") == b""


def test_update_ref_ambiguous_success_is_reconciled(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original = writer._git_write
    def fail(root, *arguments, **kwargs):
        result = original(root, *arguments, **kwargs)
        if arguments[0] == "update-ref":
            raise OSError("update-ref succeeded but caller lost response")
        return result
    monkeypatch.setattr(writer, "_git_write", fail)
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_WRITE_RECOVERY_PENDING"
    monkeypatch.setattr(writer, "_git_write", original)
    assert writer.write_recipe(config, request, state)["detail"]
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"


def test_unreadable_head_after_ambiguous_ref_publication_remains_retryable(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original_write, original_read = writer._git_write, writer._git
    attempted = []
    def ambiguous(root, *arguments, **kwargs):
        result = original_write(root, *arguments, **kwargs)
        if arguments[0] == "update-ref":
            attempted.append(True)
            raise OSError("lost ref response")
        return result
    def unreadable(root, *arguments, **kwargs):
        if attempted and arguments == ("rev-parse", "HEAD"):
            raise AgentError("CODE_VERSION_UNAVAILABLE", "temporary Git read failure")
        return original_read(root, *arguments, **kwargs)
    monkeypatch.setattr(writer, "_git_write", ambiguous)
    monkeypatch.setattr(writer, "_git", unreadable)
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_WRITE_RECOVERY_PENDING"
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"
    # A retry that cannot yet read HEAD also stays pending, never REJECTED.
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_WRITE_RECOVERY_PENDING"
    monkeypatch.setattr(writer, "_git_write", original_write)
    monkeypatch.setattr(writer, "_git", original_read)
    assert writer.write_recipe(config, request, state)["detail"]
    assert git(checkout, "status", "--porcelain") == b""


def test_lock_ownership_record_failure_never_leaves_unowned_git_lock(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original = writer._record
    def fail(record_path, receipt):
        if "index_lock" in receipt:
            raise OSError("cannot journal lock ownership")
        return original(record_path, receipt)
    monkeypatch.setattr(writer, "_record", fail)
    with pytest.raises(OSError, match="ownership"):
        writer.write_recipe(config, request, state)
    assert not (checkout / ".git/index.lock").exists()
    assert not list((checkout / ".git").glob(".vla-recipe-index-*"))
    assert path.read_text() == SOURCE


def test_sibling_profile_in_same_git_repository_blocks_write_and_shares_lock(fixture, tmp_path):
    config, checkout, path, state = fixture
    sibling = checkout / "another_project"
    sibling.mkdir()
    config["profiles"].append({"id": "sibling", "adapter": "openpi-sft", "code_dir": str(sibling)})
    request = payload(config)
    state.create_attempt({"attempt_id": "sibling-live", "profile_id": "sibling", "code_expectation": {"repository_root": str(checkout)}}, str(tmp_path / "out"))
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_CHECKOUT_BUSY"
    with writer.checkout_lock(config, str(sibling)):
        with pytest.raises(AgentError) as failure:
            with writer.checkout_lock(config, str(checkout)):
                pytest.fail("same repository must share one lock")
    assert failure.value.code == "RECIPE_WRITE_BUSY"


def test_durable_result_is_idempotent_and_operation_identity_is_bound(fixture):
    config, checkout, path, state = fixture
    request = payload(config)
    result = writer.write_recipe(config, request, state)
    request["expires_at"] = "2000-01-01T00:00:00Z"
    assert writer.write_recipe(config, request, State(config["state_dir"])) == result
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"2"
    request["config_name"] = "different_config"
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_OPERATION_CONFLICT"


def test_new_expired_request_and_busy_git_index_make_no_changes(fixture):
    config, checkout, path, state = fixture
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, payload(config, expires_at="2000-01-01T00:00:00Z"), state)
    assert failure.value.code == "RECIPE_QUERY_EXPIRED"
    lock = checkout / ".git/index.lock"
    lock.write_text("somebody else's lock")
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, payload(config), state)
    assert failure.value.code == "RECIPE_GIT_BUSY"
    assert lock.read_text() == "somebody else's lock" and path.read_text() == SOURCE


def test_checkout_start_and_write_share_same_lock(fixture):
    config, checkout, path, state = fixture
    with writer.checkout_lock(config, str(checkout)):
        with pytest.raises(AgentError) as failure:
            writer.write_recipe(config, payload(config), state)
    assert failure.value.code == "RECIPE_WRITE_BUSY"


@pytest.mark.parametrize("suffix", ["]\n", "] # registry\n"])
def test_static_single_line_registry_without_trailing_comma(fixture, suffix):
    config, checkout, path, state = fixture
    path.write_text('_CONFIGS = [TrainConfig(name="base_config", batch_size=72)' + suffix)
    git(checkout, "add", str(path))
    git(checkout, "commit", "-m", "Small registry")
    request = payload(config)
    request["definition"]["overrides"] = {"batch_size": 8}
    request["definition"]["digest"] = definition_digest(request["definition"])
    result = writer.write_recipe(config, request, state)
    ast.parse(path.read_bytes())
    assert fields(result)["batch_size"] == 8


def test_source_changed_after_git_objects_prepared_is_preserved(fixture, monkeypatch):
    config, checkout, path, state = fixture
    request = payload(config)
    original = writer._index_lock
    external = SOURCE + "# external concurrent edit\n"
    def edit(*args):
        result = original(*args)
        path.write_text(external)
        return result
    monkeypatch.setattr(writer, "_index_lock", edit)
    with pytest.raises(AgentError):
        writer.write_recipe(config, request, state)
    assert path.read_text() == external
    assert git(checkout, "rev-list", "--count", "HEAD").strip() == b"1"


@pytest.mark.parametrize("default", [
    "dc.field(default_factory=opt.Cosine)",
    "dc.field(default_factory=lambda: opt.Cosine(warmup=5))",
    "dc.field(default=opt.Cosine(warmup=5))",
])
def test_dataclass_default_factories_become_explicit_constructors(fixture, default):
    config, checkout, path, state = fixture
    path.write_text(f'''import dataclasses as dc
class TrainConfig:
    lr_schedule: object = {default}
_CONFIGS = [TrainConfig(name="base_config", batch_size=72)]
''')
    git(checkout, "add", str(path))
    git(checkout, "commit", "-m", "Default factory")
    request = payload(config)
    request["definition"]["overrides"] = {"lr_schedule.peak_lr": 0.00002}
    request["definition"]["digest"] = definition_digest(request["definition"])
    result = writer.write_recipe(config, request, state)
    assert fields(result)["lr_schedule.peak_lr"] == 0.00002
    assert "lr_schedule=opt.Cosine(" in result["detail"]["source_code"]
    assert "dc.field(" not in result["detail"]["source_code"]


def test_complex_default_factory_rejected_instead_of_mutating_field_constructor(fixture):
    config, checkout, path, state = fixture
    path.write_text('''import dataclasses
class TrainConfig:
    lr_schedule: object = dataclasses.field(default_factory=functools.partial(opt.Cosine, 5))
_CONFIGS = [TrainConfig(name="base_config", batch_size=72)]
''')
    git(checkout, "add", str(path))
    git(checkout, "commit", "-m", "Complex factory")
    request = payload(config)
    request["definition"]["overrides"] = {"lr_schedule.peak_lr": 0.00002}
    request["definition"]["digest"] = definition_digest(request["definition"])
    with pytest.raises(AgentError) as failure:
        writer.write_recipe(config, request, state)
    assert failure.value.code == "RECIPE_WRITE_UNSUPPORTED"
