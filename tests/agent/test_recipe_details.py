import hashlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vla_agent import recipe_details as details_module
from vla_agent.main import Agent
from vla_agent.recipe_details import recipe_details
from vla_agent.safety import AgentError


def query(**overrides):
    return {"profile_id": "openpi-native-1", "recipe": "pi0_test",
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(), **overrides}


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=True).stdout


def checkout(tmp_path, source, *, layout="src/openpi/training/config.py"):
    root = tmp_path / "checkout"
    path = root / layout
    path.parent.mkdir(parents=True)
    path.write_text(source)
    for arguments in (("init", "-b", "test"), ("config", "user.name", "Recipe Tests"),
                      ("config", "user.email", "recipe@example.invalid"), ("add", "."), ("commit", "-m", "Fixture")):
        git(root, *arguments)
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi-native-1", "adapter": "openpi-sft",
              "execution_mode": "native", "code_dir": str(root), "python": "/never/execute/me"}]}
    return config, path


def by_path(result):
    return {field["path"]: field for field in result["fields"]}


@pytest.mark.parametrize("layout", ("src/openpi/training/config.py", "openpi/training/config.py"))
def test_recipe_nested_fields_defaults_and_pinned_source_without_importing(tmp_path, layout):
    config, path = checkout(tmp_path, '''
import jax
raise RuntimeError("must never execute")
class TrainConfig:
    num_train_steps: int = 30000
    batch_size: int = 32
    seed: int = 42
    ema_decay: float | None = None
    model: object = dataclasses.field(default_factory=initialize_gpu)
_CONFIGS = [TrainConfig(
    name="pi0_test", batch_size=4,
    model=Pi0Config(action_dim=32, action_horizon=10, freeze=make_freeze()),
    lr_schedule=CosineDecaySchedule(peak_lr=5e-5),
    optimizer=AdamW(clip_gradient_norm=1.0),
    data=CombinedData(datasets=[UmiData(repo_id="/data2/中文")]),
)]
''', layout=layout)
    result = recipe_details(config, query())
    fields = by_path(result)
    assert fields["model.action_dim"]["value"] == 32
    assert fields["model.freeze"]["kind"] == "expression" and fields["model.freeze"]["expression"] == "make_freeze()"
    assert fields["data.datasets[0].repo_id"]["value"] == "/data2/中文"
    assert fields["lr_schedule.peak_lr"]["value"] == 5e-5
    assert fields["optimizer.clip_gradient_norm"]["value"] == 1.0
    assert fields["batch_size"]["value"] == 4 and fields["batch_size"]["origin"] == "recipe"
    assert fields["num_train_steps"]["value"] == 30000 and fields["num_train_steps"]["origin"] == "default"
    assert fields["seed"]["value"] == 42 and fields["ema_decay"]["value"] is None
    assert result["warnings"] == []
    assert result["source"] == {
        "path": layout, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "line": 10, "end_line": 16,
        "commit_sha": git(path.parents[3] if layout.startswith("src/") else path.parents[2], "rev-parse", "HEAD").decode().strip(),
        "code_dir": config["profiles"][0]["code_dir"], "dirty": False,
        "changes_digest": hashlib.sha256(b"STAGED\0\0UNSTAGED\0").hexdigest(),
    }
    assert result["source_code"].startswith('TrainConfig(') and result["source_code"].endswith(')')
    assert "class TrainConfig" not in result["source_code"] and "raise RuntimeError" not in result["source_code"]


def test_unresolved_default_factory_is_expression_and_never_called(tmp_path):
    config, _ = checkout(tmp_path, '''
class TrainConfig:
    batch_size: int = 32
    lr_schedule: object = dataclasses.field(default_factory=initialize_gpu)
_CONFIGS = [TrainConfig(name="pi0_test")]
''')
    result = recipe_details(config, query())
    assert by_path(result)["lr_schedule"]["kind"] == "expression"
    assert "value" not in by_path(result)["lr_schedule"]
    assert "UNRESOLVED_DEFAULTS" in result["warnings"]


def test_literal_shapes_preserve_values_and_leave_expressions_unresolved(tmp_path):
    config, _ = checkout(tmp_path, '''
class TrainConfig: pass
_CONFIGS = [TrainConfig(name="pi0_test", seed=-5, scalar=+1.5, flags=[True, None, [1, "text"]],
                       dynamic=2*3, mapping={"x": 1}, infinite=1e999, unsafe_integer=9007199254740992)]
''')
    fields = by_path(recipe_details(config, query()))
    assert fields["seed"]["value"] == -5 and fields["scalar"]["value"] == 1.5
    assert fields["flags"]["value"] == [True, None, [1, "text"]]
    for name in ("dynamic", "mapping", "infinite", "unsafe_integer"):
        assert fields[name]["kind"] == "expression" and "value" not in fields[name]


def test_git_pin_matches_existing_staged_unstaged_snapshot_digest(tmp_path):
    config, path = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="pi0_test", batch_size=2)]\n')
    root = Path(config["profiles"][0]["code_dir"])
    path.write_text('_CONFIGS = [TrainConfig(name="pi0_test", batch_size=3)]\n')
    git(root, "add", ".")
    path.write_text('_CONFIGS = [TrainConfig(name="pi0_test", batch_size=4)]\n')
    result = recipe_details(config, query())
    staged = git(root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary", "--")
    unstaged = git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--")
    assert result["source"]["changes_digest"] == hashlib.sha256(b"STAGED\0" + staged + b"\0UNSTAGED\0" + unstaged).hexdigest()
    assert result["source"]["dirty"] is True
    assert by_path(result)["batch_size"]["value"] == 4


def test_primary_registry_binding_allows_aliases_and_ignores_unrelated_names(tmp_path):
    config, _ = checkout(tmp_path, '''
unrelated = TrainConfig(name="other")
BASE = [TrainConfig(name="pi0_test", batch_size=2)]
alias = BASE
BASE = [TrainConfig(name="rebound")]
_CONFIGS = alias
''')
    assert by_path(recipe_details(config, query()))["batch_size"]["value"] == 2


@pytest.mark.parametrize("source", [
    '_CONFIGS = make_configs()\n',
    'from openpi.training.extra import EXTRA\n_CONFIGS = EXTRA\n',
    'unrelated=TrainConfig(name="pi0_test")\n_CONFIGS = []\n',
    '_CONFIGS = [TrainConfig(name="pi0_test"), TrainConfig(name="pi0_test")]\n',
    'unrelated=TrainConfig(name="pi0_test")\n_CONFIGS=[TrainConfig(name="pi0_test")]\n',
    '_CONFIGS = [TrainConfig(*args, name="pi0_test")]\n',
    '_CONFIGS = [TrainConfig(name="pi0_test", **kwargs)]\n',
    '_CONFIGS = [TrainConfig(name="pi0_test", batch_size=1, batch_size=2)]\n',
])
def test_unsupported_or_ambiguous_details_never_guess_fields(tmp_path, source):
    config, path = checkout(tmp_path, source)
    (path.parent / "extra.py").write_text('EXTRA = [TrainConfig(name="pi0_test", batch_size=4)]\n')
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_DETAIL_UNSUPPORTED"


def test_absent_recipe_reports_not_found_when_registry_is_complete(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="other")]\n')
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_NOT_FOUND"


@pytest.mark.parametrize("overrides", [{"recipe": "bad.name"}, {"recipe": None}, {"profile_id": ""},
    {"profile_id": "bad\n"}, {"expires_at": None}, {"expires_at": "2030-01-01T00:00:00"},
    {"expires_at": "2030-01-01T00:00:00+01:00"}])
def test_invalid_requests_do_not_access_files(monkeypatch, overrides):
    monkeypatch.setattr(details_module, "allowed_path", lambda *_args: pytest.fail("must not read invalid requests"))
    with pytest.raises(AgentError) as failure:
        recipe_details({"profiles": [], "roots": []}, query(**overrides))
    assert failure.value.code == "INVALID_RECIPE_DETAIL"


def test_expired_and_unknown_profile_requests_do_not_access_files(monkeypatch):
    monkeypatch.setattr(details_module, "allowed_path", lambda *_args: pytest.fail("must not read invalid requests"))
    with pytest.raises(AgentError) as failure:
        recipe_details({"profiles": [], "roots": []}, query(expires_at="2000-01-01T00:00:00Z"))
    assert failure.value.code == "RECIPE_QUERY_EXPIRED"
    with pytest.raises(AgentError) as failure:
        recipe_details({"profiles": [], "roots": []}, query())
    assert failure.value.code == "PROFILE_UNAVAILABLE"


def test_source_cannot_escape_checkout_through_a_symlink(tmp_path):
    config, path = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="pi0_test")]\n')
    outside = tmp_path / "private.py"
    outside.write_text('_CONFIGS = [TrainConfig(name="pi0_test", private="secret")]\n')
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "PATH_OUTSIDE_ROOTS" and "secret" not in str(failure.value)


def test_non_git_checkout_requires_a_real_code_version(tmp_path):
    root = tmp_path / "checkout"
    path = root / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text('_CONFIGS=[TrainConfig(name="pi0_test")]\n')
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi-native-1", "adapter": "openpi-sft", "code_dir": str(root)}]}
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "CODE_VERSION_UNAVAILABLE"


def test_source_edit_during_detail_capture_is_rejected(tmp_path, monkeypatch):
    config, path = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test", batch_size=2)]\n')
    original, reads = details_module._source, []
    def changed(*args):
        if reads:
            path.write_text('_CONFIGS=[TrainConfig(name="pi0_test", batch_size=3)]\n')
        reads.append(True)
        return original(*args)
    monkeypatch.setattr(details_module, "_source", changed)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_SOURCE_CHANGED"


def test_tracked_code_change_during_detail_capture_is_rejected(tmp_path, monkeypatch):
    config, _ = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test")]\n')
    original, pins = details_module._git_pin, []
    def changed(*args):
        pin = original(*args)
        if pins:
            pin["commit_sha"] = "a" * 40
        pins.append(True)
        return pin
    monkeypatch.setattr(details_module, "_git_pin", changed)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_SOURCE_CHANGED"


def test_detail_bounds_and_expression_truncation_are_explicit(tmp_path, monkeypatch):
    config, _ = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test", text="' + 'x' * 2200 + '", batch_size=2)]\n')
    result = recipe_details(config, query())
    assert len(by_path(result)["text"]["expression"]) == details_module.MAX_EXPRESSION_CHARS
    assert "DETAIL_EXPRESSION_TRUNCATED" in result["warnings"]
    monkeypatch.setattr(details_module, "MAX_FIELDS", 1)
    result = recipe_details(config, query())
    assert len(result["fields"]) == 1 and "DETAIL_FIELDS_TRUNCATED" in result["warnings"]
    monkeypatch.setattr(details_module, "MAX_AST_NODES", 1)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_SOURCE_TOO_LARGE"


def test_oversized_snippets_are_rejected_instead_of_returning_partial_source(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test", text="' + 'x' * 33000 + '")]\n')
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_DETAIL_UNSUPPORTED"


def test_git_diff_output_is_bounded(tmp_path, monkeypatch):
    config, path = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test")]\n')
    path.write_text(path.read_text() + '# ' + 'x' * 10000)
    monkeypatch.setattr(details_module, "MAX_GIT_DIFF_BYTES", 100)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "CODE_VERSION_UNAVAILABLE"


def test_git_time_is_bounded(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    monkeypatch.setattr(details_module.subprocess, "Popen", lambda *_args, **kwargs: real_popen([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs))
    monkeypatch.setattr(details_module, "GIT_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(AgentError) as failure:
        details_module._git(tmp_path, "rev-parse", "HEAD")
    assert failure.value.code == "CODE_VERSION_UNAVAILABLE"


def test_display_controls_are_escaped_and_not_executed(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS=[TrainConfig(name="pi0_test", text="ab\u202ecd")]\n')
    result = recipe_details(config, query())
    assert "\u202e" not in result["source_code"] and "\\u202e" in result["source_code"]
    assert "DETAIL_TEXT_SANITIZED" in result["warnings"]


def test_recipe_detail_command_is_persisted_across_agent_restarts(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    profile["adapter"] = "openpi-sft"
    path = Path(profile["code_dir"]) / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text('_CONFIGS=[TrainConfig(name="pi0_test", batch_size=2)]\n')
    agent = Agent(config)
    command = {"id": "recipe-detail", "type": "RECIPE_DETAIL", "payload": query(profile_id=profile["id"])}
    acknowledgement = agent.handle_command(command)
    assert acknowledgement["status"] == "APPLIED"
    assert by_path(acknowledgement["result"])["batch_size"]["value"] == 2
    restarted = Agent(config)
    monkeypatch.setattr("vla_agent.main.recipe_details", lambda *_args: pytest.fail("persisted details must not reread"))
    assert restarted.handle_command(command) == acknowledgement
    agent.runtime_monitor.close()
    restarted.runtime_monitor.close()
