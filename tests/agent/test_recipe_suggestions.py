from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vla_agent import recipe_suggestions
from vla_agent.main import Agent
from vla_agent.recipe_suggestions import suggest_recipes
from vla_agent.safety import AgentError


def query(prefix="", **overrides):
    return {"profile_id": "openpi-native-1", "prefix": prefix, "limit": 50,
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(), **overrides}


def checkout(tmp_path, source, *, layout="src/openpi/training/config.py"):
    repo = tmp_path / "code"
    path = repo / layout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return {"roots": [str(tmp_path)], "profiles": [{"id": "openpi-native-1", "code_dir": str(repo),
            "adapter": "openpi-sft", "python": "/never/execute/me", "execution_mode": "native"}]}, path


@pytest.mark.parametrize("layout", recipe_suggestions.SOURCE_PATHS)
def test_only_registry_bound_literals_are_discovered_without_execution(tmp_path, monkeypatch, layout):
    config, _ = checkout(tmp_path, '''
import jax
raise RuntimeError("this must never execute")
UNRELATED = TrainConfig(name="not_a_recipe")
class Example:
    TRAINING = [TrainConfig(name="also_not_a_recipe")]
_CONFIGS = [
    TrainConfig(name="pi0_beta", expensive=initialize_gpu()),
    TrainConfig(name="pi0_alpha"),
    FakeConfig(name="fake"),
    *roboarena_config.get_roboarena_configs(),
]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
''', layout=layout)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("must not invoke the interpreter"))
    result = suggest_recipes(config, query("pi0_"))
    assert result == {"profile_id": "openpi-native-1", "prefix": "pi0_", "entries": [
        {"name": "pi0_alpha"}, {"name": "pi0_beta"}], "truncated": False,
        "source": layout, "warnings": ["DYNAMIC_REGISTRY_UNSUPPORTED"]}


def test_registry_lists_aliases_and_mutations_follow_source_order(tmp_path):
    config, _ = checkout(tmp_path, '''
EXTRA = [TrainConfig(name="extra")]
ALIAS = EXTRA
EXTRA.append(TrainConfig(name="appended"))
_CONFIGS: list[TrainConfig] = [TrainConfig(name="initial"), *ALIAS]
_CONFIGS += [TrainConfig(name="added")]
_CONFIGS.extend((TrainConfig(name="extended"),))
_CONFIGS.append(TrainConfig(name="final"))
_CONFIGS = _CONFIGS + [TrainConfig(name="summed")]
EXTRA = [TrainConfig(name="not_added")]
''')
    result = suggest_recipes(config, query())
    assert [entry["name"] for entry in result["entries"]] == ["added", "appended", "extended", "extra", "final", "initial", "summed"]
    assert result["warnings"] == [] and not result["truncated"]


def test_only_final_registry_assignment_is_used(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="old")]\n_CONFIGS = [TrainConfig(name="new")]\n')
    assert suggest_recipes(config, query())["entries"] == [{"name": "new"}]


def test_assignment_time_bindings_preserve_aliases_across_rebinding(tmp_path):
    config, _ = checkout(tmp_path, '''
base = [TrainConfig(name="pi0_before")]
alias = base
copied = [*base]
base.append(TrainConfig(name="pi0_appended"))
base = [TrainConfig(name="pi0_after")]
alias = alias
_CONFIGS = alias + copied
''')
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "pi0_appended"}, {"name": "pi0_before"}]
    assert result["warnings"] == []


def test_list_copies_do_not_pick_up_subsequent_alias_mutations(tmp_path):
    config, _ = checkout(tmp_path, '''
base = [TrainConfig(name="before")]
alias = base
copied = [*alias]
summed = alias + [TrainConfig(name="summed")]
alias.append(TrainConfig(name="later"))
_CONFIGS = copied + summed
''')
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "before"}, {"name": "summed"}]
    assert result["warnings"] == []


def test_local_imported_registry_lists_are_bounded_and_not_executed(tmp_path):
    config, path = checkout(tmp_path, '''
from openpi.training.extra import EXTRA as imported
import openpi.training.more as more
from .relative import RECIPES
_CONFIGS = imported + [*more.EXTRA, *RECIPES, TrainConfig(name="local")]
''')
    (path.parent / "extra.py").write_text('raise RuntimeError("do not import")\nEXTRA=[TrainConfig(name="extra")]\n')
    (path.parent / "more.py").write_text('EXTRA = [TrainConfig(name="more")]\n')
    (path.parent / "relative.py").write_text('RECIPES = [TrainConfig(name="relative")]\n')
    result = suggest_recipes(config, query())
    assert [entry["name"] for entry in result["entries"]] == ["extra", "local", "more", "relative"]
    assert result["warnings"] == []


def test_conditional_and_generated_recipes_are_not_fabricated(tmp_path):
    config, _ = checkout(tmp_path, '''
_CONFIGS = [TrainConfig(name="certain")]
if maybe():
    _CONFIGS.append(TrainConfig(name="conditional"))
for name in ["generated"]:
    _CONFIGS.append(TrainConfig(name=name))
''')
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "certain"}]
    assert result["warnings"] == ["DYNAMIC_REGISTRY_UNSUPPORTED"]


def test_invalid_literal_names_and_dynamic_names_have_safe_warnings(tmp_path):
    config, _ = checkout(tmp_path, '''
NAME = "dynamic"
_CONFIGS = [TrainConfig(name="valid-name_1"), TrainConfig(name="bad.name"),
            TrainConfig(name="bad\\nname"), TrainConfig(name=NAME), TrainConfig(name=123)]
''')
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "valid-name_1"}]
    assert result["warnings"] == ["DYNAMIC_REGISTRY_UNSUPPORTED", "INVALID_RECIPE_NAME"]


def test_empty_or_unsupported_registry_reports_static_limit(tmp_path):
    config, path = checkout(tmp_path, '_CONFIGS = []\n')
    assert suggest_recipes(config, query())["warnings"] == []
    path.write_text('_CONFIGS = build_configs()\n')
    result = suggest_recipes(config, query())
    assert result["entries"] == [] and result["warnings"] == ["DYNAMIC_REGISTRY_UNSUPPORTED"]


def test_prefix_case_sorting_and_result_limit(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS = [' + ','.join(f'TrainConfig(name="pi0_{index:02}")' for index in range(60)) + ']\n')
    result = suggest_recipes(config, query("pi0_", limit=1000))
    assert len(result["entries"]) == 50 and result["truncated"]
    assert suggest_recipes(config, query("PI0"))["entries"] == result["entries"]
    assert suggest_recipes(config, query("pi0_1"))["entries"] == [{"name": f"pi0_{index}"} for index in range(10, 20)]
    assert len(suggest_recipes(config, query(limit=0))["entries"]) == 1


def test_keyword_matches_middle_case_insensitively_and_prioritizes_prefix_before_limit(tmp_path):
    names = ["aaa_elec_meta", "pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta", "elec_b", "Elec_a", "unrelated"]
    config, _ = checkout(tmp_path, '_CONFIGS = [' + ','.join(f'TrainConfig(name="{name}")' for name in names) + ']\n')
    expected = ["Elec_a", "elec_b", "aaa_elec_meta", "pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta"]
    for keyword in ("elec", "ELEC", "EleC"):
        result = suggest_recipes(config, query(keyword))
        assert [entry["name"] for entry in result["entries"]] == expected
        assert result["prefix"] == keyword
        limited = suggest_recipes(config, query(keyword, limit=2))
        assert [entry["name"] for entry in limited["entries"]] == expected[:2]
        assert limited["truncated"] is True
    assert suggest_recipes(config, query("no_match"))["entries"] == []


@pytest.mark.parametrize("prefix", [None, 4, "a.b", "a b", "a/b", "a\n", "a\x00", "a" * 256, "汉字"])
def test_invalid_prefix_rejected_before_file_access(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(recipe_suggestions, "allowed_path", lambda *_args: pytest.fail("must not read invalid requests"))
    with pytest.raises(AgentError) as failure:
        suggest_recipes({"profiles": [], "roots": []}, query(prefix))
    assert failure.value.code == "INVALID_RECIPE_QUERY"


@pytest.mark.parametrize("limit", [True, False, 1.5, "2", None])
def test_invalid_limit_rejected(limit):
    with pytest.raises(AgentError) as failure:
        suggest_recipes({"profiles": [], "roots": []}, query(limit=limit))
    assert failure.value.code == "INVALID_RECIPE_QUERY"


@pytest.mark.parametrize("expiry", [None, "not-a-date", "2030-01-01T00:00:00", "2030-01-01T00:00:00+01:00"])
def test_invalid_expiry_rejected(expiry):
    with pytest.raises(AgentError) as failure:
        suggest_recipes({"profiles": [], "roots": []}, query(expires_at=expiry))
    assert failure.value.code == "INVALID_RECIPE_QUERY"


def test_expired_query_never_accesses_sources(monkeypatch):
    monkeypatch.setattr(recipe_suggestions, "allowed_path", lambda *_args: pytest.fail("must not resolve expired requests"))
    with pytest.raises(AgentError) as failure:
        suggest_recipes({"profiles": [], "roots": []}, query(expires_at="2000-01-01T00:00:00Z"))
    assert failure.value.code == "RECIPE_QUERY_EXPIRED"


def test_registered_profile_ids_preserve_existing_schema_compatibility(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="recipe")]\n')
    profile_id = "environment. with spaces " + "x" * 170
    config["profiles"][0]["id"] = profile_id
    assert suggest_recipes(config, query(profile_id=profile_id))["entries"] == [{"name": "recipe"}]
    for invalid in ("", "x" * 201, None, "unsafe\x00"):
        with pytest.raises(AgentError) as failure:
            suggest_recipes(config, query(profile_id=invalid))
        assert failure.value.code == "INVALID_RECIPE_QUERY"


def test_unknown_wrong_adapter_and_non_native_profiles_rejected(tmp_path):
    config, _ = checkout(tmp_path, '_CONFIGS = []\n')
    for overrides in ({"id": "different"}, {"adapter": "smoke"}, {"execution_mode": "docker"}):
        original = dict(config["profiles"][0])
        config["profiles"][0].update(overrides)
        with pytest.raises(AgentError) as failure:
            suggest_recipes(config, query())
        assert failure.value.code == "PROFILE_UNAVAILABLE"
        config["profiles"][0] = original


def test_config_symlinks_cannot_escape_checkout_even_inside_another_root(tmp_path):
    config, path = checkout(tmp_path, '_CONFIGS = []\n')
    outside = tmp_path / "private.py"
    outside.write_text('_CONFIGS = [TrainConfig(name="private")]\n')
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"


def test_import_symlink_cannot_escape_checkout(tmp_path):
    config, path = checkout(tmp_path, 'from openpi.training.extra import EXTRA\n_CONFIGS = EXTRA\n')
    outside = tmp_path / "private.py"
    outside.write_text('EXTRA = [TrainConfig(name="private")]\n')
    (path.parent / "extra.py").symlink_to(outside)
    result = suggest_recipes(config, query())
    assert result["entries"] == [] and result["warnings"] == ["DYNAMIC_REGISTRY_UNSUPPORTED"]


def test_missing_unreadable_non_regular_and_invalid_sources_are_safe(tmp_path, monkeypatch):
    config, path = checkout(tmp_path, '_CONFIGS = []\n')
    path.write_text('not valid python!!!')
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_INVALID"
    path.unlink()
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_UNAVAILABLE"
    path.mkdir()
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_UNAVAILABLE"
    path.rmdir()
    path.write_text('_CONFIGS = []\n')
    monkeypatch.setattr(recipe_suggestions.os, "open", lambda *_args: (_ for _ in ()).throw(PermissionError("private")))
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_UNAVAILABLE" and "private" not in str(failure.value)


def test_source_size_and_ast_node_limits(tmp_path, monkeypatch):
    config, path = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="small")]\n')
    monkeypatch.setattr(recipe_suggestions, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_TOO_LARGE"
    monkeypatch.setattr(recipe_suggestions, "MAX_SOURCE_BYTES", 1024)
    monkeypatch.setattr(recipe_suggestions, "MAX_AST_NODES", 2)
    with pytest.raises(AgentError) as failure:
        suggest_recipes(config, query())
    assert failure.value.code == "RECIPE_SOURCE_TOO_LARGE"


def test_import_file_and_time_budgets_report_truncation(tmp_path, monkeypatch):
    config, path = checkout(tmp_path, 'from openpi.training.extra import EXTRA\n_CONFIGS = [TrainConfig(name="local"), *EXTRA]\n')
    (path.parent / "extra.py").write_text('EXTRA = [TrainConfig(name="extra")]\n')
    monkeypatch.setattr(recipe_suggestions, "MAX_SOURCE_FILES", 1)
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "local"}]
    assert result["truncated"] and result["warnings"] == ["DISCOVERY_LIMIT_REACHED"]
    monkeypatch.setattr(recipe_suggestions, "SCAN_BUDGET_SECONDS", -1)
    result = suggest_recipes(config, query())
    assert result["truncated"] and result["warnings"] == ["DISCOVERY_LIMIT_REACHED"]


def test_cyclic_aliases_and_imports_do_not_recurse_forever(tmp_path):
    config, path = checkout(tmp_path, 'from openpi.training.extra import EXTRA\nA=B\nB=A\n_CONFIGS = [*A, *EXTRA, TrainConfig(name="local")]\n')
    (path.parent / "extra.py").write_text('from openpi.training.config import _CONFIGS\nEXTRA = _CONFIGS\n')
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "local"}]
    assert result["warnings"] == ["DYNAMIC_REGISTRY_UNSUPPORTED"]


def test_registry_alias_doubling_never_expands_without_bound(tmp_path, monkeypatch):
    config, _ = checkout(tmp_path, '_CONFIGS = [TrainConfig(name="one")]\nALIAS = _CONFIGS\n' + '_CONFIGS += ALIAS\n' * 100)
    monkeypatch.setattr(recipe_suggestions, "MAX_COLLECTED_NAMES", 10)
    result = suggest_recipes(config, query())
    assert result["entries"] == [{"name": "one"}]
    assert result["truncated"] and result["warnings"] == ["DISCOVERY_LIMIT_REACHED"]


def test_recipe_command_persists_across_restarts(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    profile["adapter"] = "openpi-sft"
    path = Path(profile["code_dir"]) / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text('_CONFIGS = [TrainConfig(name="test_recipe")]\n')
    agent = Agent(config)
    command = {"id": "recipe-query", "type": "RECIPE_SUGGEST", "payload": query(profile_id=profile["id"])}
    acknowledgement = agent.handle_command(command)
    assert acknowledgement["status"] == "APPLIED"
    assert acknowledgement["result"]["entries"] == [{"name": "test_recipe"}]
    rejected = {"id": "recipe-rejected", "type": "RECIPE_SUGGEST", "payload": query("invalid.name")}
    failure = agent.handle_command(rejected)
    assert failure["status"] == "REJECTED" and failure["result"]["error"]["code"] == "INVALID_RECIPE_QUERY"
    restarted = Agent(config)
    monkeypatch.setattr("vla_agent.main.suggest_recipes", lambda *_args: pytest.fail("persisted commands must not rescan"))
    assert restarted.handle_command(command) == acknowledgement
    assert restarted.handle_command(rejected) == failure
    agent.runtime_monitor.close()
    restarted.runtime_monitor.close()
