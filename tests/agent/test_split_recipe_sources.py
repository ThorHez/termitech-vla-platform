"""Split recipe discovery, actual-source details and source-only writeback."""

import hashlib
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vla_adapters.recipe_definitions import definition_digest
from vla_agent import recipe_details as details_module
from vla_agent import recipe_writeback as writer
from vla_agent.recipe_details import recipe_details
from vla_agent.recipe_suggestions import suggest_recipes
from vla_agent.safety import AgentError
from vla_agent.state import State


FAMILIES = ("sft", "recap", "mem", "rlt", "uot")
PRIMARY = '''# A static read must never execute this module.
raise RuntimeError("never import training or initialize a GPU")
import dataclasses
class TrainConfig:
    num_train_steps: int = 30000
    seed: int = 42
    optimizer: object = dataclasses.field(default_factory=_optimizer.AdamW)
    lr_schedule: object = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
{imports}
_CONFIGS = [{members}]
'''
PART = '''from openpi.training.config import TrainConfig, _optimizer
raise RuntimeError("never execute a recipe part")
_CONFIGS = [TrainConfig(name="{name}", batch_size=8,
                        data=UmiData(image_horizon=1))]
'''


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def query(**changes):
    return {"profile_id": "default", "recipe": "sft_base", "prefix": "", "limit": 50,
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), **changes}


@pytest.fixture(params=("src/", ""))
def split_checkout(tmp_path, request):
    root = tmp_path / "checkout"
    primary = root / (request.param + "openpi/training/config.py")
    primary.parent.mkdir(parents=True)
    parts = primary.parent / "recipes";parts.mkdir()
    primary.write_text(PRIMARY.format(
        imports="\n".join(f"from openpi.training.recipes.{family} import _CONFIGS as _{family.upper()}_CONFIGS" for family in FAMILIES),
        members=", ".join(f"*_{family.upper()}_CONFIGS" for family in FAMILIES)))
    for family in FAMILIES:
        (parts / (family + ".py")).write_text(PART.format(name=family + "_base"))
    (root / "unrelated.txt").write_text("original\n")
    for args in (("init", "-b", "test"), ("config", "user.name", "Recipe Tests"),
                 ("config", "user.email", "recipe@example.invalid"), ("add", "."), ("commit", "-m", "Fixture")):
        git(root, *args)
    config = {"roots": [str(tmp_path)], "state_dir": str(tmp_path / "state"), "profiles": [
        {"id": "default", "adapter": "openpi-sft", "execution_mode": "native", "code_dir": str(root)}]}
    return config, root, primary, parts, State(config["state_dir"])


def request_for(config, **changes):
    detail = recipe_details(config, query())
    definition = {"schema_version": 1, "id": "revision-1", "recipe_id": "recipe-1", "name": "Saved recipe", "revision": 1,
                  "base_recipe": "sft_base", "node_id": "node-1", "profile_id": "default", "source": detail["source"],
                  "overrides": {"num_train_steps": 100, "optimizer.clip_gradient_norm": 2.0, "lr_schedule.peak_lr": 2e-5}}
    definition["digest"] = definition_digest(definition)
    return {"operation_id": "split-write-1", "profile_id": "default", "definition": definition,
            "config_name": "new_sft", "expires_at": query()["expires_at"], **changes}


def test_all_split_families_are_discovered_without_importing(split_checkout):
    config, *_ = split_checkout
    result = suggest_recipes(config, query())
    assert {item["name"] for item in result["entries"]} == {family + "_base" for family in FAMILIES}
    assert result["warnings"] == [] and result["truncated"] is False


@pytest.mark.parametrize("family", FAMILIES)
def test_split_details_use_actual_part_and_primary_defaults(split_checkout, family):
    config, root, primary, parts, _ = split_checkout
    result = recipe_details(config, query(recipe=family + "_base"))
    source = parts / (family + ".py")
    assert result["source"]["path"] == source.relative_to(root).as_posix()
    assert result["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    fields = {item["path"]: item for item in result["fields"]}
    assert fields["batch_size"]["value"] == 8 and fields["batch_size"]["origin"] == "recipe"
    assert fields["num_train_steps"]["value"] == 30000 and fields["num_train_steps"]["origin"] == "default"
    assert fields["seed"]["value"] == 42
    assert fields["optimizer"]["expression"] == "dataclasses.field(default_factory=_optimizer.AdamW)"
    assert "never execute" not in result["source_code"] and "class TrainConfig" not in result["source_code"]


def test_imported_but_unregistered_parts_cannot_supply_details(split_checkout):
    config, _, primary, _, _ = split_checkout
    primary.write_text(primary.read_text().replace("*_MEM_CONFIGS, ", ""))
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query(recipe="mem_base"))
    assert failure.value.code == "RECIPE_NOT_FOUND"


def test_duplicate_recipe_across_parts_is_ambiguous(split_checkout):
    config, _, _, parts, _ = split_checkout
    (parts / "recap.py").write_text(PART.format(name="sft_base"))
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_DETAIL_UNSUPPORTED"


@pytest.mark.parametrize("target", ("primary", "part"))
def test_changed_split_source_or_shared_defaults_is_rejected(split_checkout, target, monkeypatch):
    config, _, primary, parts, _ = split_checkout
    selected = primary if target == "primary" else parts / "sft.py"
    real, reads = details_module._source, 0
    def changing(path, *args):
        nonlocal reads
        if path == selected:
            reads += 1
            if reads == 2:selected.write_text(selected.read_text()+"\n# concurrent edit\n")
        return real(path, *args)
    monkeypatch.setattr(details_module, "_source", changing)
    with pytest.raises(AgentError) as failure:
        recipe_details(config, query())
    assert failure.value.code == "RECIPE_SOURCE_CHANGED"


def test_split_part_symlink_cannot_disclose_another_checkout(split_checkout, tmp_path):
    config, _, _, parts, _ = split_checkout
    private = tmp_path / "private.py";private.write_text('_CONFIGS=[TrainConfig(name="sft_base", secret="do-not-disclose")]')
    part=parts/'sft.py';part.unlink();part.symlink_to(private)
    with pytest.raises(AgentError) as failure:recipe_details(config,query())
    assert failure.value.code == "RECIPE_DETAIL_UNSUPPORTED" and "do-not-disclose" not in str(failure.value)


def test_split_writeback_commits_only_the_selected_part_and_inherited_defaults(split_checkout):
    config, root, primary, parts, state = split_checkout
    before = primary.read_bytes()
    unrelated = root / "unrelated.txt";unrelated.write_text("staged\n");git(root,"add","unrelated.txt");unrelated.write_text("unstaged\n")
    index_before=git(root,"diff","--cached","--binary")
    request=request_for(config)
    result=writer.write_recipe(config,request,state)
    relative=(parts/'sft.py').relative_to(root).as_posix()
    assert result["changed_paths"]==[relative] and primary.read_bytes()==before
    assert git(root,"diff-tree","--no-commit-id","--name-only","-r","HEAD").decode().splitlines()==[relative]
    assert git(root,"diff","--cached","--binary")==index_before
    assert git(root,"show","HEAD:unrelated.txt")==b"original\n"
    fields={item['path']:item for item in result['detail']['fields']}
    assert fields['optimizer.clip_gradient_norm']['value']==2.0
    assert fields['lr_schedule.peak_lr']['value']==2e-5
    assert fields['num_train_steps']['value']==100
    assert result['detail']['source']['path']==relative
    assert writer.write_recipe(config,request,state)==result
    assert git(root,'rev-list','--count','HEAD').strip()==b'2'


def test_new_name_conflict_in_other_family_prevents_any_commit(split_checkout):
    config, root, _, parts, state=split_checkout
    before=(parts/'sft.py').read_bytes()
    with pytest.raises(AgentError) as failure:writer.write_recipe(config,request_for(config,config_name='uot_base'),state)
    assert failure.value.code=='RECIPE_NAME_CONFLICT'
    assert (parts/'sft.py').read_bytes()==before and git(root,'rev-list','--count','HEAD').strip()==b'1'


def test_other_recipe_part_changes_before_publication_abort_only_this_write(split_checkout, monkeypatch):
    config, root, _, parts, state=split_checkout
    request=request_for(config);before=(parts/'sft.py').read_bytes()
    original=writer._idle;calls=0
    def mutate(*args):
        nonlocal calls
        calls+=1
        if calls==2:(parts/'recap.py').write_text((parts/'recap.py').read_text()+'\n# unrelated concurrent edit\n')
        return original(*args)
    monkeypatch.setattr(writer,'_idle',mutate)
    with pytest.raises(AgentError) as failure:writer.write_recipe(config,request,state)
    assert failure.value.code=='RECIPE_SOURCE_CHANGED'
    assert (parts/'sft.py').read_bytes()==before and git(root,'rev-list','--count','HEAD').strip()==b'1'
    assert (parts/'recap.py').read_text().endswith('# unrelated concurrent edit\n')
