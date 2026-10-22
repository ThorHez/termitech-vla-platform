import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from vla_adapters import build_launch
from vla_adapters.openpi_entry import configuration_snapshot, resolve_config
from vla_adapters.recipe_definitions import (apply_overrides, definition_digest, is_asset_managed_field, resolve_definition, validate_definition,
                                            validate_overrides, validate_overrides_against_fields)


def definition(code_dir="/code", overrides=None):
    value = {"schema_version": 1, "id": "revision-1", "recipe_id": "recipe-1", "name": "低学习率配方", "revision": 1,
             "base_recipe": "base_config", "node_id": "node-1", "profile_id": "profile-1",
             "source": {"path": "src/openpi/training/config.py", "sha256": "a" * 64, "line": 1, "end_line": 10,
                        "commit_sha": "b" * 40, "code_dir": code_dir, "changes_digest": "c" * 64, "dirty": False},
             "overrides": overrides or {"batch_size": 36, "num_workers": 4, "lr_schedule.peak_lr": 0.00002}}
    value["digest"] = definition_digest(value)
    return value


@pytest.mark.parametrize("overrides", [{"__class__": 1}, {"data.__class__": "/tmp"}, {"batch_size": True},
    {"num_workers": -1}, {"seed": 1.2}, {"ema_decay": 1}, {"lr_schedule.peak_lr": float("nan")},
    {"optimizer.clip_gradient_norm": "__import__('os')"}, {"keep_period": None}, {"batch_size": 10 ** 1000}])
def test_only_safe_paths_and_bounded_values_are_accepted(overrides):
    with pytest.raises(ValueError, match="RECIPE_OVERRIDES_INVALID"):
        validate_overrides(overrides)


def test_generic_edits_require_observed_literals_and_matching_types():
    fields = [{"path": "data.datasets[1].base_config.prompt_from_task", "kind": "literal", "value": True},
              {"path": "data.datasets[1].base_config.action_sequence_keys", "kind": "literal", "value": []},
              {"path": "model.freeze", "kind": "expression", "expression": "make_freeze()"},
              {"path": "data.state_pad_dim", "kind": "literal", "value": 128}]
    edits = {"data.datasets[1].base_config.prompt_from_task": False,
             "data.datasets[1].base_config.action_sequence_keys": ["actions"], "data.state_pad_dim": 64}
    assert validate_overrides_against_fields(edits, fields) == edits
    for invalid in ({"data.missing": 1}, {"model.freeze": True}, {"data.state_pad_dim": "64"},
                    {"data.datasets[1].base_config.prompt_from_task": "false"}):
        with pytest.raises(ValueError, match="RECIPE_FIELD_UNSUPPORTED"):
            validate_overrides_against_fields(invalid, fields)
    assert validate_overrides_against_fields({"num_train_steps": 100}, []) == {"num_train_steps": 100}


@pytest.mark.parametrize("path", ["data", "name", "checkpoint_base_dir", "data.x.__dict__", "data.x[64]", "model..x", "data.x[01]", "data.x()"])
def test_generic_paths_cannot_replace_runtime_controls_or_expressions(path):
    with pytest.raises(ValueError, match="RECIPE_OVERRIDES_INVALID"):
        validate_overrides({path: 1})


def test_literal_edits_are_bounded_and_parent_child_overrides_are_rejected():
    for edits in ({"data.values": [0] * 65}, {"data.label": "x" * 1025},
                  {"data.values": [1], "data.values[0]": 2}, {"data.value": float("inf")}):
        with pytest.raises(ValueError, match="RECIPE_OVERRIDES_INVALID"):
            validate_overrides(edits)


@pytest.mark.parametrize("path,value", [
    ("data.repo_id", "/hidden"), ("data.datasets[1].repo_id", "/hidden"),
    ("data.datasets[1].assets.asset_id", "other"), ("data.datasets[1].assets.assets_dir", "/hidden"),
    ("data.base_config.norm_stats", {}), ("data.norm_stats_path", "/hidden"),
    ("data.datasets[0].base_config.rlds_data_dir", "/hidden"), ("data.base_config.filter_dict_path", "/hidden"),
    ("data.weights", [0.5, 2]), ("data.weights[0]", 2), ("data.datasets", []),
    ("data.datasets[1]", {"repo_id": "/hidden"}), ("model.weights", "/hidden"),
    ("model.pretrained_path", "/hidden"), ("model.weight_loader.params_path", "/hidden"),
])
def test_asset_managed_fields_are_never_recipe_overrides_even_when_observed(path, value):
    assert is_asset_managed_field(path)
    for validate in (lambda: validate_overrides({path: value}),
                     lambda: validate_overrides_against_fields({path: value}, [{"path": path, "kind": "literal", "value": value}])):
        with pytest.raises(ValueError, match="RECIPE_ASSET_FIELDS_MANAGED"):
            validate()


def test_asset_keys_cannot_be_smuggled_inside_a_parent_literal():
    with pytest.raises(ValueError, match="RECIPE_ASSET_FIELDS_MANAGED"):
        validate_overrides({"data.base_config": {"prompt_from_task": True, "repo_id": "/hidden"}})


def test_legacy_asset_overrides_are_readable_but_not_executable():
    saved = definition(overrides={"data.repo_id": "/legacy", "batch_size": 16})
    assert validate_definition(saved) == saved
    profile = {"id": "profile-1", "adapter": "openpi-sft", "metadata": {"validated_recipe_revisions": [saved["digest"]]}}
    with pytest.raises(ValueError, match="RECIPE_ASSET_FIELDS_MANAGED"):
        resolve_definition(profile, saved)
    with pytest.raises(ValueError, match="RECIPE_ASSET_FIELDS_MANAGED"):
        apply_overrides(Config(), saved["overrides"])


def test_nested_sequence_edits_preserve_base_dataclasses_and_container_types():
    @dataclasses.dataclass(frozen=True)
    class Child:
        prompt_from_task: bool = True
    @dataclasses.dataclass(frozen=True)
    class Multi:
        datasets: tuple = (Child(), Child())
        state_pad_dim: int = 128
        weights: list = dataclasses.field(default_factory=lambda: [1, 2])
    base = Config(data=Multi())
    edited = apply_overrides(base, {"data.datasets[1].prompt_from_task": False, "data.state_pad_dim": 64})
    assert edited.data.datasets[1].prompt_from_task is False and isinstance(edited.data.datasets, tuple)
    assert edited.data.state_pad_dim == 64 and edited.data.weights == [1, 2]
    assert base.data.datasets[1].prompt_from_task is True and base.data.weights == [1, 2]


def test_revision_content_is_immutable_and_old_recipe_approval_does_not_authorize_clone():
    saved = definition()
    profile = {"id": "profile-1", "adapter": "openpi-sft", "recipe": "base_config",
               "metadata": {"validated": True, "validated_recipes": ["base_config"]}}
    with pytest.raises(ValueError, match="PROFILE_NOT_VALIDATED"):
        resolve_definition(profile, saved)
    profile["metadata"]["validated_recipe_revisions"] = [saved["digest"]]
    assert resolve_definition(profile, saved) == "base_config"
    changed = copy.deepcopy(saved)
    changed["overrides"]["batch_size"] = 72
    with pytest.raises(ValueError, match="RECIPE_DEFINITION_INVALID"):
        resolve_definition(profile, changed)
    changed["digest"] = definition_digest(changed)
    with pytest.raises(ValueError, match="PROFILE_NOT_VALIDATED"):
        resolve_definition(profile, changed)
    profile["id"] = "another-profile"
    with pytest.raises(ValueError, match="RECIPE_DEFINITION_MISMATCH"):
        resolve_definition(profile, saved)


@pytest.mark.parametrize("field,value", [("source", {}), ("revision", True), ("base_recipe", "../config"), ("digest", "forged")])
def test_invalid_revision_identity_is_rejected(field, value):
    saved = definition()
    saved[field] = value
    with pytest.raises(ValueError, match="RECIPE_DEFINITION_INVALID"):
        validate_definition(saved)


def test_custom_name_is_not_passed_as_code_config_and_task_parameters_take_precedence(tmp_path):
    saved = definition(str(tmp_path), {"batch_size": 36, "num_train_steps": 60000, "lr_schedule.peak_lr": 0.00002})
    profile = {"id": "profile-1", "adapter": "openpi-sft", "code_dir": str(tmp_path), "python": "/usr/bin/python3",
               "metadata": {"validated_recipe_revisions": [saved["digest"]]}}
    inputs = {role: {"path": str(tmp_path / role)} for role in ["dataset", "model", "norm"]}
    launch = build_launch(profile, "openpi-sft", inputs, {"batch_size": 8}, str(tmp_path / "output"), ["GPU-a"],
                          recipe="base_config", recipe_definition=saved)
    spec = json.loads(launch["argv"][-1])
    assert spec["recipe"] == "base_config"
    assert spec["recipe_definition"] == saved
    assert spec["parameters"] == {"batch_size": 8, "steps": 60000, "learning_rate": 0.00002}


@dataclasses.dataclass
class Schedule:
    peak_lr: float = 0.1


@dataclasses.dataclass
class Optimizer:
    clip_gradient_norm: float = 1


@dataclasses.dataclass
class Data:
    repo_id: str = "/old"
    assets: object = None


@dataclasses.dataclass
class Config:
    data: object = dataclasses.field(default_factory=Data)
    lr_schedule: object = dataclasses.field(default_factory=Schedule)
    optimizer: object = dataclasses.field(default_factory=Optimizer)
    num_workers: int = 8
    weight_loader: object = None
    num_train_steps: int = 60000
    batch_size: int = 72
    fsdp_devices: int = 8
    checkpoint_base_dir: str = "/old/checkpoints"
    exp_name: str = "old"
    resume: bool = True
    overwrite: bool = True
    wandb_enabled: bool = True


def test_numeric_nested_edits_preserve_base_objects_and_fail_unknown_factories():
    base = Config()
    edited = apply_overrides(base, {"num_workers": 2, "optimizer.clip_gradient_norm": 0.5})
    assert edited.num_workers == 2 and edited.optimizer.clip_gradient_norm == 0.5
    assert base.num_workers == 8 and base.optimizer.clip_gradient_norm == 1
    with pytest.raises(ValueError, match="RECIPE_FIELD_UNSUPPORTED"):
        apply_overrides(Config(optimizer=object()), {"optimizer.clip_gradient_norm": 0.5})


def test_wrapper_applies_saved_changes_then_explicit_assets_and_parameters(tmp_path):
    for name in ["dataset/meta", "norm", "model/params"]:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "dataset/meta/info.json").write_text("{}")
    (tmp_path / "norm/norm_stats.json").write_text("{}")
    saved = definition(str(tmp_path))
    spec = {"recipe": "base_config", "recipe_definition": saved,
            "inputs": {role: {"path": str(tmp_path / role)} for role in ["dataset", "model", "norm"]},
            "parameters": {"steps": 10, "batch_size": 4, "learning_rate": 0.00001}, "gpu_count": 2,
            "output_dir": str(tmp_path / "output")}
    result = resolve_config(Config(), spec, SimpleNamespace(AssetsConfig=lambda **kw: kw),
                            SimpleNamespace(CheckpointWeightLoader=lambda p: p))
    assert result.num_workers == 4 and result.lr_schedule.peak_lr == 0.00001
    assert result.batch_size == 4 and result.fsdp_devices == 2
    assert result.data.repo_id == str(tmp_path / "dataset")
    multi = Config(data=SimpleNamespace(datasets=[Data()], repo_id="multi"))
    with pytest.raises(ValueError, match="DATASET_BINDINGS_REQUIRED"):
        resolve_config(multi, spec, None, None)


def test_configuration_snapshot_does_not_call_arbitrary_object_repr():
    class CodeObject:
        def __repr__(self):
            raise AssertionError("repr is executable code")
    result = configuration_snapshot(Config(weight_loader=CodeObject()))
    assert result["configuration"]["batch_size"] == 72
    assert result["configuration"]["weight_loader"]["unresolved"] == "code_object"
