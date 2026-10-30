import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import build_launch, resolve_recipe
from vla_adapters.automatic_config import automatic_data_layout, automatic_recipe_name, build_automatic_base, normalize_auto_config
from vla_adapters.generated_config import generated_training_config
from vla_adapters import openpi_entry
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.training_plan import fingerprint, inspect_training_config
from test_data_processors import processor_module  # noqa: F401
from test_training_plan import Config, training  # noqa: F401
from test_composed_processing_contract import profile as composed_profile


@dataclasses.dataclass
class Pi0Config:
    pi05: bool = False
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    image_keys: tuple | None = None
    depth_keys: tuple | None = None

    def get_freeze_filter(self):
        return {"filter": "lora", "model": self.paligemma_variant}


@dataclasses.dataclass
class Pi0ValueConfig(Pi0Config):
    pass


@dataclasses.dataclass
class AutomaticTrainConfig(Config):
    name: str = ""
    model: object = None
    optimizer: object = None
    freeze_filter: object = "full"
    ema_decay: float | None = 0.99
    val_ratio: float = 0.1
    eval_interval: int = 1000
    eval_batches: int = 10


@dataclasses.dataclass
class CheckpointWeightLoaderWithValueHead:
    params_path: str


@pytest.fixture
def automatic(training, processor_module):
    _, spec, module, loaders = copy.deepcopy(training)
    module.DataConfigFactory = processor_module.DataConfigFactory
    module.DataConfig = processor_module.DataConfig
    module.IndependentFactory = processor_module.IndependentFactory
    module.RequiredACPFactory = processor_module.RequiredACPFactory
    module.TrainConfig = AutomaticTrainConfig
    module.pi0_config = SimpleNamespace(Pi0Config=Pi0Config)
    module.pi0_value = SimpleNamespace(Pi0ValueConfig=Pi0ValueConfig)
    loaders.CheckpointWeightLoaderWithValueHead = CheckpointWeightLoaderWithValueHead
    spec["auto_config"] = normalize_auto_config({"model_type": "pi0", "action_horizon": 16})
    spec["recipe"] = automatic_recipe_name(spec["auto_config"])
    for row in spec["dataset_bindings"]:
        row.pop("recipe_index")
        row["processor"] = {"name": "IndependentFactory", "parameters": {"image_horizon": 2}}
    return spec, module, loaders


def test_auto_choices_are_explicit_stable_and_do_not_mutate_input():
    partial = {"model_type": "pi0"}
    clean = normalize_auto_config(partial)
    assert partial == {"model_type": "pi0"}
    assert clean == {"model_type": "pi0", "finetune_mode": "full", "action_horizon": 50, "action_dim": 32}
    assert normalize_auto_config(None) is None
    assert automatic_recipe_name(clean) == "auto_pi0_full_h50_d32_sft"
    assert automatic_recipe_name(clean, "value").endswith("_value")


@pytest.mark.parametrize("value", [[], "pi05", {"extra": 1}, {"action_horizon": True},
    {"action_horizon": 1001}, {"action_dim": 0}, {"action_dim": 4097}, {"model_type": []},
    {"model_type": "mem"}, {"finetune_mode": "head-only"}, {"finetune_mode": []}])
def test_auto_choices_reject_ambiguous_or_unsupported_values(value):
    with pytest.raises(ValueError, match="AUTO_CONFIG_INVALID"):
        normalize_auto_config(value)


def test_composed_layout_selects_declared_rgb_depth_and_retains_long_pi05_state():
    one = composed_profile()["processing"]
    two = copy.deepcopy(one)
    one["cameras"][0].update(slot="base_0_rgb", kind="rgb")
    two["cameras"][0].update(slot="left_wrist_0_rgb", kind="depth")
    two["state_history"] = 6
    rows = [{"processor": {"name": "ComposableDataConfig", "parameters": {"composition": value}}}
            for value in (one, two)]
    vision, padding = automatic_data_layout(normalize_auto_config({}), rows)
    assert vision == {"image_keys": ("base_0_rgb",), "depth_keys": ("left_wrist_0_rgb",)}
    assert padding == 42 and padding > 32
    two["cameras"][0]["slot"] = "base_0_rgb"
    with pytest.raises(ValueError, match="AUTO_CONFIG_VISION_CONFLICT"):
        automatic_data_layout(normalize_auto_config({}), rows)


def test_automatic_layout_rejects_pi0_state_truncation_and_unknown_pi05_mixture_shape():
    composition = composed_profile()["processing"]
    composition["state_history"] = 6
    rows = [{"processor": {"name": "ComposableDataConfig", "parameters": {"composition": composition}}}]
    with pytest.raises(ValueError, match="AUTO_CONFIG_DIMENSION_MISMATCH"):
        automatic_data_layout(normalize_auto_config({"model_type": "pi0"}), rows)
    with pytest.raises(ValueError, match="AUTO_CONFIG_STATE_UNSUPPORTED"):
        automatic_data_layout(normalize_auto_config({}), [
            {"processor": {"name": "IndependentFactory", "parameters": {}}},
            {"processor": {"name": "IndependentFactory", "parameters": {}}}])


def test_auto_builder_uses_real_model_and_independent_factories_with_per_dataset_norms(automatic):
    spec, module, loaders = automatic
    base = build_automatic_base(spec, module, loaders)
    config = resolve_config(base, spec, module, loaders)
    assert not config.model.pi05 and config.model.action_horizon == 16 and config.model.action_dim == 32
    assert config.data.use_merged_norm_stats is False
    assert config.data.state_pad_dim == 32 and config.data.weights == [1, 20, 50]
    assert [child.repo_id for child in config.data.datasets] == [spec["inputs"][f"dataset_{i}"]["path"] for i in range(3)]
    assert [child.assets.assets_dir for child in config.data.datasets] == [str(Path(spec["inputs"][f"norm_{i}"]["path"]).parent) for i in range(3)]
    assert config.batch_size == spec["parameters"]["batch_size"]
    assert config.num_train_steps == spec["parameters"]["steps"]
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["auto_config"] == spec["auto_config"]
    assert all(row["processor"]["name"] == "IndependentFactory" for row in plan["datasets"])
    assert "generated_config" in plan


def test_auto_single_and_lora_do_not_inherit_recipe_defaults(automatic):
    spec, module, loaders = automatic
    spec["dataset_bindings"] = spec["dataset_bindings"][:1]
    spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in {"model", "dataset_0", "norm_0"}}
    spec["auto_config"]["finetune_mode"] = "lora"
    spec["recipe"] = automatic_recipe_name(spec["auto_config"])
    config = resolve_config(build_automatic_base(spec, module, loaders), spec, module, loaders)
    assert not hasattr(config.data, "datasets")
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.freeze_filter["filter"] == "lora" and config.ema_decay is None


@pytest.mark.parametrize("count,entry", [(1, "train.py"), (2, "train_multi_dataset.py")])
def test_auto_dataset_count_freezes_matching_default_entrypoint(automatic, count, entry):
    spec, module, loaders = automatic
    spec["dataset_bindings"] = spec["dataset_bindings"][:count]
    roles = {"model", *(f"dataset_{i}" for i in range(count)), *(f"norm_{i}" for i in range(count))}
    spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in roles}
    config = resolve_config(build_automatic_base(spec, module, loaders), spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert len(plan["datasets"]) == count
    assert plan["entrypoint"] == str(Path(spec["code_dir"]) / "scripts" / entry)
    assert plan["requested_entrypoint"] is None


def test_auto_value_builds_correct_model_and_loader_and_rejects_lora(automatic):
    spec, module, loaders = automatic
    spec.update(template_id="recap", training_stage="value", stage_config={})
    spec["recipe"] = automatic_recipe_name(spec["auto_config"], "value")
    base = build_automatic_base(spec, module, loaders)
    assert isinstance(base.model, Pi0ValueConfig)
    assert isinstance(base.weight_loader, CheckpointWeightLoaderWithValueHead)
    spec["auto_config"]["finetune_mode"] = "lora"
    spec["recipe"] = automatic_recipe_name(spec["auto_config"], "value")
    with pytest.raises(ValueError, match="AUTO_CONFIG_UNSUPPORTED"):
        build_automatic_base(spec, module, loaders)


def test_auto_offline_stage_supplies_required_acp_constructor_parameter(automatic):
    spec, module, loaders = automatic
    spec.update(template_id="recap", training_stage="offline_rl", stage_config={"acp_dropout": 0.2})
    spec["recipe"] = automatic_recipe_name(spec["auto_config"], "offline_rl")
    for row in spec["dataset_bindings"]:
        row["processor"] = {"name": "RequiredACPFactory", "parameters": {}}
    config = resolve_config(build_automatic_base(spec, module, loaders), spec, module, loaders)
    assert all(factory.acp_dropout == 0.2 for factory in config.data.datasets)


def test_auto_builder_rejects_legacy_data_indices_and_named_recipe_conflict(automatic):
    spec, module, loaders = automatic
    spec["recipe"] = "legacy_recipe"
    with pytest.raises(ValueError, match="AUTO_CONFIG_CONFLICT"):
        build_automatic_base(spec, module, loaders)
    spec["recipe"] = automatic_recipe_name(spec["auto_config"])
    spec["dataset_bindings"][0].pop("processor")
    spec["dataset_bindings"][0]["recipe_index"] = 0
    with pytest.raises(ValueError, match="AUTO_CONFIG_PROCESSOR_REQUIRED"):
        build_automatic_base(spec, module, loaders)


def test_auto_launch_freezes_choices_and_internal_identity_without_profile_recipe(automatic):
    spec, _, _ = automatic
    profile = {"adapter": "openpi-sft", "code_dir": spec["code_dir"], "python": sys.executable}
    launch = build_launch(profile, "openpi-sft", spec["inputs"], spec["parameters"], spec["output_dir"], ["gpu-0"],
                          dataset_bindings=spec["dataset_bindings"], auto_config=spec["auto_config"])
    frozen = json.loads(launch["argv"][-1])
    assert frozen["auto_config"] == spec["auto_config"] and frozen["recipe"] == spec["recipe"]
    assert resolve_recipe(profile, auto_config={}) == "auto_pi05_full_h50_d32_sft"
    with pytest.raises(ValueError, match="AUTO_CONFIG_CONFLICT"):
        resolve_recipe(profile, "old", auto_config={})


def test_prepare_automatic_never_looks_up_named_recipe(automatic, monkeypatch):
    spec, module, loaders = automatic
    module.__file__ = str(Path(spec["code_dir"]) / "config.py")
    module.get_config = lambda name: (_ for _ in ()).throw(AssertionError("must not use a named code recipe"))
    root = ModuleType("openpi")
    package = ModuleType("openpi.training")
    root.training = package
    package.config = module
    package.weight_loaders = loaders
    for name, value in (("openpi", root), ("openpi.training", package),
                        ("openpi.training.config", module), ("openpi.training.weight_loaders", loaders)):
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(openpi_entry, "configure_cache", lambda spec: None)
    config, definition = openpi_entry.prepare_config(spec)
    assert definition is None and not config.model.pi05 and len(config.data.datasets) == 3


def test_generated_python_is_faithful_stable_across_attempt_paths_and_changes_with_effective_config(training, monkeypatch):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    generated = generated_training_config(config, spec)
    ast.parse(generated["source"])
    assert generated["sha256"] == hashlib.sha256(generated["source"].encode()).hexdigest()
    assert spec["output_dir"] not in generated["source"]
    assert "str(Path(output_dir) / 'checkpoints')" in generated["source"]
    assert "registered" in generated["source"] and "bounded review projection" in generated["source"]
    assert "def native_configuration" in generated["source"] and "return Config(" in generated["source"]
    assert spec["inputs"]["dataset_2"]["path"] in generated["source"]
    first = inspect_training_config(config, spec, checks=[])
    other = {**spec, "output_dir": spec["output_dir"] + "-actual-attempt", "expected_plan_digest": "0" * 64}
    changed_output_config = resolve_config(base, other, module, loaders)
    assert fingerprint(first) == fingerprint(inspect_training_config(changed_output_config, other, checks=[]))
    changed_optimizer_config = dataclasses.replace(config, ema_decay=0.5) if hasattr(config, "ema_decay") else dataclasses.replace(config, save_interval=73)
    assert generated_training_config(changed_optimizer_config, spec)["sha256"] != generated["sha256"]
    calls = []
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda frozen: (calls.append(frozen) or config, None))
    namespace = {}
    exec(compile(generated["source"], "training_config.py", "exec"), namespace)
    assert namespace["build_config"](other["output_dir"]) is config
    assert calls[0]["output_dir"] == other["output_dir"] and calls[0]["inputs"] == spec["inputs"]
    native = namespace["native_configuration"](other["output_dir"], config)
    assert native.checkpoint_base_dir == other["output_dir"] + "/checkpoints"
    assert native.data == config.data and native.lr_schedule == config.lr_schedule
