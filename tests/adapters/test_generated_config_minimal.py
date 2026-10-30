"""Recipe-focused review code keeps active choices without unrelated task knobs."""

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from vla_adapters import openpi_entry
from vla_adapters.generated_config import generated_training_config


@dataclasses.dataclass
class AssetsConfig:
    assets_dir: str
    asset_id: str = "."


@dataclasses.dataclass
class ComposableDataConfig:
    repo_id: str
    assets: AssetsConfig
    composition: dict
    acp_dropout: float = 0.1
    training_stage: str = "sft"


@dataclasses.dataclass
class Pi0Config:
    pi05: bool = False
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"


@dataclasses.dataclass
class Pi0ValueConfig(Pi0Config):
    pass


@dataclasses.dataclass
class ShellgameCupEvalConfig:
    enabled: bool = False
    raw_dataset_root: str = ""
    interval: int = 1000


@dataclasses.dataclass
class ShellgameMemoryClassifierConfig:
    enabled: bool = False
    episodes_metadata_path: str = ""
    loss_weight: float = 1.0


@dataclasses.dataclass
class UOTConfig:
    enabled: bool = False
    alignment_weight: float = 0.1


@dataclasses.dataclass
class Schedule:
    peak_lr: float = 2.5e-5
    warmup_steps: int = 1000


@dataclasses.dataclass
class TrainConfig:
    name: str
    model: object
    data: object
    batch_size: int = 32
    num_train_steps: int = 30000
    seed: int = 42
    lr_schedule: Schedule = dataclasses.field(default_factory=Schedule)
    checkpoint_base_dir: str = "./checkpoints"
    pytorch_weight_path: str | None = None
    pytorch_training_precision: str = "bfloat16"
    c_fail_coef: float = 1.0
    value_clip_min: float = -1.0
    value_clip_max: float = 0.0
    episode_metadata_path: str | None = None
    task_max_lengths: dict | None = None
    val_ratio: float = 0.0
    eval_interval: int = 1000
    eval_batches: int = 10
    eval_batch_size: int | None = None
    shellgame_cup_eval: ShellgameCupEvalConfig = dataclasses.field(default_factory=ShellgameCupEvalConfig)
    shellgame_memory_classifier: ShellgameMemoryClassifierConfig = dataclasses.field(
        default_factory=ShellgameMemoryClassifierConfig)
    uot: UOTConfig | None = None
    custom_runtime_flag: str = "default"


def forbidden_default_factory():
    raise AssertionError("Review generation must not call arbitrary recipe default factories")


@dataclasses.dataclass
class CustomFactoryConfig(TrainConfig):
    custom_extension: object = dataclasses.field(default_factory=forbidden_default_factory)


class OpaqueDefaultFactory:
    def __call__(self):
        raise AssertionError("Review generation must not execute opaque default factories")

    def __eq__(self, other):
        raise AssertionError("Review generation must not compare opaque default factories")


@dataclasses.dataclass
class OpaqueFactoryConfig(TrainConfig):
    custom_extension: object = dataclasses.field(default_factory=OpaqueDefaultFactory())


# Match the identities of the known OpenPI types without importing JAX/OpenPI.
# Other recipe classes remain in this test module to exercise custom types.
TrainConfig.__module__ = "openpi.training.config"
ComposableDataConfig.__module__ = "openpi.training.config"


@pytest.fixture
def focused_config(tmp_path, monkeypatch):
    root_module = ModuleType("openpi")
    root_module.__path__ = []
    training_module = ModuleType("openpi.training")
    training_module.__path__ = []
    config_module = ModuleType("openpi.training.config")
    root_module.training = training_module
    training_module.config = config_module
    config_module.TrainConfig = TrainConfig
    config_module.ComposableDataConfig = ComposableDataConfig
    for name, module in [("openpi", root_module), ("openpi.training", training_module),
                         ("openpi.training.config", config_module)]:
        monkeypatch.setitem(sys.modules, name, module)
    dataset = tmp_path / "selected-dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({"fps": 20}))
    composition = {"schema_version": 1, "action_key": "action",
                   "state_fields": [{"key": "observation.state", "width": 14}],
                   "action_layout": [{"width": 14}]}
    data = ComposableDataConfig(str(dataset), AssetsConfig(str(dataset / "norm_raw14")), composition)
    model = Pi0Config(pi05=True, action_dim=32, action_horizon=16)
    config = TrainConfig("auto_pi05_full_h16_d32_sft", model, data, batch_size=8,
                         num_train_steps=10000, lr_schedule=Schedule(peak_lr=5e-5),
                         checkpoint_base_dir=str(tmp_path / "outputs/checkpoints"))
    spec = {
        "code_dir": str(tmp_path / "registered-openpi"), "recipe": config.name,
        "entrypoint": "scripts/train.py", "template_id": "openpi-sft",
        "auto_config": {"model_type": "pi05", "finetune_mode": "full", "action_dim": 32,
                        "action_horizon": 16},
        "inputs": {"dataset_0": {"id": "selected-dataset", "path": str(dataset),
                                  "dataset_profile": {"fps": 20}},
                   "norm_0": {"id": "selected-norm", "path": str(dataset / "norm_raw14/norm_stats.json")},
                   "model": {"id": "selected-model", "path": str(tmp_path / "selected-model")}},
        "dataset_bindings": [{"dataset_asset_id": "selected-dataset", "norm_asset_id": "selected-norm",
                              "weight": 1, "processor": {"name": "ComposableDataConfig",
                                  "parameters": {"composition": copy.deepcopy(composition)}}}],
        "parameters": {"steps": 10000, "batch_size": 8, "learning_rate": 5e-5},
        "gpu_count": 1, "allowed_roots": [str(tmp_path)], "output_dir": str(tmp_path / "outputs"),
    }
    return config, spec


def exported(config, spec, monkeypatch):
    generated = generated_training_config(config, spec)
    calls = []
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda request: (calls.append(request) or config, None))
    namespace = {}
    exec(compile(generated["source"], "minimal_training_config.py", "exec"), namespace)
    return generated, namespace, calls


def native_keywords(source, class_name):
    return [keyword.arg for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == class_name for keyword in node.keywords]


def test_sft_export_contains_no_inactive_stage_or_task_fields_anywhere(focused_config):
    config, spec = focused_config
    source = generated_training_config(config, spec)["source"]
    for field in ["shellgame_cup_eval", "shellgame_memory_classifier", "ShellgameCupEvalConfig",
                  "ShellgameMemoryClassifierConfig", "UOTConfig", "acp_dropout", "c_fail_coef",
                  "value_clip_min", "value_clip_max", "episode_metadata_path", "task_max_lengths",
                  "eval_interval", "eval_batches", "eval_batch_size", "pytorch_training_precision"]:
        assert field not in source, field
    assert "uot" not in native_keywords(source, "TrainConfig")


def test_sft_review_and_native_builder_preserve_active_model_data_and_training_choices(focused_config, monkeypatch):
    config, spec = focused_config
    original_config, original_spec = copy.deepcopy(config), copy.deepcopy(spec)
    generated, namespace, calls = exported(config, spec, monkeypatch)
    target = str(Path(spec["output_dir"]).with_name("new-attempt"))
    native = namespace["native_configuration"](target, config)
    assert native.model == config.model
    assert native.data == config.data
    assert native.batch_size == 8 and native.num_train_steps == 10000
    assert native.lr_schedule == config.lr_schedule
    assert native.checkpoint_base_dir == target + "/checkpoints"
    assert namespace["build_config"](target) is config
    assert calls == [{**spec, "output_dir": target}]
    assert config == original_config and spec == original_spec
    assert spec["output_dir"] not in generated["source"]


def test_unknown_nondefault_recipe_fields_remain_reviewable_and_reconstructable(focused_config, monkeypatch):
    config, spec = focused_config
    config.custom_runtime_flag = "custom-recipe-choice"
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "custom_runtime_flag" in generated["source"]
    assert "custom-recipe-choice" in generated["source"]
    assert namespace["native_configuration"](spec["output_dir"], config).custom_runtime_flag == "custom-recipe-choice"


def test_safe_default_values_are_not_repeated_in_native_constructors(focused_config):
    config, spec = focused_config
    source = generated_training_config(config, spec)["source"]
    assert "custom_runtime_flag" not in native_keywords(source, "TrainConfig")
    assert "paligemma_variant" not in native_keywords(source, "Pi0Config")
    assert "warmup_steps" not in native_keywords(source, "Schedule")


def test_default_factory_is_never_executed_to_decide_whether_a_field_is_minimal(focused_config, monkeypatch):
    base, spec = focused_config
    config = CustomFactoryConfig(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
                                 custom_extension={"explicit_custom_choice": True})
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "custom_extension" in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert native.custom_extension == {"explicit_custom_choice": True}


def test_opaque_default_factory_equality_is_never_invoked_by_generation(focused_config, monkeypatch):
    base, spec = focused_config
    config = OpaqueFactoryConfig(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
                                 custom_extension={"explicit_custom_choice": "opaque-factory"})
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "custom_extension" in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert native.custom_extension == {"explicit_custom_choice": "opaque-factory"}


@pytest.mark.parametrize("dropout", [0.1, 0.7])
def test_same_named_custom_composable_processor_keeps_its_dropout_in_sft(focused_config, monkeypatch, dropout):
    config, spec = focused_config
    custom_module = ModuleType("custom_recipe_config")
    custom_class = dataclasses.make_dataclass("ComposableDataConfig", [
        ("repo_id", str), ("acp_dropout", float, 0.1), ("training_stage", str, "sft"),
    ], namespace={"__module__": custom_module.__name__})
    custom_class.__module__ = custom_module.__name__
    custom_module.ComposableDataConfig = custom_class
    monkeypatch.setitem(sys.modules, custom_module.__name__, custom_module)
    config.data = custom_class(config.data.repo_id, acp_dropout=dropout)
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "acp_dropout" in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert type(native.data) is custom_class and native.data.acp_dropout == dropout


def test_required_known_root_fields_are_kept_despite_matching_inactive_stage_names(focused_config, monkeypatch):
    base, spec = focused_config
    required_class = dataclasses.make_dataclass("TrainConfig", [
        ("name", str), ("model", object), ("data", object), ("c_fail_coef", float),
        ("eval_interval", int), ("pytorch_training_precision", str), ("shellgame_cup_eval", object),
    ], namespace={"__module__": "openpi.training.config"})
    required_class.__module__ = "openpi.training.config"
    monkeypatch.setattr(sys.modules["openpi.training.config"], "TrainConfig", required_class)
    config = required_class(base.name, base.model, base.data, 2.0, 37, "float32", None)
    generated, namespace, _ = exported(config, spec, monkeypatch)
    arguments = native_keywords(generated["source"], "TrainConfig")
    assert {"c_fail_coef", "eval_interval", "pytorch_training_precision", "shellgame_cup_eval"} <= set(arguments)
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert type(native) is required_class
    assert native.c_fail_coef == 2.0 and native.eval_interval == 37
    assert native.pytorch_training_precision == "float32" and native.shellgame_cup_eval is None


def test_disabled_extensions_and_inactive_acp_remain_absent_even_with_stale_nondefault_values(focused_config):
    config, spec = focused_config
    config.shellgame_cup_eval.raw_dataset_root = "/stale-unused-shellgame-root"
    config.shellgame_memory_classifier.episodes_metadata_path = "/stale-unused-classifier.jsonl"
    config.data.acp_dropout = 0.9
    source = generated_training_config(config, spec)["source"]
    for value in ["shellgame_cup_eval", "shellgame_memory_classifier", "acp_dropout",
                  "/stale-unused-shellgame-root", "/stale-unused-classifier.jsonl"]:
        assert value not in source


def test_acp_retains_requested_dropout_while_omitting_unrelated_shellgame_and_value(focused_config, monkeypatch):
    config, spec = focused_config
    config.data.training_stage = "offline_rl"
    config.data.acp_dropout = 0.3
    spec.update(template_id="recap", training_stage="offline_rl", stage_config={"acp_dropout": 0.3})
    spec["recipe"] = config.name = "auto_pi05_full_h16_d32_offline_rl"
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "acp_dropout" in generated["source"]
    assert "shellgame_cup_eval" not in generated["source"]
    assert "value_clip_min" not in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert native.data.training_stage == "offline_rl" and native.data.acp_dropout == 0.3


def test_value_recipe_keeps_active_value_loss_and_validation_parameters(focused_config, monkeypatch):
    config, spec = focused_config
    config.model = Pi0ValueConfig(pi05=True, action_horizon=16)
    config.data.training_stage = "value"
    config.c_fail_coef, config.value_clip_min, config.value_clip_max = 2.0, -2.0, 1.0
    config.val_ratio, config.eval_interval, config.eval_batches, config.eval_batch_size = 0.2, 73, 4, 2
    config.episode_metadata_path = "/registered/episode-values.jsonl"
    config.task_max_lengths = {0: 900}
    spec.update(template_id="recap", training_stage="value", stage_config={"c_fail_coef": 2.0})
    spec["entrypoint"] = "scripts/train_value.py"
    spec["recipe"] = config.name = "auto_pi05_full_h16_d32_value"
    generated, namespace, _ = exported(config, spec, monkeypatch)
    for field in ["c_fail_coef", "value_clip_min", "value_clip_max", "episode_metadata_path",
                  "task_max_lengths", "val_ratio", "eval_interval", "eval_batches", "eval_batch_size"]:
        assert field in generated["source"], field
    assert "shellgame_cup_eval" not in generated["source"] and "acp_dropout" not in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert native.c_fail_coef == 2.0 and native.value_clip_min == -2.0 and native.value_clip_max == 1.0
    assert native.val_ratio == 0.2 and native.eval_interval == 73 and native.eval_batch_size == 2


@pytest.mark.parametrize("field,class_name", [
    ("shellgame_cup_eval", "ShellgameCupEvalConfig"),
    ("shellgame_memory_classifier", "ShellgameMemoryClassifierConfig"),
])
def test_enabled_mem_task_extension_is_exported_instead_of_filtered(focused_config, monkeypatch, field, class_name):
    config, spec = focused_config
    enabled = dataclasses.replace(getattr(config, field), enabled=True)
    setattr(config, field, enabled)
    config.val_ratio = 0.1
    spec["entrypoint"] = "scripts/mem/train_pi0_mem_compress.py"
    spec.pop("auto_config")
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert field in generated["source"] and class_name in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert getattr(native, field) == enabled


def test_uot_recipe_preserves_enabled_alignment_configuration(focused_config, monkeypatch):
    config, spec = focused_config
    config.uot = UOTConfig(enabled=True, alignment_weight=0.25)
    spec.pop("auto_config")
    spec["entrypoint"] = "scripts/train_uot.py"
    generated, namespace, _ = exported(config, spec, monkeypatch)
    assert "UOTConfig" in generated["source"]
    native = namespace["native_configuration"](spec["output_dir"], config)
    assert native.uot == config.uot


def test_minimal_export_digest_is_stable_for_attempt_paths_and_tracks_active_changes(focused_config):
    config, spec = focused_config
    generated = generated_training_config(config, spec)
    assert generated["sha256"] == hashlib.sha256(generated["source"].encode()).hexdigest()
    moved = {**spec, "output_dir": spec["output_dir"] + "-another-attempt"}
    moved_config = dataclasses.replace(config, checkpoint_base_dir=moved["output_dir"] + "/checkpoints")
    assert generated_training_config(moved_config, moved)["sha256"] == generated["sha256"]
    changed = dataclasses.replace(config, lr_schedule=Schedule(peak_lr=8e-5))
    assert generated_training_config(changed, spec)["sha256"] != generated["sha256"]


def test_changed_declared_and_effective_defaults_change_digest_even_when_native_code_is_identical(focused_config, monkeypatch):
    base, spec = focused_config
    module = ModuleType("changing_recipe_defaults")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    variants = []
    for beta2 in [0.999, 0.99]:
        optimizer_class = dataclasses.make_dataclass("Optimizer", [("beta2", float, beta2)])
        optimizer_class.__module__ = module.__name__
        module.Optimizer = optimizer_class
        config_class = dataclasses.make_dataclass("OptimizerTrainConfig", [
            ("optimizer", object, dataclasses.field(default_factory=optimizer_class)),
        ], bases=(TrainConfig,))
        config_class.__module__ = module.__name__
        module.OptimizerTrainConfig = config_class
        config = config_class(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
                              optimizer=optimizer_class())
        generated, namespace, _ = exported(config, spec, monkeypatch)
        assert "beta2" not in native_keywords(generated["source"], "Optimizer")
        native = namespace["native_configuration"](spec["output_dir"], config)
        assert native.optimizer.beta2 == beta2
        tree = ast.parse(generated["source"])
        native_ast = ast.dump(next(node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name == "native_configuration"),
                              include_attributes=False)
        variants.append((generated, namespace, native_ast))
    first, second = variants
    assert first[2] == second[2], "Both optimizer parameters are omitted as declared defaults"
    assert first[1]["RESOLVED_CONFIGURATION_SHA256"] != second[1]["RESOLVED_CONFIGURATION_SHA256"]
    assert first[0]["sha256"] != second[0]["sha256"]
