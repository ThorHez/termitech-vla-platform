"""Per-run checkpoint/loading choices must not replace the selected data recipe."""
import ast
import copy
import dataclasses
import json
import sys

import pytest

from vla_adapters import build_launch, get_templates, validate_parameters
from vla_adapters.automatic_config import build_automatic_base
from vla_adapters.generated_config import generated_training_config
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.recipe_definitions import recipe_parameters
from vla_adapters.runtime_options import (RUNTIME_PARAMETER_RULES, runtime_parameter_overrides,
                                          validate_runtime_parameter_relations)
from test_automatic_config import AutomaticTrainConfig, automatic  # noqa: F401
from test_data_processors import processor_module  # noqa: F401
from test_generated_config_minimal import focused_config  # noqa: F401
from test_training_plan import Config, training  # noqa: F401


RUNTIME = {"save_interval": 2000, "keep_period": 20000, "num_workers": 8}


def test_only_openpi_templates_offer_optional_runtime_choices():
    templates = {row["id"]: row for row in get_templates()}
    assert not set(RUNTIME) & templates["smoke"]["parameters"].keys()
    for name in ("openpi-sft", "recap"):
        rules = templates[name]["parameters"]
        assert {key: rules[key]["default"] for key in RUNTIME} == {
            "save_interval": 1000, "keep_period": 5000, "num_workers": 2}
        assert all(rules[key]["optional"] and rules[key]["required"] is False for key in RUNTIME)


def test_old_requests_do_not_acquire_explicit_runtime_overrides():
    assert validate_parameters("openpi-sft", {}) == {"steps": 10000, "batch_size": 8, "learning_rate": 5e-5}
    assert runtime_parameter_overrides({"steps": 80_000}) == {}
    assert validate_parameters("recap", RUNTIME) == {
        "steps": 10000, "batch_size": 8, "learning_rate": 5e-5, **RUNTIME}
    assert validate_parameters("openpi-sft", {"num_workers": 0})["num_workers"] == 0
    with pytest.raises(ValueError, match="Unknown training parameters"):
        validate_parameters("smoke", RUNTIME)
    with pytest.raises(ValueError, match="Unknown training parameters"):
        validate_parameters("openpi-sft", {"num_workers_command": "anything"})


@pytest.mark.parametrize("name,bad", [
    ("save_interval", 0), ("save_interval", 1_000_001), ("save_interval", 2.5),
    ("keep_period", 0), ("keep_period", 10_000_001), ("keep_period", None),
    ("num_workers", -1), ("num_workers", 129), ("num_workers", True),
    ("num_workers", float("nan")), ("num_workers", "8"),
])
def test_runtime_parameters_are_bounded_and_native_resolver_rechecks_them(name, bad):
    with pytest.raises(ValueError):
        validate_parameters("openpi-sft", {name: bad})
    with pytest.raises(ValueError, match="TRAINING_PARAMETER_INVALID"):
        runtime_parameter_overrides({name: bad})


@dataclasses.dataclass
class RuntimeConfig(Config):
    save_interval: int = 6000
    keep_period: int = 24000
    num_workers: int = 12


def test_old_advanced_recipe_retains_runtime_values_until_run_explicitly_overrides(training):
    base, spec, module, loaders = training
    base = RuntimeConfig(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)})
    base = dataclasses.replace(base, save_interval=6000, keep_period=24000)
    original = copy.deepcopy(base)
    inherited = resolve_config(base, spec, module, loaders)
    assert (inherited.save_interval, inherited.keep_period, inherited.num_workers) == (6000, 24000, 12)
    spec = {**spec, "parameters": {**spec["parameters"], **RUNTIME}}
    overridden = resolve_config(base, spec, module, loaders)
    assert {key: getattr(overridden, key) for key in RUNTIME} == RUNTIME
    assert overridden.data == inherited.data
    assert base == original


def test_automatic_assembly_uses_same_selected_assets_with_new_runtime_parameters(automatic):
    spec, module, loaders = automatic
    module.TrainConfig = dataclasses.make_dataclass("TrainConfig", [
        ("save_interval", int, dataclasses.field(default=1000)),
        ("keep_period", int, dataclasses.field(default=5000)),
        ("num_workers", int, dataclasses.field(default=2)),
    ], bases=(AutomaticTrainConfig,))
    original = copy.deepcopy(spec)
    base = build_automatic_base(spec, module, loaders)
    inherited = resolve_config(base, spec, module, loaders)
    assert (inherited.save_interval, inherited.keep_period, inherited.num_workers) == (1000, 5000, 2)
    chosen = {**spec, "parameters": {**spec["parameters"], **RUNTIME, "steps": 80000, "batch_size": 128}}
    resolved = resolve_config(build_automatic_base(chosen, module, loaders), chosen, module, loaders)
    assert {key: getattr(resolved, key) for key in RUNTIME} == RUNTIME
    assert resolved.num_train_steps == 80000 and resolved.batch_size == 128
    assert resolved.data == inherited.data
    assert spec == original


def test_saved_recipe_numeric_runtime_parameters_are_available_as_defaults():
    assert recipe_parameters({"overrides": {**RUNTIME, "batch_size": 128, "seed": 42}}) == {
        "batch_size": 128, **RUNTIME}


def test_launch_freezes_requested_runtime_values_without_editing_assets(training):
    _, spec, _, _ = training
    profile = {"adapter": "openpi-sft", "code_dir": spec["code_dir"], "python": sys.executable}
    parameters = {"steps": 80000, "batch_size": 128, "learning_rate": 5e-5, **RUNTIME}
    launch = build_launch(profile, "openpi-sft", spec["inputs"], parameters, spec["output_dir"],
                          ["GPU-a", "GPU-b", "GPU-c", "GPU-d"], recipe=spec["recipe"],
                          dataset_bindings=spec["dataset_bindings"])
    frozen = json.loads(launch["argv"][-1])
    assert frozen["parameters"] == parameters
    assert frozen["inputs"] == spec["inputs"]
    assert frozen["dataset_bindings"] == spec["dataset_bindings"]


def test_explicit_defaults_stay_visible_in_minimal_config_and_confirmation_digest(focused_config, monkeypatch):
    config, spec = focused_config
    native_type = dataclasses.make_dataclass("TrainConfig", [
        ("save_interval", int, dataclasses.field(default=1000)),
        ("keep_period", int, dataclasses.field(default=5000)),
        ("num_workers", int, dataclasses.field(default=2)),
    ], bases=(type(config),), namespace={"__module__": "openpi.training.config"})
    monkeypatch.setattr(sys.modules["openpi.training.config"], "TrainConfig", native_type)
    values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    config = native_type(**values)
    implicit = generated_training_config(config, spec)
    defaults = {name: rule["default"] for name, rule in RUNTIME_PARAMETER_RULES.items() if name in RUNTIME}
    spec = {**spec, "parameters": {**spec["parameters"], **defaults}}
    explicit = generated_training_config(config, spec)
    keywords = {keyword.arg: keyword.value for node in ast.walk(ast.parse(explicit["source"]))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"
                for keyword in node.keywords}
    assert all(ast.literal_eval(keywords[name]) == value for name, value in defaults.items())
    assert implicit["sha256"] != explicit["sha256"]
    changed = generated_training_config(dataclasses.replace(config, num_workers=8), {
        **spec, "parameters": {**spec["parameters"], "num_workers": 8}})
    assert changed["sha256"] != explicit["sha256"]


@dataclasses.dataclass
class CosineDecaySchedule:
    warmup_steps: int = 1000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30000
    decay_lr: float = 2.5e-6


@dataclasses.dataclass
class AdamW:
    clip_gradient_norm: float = 1.0
    weight_decay: float = 1e-10


@dataclasses.dataclass
class LearningConfig(RuntimeConfig):
    lr_schedule: object = dataclasses.field(default_factory=CosineDecaySchedule)
    optimizer: object = dataclasses.field(default_factory=AdamW)
    ema_decay: float | None = 0.99
    log_interval: int = 100


def learning_base(training):
    original, spec, module, loaders = training
    values = {field.name: getattr(original, field.name) for field in dataclasses.fields(original)}
    values["lr_schedule"] = CosineDecaySchedule()
    return LearningConfig(**values), spec, module, loaders


def test_full_user_example_changes_training_choices_without_replacing_model_or_datasets(training):
    base, spec, module, loaders = learning_base(training)
    original = copy.deepcopy(base)
    chosen = {"steps": 60000, "batch_size": 72, "learning_rate": 3e-5,
              "warmup_steps": 2000, "decay_steps": 50000, "decay_lr": 1e-5,
              "clip_gradient_norm": 1.0, "ema_decay": 0.999, "num_workers": 8,
              "fsdp_devices": 8, "log_interval": 10, "keep_period": 30000}
    inherited = resolve_config(base, spec, module, loaders)
    spec = {**spec, "parameters": validate_parameters("openpi-sft", chosen)}
    result = resolve_config(base, spec, module, loaders)
    assert result.lr_schedule == CosineDecaySchedule(2000, 3e-5, 50000, 1e-5)
    assert result.optimizer == AdamW(1.0)
    assert (result.ema_decay, result.fsdp_devices, result.num_workers, result.log_interval) == (0.999, 8, 8, 10)
    assert result.num_train_steps == 60000 and result.batch_size == 72
    assert result.data == inherited.data and base == original


@pytest.mark.parametrize("parameters", [
    {"warmup_steps": 2000, "decay_steps": 2000},
    {"warmup_steps": 2000, "decay_steps": 1000},
    {"learning_rate": 3e-5, "decay_lr": 4e-5},
    {"learning_rate": 0, "decay_lr": 0},
    {"clip_gradient_norm": 0}, {"clip_gradient_norm": -1},
    {"ema_decay": 1}, {"ema_decay": -0.1},
    {"ema_decay": float("nan")}, {"ema_decay": True}, {"fsdp_devices": 3},
    {"fsdp_devices": 16}, {"fsdp_devices": 0}, {"log_interval": 0},
    {"decay_steps": 10 ** 1000}, {"clip_gradient_norm": 10 ** 1000},
])
def test_cross_field_and_numeric_errors_are_rejected_before_preflight(parameters):
    with pytest.raises(ValueError, match="TRAINING_PARAMETER_INVALID"):
        validate_runtime_parameter_relations(parameters, gpu_count=8)
    with pytest.raises(ValueError):
        normalized = validate_parameters("openpi-sft", parameters)
        validate_runtime_parameter_relations(normalized, gpu_count=8)


def test_boundary_schedule_ema_and_fsdp_options_are_valid(training):
    base, spec, module, loaders = learning_base(training)
    spec = {**spec, "parameters": {**spec["parameters"], "warmup_steps": 0, "decay_steps": 1,
                                   "decay_lr": 0, "ema_decay": 0, "fsdp_devices": 4,
                                   "clip_gradient_norm": 1e-12}}
    result = resolve_config(base, spec, module, loaders)
    assert result.lr_schedule.warmup_steps == 0 and result.lr_schedule.decay_lr == 0
    assert result.fsdp_devices == 4 and result.ema_decay == 0
    assert result.optimizer.clip_gradient_norm == 1e-12


def test_omitted_ema_preserves_lora_disable_and_fsdp_uses_requested_gpu_count(training):
    base, spec, module, loaders = learning_base(training)
    base = dataclasses.replace(base, ema_decay=None, fsdp_devices=1)
    result = resolve_config(base, spec, module, loaders)
    assert result.ema_decay is None and result.fsdp_devices == spec["gpu_count"]
    overridden = resolve_config(base, {**spec, "parameters": {**spec["parameters"], "ema_decay": 0.999}}, module, loaders)
    assert overridden.ema_decay == 0.999 and base.ema_decay is None
    for key in ("ema_decay", "fsdp_devices"):
        assert RUNTIME_PARAMETER_RULES[key]["default"] is None
        assert key not in validate_parameters("openpi-sft", {})


def test_effective_inherited_schedule_values_are_checked_when_one_field_is_changed(training):
    base, spec, module, loaders = learning_base(training)
    for changed in ({"warmup_steps": 40000}, {"decay_steps": 500},
                    {"warmup_steps": 1, "learning_rate": 1e-7}):
        with pytest.raises(ValueError, match="TRAINING_PARAMETER_INVALID"):
            resolve_config(base, {**spec, "parameters": {**spec["parameters"], **changed}}, module, loaders)


def test_schedule_and_optimizer_types_are_preserved_and_unsupported_fields_fail_explicitly(training):
    @dataclasses.dataclass
    class RsqrtDecaySchedule:
        warmup_steps: int = 1000
        peak_lr: float = 5e-5
        timescale: float = 10000

    @dataclasses.dataclass
    class SGD:
        momentum: float = 0.9

    base, spec, module, loaders = learning_base(training)
    alternate = dataclasses.replace(base, lr_schedule=RsqrtDecaySchedule(), optimizer=SGD())
    inherited = resolve_config(alternate, spec, module, loaders)
    assert type(inherited.lr_schedule) is RsqrtDecaySchedule and type(inherited.optimizer) is SGD
    warmed = resolve_config(alternate, {**spec, "parameters": {**spec["parameters"], "warmup_steps": 0}}, module, loaders)
    assert type(warmed.lr_schedule) is RsqrtDecaySchedule and warmed.lr_schedule.warmup_steps == 0
    for changed in ({"decay_steps": 50000}, {"decay_lr": 1e-6}, {"clip_gradient_norm": 1.0}):
        with pytest.raises(ValueError, match="TRAINING_PARAMETER_UNSUPPORTED"):
            resolve_config(alternate, {**spec, "parameters": {**spec["parameters"], **changed}}, module, loaders)


def test_saved_recipe_parameter_paths_include_nested_training_controls():
    assert recipe_parameters({"overrides": {"lr_schedule.warmup_steps": 2000,
        "lr_schedule.decay_steps": 50000, "lr_schedule.decay_lr": 1e-5,
        "optimizer.clip_gradient_norm": 1.0, "ema_decay": 0.999,
        "fsdp_devices": 8, "log_interval": 10}}) == {
        "warmup_steps": 2000, "decay_steps": 50000, "decay_lr": 1e-5,
        "clip_gradient_norm": 1.0, "ema_decay": 0.999, "fsdp_devices": 8, "log_interval": 10}


def test_explicit_nested_defaults_are_pinned_in_review_and_change_confirmation_digest(focused_config, monkeypatch):
    config, spec = focused_config
    native_type = dataclasses.make_dataclass("TrainConfig", [
        ("optimizer", object, dataclasses.field(default_factory=AdamW)),
        ("ema_decay", float, dataclasses.field(default=0.99)),
        ("fsdp_devices", int, dataclasses.field(default=1)),
        ("log_interval", int, dataclasses.field(default=100)),
    ], bases=(type(config),), namespace={"__module__": "openpi.training.config"})
    monkeypatch.setattr(sys.modules["openpi.training.config"], "TrainConfig", native_type)
    config = native_type(**{field.name: getattr(config, field.name) for field in dataclasses.fields(config)})
    config = dataclasses.replace(config, lr_schedule=CosineDecaySchedule(peak_lr=spec["parameters"]["learning_rate"]))
    implicit = generated_training_config(config, spec)
    pinned = {"warmup_steps": 1000, "decay_steps": 30000, "decay_lr": 2.5e-6,
              "clip_gradient_norm": 1.0, "ema_decay": 0.99, "fsdp_devices": 1, "log_interval": 100}
    explicit = generated_training_config(config, {**spec, "parameters": {**spec["parameters"], **pinned}})
    calls = {node.func.id: {keyword.arg: keyword.value for keyword in node.keywords}
             for node in ast.walk(ast.parse(explicit["source"]))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for key in ("warmup_steps", "decay_steps", "decay_lr"):
        assert ast.literal_eval(calls["CosineDecaySchedule"][key]) == pinned[key]
    assert ast.literal_eval(calls["AdamW"]["clip_gradient_norm"]) == 1.0
    assert all(ast.literal_eval(calls["TrainConfig"][key]) == pinned[key] for key in ("ema_decay", "fsdp_devices", "log_interval"))
    assert implicit["sha256"] != explicit["sha256"]
