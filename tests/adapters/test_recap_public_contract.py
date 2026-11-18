import copy
import json

import pytest

from vla_adapters import build_launch, get_templates, validate_parameters
from vla_adapters.recap import (ACP_ENTRYPOINT, ACP_MULTI_ENTRYPOINT, RECAP_TORCH_ENTRY, STAGE_ANNOTATION_ENTRY,
                                VALUE_ANNOTATION_ENTRY, recap_stages, runtime_adapter_for, validate_training_stage)


def test_recap_exposes_distinct_train_and_annotation_stages_with_independent_configuration():
    template = next(template for template in get_templates() if template["id"] == "recap")
    assert template["status"] == "AVAILABLE" and template["runtime_adapter"] == "openpi-sft"
    value, value_annotation, offline, stage_progress, annotation = template["stages"]
    assert value["id"] == "value" and offline["id"] == "offline_rl"
    assert set(value["parameters"]) == {"val_ratio", "eval_interval", "eval_batches"}
    assert value_annotation["id"] == "value_annotation"
    assert value_annotation["default_entrypoint"] == VALUE_ANNOTATION_ENTRY
    assert {key: rule["default"] for key, rule in value_annotation["parameters"].items()} == {"n_step": 50, "positive_ratio": .3}
    assert set(offline["parameters"]) == {"acp_dropout"}
    assert stage_progress["id"] == "stage_progress"
    assert {key: rule["default"] for key, rule in stage_progress["parameters"].items()} == {
        "val_ratio": 0, "eval_interval": 1000, "eval_batches": 10}
    assert annotation["id"] == "stage_annotation"
    assert {key: row["default"] for key, row in annotation["parameters"].items()} == {
        "n_step": 50, "positive_ratio": .3, "stage_count": 2}
    assert "backend" not in template
    assert offline["backend"] == "jax"
    assert (offline["default_entrypoint"], offline["default_multi_entrypoint"]) == (ACP_ENTRYPOINT, ACP_MULTI_ENTRYPOINT)
    assert all(stage["backend"] == "pytorch" for stage in (value, value_annotation, stage_progress, annotation))
    assert all(stage["default_entrypoint"] == stage["default_multi_entrypoint"] == RECAP_TORCH_ENTRY
               for stage in (value, stage_progress))
    assert annotation["default_entrypoint"] == STAGE_ANNOTATION_ENTRY
    assert {"fsdp_devices", "ema_decay"}.isdisjoint(template["parameters"])
    assert runtime_adapter_for("recap") == "openpi-sft"
    value["parameters"].clear()
    assert recap_stages()[0]["parameters"]


@pytest.mark.parametrize(("stage", "config"), [
    ("value", {"eval_interval": 42, "val_ratio": .2}),
    ("value_annotation", {"n_step": 25, "positive_ratio": .4}),
    ("offline_rl", {"acp_dropout": .3}),
    ("stage_progress", {"val_ratio": .15, "eval_interval": 40, "eval_batches": 2}),
    ("stage_annotation", {"n_step": 25, "positive_ratio": .4, "stage_count": 3}),
])
def test_stage_settings_remain_separate_from_common_training_parameters(tmp_path, stage, config):
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": "/usr/bin/python3"}
    bindings = [{"dataset_asset_id": "dataset-id", "norm_asset_id": "norm-id", "recipe_index": 0, "weight": 1}]
    inputs = {"model": {"id": "model-id", "path": str(tmp_path / "model")},
              "dataset_0": {"id": "dataset-id", "path": str(tmp_path / "dataset")},
              "norm_0": {"id": "norm-id", "path": str(tmp_path / "norm")}}
    original = copy.deepcopy(inputs)
    launch = build_launch(profile, "recap", inputs, {"steps": 7}, str(tmp_path / "output"), ["gpu-0"],
                          recipe="selected_recipe", dataset_bindings=bindings,
                          training_stage=stage, stage_config=config)
    spec = json.loads(launch["argv"][-1])
    assert spec["template_id"] == "recap" and spec["training_stage"] == stage
    expected = ("jax", "cuda") if stage == "offline_rl" else ("pytorch", "cpu")
    assert (spec["recap_backend"], launch["env"]["JAX_PLATFORMS"]) == expected
    assert spec["stage_config"] == config and set(spec["parameters"]) == {"steps", "batch_size", "learning_rate"}
    assert inputs == original
    if config:
        with pytest.raises(ValueError, match="Unknown training parameters"):
            validate_parameters("recap", config)


def test_jax_acp_rejects_torch_entrypoint_except_for_already_frozen_run(tmp_path):
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": "/usr/bin/python3"}
    bindings = [{"dataset_asset_id": "dataset-a", "norm_asset_id": "norm-a", "recipe_index": 0, "weight": 1}]
    inputs = {role: {"id": identifier, "path": str(tmp_path / role)} for role, identifier in {
        "model": "model-a", "dataset_0": "dataset-a", "norm_0": "norm-a"}.items()}
    options = {"recipe": "policy", "dataset_bindings": bindings, "training_stage": "offline_rl", "entrypoint": RECAP_TORCH_ENTRY}
    with pytest.raises(ValueError, match="RECAP_STAGE_ENTRYPOINT_MISMATCH"):
        build_launch(profile, "recap", inputs, {}, str(tmp_path / "output"), ["gpu-0"], **options)
    frozen = build_launch(profile, "recap", inputs, {}, str(tmp_path / "output"), ["gpu-0"],
                          frozen_recap_backend="pytorch", **options)
    assert frozen["env"]["JAX_PLATFORMS"] == "cpu"
    assert json.loads(frozen["argv"][-1])["recap_backend"] == "pytorch"


@pytest.mark.parametrize("template,stage,config,code", [
    ("recap", None, None, "RECAP_STAGE_REQUIRED"),
    ("recap", "sft", {}, "RECAP_STAGE_INVALID"),
    ("openpi-sft", "value", {}, "TRAINING_STAGE_UNSUPPORTED"),
    ("smoke", None, {}, "TRAINING_STAGE_UNSUPPORTED"),
    ("recap", "value", {"acp_dropout": 0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value_annotation", {"stage_count": 2}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"val_ratio": 0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"eval_interval": 1.5}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"eval_batches": True}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"val_ratio": float("nan")}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"val_ratio": 10 ** 400}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"acp_dropout": -0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"acp_dropout": 1.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_annotation", {"positive_ratio": 0}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_annotation", {"stage_count": 2.5}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_progress", {"n_step": 50}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_progress", {"acp_dropout": .1}, "RECAP_STAGE_CONFIG_INVALID"),
])
def test_missing_stage_or_wrong_stage_settings_are_rejected(template, stage, config, code):
    with pytest.raises(ValueError, match=code):
        validate_training_stage(template, stage, config)


def test_absent_optional_stage_overrides_preserve_recipe_defaults():
    assert validate_training_stage("recap", "value") == ("value", {})
    assert validate_training_stage("openpi-sft") == (None, None)
    assert validate_training_stage("smoke") == (None, None)
