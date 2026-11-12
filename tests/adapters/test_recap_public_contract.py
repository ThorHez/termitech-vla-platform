import copy
import json

import pytest

from vla_adapters import build_launch, get_templates, validate_parameters
from vla_adapters.recap import RECAP_TORCH_ENTRY, recap_stages, runtime_adapter_for, validate_training_stage


def test_recap_exposes_three_distinct_stages_with_independent_configuration():
    template = next(template for template in get_templates() if template["id"] == "recap")
    assert template["status"] == "AVAILABLE" and template["runtime_adapter"] == "openpi-sft"
    value, offline, stage_progress = template["stages"]
    assert value["id"] == "value" and offline["id"] == "offline_rl"
    assert set(value["parameters"]) == {"val_ratio", "eval_interval", "eval_batches"}
    assert set(offline["parameters"]) == {"acp_dropout"}
    assert stage_progress["id"] == "stage_progress"
    assert {key: row["default"] for key, row in stage_progress["parameters"].items()} == {
        "n_step": 50, "positive_ratio": .3, "stage_count": 2}
    assert template["backend"] == "pytorch"
    assert all(stage["default_entrypoint"] == stage["default_multi_entrypoint"] == RECAP_TORCH_ENTRY for stage in template["stages"])
    assert {"fsdp_devices", "ema_decay"}.isdisjoint(template["parameters"])
    assert runtime_adapter_for("recap") == "openpi-sft"
    value["parameters"].clear()
    assert recap_stages()[0]["parameters"]


@pytest.mark.parametrize(("stage", "config"), [
    ("value", {"eval_interval": 42, "val_ratio": .2}),
    ("offline_rl", {"acp_dropout": .3}),
    ("stage_progress", {"n_step": 25, "positive_ratio": .4, "stage_count": 3}),
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
    assert spec["recap_backend"] == "pytorch" and launch["env"]["JAX_PLATFORMS"] == "cpu"
    assert spec["stage_config"] == config and set(spec["parameters"]) == {"steps", "batch_size", "learning_rate"}
    assert inputs == original
    with pytest.raises(ValueError, match="Unknown training parameters"):
        validate_parameters("recap", config)


@pytest.mark.parametrize("template,stage,config,code", [
    ("recap", None, None, "RECAP_STAGE_REQUIRED"),
    ("recap", "sft", {}, "RECAP_STAGE_INVALID"),
    ("openpi-sft", "value", {}, "TRAINING_STAGE_UNSUPPORTED"),
    ("smoke", None, {}, "TRAINING_STAGE_UNSUPPORTED"),
    ("recap", "value", {"acp_dropout": 0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"val_ratio": 0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"eval_interval": 1.5}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"eval_batches": True}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"val_ratio": float("nan")}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "value", {"val_ratio": 10 ** 400}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"acp_dropout": -0.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "offline_rl", {"acp_dropout": 1.1}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_progress", {"positive_ratio": 0}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_progress", {"stage_count": 2.5}, "RECAP_STAGE_CONFIG_INVALID"),
    ("recap", "stage_progress", {"acp_dropout": .1}, "RECAP_STAGE_CONFIG_INVALID"),
])
def test_missing_stage_or_wrong_stage_settings_are_rejected(template, stage, config, code):
    with pytest.raises(ValueError, match=code):
        validate_training_stage(template, stage, config)


def test_absent_optional_stage_overrides_preserve_recipe_defaults():
    assert validate_training_stage("recap", "value") == ("value", {})
    assert validate_training_stage("openpi-sft") == (None, None)
    assert validate_training_stage("smoke") == (None, None)
