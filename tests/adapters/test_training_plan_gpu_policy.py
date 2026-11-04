"""Shared GPU policy stays launch metadata and participates in config confirmation."""
import ast
import json

import pytest

from vla_adapters import build_launch
from vla_adapters.generated_config import generated_training_config
from tests.adapters.test_generated_config_minimal import focused_config


POLICY = {"mode": "shared", "memory_mb": 40960}


def test_shared_policy_is_carried_in_launch_spec_and_exclusive_launch_stays_unchanged(tmp_path):
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": "/usr/bin/python3", "recipe": "fixture_recipe"}
    inputs = {role: {"path": str(tmp_path / role)} for role in ("dataset", "model", "norm")}
    args = (profile, "openpi-sft", inputs, {}, str(tmp_path / "output"), ["GPU-b", "GPU-a"])
    old = build_launch(*args)
    assert old == build_launch(*args, training_gpu_policy=None)
    assert "training_gpu_policy" not in json.loads(old["argv"][-1])
    shared = build_launch(*args, training_gpu_policy=POLICY)
    assert json.loads(shared["argv"][-1])["training_gpu_policy"] == POLICY
    assert shared["env"] == old["env"]  # Runtime NVML must choose the fraction, not a guessed adapter environment.


@pytest.mark.parametrize("template,assigned", [("openpi-norm", []), ("smoke", ["GPU-a"]), ("openpi-sft", [])])
def test_cpu_smoke_and_zero_device_launches_cannot_accept_a_shared_budget(tmp_path, template, assigned):
    with pytest.raises(ValueError, match="TRAINING_GPU_POLICY_INVALID"):
        build_launch({"adapter": "openpi-sft"}, template, {}, {}, str(tmp_path), assigned, training_gpu_policy=POLICY)


def test_confirmed_generated_source_freezes_gpu_budget_without_adding_a_trainconfig_field(focused_config):
    config, spec = focused_config
    old = generated_training_config(config, spec)
    shared_spec = {**spec, "training_gpu_policy": POLICY}
    shared = generated_training_config(config, shared_spec)
    changed = generated_training_config(config, {**shared_spec, "training_gpu_policy": {**POLICY, "memory_mb": 30000}})
    assert len({old["sha256"], shared["sha256"], changed["sha256"]}) == 3
    tree = ast.parse(shared["source"])
    frozen = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST" for target in node.targets))
    assert frozen["training_gpu_policy"] == POLICY
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"]
    assert calls and all(keyword.arg != "training_gpu_policy" for call in calls for keyword in call.keywords)
    assert "training_gpu_policy" not in spec and "training_gpu_policy" not in old["source"]
