"""Small launch adapters. Training libraries stay in the registered environment."""

import json
import math
import re
from pathlib import Path

from .entrypoints import validate_entrypoint
from .dataset_bindings import validate_binding_assets
from .recap import recap_stages, runtime_adapter_for, validate_training_stage


def get_templates() -> list[dict]:
    parameters = {
        "steps": {"type": "integer", "default": 10000, "min": 1, "max": 1000000, "label": "训练步数"},
        "batch_size": {"type": "integer", "default": 8, "min": 1, "max": 4096, "label": "批次大小"},
        "learning_rate": {"type": "number", "default": 0.00005, "min": 0.00000001, "max": 0.1, "label": "峰值学习率"},
    }
    return [
        {"id": "smoke", "name": "平台链路测试", "algorithm": "SMOKE", "status": "AVAILABLE",
         "description": "CPU 合成数据测试，验证调度、日志、指标与代码追溯；不产生 VLA 模型。",
         "parameters": {**parameters, "steps": {**parameters["steps"], "default": 30, "max": 10000}},
         "required_inputs": [], "min_gpus": 0},
        {"id": "openpi-sft", "name": "OpenPI / UMI 监督微调", "algorithm": "SFT", "status": "AVAILABLE",
         "description": "绑定本机配方、数据、初始权重与归一化统计，由 Agent 自动检查配置和资产，通过后可提交训练。",
         "parameters": parameters, "required_inputs": ["dataset", "model", "norm"], "min_gpus": 1},
        {"id": "mem", "name": "MEM 记忆训练", "algorithm": "MEM", "status": "PLANNED",
         "description": "待接入 mem_exp 两阶段训练与记忆产物选择。", "parameters": {}, "required_inputs": []},
        {"id": "recap", "name": "ReCap 策略提升", "algorithm": "RECAP", "status": "AVAILABLE",
         "description": "先选择价值函数训练或离线强化学习阶段，再配置对应的配方、数据、权重与参数。两阶段之间需完成离线优势标注。",
         "runtime_adapter": "openpi-sft", "stages": recap_stages(),
         "parameters": parameters, "required_inputs": ["dataset", "model", "norm"], "min_gpus": 1},
        {"id": "rlt", "name": "RLT 离线训练", "algorithm": "RLT", "status": "PLANNED",
         "description": "待确认代码入口与 replay 数据协议。", "parameters": {}, "required_inputs": []},
    ]


def validate_parameters(template_id: str, parameters: dict) -> dict:
    template = next((t for t in get_templates() if t["id"] == template_id), None)
    if not template or template["status"] != "AVAILABLE":
        raise ValueError("TEMPLATE_UNAVAILABLE")
    rules = template["parameters"]
    if set(parameters) - set(rules):
        raise ValueError("Unknown training parameters")
    result = {}
    for name, rule in rules.items():
        value = parameters.get(name, rule["default"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid parameter: {name}")
        if rule["type"] == "integer" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not rule["min"] <= value <= rule["max"]:
            raise ValueError(f"{name} outside permitted range")
        result[name] = value
    return result


def resolve_recipe(profile: dict, requested: str | None = None, require_validated: bool = True) -> str:
    """Resolve a task recipe without changing the registered runtime or its trust boundary."""
    recipe = requested if requested is not None else profile.get("recipe")
    if profile.get("adapter") == "smoke" and requested is None:
        recipe = "smoke"
    if recipe is None or recipe == "":
        raise ValueError("RECIPE_REQUIRED: select a training recipe for this task")
    if not isinstance(recipe, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,255}", recipe) is None:
        raise ValueError("RECIPE_INVALID: recipe must be an ASCII named configuration")
    if profile.get("adapter") == "smoke":
        if recipe != "smoke":
            raise ValueError("RECIPE_INVALID: the smoke adapter only supports the smoke recipe")
        return recipe
    if profile.get("adapter") != "openpi-sft":
        raise ValueError("PROFILE_TEMPLATE_MISMATCH: unsupported training adapter")
    if require_validated:
        metadata = profile.get("metadata") or {}
        validated = metadata.get("validated_recipes", [])
        listed = isinstance(validated, list) and recipe in validated
        # A legacy flag authorizes its original recipe, never a task override.
        legacy = metadata.get("validated") is True and recipe == profile.get("recipe")
        if not (listed or legacy):
            raise ValueError("PROFILE_NOT_VALIDATED: administrator must validate this local recipe first")
    return recipe


def build_launch(profile: dict, template_id: str, inputs: dict, parameters: dict,
                 output_dir: str, gpu_uuids: list[str], recipe: str | None = None,
                 recipe_definition: dict | None = None, entrypoint: str | None = None,
                 dataset_bindings: list[dict] | None = None,
                 training_stage: str | None = None, stage_config: dict | None = None) -> dict:
    """Build argv; Agent preflight and launch recheck validate the selected recipe."""
    from .recipe_definitions import recipe_parameters, resolve_definition
    parameters = validate_parameters(template_id, {**recipe_parameters(recipe_definition), **parameters})
    training_stage, stage_config = validate_training_stage(template_id, training_stage, stage_config)
    entrypoint = validate_entrypoint(entrypoint)
    dataset_bindings = validate_binding_assets(dataset_bindings, inputs)
    if template_id not in {"openpi-sft", "recap"} and dataset_bindings is not None:
        raise ValueError("DATASET_BINDINGS_INVALID: 该训练模板不支持多数据集绑定")
    if template_id == "recap" and dataset_bindings is None:
        raise ValueError("DATASET_BINDINGS_INVALID: ReCap 每组数据集必须明确绑定归一化参数")
    if template_id == "smoke" and entrypoint is not None:
        raise ValueError("INVALID_ENTRYPOINT: 平台链路测试使用固定测试入口")
    if profile.get("adapter") != runtime_adapter_for(template_id):
        raise ValueError("PROFILE_TEMPLATE_MISMATCH")
    if profile.get("execution_mode", "native") != "native":
        raise ValueError("EXECUTION_MODE_UNSUPPORTED: this release uses registered native environments")
    recipe = (resolve_definition(profile, recipe_definition, recipe, require_validated=False)
              if recipe_definition is not None else resolve_recipe(profile, recipe, require_validated=False))
    code_dir = Path(profile["code_dir"]).resolve(strict=True)
    python = str(Path(profile["python"]).absolute())
    output = Path(output_dir).resolve()
    if template_id == "smoke":
        script = (code_dir / "smoke_training.py").resolve(strict=True)
        if not script.is_relative_to(code_dir):
            raise ValueError("ENTRYPOINT_OUTSIDE_CODE_DIR")
        argv = [python, "-u", str(script), "--output-dir", str(output)]
        for key, value in parameters.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
    else:
        if not gpu_uuids:
            raise ValueError("GPU_REQUIRED")
        for role in (("model",) if dataset_bindings is not None else ("dataset", "model", "norm")):
            if not inputs.get(role, {}).get("path"):
                raise ValueError(f"Missing required {role} asset")
        if parameters["batch_size"] % len(gpu_uuids):
            raise ValueError("batch_size must be divisible by GPU count")
        # The wrapper is standard-library-only until it enters the registered interpreter.
        wrapper = Path(__file__).with_name("openpi_entry.py").resolve()
        spec = {"code_dir": str(code_dir), "recipe": recipe, "inputs": inputs,
                "parameters": parameters, "output_dir": str(output), "gpu_count": len(gpu_uuids)}
        if template_id == "recap":
            spec.update({"template_id": "recap", "training_stage": training_stage, "stage_config": stage_config})
        if dataset_bindings is not None:
            spec["dataset_bindings"] = dataset_bindings
        if entrypoint is not None:
            spec["entrypoint"] = entrypoint
        if recipe_definition is not None:
            spec["recipe_definition"] = recipe_definition
        argv = [python, "-u", str(wrapper), "--spec-json", json.dumps(spec, ensure_ascii=False)]
    return {"argv": argv, "cwd": str(code_dir), "env": {
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids), "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }}
