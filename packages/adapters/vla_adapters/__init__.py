"""Small launch adapters. Training libraries stay in the registered environment."""

import json
import math
from pathlib import Path


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
         "description": "使用管理员已验证的本机 recipe 和环境，显式绑定数据、初始权重和归一化统计。需完成对应机器训练验收。",
         "parameters": parameters, "required_inputs": ["dataset", "model", "norm"], "min_gpus": 1},
        {"id": "mem", "name": "MEM 记忆训练", "algorithm": "MEM", "status": "PLANNED",
         "description": "待接入 mem_exp 两阶段训练与记忆产物选择。", "parameters": {}, "required_inputs": []},
        {"id": "recap", "name": "ReCap 策略提升", "algorithm": "RECAP", "status": "PLANNED",
         "description": "待接入价值训练、本机派生标注和 ACP 训练。", "parameters": {}, "required_inputs": []},
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


def build_launch(profile: dict, template_id: str, inputs: dict, parameters: dict,
                 output_dir: str, gpu_uuids: list[str]) -> dict:
    """Build argv without shell expansion; only trusted profile recipes can execute."""
    parameters = validate_parameters(template_id, parameters)
    if profile.get("adapter") != template_id:
        raise ValueError("PROFILE_TEMPLATE_MISMATCH")
    if profile.get("execution_mode", "native") != "native":
        raise ValueError("EXECUTION_MODE_UNSUPPORTED: this release uses registered native environments")
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
        recipe = profile.get("recipe")
        if not isinstance(recipe, str) or not recipe or not all(c.isalnum() or c in "_-" for c in recipe):
            raise ValueError("Registered recipe must be a named openpi config")
        if not profile.get("metadata", {}).get("validated", False):
            raise ValueError("PROFILE_NOT_VALIDATED: administrator must validate the local recipe first")
        for role in ("dataset", "model", "norm"):
            if not inputs.get(role, {}).get("path"):
                raise ValueError(f"Missing required {role} asset")
        if parameters["batch_size"] % len(gpu_uuids):
            raise ValueError("batch_size must be divisible by GPU count")
        # The wrapper is standard-library-only until it enters the registered interpreter.
        wrapper = Path(__file__).with_name("openpi_entry.py").resolve()
        spec = {"code_dir": str(code_dir), "recipe": recipe, "inputs": inputs,
                "parameters": parameters, "output_dir": str(output), "gpu_count": len(gpu_uuids)}
        argv = [python, "-u", str(wrapper), "--spec-json", json.dumps(spec, ensure_ascii=False)]
    return {"argv": argv, "cwd": str(code_dir), "env": {
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids), "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }}
