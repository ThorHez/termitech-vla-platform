"""Public ReCap stage configuration, shared across control and Agent boundaries."""

import copy
import math


STAGES = (
    {"id": "value", "name": "价值函数训练", "runtime_adapter": "openpi-sft",
     "description": "训练价值模型。数据必须已经包含 value_target；产出的价值模型用于后续离线优势标注。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": "scripts/train_value.py", "default_multi_entrypoint": "scripts/train_value_multi_dataset.py",
     "recipe_prefixes": ["pi0_value_"],
     "parameters": {
         "val_ratio": {"type": "number", "label": "验证集比例", "default": 0.1, "min": 0, "max": 0.9},
         "eval_interval": {"type": "integer", "label": "验证间隔（步）", "default": 1000, "min": 1, "max": 1000000},
         "eval_batches": {"type": "integer", "label": "每次验证批数", "default": 10, "min": 1, "max": 10000},
     }},
    {"id": "offline_rl", "name": "离线强化学习（ACP 策略训练）", "runtime_adapter": "openpi-sft",
     "description": "使用已完成离线优势标注的数据训练 ACP 策略。数据必须包含 is_positive；初始权重选择策略模型。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": "scripts/train.py", "default_multi_entrypoint": "scripts/train_multi_dataset.py",
     "recipe_prefixes": ["pi05_acp_"],
     "parameters": {
         "acp_dropout": {"type": "number", "label": "优势条件丢弃概率", "default": 0.1, "min": 0, "max": 1},
     }},
)


def recap_stages():
    return copy.deepcopy(list(STAGES))


def runtime_adapter_for(template_id):
    return "openpi-sft" if template_id == "recap" else template_id


def validate_training_stage(template_id, training_stage=None, stage_config=None):
    if template_id != "recap":
        if training_stage is not None or stage_config is not None:
            raise ValueError("TRAINING_STAGE_UNSUPPORTED: 只有 ReCap 模板支持分阶段配置")
        return None, None
    if training_stage is None:
        raise ValueError("RECAP_STAGE_REQUIRED: 请选择价值函数训练或离线强化学习阶段")
    stage = next((item for item in STAGES if item["id"] == training_stage), None)
    if stage is None:
        raise ValueError("RECAP_STAGE_INVALID: 不支持的 ReCap 训练阶段")
    config = {} if stage_config is None else stage_config
    if not isinstance(config, dict) or set(config) - set(stage["parameters"]):
        raise ValueError("RECAP_STAGE_CONFIG_INVALID: 当前阶段包含不支持的配置项")
    clean = {}
    for name, value in config.items():
        rule = stage["parameters"][name]
        try:
            finite = isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if (isinstance(value, bool) or not finite
                or rule["type"] == "integer" and not isinstance(value, int)
                or not rule["min"] <= value <= rule["max"]):
            raise ValueError(f"RECAP_STAGE_CONFIG_INVALID: 阶段参数 {name} 超出范围或类型无效")
        clean[name] = float(value) if rule["type"] == "number" else value
    return training_stage, clean
