"""Optional per-run controls, independent of data/model recipe selection."""

import ast
import math


RUNTIME_PARAMETER_RULES = {
    "warmup_steps": {
        "type": "integer", "default": 1000, "min": 0, "max": 10_000_000,
        "label": "学习率预热步数", "optional": True, "required": False,
        "hint": "从预热阶段升至峰值学习率的步数；必须小于学习率衰减总步数。",
    },
    "decay_steps": {
        "type": "integer", "default": 30000, "min": 1, "max": 100_000_000,
        "label": "学习率衰减总步数", "optional": True, "required": False,
        "hint": "余弦学习率计划的总步数（含预热）；须大于预热步数，独立于训练总步数。",
    },
    "decay_lr": {
        "type": "number", "default": 2.5e-6, "min": 0, "max": 0.1,
        "label": "最终学习率", "optional": True, "required": False,
        "hint": "余弦衰减结束时的学习率，不能超过峰值学习率。",
    },
    "clip_gradient_norm": {
        "type": "number", "default": 1.0, "min": 0, "max": 1000, "exclusive_min": True,
        "label": "梯度裁剪阈值", "optional": True, "required": False,
        "hint": "按全局梯度范数裁剪的正数阈值；修改现有优化器参数，不切换优化器类型。",
    },
    "ema_decay": {
        "type": "number", "default": None, "min": 0, "max": 1, "exclusive_max": True,
        "label": "EMA 衰减系数", "optional": True, "required": False,
        "hint": "留空继承配置：全量训练通常为 0.99，LoRA 默认关闭；显式启用 EMA 会增加模型副本的显存开销。",
    },
    "fsdp_devices": {
        "type": "integer", "default": None, "min": 1, "max": 4096,
        "label": "FSDP 分片设备数", "optional": True, "required": False,
        "hint": "留空使用申请的 GPU 数量；明确填写时必须是 GPU 数量的正整数因数，不指定物理卡号。",
    },
    "log_interval": {
        "type": "integer", "default": 100, "min": 1, "max": 1_000_000,
        "label": "日志记录间隔", "optional": True, "required": False,
        "hint": "每隔多少训练步记录一次训练指标。",
    },
    "save_interval": {
        "type": "integer", "default": 1000, "min": 1, "max": 1_000_000,
        "label": "检查点保存间隔", "optional": True, "required": False,
        "hint": "每隔多少训练步保存一次检查点；留空沿用训练环境或所选配方。",
    },
    "keep_period": {
        "type": "integer", "default": 5000, "min": 1, "max": 10_000_000,
        "label": "检查点保留周期", "optional": True, "required": False,
        "hint": "已保存且步数能被此周期整除的检查点长期保留，另保留最新一份；不会额外触发保存。",
    },
    "num_workers": {
        "type": "integer", "default": 2, "min": 0, "max": 128,
        "label": "数据加载进程数", "optional": True, "required": False,
        "hint": "数据加载的子进程数量；0 表示在训练主进程加载，留空沿用训练环境或所选配方。",
    },
}

SCHEDULE_PARAMETERS = frozenset({"warmup_steps", "decay_steps", "decay_lr"})
OPTIMIZER_PARAMETERS = frozenset({"clip_gradient_norm"})


def normalize_normalization_method(value, *, template_id=None):
    """Absent choices retain the registered recipe/checkpoint's legacy behavior."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in {"minmax", "quantile", "zscore"}:
        raise ValueError("NORMALIZATION_METHOD_INVALID: 归一化方法只支持 minmax、quantile 或 zscore")
    if template_id is not None and template_id not in {"openpi-sft", "recap"}:
        raise ValueError("NORMALIZATION_METHOD_UNSUPPORTED: 该训练模板不支持选择归一化方法")
    return value


def validate_generated_normalization_method(generated, method):
    """Confirm the reviewed Python freezes an explicit choice without executing it."""
    method = normalize_normalization_method(method)
    if method is None:
        return
    source = generated.get("source") if isinstance(generated, dict) else None
    try:
        if not isinstance(source, str) or len(source.encode("utf-8")) > 384 * 1024:
            raise ValueError("missing bounded configuration source")
        assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST" for target in node.targets)]
        if len(assignments) != 1:
            raise ValueError("missing frozen request")
        frozen = ast.literal_eval(assignments[0].value)
        if not isinstance(frozen, dict) or frozen.get("normalization_method") != method:
            raise ValueError("different normalization method")
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError("NORMALIZATION_METHOD_MISMATCH: generated FROZEN_REQUEST does not freeze normalization_method") from exc


def runtime_parameter_overrides(parameters):
    """Keep absence meaningful: never overwrite a recipe with template defaults."""
    result = {}
    for name, rule in RUNTIME_PARAMETER_RULES.items():
        if name not in parameters:
            continue
        value = parameters[name]
        valid_type = type(value) is int if rule["type"] == "integer" else type(value) in (int, float)
        if (not valid_type or type(value) is float and not math.isfinite(value)
                or not rule["min"] <= value <= rule["max"]
                or rule.get("exclusive_min") and value == rule["min"]
                or rule.get("exclusive_max") and value == rule["max"]):
            kind = "整数" if rule["type"] == "integer" else "有限数值"
            raise ValueError(f"TRAINING_PARAMETER_INVALID: {name} 必须是有效范围内的{kind}")
        result[name] = value
    return result


def validate_runtime_parameter_relations(parameters, *, gpu_count=None):
    """Validate explicit cross-field choices at control and Agent boundaries.

    Omitted values still mean inheritance; the registered interpreter checks
    relationships again after it has resolved those inherited native values.
    """
    values = runtime_parameter_overrides(parameters)
    if ("warmup_steps" in values and "decay_steps" in values
            and values["decay_steps"] <= values["warmup_steps"]):
        raise ValueError("TRAINING_PARAMETER_INVALID: 学习率衰减总步数必须大于预热步数")
    if "decay_lr" in values and "learning_rate" in parameters:
        peak = parameters["learning_rate"]
        if (type(peak) not in (int, float) or type(peak) is float and not math.isfinite(peak)
                or peak <= 0 or values["decay_lr"] > peak):
            raise ValueError("TRAINING_PARAMETER_INVALID: 最终学习率不能超过正数峰值学习率")
    if "fsdp_devices" in values and gpu_count is not None:
        if type(gpu_count) is not int or gpu_count <= 0 or gpu_count % values["fsdp_devices"]:
            raise ValueError("TRAINING_PARAMETER_INVALID: FSDP 分片设备数必须整除申请的 GPU 数量")
