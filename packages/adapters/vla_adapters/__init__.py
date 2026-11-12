"""Small launch adapters. Training libraries stay in the registered environment."""

import json
import math
import re
from pathlib import Path

from .entrypoints import validate_entrypoint
from .dataset_bindings import validate_binding_assets
from .norm_datasets import normalize_norm_datasets, validate_norm_assets, validate_norm_configuration
from .recap import (STAGE_PROGRESS, recap_stages, runtime_adapter_for, validate_recap_torch_selection,
                    validate_stage_progress_selection, validate_training_stage)
from .automatic_config import automatic_recipe_name, normalize_auto_config
from .runtime_options import RUNTIME_PARAMETER_RULES, normalize_normalization_method, validate_runtime_parameter_relations
from .resume import normalize_resume_source


def get_templates() -> list[dict]:
    parameters = {
        "steps": {"type": "integer", "default": 10000, "min": 1, "max": 1000000, "label": "训练步数"},
        "batch_size": {"type": "integer", "default": 8, "min": 1, "max": 4096, "label": "批次大小"},
        "learning_rate": {"type": "number", "default": 0.00005, "min": 0.00000001, "max": 0.1, "label": "峰值学习率"},
    }
    openpi_parameters = {**parameters, **{name: dict(rule) for name, rule in RUNTIME_PARAMETER_RULES.items()}}
    recap_parameters = {name: rule for name, rule in openpi_parameters.items() if name not in {"ema_decay", "fsdp_devices"}}
    return [
        {"id": "smoke", "name": "平台链路测试", "algorithm": "SMOKE", "status": "AVAILABLE",
         "description": "CPU 合成数据测试，验证调度、日志、指标与代码追溯；不产生 VLA 模型。",
         "parameters": {**parameters, "steps": {**parameters["steps"], "default": 30, "max": 10000}},
         "required_inputs": [], "min_gpus": 0},
        {"id": "openpi-sft", "name": "OpenPI / UMI 监督微调", "algorithm": "SFT", "status": "AVAILABLE",
         "description": "绑定本机配方、数据、初始权重与归一化统计，由 Agent 自动检查配置和资产，通过后可提交训练。",
         "parameters": openpi_parameters, "required_inputs": ["dataset", "model", "norm"], "min_gpus": 1},
        {"id": "openpi-norm", "name": "计算 norm_stats", "algorithm": "NORM", "status": "AVAILABLE",
         "description": "按各数据集的数据处理方式完整计算 state / actions 统计；多选数据集时逐个独立计算，各自登记 norm_stats，无需初始权重、已有归一化参数或混采权重。",
         "runtime_adapter": "openpi-sft", "required_inputs": ["dataset"], "min_gpus": 0,
         "parameters": {
             "batch_size": {"type": "integer", "default": 32, "min": 1, "max": 4096, "label": "统计批次大小"},
             "num_workers": {"type": "integer", "default": 0, "min": 0, "max": 64, "label": "数据加载进程数"},
         }},
        {"id": "mem", "name": "MEM 记忆训练", "algorithm": "MEM", "status": "PLANNED",
         "description": "待接入 mem_exp 两阶段训练与记忆产物选择。", "parameters": {}, "required_inputs": []},
        {"id": "recap", "name": "ReCap 策略提升", "algorithm": "RECAP", "status": "AVAILABLE",
         "description": "使用 PyTorch 训练回报价值模型、用 Kai0 阶段进度生成优势数据，或用已标注数据训练 ACP 策略。",
         "runtime_adapter": "openpi-sft", "backend": "pytorch", "stages": recap_stages(),
         "parameters": recap_parameters, "required_inputs": ["dataset", "model", "norm"], "min_gpus": 1},
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
        if rule.get("optional") and name not in parameters:
            continue
        value = parameters.get(name, rule["default"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Invalid parameter: {name}")
        if rule["type"] == "integer" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not rule["min"] <= value <= rule["max"]:
            raise ValueError(f"{name} outside permitted range")
        result[name] = value
    validate_runtime_parameter_relations(result)
    return result


def resolve_recipe(profile: dict, requested: str | None = None, require_validated: bool = True,
                   auto_config: dict | None = None, training_stage: str | None = None) -> str:
    """Resolve a task recipe without changing the registered runtime or its trust boundary."""
    auto_config = normalize_auto_config(auto_config)
    if auto_config is not None:
        if profile.get("adapter") != "openpi-sft":
            raise ValueError("AUTO_CONFIG_UNSUPPORTED: 自动配置需要 OpenPI 运行环境")
        name = automatic_recipe_name(auto_config, training_stage)
        if requested not in (None, "", name):
            raise ValueError("AUTO_CONFIG_CONFLICT: 自动配置不能同时选择其他代码配方")
        return name
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
                 training_stage: str | None = None, stage_config: dict | None = None,
                 auto_config: dict | None = None, normalization_method: str | None = None,
                 norm_datasets: list[dict] | None = None, training_gpu_policy: dict | None = None,
                 resume_source: dict | None = None) -> dict:
    """Build argv; Agent preflight and launch recheck validate the selected recipe."""
    from .recipe_definitions import recipe_parameters, resolve_definition
    from vla_protocol.training_gpu_policy import normalize_training_gpu_policy
    training_gpu_policy = normalize_training_gpu_policy(training_gpu_policy)
    if training_gpu_policy is not None and (template_id not in {"openpi-sft", "recap"}
            or profile.get("adapter") != "openpi-sft" or not gpu_uuids):
        raise ValueError("TRAINING_GPU_POLICY_INVALID: 共享显存仅支持申请 GPU 的 OpenPI 训练")
    parameters = validate_parameters(template_id, {**recipe_parameters(recipe_definition), **parameters})
    normalization_method = normalize_normalization_method(normalization_method, template_id=template_id)
    norm_datasets = normalize_norm_datasets(norm_datasets)
    if norm_datasets is not None and template_id != "openpi-norm":
        raise ValueError("NORM_DATASETS_UNSUPPORTED: 只有统计模板支持独立统计数据集列表")
    training_stage, stage_config = validate_training_stage(template_id, training_stage, stage_config)
    auto_config = normalize_auto_config(auto_config)
    if auto_config is not None:
        if template_id not in {"openpi-sft", "recap", "openpi-norm"} or recipe_definition is not None:
            raise ValueError("AUTO_CONFIG_CONFLICT: 当前模板或自定义配方不能使用自动配置")
    entrypoint = validate_entrypoint(entrypoint)
    if template_id == "recap":
        validate_recap_torch_selection(auto_config, entrypoint)
    resume_source = normalize_resume_source(resume_source, inputs, template_id=template_id, entrypoint=entrypoint)
    dataset_bindings = validate_binding_assets(dataset_bindings, inputs)
    validate_stage_progress_selection(training_stage, dataset_bindings, auto_config)
    if auto_config is not None and template_id != "openpi-norm" and (not dataset_bindings or any("processor" not in row for row in dataset_bindings)):
        raise ValueError("AUTO_CONFIG_PROCESSOR_REQUIRED: 自动配置要求每组数据使用独立数据处理器")
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
    if template_id == "openpi-norm":
        if recipe_definition is not None or entrypoint is not None:
            raise ValueError("NORM_CONFIG_UNSUPPORTED: 统计模板使用自动配置或代码配方和固定统计入口")
        if gpu_uuids:
            raise ValueError("NORM_GPU_UNSUPPORTED: 统计任务使用 CPU，请申请 0 张 GPU")
        norm_datasets = validate_norm_assets(norm_datasets, inputs)
        auto_config = validate_norm_configuration(auto_config, norm_datasets)
        recipe = resolve_recipe(profile, recipe, require_validated=False, auto_config=auto_config)
        spec = {"template_id": template_id, "code_dir": str(Path(profile["code_dir"]).resolve(strict=True)),
                "recipe": recipe, "inputs": inputs, "parameters": parameters,
                "output_dir": str(Path(output_dir).resolve()), "gpu_count": 0}
        if norm_datasets is not None:
            spec["norm_datasets"] = norm_datasets
        if auto_config is not None:
            spec["auto_config"] = auto_config
        return {"argv": [str(Path(profile["python"]).absolute()), "-u",
                         str(Path(__file__).with_name("norm_entry.py").resolve()),
                         "--spec-json", json.dumps(spec, ensure_ascii=False)],
                "cwd": spec["code_dir"], "env": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}}
    recipe = (resolve_definition(profile, recipe_definition, recipe, require_validated=False)
              if recipe_definition is not None else resolve_recipe(profile, recipe, require_validated=False,
                                                                     auto_config=auto_config, training_stage=training_stage))
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
        validate_runtime_parameter_relations(parameters, gpu_count=len(gpu_uuids))
        # The wrapper is standard-library-only until it enters the registered interpreter.
        wrapper = Path(__file__).with_name("openpi_entry.py").resolve()
        spec = {"code_dir": str(code_dir), "recipe": recipe, "inputs": inputs,
                "parameters": parameters, "output_dir": str(output), "gpu_count": len(gpu_uuids)}
        if training_gpu_policy is not None:
            spec["training_gpu_policy"] = training_gpu_policy
        if template_id == "recap":
            spec.update({"template_id": "recap", "training_stage": training_stage, "stage_config": stage_config,
                         "recap_backend": "pytorch"})
            if training_stage == STAGE_PROGRESS:
                spec["profile_id"] = profile.get("id")
        if dataset_bindings is not None:
            spec["dataset_bindings"] = dataset_bindings
        if entrypoint is not None:
            spec["entrypoint"] = entrypoint
        if recipe_definition is not None:
            spec["recipe_definition"] = recipe_definition
        if auto_config is not None:
            spec["auto_config"] = auto_config
        if normalization_method is not None:
            spec["normalization_method"] = normalization_method
        if resume_source is not None:
            spec.update(template_id=template_id, resume_source=resume_source)
        argv = [python, "-u", str(wrapper), "--spec-json", json.dumps(spec, ensure_ascii=False)]
    return {"argv": argv, "cwd": str(code_dir), "env": {
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids), "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        **({"JAX_PLATFORMS": "cpu"} if template_id == "recap" else {}),
    }}
