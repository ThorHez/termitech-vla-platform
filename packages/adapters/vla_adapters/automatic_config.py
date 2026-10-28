"""Data-only automatic model choices; construction runs in the training Python."""

import dataclasses
from pathlib import Path

if __package__:
    from .dataset_bindings import validate_binding_assets
    from .data_processors import instantiate_data_processor
    from .composition_spec import COMPOSABLE_PROCESSOR, SLOTS, normalize_composition
else:
    from dataset_bindings import validate_binding_assets
    from data_processors import instantiate_data_processor
    from composition_spec import COMPOSABLE_PROCESSOR, SLOTS, normalize_composition


AUTO_CONFIG_DEFAULTS = {"model_type": "pi05", "finetune_mode": "full", "action_horizon": 50, "action_dim": 32}


def normalize_auto_config(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(AUTO_CONFIG_DEFAULTS):
        raise ValueError("AUTO_CONFIG_INVALID: 自动配置只接受模型、微调方式、动作预测长度与模型动作维度")
    clean = {**AUTO_CONFIG_DEFAULTS, **value}
    if (not isinstance(clean["model_type"], str) or clean["model_type"] not in {"pi0", "pi05"}
            or not isinstance(clean["finetune_mode"], str) or clean["finetune_mode"] not in {"full", "lora"}):
        raise ValueError("AUTO_CONFIG_INVALID: 自动配置支持 Pi0/Pi05 与 full/lora 微调")
    if (type(clean["action_horizon"]) is not int or not 1 <= clean["action_horizon"] <= 1000
            or type(clean["action_dim"]) is not int or not 1 <= clean["action_dim"] <= 4096):
        raise ValueError("AUTO_CONFIG_INVALID: 预测长度应为 1–1000，模型动作维度应为 1–4096 的整数")
    return clean


def automatic_recipe_name(auto_config, training_stage=None):
    clean = normalize_auto_config(auto_config)
    if clean is None or training_stage not in (None, "sft", "value", "offline_rl"):
        raise ValueError("AUTO_CONFIG_INVALID: 自动配置或训练阶段无效")
    return (f"auto_{clean['model_type']}_{clean['finetune_mode']}_h{clean['action_horizon']}"
            f"_d{clean['action_dim']}_{training_stage or 'sft'}")


def automatic_data_layout(auto_config, bindings):
    """Use declared processing semantics, never raw dataset width guesses."""
    slots, state_widths = {}, []
    compositions = []
    independent = False
    for row in bindings:
        processor = row["processor"]
        if processor["name"] != COMPOSABLE_PROCESSOR:
            independent = True
            continue
        composition = normalize_composition(processor["parameters"]["composition"])
        compositions.append(composition)
        state_widths.append(composition["state_history"] * sum(item["width"] for item in composition["state_fields"]))
        if sum(item["width"] for item in composition["action_layout"]) > auto_config["action_dim"]:
            raise ValueError("AUTO_CONFIG_DIMENSION_MISMATCH: 模型动作维度小于已登记动作布局")
        for camera in composition["cameras"]:
            slot, kind = camera["slot"], camera["kind"]
            if slot in slots and slots[slot] != kind:
                raise ValueError("AUTO_CONFIG_VISION_CONFLICT: 混采数据在相同模型视觉槽声明了不同 RGB/深度类型")
            slots[slot] = kind
    if independent and compositions:
        # Legacy processors have no shared contract describing their emitted
        # view keys. Retain standard RGB slots and refuse ambiguous depth mixes.
        if any(kind == "depth" for kind in slots.values()):
            raise ValueError("AUTO_CONFIG_VISION_UNSUPPORTED: 组合深度输入与独立处理器混采需要高级配方明确视觉语义")
        slots = dict.fromkeys(SLOTS, "rgb")
    if auto_config["model_type"] == "pi0" and any(width > auto_config["action_dim"] for width in state_widths):
        raise ValueError("AUTO_CONFIG_DIMENSION_MISMATCH: Pi0 状态历史展平后超过模型动作维度，禁止截断")
    if len(bindings) > 1 and auto_config["model_type"] == "pi05" and independent:
        raise ValueError("AUTO_CONFIG_STATE_UNSUPPORTED: Pi05 自动混采需要每组登记组合处理以确定状态输出宽度；其他处理器请使用高级配方")
    padding = max(state_widths) if auto_config["model_type"] == "pi05" and state_widths else auto_config["action_dim"]
    vision = ({"image_keys": tuple(slot for slot in SLOTS if slots.get(slot) == "rgb"),
               "depth_keys": tuple(slot for slot in SLOTS if slots.get(slot) == "depth")} if slots else {})
    return vision, padding


def build_automatic_base(spec, config_module, weight_loaders):
    """Build a TrainConfig without looking up any named code recipe.

    Only model classes already imported by the registered OpenPI module are
    used. Actual local data/normalization paths are applied by resolve_config.
    Every data row must own a processor; a recipe index has no automatic base.
    """
    clean = normalize_auto_config(spec.get("auto_config"))
    stage = spec.get("training_stage")
    name = automatic_recipe_name(clean, stage)
    if spec.get("recipe") != name or spec.get("recipe_definition") is not None:
        raise ValueError("AUTO_CONFIG_CONFLICT: 自动配置不能同时指定代码配方或自定义配方版本")
    bindings = validate_binding_assets(spec.get("dataset_bindings"), spec["inputs"])
    if not bindings or any("processor" not in row for row in bindings):
        raise ValueError("AUTO_CONFIG_PROCESSOR_REQUIRED: 自动配置要求每组数据使用独立数据处理器")
    model_module = getattr(config_module, "pi0_value" if stage == "value" else "pi0_config", None)
    model_cls = getattr(model_module, "Pi0ValueConfig" if stage == "value" else "Pi0Config", None)
    if not isinstance(model_cls, type) or not dataclasses.is_dataclass(model_cls):
        raise ValueError("AUTO_CONFIG_UNSUPPORTED: 训练环境未导出所选 Pi0/Pi05 模型配置")
    kwargs = {"pi05": clean["model_type"] == "pi05", "action_dim": clean["action_dim"],
              "action_horizon": clean["action_horizon"]}
    vision, state_padding = automatic_data_layout(clean, bindings)
    kwargs.update(vision)
    if clean["finetune_mode"] == "lora":
        if stage == "value":
            raise ValueError("AUTO_CONFIG_UNSUPPORTED: 自动价值训练暂不支持 LoRA，请使用全量微调或高级代码配方")
        kwargs.update(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    try:
        model = model_cls(**kwargs)
        # Reuse the existing registered processor implementation. resolve_config
        # rebinds every selected row and applies the common asset/stage checks.
        processor = bindings[0]["processor"]
        processor_cls = vars(config_module).get(processor["name"])
        stage_config = spec.get("stage_config") or {}
        if (processor["name"] != "ComposableDataConfig" and stage == "offline_rl" and "acp_dropout" in stage_config
                and dataclasses.is_dataclass(processor_cls)
                and any(field.name == "acp_dropout" and field.init for field in dataclasses.fields(processor_cls))):
            processor = {**processor, "parameters": {"acp_dropout": stage_config["acp_dropout"], **processor["parameters"]}}
        norm = Path(spec["inputs"]["norm_0"]["path"])
        seed = instantiate_data_processor(processor, config_module, spec["code_dir"],
                                          dataset_path=spec["inputs"]["dataset_0"]["path"],
                                          norm_dir=norm.parent if norm.is_file() else norm,
                                          training_stage=stage, stage_config=stage_config)
        if len(bindings) > 1:
            multi = getattr(config_module, "MultiDataConfigFactory", None)
            if not isinstance(multi, type) or not dataclasses.is_dataclass(multi):
                raise ValueError("AUTO_CONFIG_UNSUPPORTED: 当前训练环境不支持多数据集工厂")
            fields = {field.name for field in dataclasses.fields(multi)}
            if not {"datasets", "weights", "state_pad_dim", "use_merged_norm_stats"} <= fields:
                raise ValueError("AUTO_CONFIG_UNSUPPORTED: 多数据集工厂必须支持独立归一化与状态填充")
            seed = multi(repo_id="multi", datasets=[seed], weights=[1.0], state_pad_dim=state_padding, use_merged_norm_stats=False)
        values = {"name": name, "model": model, "data": seed}
        if clean["finetune_mode"] == "lora":
            values.update(freeze_filter=model.get_freeze_filter(), ema_decay=None)
        if stage == "value":
            loader = getattr(weight_loaders, "CheckpointWeightLoaderWithValueHead", None)
            if loader is None:
                raise ValueError("AUTO_CONFIG_UNSUPPORTED: 当前训练环境缺少价值头权重加载器")
            values["weight_loader"] = loader("")
        return config_module.TrainConfig(**values)
    except (AttributeError, TypeError) as exc:
        raise ValueError("AUTO_CONFIG_UNSUPPORTED: 训练环境无法构造所选模型、微调方式或数据容器") from exc
