"""Executed by the registered OpenPI Python, never imported by the control plane.

Resolves and checks local configuration in CPU-only preflight mode. Real runs use
the same resolver and preserve the recipe's single- or multi-dataset semantics.
"""

import argparse
import dataclasses
import importlib.util
import json
import logging
import math
import hashlib
import os
import stat
import sys
import inspect
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

if __package__:
    from .automatic_config import build_automatic_base, normalize_auto_config
    from .cache_contract import configure_cache
    from .data_processors import instantiate_data_processor
    from .dataset_bindings import validate_binding_assets
    from .dataset_fps import install_registered_fps, registered_fps_context
    from .entrypoints import read_entrypoint
    from .recipe_definitions import apply_overrides, validate_definition
    from .recap_runtime import apply_stage, stage_request
    from .training_plan import fingerprint, inspect_training_config, local_path, require_lerobot_source, training_entrypoint
else:
    from automatic_config import build_automatic_base, normalize_auto_config
    from cache_contract import configure_cache
    from data_processors import instantiate_data_processor
    from dataset_bindings import validate_binding_assets
    from dataset_fps import install_registered_fps, registered_fps_context
    from entrypoints import read_entrypoint
    from recipe_definitions import apply_overrides, validate_definition
    from recap_runtime import apply_stage, stage_request
    from training_plan import fingerprint, inspect_training_config, local_path, require_lerobot_source, training_entrypoint


def resolve_config(base, spec, config_module, weight_loaders):
    """Explicit asset bindings override recipe defaults; never modify the recipe."""
    if spec.get("recipe_definition") is not None:
        definition = validate_definition(spec["recipe_definition"])
        if definition["base_recipe"] != spec["recipe"]:
            raise ValueError("RECIPE_DEFINITION_MISMATCH: saved recipe has a different base config")
        base = apply_overrides(base, definition["overrides"])
    inputs = spec["inputs"]
    bindings = validate_binding_assets(spec.get("dataset_bindings"), inputs)
    stage, stage_config = stage_request(spec)
    if stage == "offline_rl" and "acp_dropout" in stage_config:
        for row in bindings or []:
            parameters = (row.get("processor") or {}).get("parameters", {})
            if "acp_dropout" in parameters and parameters["acp_dropout"] != stage_config["acp_dropout"]:
                raise ValueError("DATA_PROCESSOR_STAGE_CONFLICT: 处理器 acp_dropout 与 ReCap 阶段配置冲突；请统一设置或从处理器参数中移除该项")
    if hasattr(getattr(base, "data", None), "datasets") and bindings is None:
        raise ValueError("DATASET_BINDINGS_REQUIRED: 多数据集配方必须逐组选择数据集和归一化资产，不能使用配方中的默认目录；请重新配置数据绑定并预检")
    model = local_path(inputs["model"]["path"], spec)

    def bound_paths(dataset_role, norm_role):
        dataset = local_path(inputs[dataset_role]["path"], spec)
        norm = local_path(inputs[norm_role]["path"], spec)
        if not dataset.is_dir() or not (dataset / "meta" / "info.json").is_file():
            raise ValueError("DATASET_FORMAT: expected a local LeRobot dataset with meta/info.json")
        if norm.is_file() and norm.name != "norm_stats.json":
            raise ValueError("NORM_FORMAT: a file asset must point to norm_stats.json explicitly")
        norm_dir = norm.parent if norm.is_file() else norm
        if not (norm_dir / "norm_stats.json").is_file():
            raise ValueError("NORM_MISSING: norm asset must contain norm_stats.json")
        return dataset, norm_dir

    pairs = [(f"dataset_{i}", f"norm_{i}") for i in range(len(bindings))] if bindings is not None else [("dataset", "norm")]
    paths = {pair: bound_paths(*pair) for pair in pairs}

    def bound_factory(factory, dataset_role, norm_role):
        dataset, norm_dir = paths[(dataset_role, norm_role)]
        if not dataclasses.is_dataclass(factory) or not hasattr(factory, "repo_id") or not hasattr(factory, "assets"):
            raise ValueError("RECIPE_UNSUPPORTED: data factory must support local repo_id and assets")
        require_lerobot_source(factory)
        changes = {"repo_id": str(dataset), "assets": config_module.AssetsConfig(assets_dir=str(norm_dir), asset_id=".")}
        base_config = getattr(factory, "base_config", None)
        if dataclasses.is_dataclass(base_config) and not isinstance(base_config, type):
            # create_base_config overwrites these fields too. Keep the frozen
            # snapshot consistent with the selected resources before execution.
            updates = {key: value for key, value in {"repo_id": str(dataset), "asset_id": ".", "norm_stats": None}.items()
                       if key in {field.name for field in dataclasses.fields(base_config)}}
            changes["base_config"] = dataclasses.replace(base_config, **updates)
        return dataclasses.replace(factory, **changes)

    def selected_factory(row, index):
        dataset_role, norm_role = f"dataset_{index}", f"norm_{index}"
        if "processor" in row:
            dataset, norm_dir = paths[(dataset_role, norm_role)]
            processor = row["processor"]
            processor_cls = vars(config_module).get(processor["name"])
            if (processor["name"] != "ComposableDataConfig" and stage == "offline_rl" and "acp_dropout" in stage_config and dataclasses.is_dataclass(processor_cls)
                    and any(field.name == "acp_dropout" and field.init for field in dataclasses.fields(processor_cls))):
                # Stage-owned parameters may be required constructor arguments.
                # The frozen processor request remains separate from stage_config.
                processor = {**processor, "parameters": {"acp_dropout": stage_config["acp_dropout"], **processor["parameters"]}}
            factory = instantiate_data_processor(processor, config_module, spec["code_dir"],
                                                dataset_path=dataset, norm_dir=norm_dir,
                                                training_stage=stage, stage_config=stage_config)
        elif hasattr(base.data, "datasets"):
            if row["recipe_index"] >= len(base.data.datasets):
                raise ValueError("DATASET_BINDINGS_INVALID: 所选数据处理配置超出了配方的数据源数量")
            factory = base.data.datasets[row["recipe_index"]]
        else:
            if row["recipe_index"] != 0:
                raise ValueError("DATASET_BINDINGS_INVALID: 单数据集配方只能绑定数据处理索引 0")
            factory = base.data
        return bound_factory(factory, dataset_role, norm_role)
    params_dir = model / "params" if (model / "params").is_dir() else model
    if not params_dir.is_dir():
        raise ValueError("MODEL_FORMAT: expected a local Orbax params directory")
    if not dataclasses.is_dataclass(base.data) or not hasattr(base.data, "repo_id"):
        raise ValueError("RECIPE_UNSUPPORTED: data factory must support a local repo_id")
    if not hasattr(base.lr_schedule, "peak_lr"):
        raise ValueError("RECIPE_UNSUPPORTED: learning rate schedule has no peak_lr")
    if hasattr(base.data, "datasets"):
        multi_type = getattr(config_module, "MultiDataConfigFactory", ())
        if not isinstance(base.data, multi_type) or not isinstance(base.data.datasets, (tuple, list)) or not base.data.datasets:
            raise ValueError("RECIPE_UNSUPPORTED: unsupported or empty multi-dataset factory")
        children = [selected_factory(row, i) for i, row in enumerate(bindings)]
        updates = {"datasets": children, "weights": [row["weight"] for row in bindings]}
        if spec.get("auto_config") is not None or any(type(child).__name__ == "ComposableDataConfig" for child in children):
            if "use_merged_norm_stats" not in {field.name for field in dataclasses.fields(base.data)}:
                raise ValueError("COMPOSITION_MIXTURE_UNSUPPORTED: 多数据集容器必须支持每个资产独立的归一化统计")
            # Different robots/layouts need their own bound statistics. A shared
            # recipe default must not merge incompatible raw feature spaces.
            updates["use_merged_norm_stats"] = False
        data = dataclasses.replace(base.data, **updates)
    else:
        if bindings is not None:
            if len(bindings) != 1:
                raise ValueError("DATASET_BINDINGS_INVALID: 单数据集配方只能绑定一组数据；混采请选择多数据集配方")
            data = selected_factory(bindings[0], 0)
        else:
            data = bound_factory(base.data, "dataset", "norm")
    schedule = dataclasses.replace(base.lr_schedule, peak_lr=spec["parameters"]["learning_rate"])
    # Keep recipe-specific initialization for newly added value / gripper heads.
    # A plain policy loader rejects those intentionally missing parameters.
    loader = base.weight_loader
    known = {"CheckpointWeightLoader", "CheckpointWeightLoaderWithValueHead",
             "CheckpointWeightLoaderWithGripperHead", "CheckpointWeightLoaderWithDiscreteHead",
             "CheckpointWeightLoaderIgnoreDiscreteHead", "CheckpointWeightLoaderIgnoreGripperHead"}
    if dataclasses.is_dataclass(loader) and type(loader).__name__ in known and hasattr(loader, "params_path"):
        loader = dataclasses.replace(loader, params_path=str(params_dir))
    else:
        loader = weight_loaders.CheckpointWeightLoader(str(params_dir))
    if stage == "value" and type(loader).__name__ != "CheckpointWeightLoaderWithValueHead":
        raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值函数配方需要 CheckpointWeightLoaderWithValueHead 初始化价值头")
    config = dataclasses.replace(base, data=data,
        weight_loader=loader,
        lr_schedule=schedule, num_train_steps=spec["parameters"]["steps"],
        batch_size=spec["parameters"]["batch_size"], fsdp_devices=spec["gpu_count"],
        checkpoint_base_dir=str(Path(spec["output_dir"]) / "checkpoints"),
        exp_name="platform", resume=False, overwrite=False, wandb_enabled=False)
    return apply_stage(config, spec)


def configuration_snapshot(config):
    """Bounded dataclass projection; factories/functions remain labelled objects."""
    count = 0

    def convert(value, depth=0):
        nonlocal count
        count += 1
        if count > 5000 or depth > 8:
            return {"unresolved": "projection_limit"}
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"unresolved": "non_finite"}
        if isinstance(value, str):
            return value[:4096]
        if isinstance(value, Path):
            return str(value)[:4096]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"_type": type(value).__module__ + "." + type(value).__qualname__,
                    **{field.name: convert(getattr(value, field.name), depth + 1) for field in dataclasses.fields(value)[:128]}}
        if isinstance(value, (tuple, list)):
            return {"items": [convert(item, depth + 1) for item in value[:128]], "truncated": len(value) > 128}
        if isinstance(value, dict):
            return {str(key)[:255]: convert(item, depth + 1) for key, item in islice(value.items(), 128)
                    if isinstance(key, (str, int))}
        return {"_type": type(value).__module__ + "." + type(value).__qualname__, "unresolved": "code_object"}

    return {"schema_version": 1, "scope": "post_merge_dataclass_before_data_factory_execution", "configuration": convert(config)}


def prepare_config(spec):
    auto_config = normalize_auto_config(spec.get("auto_config"))
    if auto_config is not None:
        spec = {**spec, "auto_config": auto_config}
    definition = spec.get("recipe_definition")
    if definition is not None:
        definition = validate_definition(definition)
        checkout = Path(spec["code_dir"]).resolve(strict=True)
        source = (checkout / definition["source"]["path"]).resolve(strict=True)
        if checkout not in source.parents or checkout != Path(definition["source"]["code_dir"]).resolve(strict=True):
            raise ValueError("RECIPE_SOURCE_CHANGED: saved recipe source is outside this checkout")
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ValueError("RECIPE_SOURCE_CHANGED: expected a bounded regular source file")
            contents = stream.read(1024 * 1024 + 1)
        if len(contents) > 1024 * 1024 or hashlib.sha256(contents).hexdigest() != definition["source"]["sha256"]:
            raise ValueError("RECIPE_SOURCE_CHANGED: saved recipe source content changed")
    # Resolve the same project cache defaults before libraries freeze paths at
    # import time, in both the CPU probe and the actual training interpreter.
    configure_cache(spec)
    install_registered_fps(spec)
    # Import through the existing environment. Agent separately verifies this origin.
    import openpi.training.config as config_module
    import openpi.training.weight_loaders as weight_loaders
    code = local_path(spec["code_dir"], spec)
    if not Path(config_module.__file__).resolve().is_relative_to(code):
        raise ValueError("MODULE_OUTSIDE_CODE_DIR: OpenPI 配置模块来自登记代码目录之外")
    base = (build_automatic_base(spec, config_module, weight_loaders) if auto_config is not None
            else config_module.get_config(spec["recipe"]))
    config = resolve_config(base, spec, config_module, weight_loaders)
    return config, definition


@contextmanager
def training_module(entry, code, source_sha256):
    """Execute exactly the verified source, without accepting stale bytecode."""
    contents = read_entrypoint(entry)
    if hashlib.sha256(contents).hexdigest() != source_sha256:
        raise ValueError("ASSETS_CHANGED_SINCE_PREFLIGHT: 训练入口源码在最终检查后发生变化，请重新预检")
    name = "vla_registered_openpi_train"
    module_spec = importlib.util.spec_from_file_location(name, entry)
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.modules.get(name)
    previous_path = list(sys.path)
    try:
        sys.path[:0] = [str(entry.parent), str(Path(code).resolve())]
        sys.modules[name] = module
        exec(compile(contents, str(entry), "exec"), module.__dict__)
        yield module
    finally:
        sys.path[:] = previous_path
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _run_spec(spec, *, validate_only=False, fps_runtime=None):
    checks = []
    if validate_only:
        result = {"schema_version": 1, "scope": "configuration_and_local_assets", "checks": checks}
        try:
            configure_cache(spec)
            with registered_fps_context(spec) as fps_receipt:
                config, _ = prepare_config(spec)
                result["plan"] = inspect_training_config(config, spec, checks=checks, fps_runtime=fps_receipt)
        except Exception as exc:
            message = str(exc)
            code = message.split(":", 1)[0]
            if not code.replace("_", "").isalnum() or code.upper() != code:
                code = "RECIPE_CONFIGURATION_INVALID"
            result["error"] = {"code": code, "message": message[:2000]}
            checks.append({"name": "training_cache" if code.startswith("TRAINING_CACHE_") else "validation",
                           "ok": False, "message": message[:2000]})
        print("VLA_PREFLIGHT:" + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        if result.get("error"):
            raise SystemExit(2)
        return

    config, definition = prepare_config(spec)
    plan = inspect_training_config(config, spec, checks=checks, fps_runtime=fps_runtime)
    if spec.get("expected_plan_digest") != fingerprint(plan):
        raise ValueError("ASSETS_CHANGED_SINCE_PREFLIGHT: 训练前解析结果与 Agent 冻结的预检不一致，请重新预检")
    entry = training_entrypoint(config, spec)
    import wandb

    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_training.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    (output / "training_config.py").write_text(plan["generated_config"]["source"], encoding="utf-8")
    effective = configuration_snapshot(config)
    effective["recipe_definition"] = definition
    effective["parameters"] = spec["parameters"]
    effective["inputs"] = spec["inputs"]
    effective["execution_plan"] = plan
    effective["execution_plan_digest"] = fingerprint(plan)
    (output / "effective_training.json").write_text(json.dumps(effective, indent=2, ensure_ascii=False))

    def log_metrics(values, step=None, **kwargs):
        metrics = {key: float(value) for key, value in values.items() if hasattr(value, "__float__")}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("NON_FINITE_METRIC: training reported NaN or infinity")
        print("VLA_EVENT:" + json.dumps({"type": "METRIC", "data": {
            **metrics, "step": step, "total_steps": config.num_train_steps}}), flush=True)

    logging.basicConfig(level=logging.INFO)
    with training_module(entry, spec["code_dir"], plan["entrypoint_sha256"]) as module:
        try:
            inspect.signature(module.main).bind(config)
        except TypeError as exc:
            raise ValueError("ENTRYPOINT_UNSUPPORTED: 训练入口 main 无法按 main(config) 调用") from exc
        # Keep local metrics even when init() replaces wandb.log.
        original_init = module.init_wandb

        def local_init(*args, **kwargs):
            result = original_init(*args, **kwargs)
            wandb.log = log_metrics
            return result

        module.init_wandb = local_init
        module.main(config)
    checkpoints = sorted((p for p in Path(config.checkpoint_dir).iterdir()
                          if p.is_dir() and p.name.isdigit() and (p / "params").is_dir()),
                         key=lambda p: int(p.name))
    if not checkpoints:
        raise RuntimeError("OUTPUT_MISSING: training exited without a completed params checkpoint")
    last = checkpoints[-1]
    artifact = {"kind": "MODEL", "name": spec["recipe"] + " / " + last.name,
                "path": str(last), "version": last.name,
                "metadata": {"format": "openpi-orbax", "recipe": spec["recipe"],
                             "step": int(last.name), "evaluated": False}}
    (output / "artifacts.json").write_text(json.dumps([artifact], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec_json)
    if args.validate_only:
        return _run_spec(spec, validate_only=True)
    # Resolve cache paths before importing LeRobot, then retain this policy
    # through entrypoint imports, Dataset construction and spawned workers.
    configure_cache(spec)
    with registered_fps_context(spec) as fps_receipt:
        return _run_spec(spec, fps_runtime=fps_receipt)


if __name__ == "__main__":
    main()
