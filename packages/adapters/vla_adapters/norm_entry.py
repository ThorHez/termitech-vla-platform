"""CPU statistics jobs, executed by the registered OpenPI interpreter."""

import argparse
from contextlib import nullcontext
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unicodedata

if __package__:
    from .cache_contract import configure_cache, inspect_numeric_cache
    from .dataset_fps import install_registered_fps, registered_fps_context
    from .entrypoints import read_entrypoint
    from .data_processors import instantiate_data_processor, processor_identity, validate_processor_instance
    from .automatic_config import build_automatic_norm_base
    from .composition_spec import normalize_composition
    from .norm_datasets import normalize_norm_datasets, validate_norm_assets, validate_norm_configuration
    from .openpi_entry import training_module
    from .training_plan import (dataset_contract, file_manifest, fingerprint, local_path, read_json,
                                require_lerobot_source, stored_action_statistics_projection)
else:
    from cache_contract import configure_cache, inspect_numeric_cache
    from dataset_fps import install_registered_fps, registered_fps_context
    from entrypoints import read_entrypoint
    from data_processors import instantiate_data_processor, processor_identity, validate_processor_instance
    from automatic_config import build_automatic_norm_base
    from composition_spec import normalize_composition
    from norm_datasets import normalize_norm_datasets, validate_norm_assets, validate_norm_configuration
    from openpi_entry import training_module
    from training_plan import (dataset_contract, file_manifest, fingerprint, local_path, read_json,
                               require_lerobot_source, stored_action_statistics_projection)


def dataset_items(spec):
    rows = normalize_norm_datasets(spec.get("norm_datasets"))
    return [("dataset", None)] if rows is None else [(f"dataset_{index}", row) for index, row in enumerate(rows)]


def norm_manifest(dataset, spec):
    """Only numeric data and its required metadata participate in statistics."""
    _, metadata = read_json(dataset / "meta/info.json", spec)
    rows, size, data = file_manifest(dataset / "data", spec)
    return [["data/" + row[0], *row[1:]] for row in rows], size, fingerprint({"metadata": metadata, "data": data})


def norm_dataset_name(asset):
    """A readable source label; asset ID remains the authoritative association."""
    path = asset.get("path")
    basename = Path(path).name if isinstance(path, str) else ""
    for candidate in (asset.get("name"), basename, asset.get("id"), "dataset"):
        if not isinstance(candidate, str):
            continue
        # Preserve ordinary Unicode while excluding controls, bidi format
        # characters and unpaired surrogates from logs/JSON/database names.
        label = " ".join("".join(character for character in candidate
            if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}).split())
        if label:
            return label[:200]


def norm_artifact_name(asset):
    suffix = " norm_stats"
    return norm_dataset_name(asset)[:200 - len(suffix)] + suffix


def prepare_config(spec):
    configure_cache(spec)
    import openpi.training.config as module

    code = local_path(spec["code_dir"], spec)
    if not Path(module.__file__).resolve().is_relative_to(code):
        raise ValueError("MODULE_OUTSIDE_CODE_DIR: OpenPI 配置模块来自登记代码目录之外")
    auto_config = validate_norm_configuration(spec.get("auto_config"), spec.get("norm_datasets"))
    base = None if auto_config is not None else module.get_config(spec["recipe"])
    items = dataset_items(spec)
    index = spec.get("norm_dataset_index", 0)
    if type(index) is not int or not 0 <= index < len(items):
        raise ValueError("NORM_DATASETS_INVALID: 统计数据集索引无效")
    role, row = items[index]
    dataset = local_path(spec["inputs"][role]["path"], spec)
    if not dataset.is_dir() or not (dataset / "meta/info.json").is_file():
        raise ValueError("DATASET_FORMAT: 统计任务需要本机 LeRobot 数据集")
    # A fresh, unpopulated asset directory prevents existing recipe statistics
    # from being loaded. Statistics are measured before Normalize/model padding.
    norm_dir = Path(spec["output_dir"]) / ".unbound"
    if row is not None and "processor" in row:
        factory = instantiate_data_processor(row["processor"], module, code,
            dataset_path=dataset, norm_dir=norm_dir, norm_stats=True)
    elif hasattr(base.data, "datasets"):
        if row is None:
            raise ValueError("NORM_RECIPE_UNSUPPORTED: 多数据集配方需要为本次统计明确选择数据处理项")
        factories = base.data.datasets
        if not isinstance(factories, (list, tuple)) or row["recipe_index"] >= len(factories):
            raise ValueError("NORM_DATASETS_INVALID: 所选数据处理项超出配方范围")
        factory = factories[row["recipe_index"]]
    else:
        if row is not None and row.get("recipe_index", 0) != 0:
            raise ValueError("NORM_DATASETS_INVALID: 单数据集配方只能选择数据处理项 0")
        factory = base.data
    if hasattr(factory, "datasets"):
        raise ValueError("NORM_PROCESSOR_UNSUPPORTED: 每个统计任务必须使用单数据集处理器，不能合并多数据集容器")
    require_lerobot_source(factory)
    assets = module.AssetsConfig(assets_dir=str(norm_dir), asset_id=".")
    changes = {"repo_id": str(dataset), "assets": assets}
    base_config = getattr(factory, "base_config", None)
    if dataclasses.is_dataclass(base_config):
        fields = {field.name for field in dataclasses.fields(base_config)}
        changes["base_config"] = dataclasses.replace(base_config, **{
            key: value for key, value in {"repo_id": str(dataset), "asset_id": ".", "norm_stats": None}.items()
            if key in fields})
    factory = dataclasses.replace(factory, **changes)
    if type(factory).__name__ != "ComposableDataConfig":
        raise ValueError("NORM_NUMERIC_ONLY_UNSUPPORTED: 该配方或处理器无法证明纯数值统计，请选择字段组合处理器；不会回退读取图像")
    # The factory stays a registered class; only its existing composition is
    # normalized for statistics. No user/model create() method is executed.
    factory = dataclasses.replace(factory, composition=normalize_composition(factory.composition, norm_stats=True))
    object.__setattr__(factory, "_norm_effective_fps", (spec["inputs"][role].get("dataset_profile") or {}).get("fps"))
    if auto_config is not None:
        base = build_automatic_norm_base({**spec, "auto_config": auto_config}, module, factory, row)
    elif dataclasses.is_dataclass(base.model):
        model_fields = {field.name for field in dataclasses.fields(base.model)}
        base = dataclasses.replace(base, model=dataclasses.replace(base.model,
            **{key: () for key in ("image_keys", "depth_keys") if key in model_fields}))
    return dataclasses.replace(base, data=factory, batch_size=spec["parameters"]["batch_size"],
                               num_workers=spec["parameters"]["num_workers"])


def generated_config(spec, *, dataset_index=None):
    frozen = {key: value for key, value in spec.items() if key not in {"output_dir", "expected_plan_digest", "norm_dataset_index"}}
    batch = spec.get("norm_datasets") is not None
    if batch:
        count = len(normalize_norm_datasets(spec["norm_datasets"]))
        default = repr(dataset_index) if dataset_index is not None else "None"
        builder = (f"def build_config(output_dir, dataset_index={default}):\n"
                   "    if dataset_index is None:\n"
                   f"        if {count} > 1:\n"
                   "            raise ValueError('Batch statistics require an explicit dataset_index; use build_configs(output_dir) to inspect all inputs.')\n"
                   "        dataset_index = 0\n"
                   "    return prepare_config({**FROZEN_REQUEST, 'output_dir': str(output_dir), 'norm_dataset_index': dataset_index})\n\n"
                   "def build_configs(output_dir):\n"
                   f"    return [build_config(str(output_dir) + '/dataset-' + str(index + 1).zfill(3), index) for index in range({count})]\n")
    else:
        builder = ("def build_config(output_dir):\n"
                   "    return prepare_config({**FROZEN_REQUEST, 'output_dir': str(output_dir)})\n")
    source = ("# Full-dataset normalization statistics; no training steps.\n"
              "import sys\n"
              f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
              "from vla_adapters.norm_entry import prepare_config\n\n"
              f"FROZEN_REQUEST = {frozen!r}\n\n"
              + builder)
    return {"schema_version": 1, "language": "python", "filename": "training_config.py",
            "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}


def inspect_dataset(config, spec, statistics, role, row):
    import numpy as np

    dataset = local_path(spec["inputs"][role]["path"], spec)
    info, info_digest = read_json(dataset / "meta/info.json", spec)
    if type(info.get("total_frames")) is not int or info["total_frames"] < 1:
        raise ValueError("DATASET_EMPTY: 数据集缺少有效的 total_frames")
    files, _, manifest = norm_manifest(dataset, spec)
    if not any(row[0].startswith("data/") and row[0].endswith(".parquet") for row in files):
        raise ValueError("DATASET_FORMAT: 数据集缺少 parquet 文件")
    composed = {}
    if row is not None and "processor" in row:
        validate_processor_instance(config.data, row["processor"], norm_stats=True)
    if type(config.data).__name__ == "ComposableDataConfig":
        # Named recipes freeze their own composition; explicit processor rows
        # additionally bind that exact composition to the requested selection.
        binding = row if row is not None and "processor" in row else {"processor": {
            "name": "ComposableDataConfig", "parameters": {"composition": config.data.composition}}}
        evidence = dataset_contract(config.data, spec, spec["inputs"][role].get("dataset_profile"),
                                    model_config=config.model, binding=binding, norm_stats=True)
        composed = {key: evidence[key] for key in ("composition", "actions_shape")}
    from openpi.training import numeric_stats
    source = local_path(numeric_stats.__file__, spec)
    if not source.is_relative_to(local_path(spec["code_dir"], spec)):
        raise ValueError("NORM_RUNTIME_UNSUPPORTED: 数值投影源码必须来自登记的训练代码目录")
    data = numeric_stats.build_numeric_statistics_config(config.data, config.model,
        effective_fps=getattr(config.data, "_norm_effective_fps", None))
    if local_path(data.repo_id, spec) != dataset:
        raise ValueError("NORM_SOURCE_MISMATCH: 配方未使用本次选择的数据集")
    projection = numeric_stats.numeric_io_evidence(data)
    if not stored_action_statistics_projection(projection):
        raise ValueError("NORM_RUNTIME_UNSUPPORTED: 当前 OpenPI 不支持按原始动作形状统计，请升级训练代码后重新预检")
    loader, _ = statistics.create_torch_dataloader(data, config.model.action_horizon,
        config.batch_size, config.model, 0, max_frames=1)
    sample = next(iter(loader))
    shapes = {}
    for key in ("state", "actions"):
        values = np.asarray(sample[key])
        if values.ndim < 2 or not values.size or not np.isfinite(values).all():
            raise ValueError(f"NORM_SAMPLE_INVALID: {key} 必须是有限的数值向量")
        shapes[key] = list(values.shape[1:])
    if composed and shapes["actions"] != composed["actions_shape"]:
        raise ValueError("NORM_SAMPLE_INVALID: 动作样本形状与数据集登记形状不一致")
    result = {"path": str(dataset), "frames": info["total_frames"], "metadata_sha256": info_digest,
              "manifest_sha256": manifest, "sample_shapes": shapes, "factory": type(config.data).__name__,
              "numeric_io": projection, **composed}
    if row is not None:
        result.update(input_role=role, dataset_asset_id=row["dataset_asset_id"])
        if "processor" in row:
            result.update(processor=row["processor"],
                          processor_identity=processor_identity(config.data, spec["code_dir"], row["processor"]["name"]))
        else:
            result["recipe_index"] = row["recipe_index"]
    if spec.get("auto_config") is not None:
        result["model_configuration"] = {"model_type": "pi05" if config.model.pi05 else "pi0",
            "action_horizon": config.model.action_horizon, "action_dim": config.model.action_dim}
    return result


def inspect_config(config, spec, statistics, entry):
    items = dataset_items(spec)
    configs = config if isinstance(config, list) else [config]
    if len(configs) != len(items):
        raise ValueError("NORM_DATASETS_INVALID: 统计配置数量与数据集数量不一致")
    datasets = [inspect_dataset(value, spec, statistics, role, row)
                for value, (role, row) in zip(configs, items)]
    paths = {role: dataset["path"] for (role, _), dataset in zip(items, datasets)}
    if len(set(paths.values())) != len(paths):
        raise ValueError("NORM_DATASETS_DUPLICATE: 多个资产指向同一数据集目录，请只选择一次")
    result = {"template_id": "openpi-norm", "recipe": spec["recipe"], "recipe_definition_digest": None,
            "input_authority": "registered_assets", "requested_inputs": spec.get("requested_inputs", paths),
            "inputs": paths, "parameters": spec["parameters"], "gpu_count": 0,
            "entrypoint": str(entry), "entrypoint_sha256": hashlib.sha256(read_entrypoint(entry)).hexdigest(),
            "adapter_sha256": fingerprint({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("norm_entry.py", "norm_datasets.py", "automatic_config.py", "data_processors.py", "composition_spec.py",
                             "dataset_contracts.py", "dataset_fps.py", "cache_contract.py", "openpi_entry.py", "training_plan.py")}),
            "datasets": datasets, "training_cache": inspect_numeric_cache(datasets, spec),
            "storage": {"required_bytes": 64 * 1024 * 1024 * len(items), "reference_checkpoint_bytes": 0, "model_params_bytes": 0},
            "generated_config": generated_config(spec)}
    if spec.get("norm_datasets") is not None:
        result.update(norm_datasets=normalize_norm_datasets(spec["norm_datasets"]), mode="per_dataset_batch")
    if spec.get("auto_config") is not None:
        result["auto_config"] = validate_norm_configuration(spec["auto_config"], spec.get("norm_datasets"))
    return result


def run_spec(spec, *, validate_only=False):
    checks = []
    try:
        rows = validate_norm_assets(spec.get("norm_datasets"), spec["inputs"])
        validate_norm_configuration(spec.get("auto_config"), rows)
        with nullcontext():
            items = dataset_items(spec)
            config = (prepare_config(spec) if rows is None else
                      [prepare_config({**spec, "norm_dataset_index": index}) for index in range(len(items))])
            entry = local_path(Path(spec["code_dir"]) / "scripts/compute_norm_stats.py", spec)
            if not entry.is_relative_to(local_path(spec["code_dir"], spec)):
                raise ValueError("NORM_ENTRYPOINT_INVALID: 统计入口必须位于登记代码目录中")
            entry_digest = hashlib.sha256(read_entrypoint(entry)).hexdigest()
            with training_module(entry, spec["code_dir"], entry_digest, module_name="compute_norm_stats") as statistics:
                if not {"config_file", "output_dir", "overwrite", "numeric_only"} <= set(inspect.signature(statistics.main).parameters):
                    raise ValueError("NORM_RUNTIME_UNSUPPORTED: 请更新 OpenPI，统计入口需支持 config_file 和独立输出目录")
                plan = inspect_config(config, spec, statistics, entry)
                for name in ("recipe_configuration", "data_contract", "dataset_samples", "norm_entrypoint", "training_cache"):
                    checks.append({"name": name, "ok": True, "message": "统计配置与所选本机数据已核验"})
                if validate_only:
                    return {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan, "checks": checks}
                if spec.get("expected_plan_digest") != fingerprint(plan):
                    raise ValueError("ASSETS_CHANGED_SINCE_PREFLIGHT: 数据、配方或统计入口已变化，请重新预检")
                output = Path(spec["output_dir"]).resolve(strict=True)
                config_path = output / "training_config.py"
                with config_path.open("x") as stream:
                    stream.write(plan["generated_config"]["source"])
                outputs = []
                for index, (role, _) in enumerate(items):
                    destination = output if len(items) == 1 else output / f"dataset-{index + 1:03d}"
                    if destination != output:
                        destination.mkdir(exist_ok=False)
                    selected_config = config_path
                    if rows is not None:
                        selected_config = destination / ("dataset_config.py" if destination == output else "training_config.py")
                        with selected_config.open("x") as stream:
                            stream.write(generated_config(spec, dataset_index=index)["source"])
                    asset = spec["inputs"][role]
                    label = norm_dataset_name(asset)
                    print(f"Computing norm_stats {index + 1}/{len(items)}: {label}", flush=True)
                    statistics.main(config_file=str(selected_config), output_dir=str(destination), overwrite=False, numeric_only=True)
                    norm = destination / "norm_stats.json"
                    verify_statistics(norm)
                    outputs.append({"kind": "NORM", "name": norm_artifact_name(asset), "path": str(norm),
                        "version": hashlib.sha256(norm.read_bytes()).hexdigest(),
                        "metadata": {"recipe": spec["recipe"], "source_dataset_asset_id": asset.get("id"),
                                     "source_dataset_name": label,
                                     "statistics_plan_digest": fingerprint(plan), "coverage": "full", "batch_index": index}})
                # Check every source again after all scans; never publish a
                # partial list when another dataset failed or changed meanwhile.
                for (role, _), dataset in zip(items, plan["datasets"]):
                    if norm_manifest(Path(plan["inputs"][role]), spec)[2] != dataset["manifest_sha256"]:
                        raise ValueError("DATASET_CHANGED: 统计过程中数据集发生变化，请重新计算")
                manifest = {"outputs": outputs}
                with (output / "artifacts.json").open("x") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, allow_nan=False)
                return plan
    except Exception as exc:
        if not validate_only:
            raise
        message = str(exc)
        checks.append({"name": "norm_configuration", "ok": False, "message": message})
        return {"schema_version": 1, "scope": "configuration_and_local_assets", "checks": checks,
                "error": {"code": message.split(":", 1)[0] if ":" in message else "NORM_PREFLIGHT_FAILED", "message": message}}


def verify_statistics(path):
    """Require finite state/action statistics before a completed job is registered."""
    import math

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("NORM_OUTPUT_INVALID: 缺少有效的 norm_stats.json")
    document = json.loads(path.read_text())
    for key in ("state", "actions"):
        stats = document["norm_stats"][key]
        width = len(stats["mean"])
        if not width:
            raise ValueError("NORM_OUTPUT_INVALID: 统计向量为空")
        for field in ("mean", "std", "q01", "q99"):
            values = stats[field]
            if not isinstance(values, list) or len(values) != width or any(
                type(value) not in (int, float) or not math.isfinite(value) or field == "std" and value < 0 for value in values
            ):
                raise ValueError("NORM_OUTPUT_INVALID: 统计向量或数值无效")
        if any(low > high for low, high in zip(stats["q01"], stats["q99"])):
            raise ValueError("NORM_OUTPUT_INVALID: q01 不能大于 q99")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_spec(json.loads(args.spec_json), validate_only=args.validate_only)
    if args.validate_only:
        print("VLA_PREFLIGHT:" + json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if result.get("error") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
