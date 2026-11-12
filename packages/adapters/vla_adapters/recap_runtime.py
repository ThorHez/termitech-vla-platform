"""ReCap stage checks in the registered interpreter, without model creation.

Value training consumes targets, stage-progress training publishes a derivative,
and ACP consumes previously annotated labels.
"""

import dataclasses
import hashlib
import importlib
import importlib.metadata
import math
import os
import tempfile
import time
from pathlib import Path

if __package__:
    from .composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from .cache_contract import _managed_path
    from .dataset_validation_cache import get_or_validate, lookup_cached
    from .recap import (RECAP_SUPERVISION, RECAP_TORCH_ENTRY, RECAP_TORCH_LOADER, STAGE_PROGRESS, STAGE_PROGRESS_DATASET, STAGE_PROGRESS_ENTRY,
                        STAGE_PROGRESS_ID_COLUMNS, STAGE_PROGRESS_KEYS, STAGE_PROGRESS_LOADER, STAGE_PROGRESS_MODEL,
                        STAGE_PROGRESS_MODULES, STAGE_PROGRESS_SUPPORT_ENTRY,
                        recap_model_matches, recap_torch_module_paths, uses_torch_recap,
                        validate_recap_torch_selection, validate_stage_progress_selection, validate_training_stage)
else:
    from composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from cache_contract import _managed_path
    from dataset_validation_cache import get_or_validate, lookup_cached
    from recap import (RECAP_SUPERVISION, RECAP_TORCH_ENTRY, RECAP_TORCH_LOADER, STAGE_PROGRESS, STAGE_PROGRESS_DATASET, STAGE_PROGRESS_ENTRY,
                       STAGE_PROGRESS_ID_COLUMNS, STAGE_PROGRESS_KEYS, STAGE_PROGRESS_LOADER, STAGE_PROGRESS_MODEL,
                       STAGE_PROGRESS_MODULES, STAGE_PROGRESS_SUPPORT_ENTRY,
                       recap_model_matches, recap_torch_module_paths, uses_torch_recap,
                       validate_recap_torch_selection, validate_stage_progress_selection, validate_training_stage)


def stage_request(spec):
    return validate_training_stage(spec.get("template_id", "openpi-sft"),
                                   spec.get("training_stage"), spec.get("stage_config"))


def _factory_mapping(factory):
    if type(factory).__name__ != COMPOSABLE_PROCESSOR:
        return getattr(factory, "mapping", None)
    spec = normalize_composition(factory.composition)
    keys = [field["key"] for field in spec["state_fields"] + spec["cameras"]]
    keys += [spec["action_key"], spec["prompt_key"], *spec["aux_keys"]]
    return {key: key for key in keys}


def apply_stage(config, spec):
    stage, overrides = stage_request(spec)
    if stage is None:
        return config
    children = getattr(config.data, "datasets", None)
    factories = children if children is not None else [config.data]
    model_type = type(config.model).__name__
    torch_recap = uses_torch_recap(spec)
    if torch_recap:
        validate_recap_torch_selection(spec.get("auto_config"), spec.get("entrypoint"))
        if any("lora" in str(getattr(config.model, key, "")) for key in ("paligemma_variant", "action_expert_variant")):
            raise ValueError("RECAP_TORCH_LORA_UNSUPPORTED: PyTorch ReCap 支持全量微调，请选择兼容的模型配方")
        if stage == "offline_rl" and model_type not in {"Pi0Config", "Pi0GripperConfig"}:
            raise ValueError("RECAP_STAGE_MODEL_UNSUPPORTED: PyTorch ACP 支持 Pi0Config 和 Pi0GripperConfig")
        if getattr(config, "recap_training_stage", None) != stage or not getattr(config, "pytorch_weight_path", None):
            raise ValueError("RECAP_STAGE_RUNTIME_UNAVAILABLE: 当前 OpenPI-UMI 未绑定 PyTorch ReCap 阶段和权重")
    if stage == STAGE_PROGRESS:
        validate_stage_progress_selection(stage, spec.get("dataset_bindings"), spec.get("auto_config"))
        if children is not None or model_type != STAGE_PROGRESS_MODEL or not getattr(config.model, "pi05", False):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: Kai0 阶段进度需要单数据集的 Pi0StageProgressConfig 配方")
        if (type(config.data).__name__ != COMPOSABLE_PROCESSOR
                or RECAP_SUPERVISION[stage] not in _factory_mapping(config.data)):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: Kai0 阶段进度需要保留 stage_progress_gt 的组合字段处理器")
        if not torch_recap and type(config.weight_loader).__name__ != STAGE_PROGRESS_LOADER:
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: Kai0 阶段进度需要支持初始化进度头的权重加载器")
        model = dataclasses.replace(config.model, **overrides)
        validate_training_stage("recap", stage, {key: getattr(model, key) for key in STAGE_PROGRESS_KEYS})
        return dataclasses.replace(config, model=model)
    if stage == "value":
        if model_type != "Pi0ValueConfig":
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值函数阶段必须使用 Pi0ValueConfig 配方")
        if any(not isinstance(_factory_mapping(factory), dict) or
               "value_target" not in _factory_mapping(factory) for factory in factories):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值函数配方必须保留 value_target")
        if any(not hasattr(config, key) for key in overrides):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值函数配方缺少所选验证参数")
        config = dataclasses.replace(config, **overrides)
        keys = ("val_ratio", "eval_interval", "eval_batches")
        if any(not hasattr(config, key) for key in keys):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值函数配方缺少验证参数")
        validate_training_stage("recap", "value", {key: getattr(config, key) for key in keys})
        return config
    if not recap_model_matches(stage, model_type) or any(
            not hasattr(factory, "acp_dropout") or
            not isinstance(_factory_mapping(factory), dict) or
            "is_positive" not in _factory_mapping(factory) for factory in factories):
        raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 离线强化学习阶段必须使用读取 is_positive 的 ACP 策略配方")
    if overrides:
        factories = [dataclasses.replace(factory, **overrides) for factory in factories]
        data = dataclasses.replace(config.data, datasets=factories) if children is not None else factories[0]
        config = dataclasses.replace(config, data=data)
    return config


def default_entrypoint(config, spec):
    stage, _ = stage_request(spec)
    if uses_torch_recap(spec):
        return RECAP_TORCH_ENTRY
    multi = hasattr(config.data, "datasets")
    if stage == "value":
        name = "train_value_multi_dataset.py" if multi else "train_value.py"
        legacy = "scripts/" + name
        current = "scripts/acp/" + name
        if spec.get("code_dir") and not (Path(spec["code_dir"]) / legacy).exists() and (Path(spec["code_dir"]) / current).exists():
            return current
        return legacy
    if stage == STAGE_PROGRESS:
        return STAGE_PROGRESS_ENTRY
    return "scripts/train_multi_dataset.py" if multi else "scripts/train.py"


def validate_stage_entrypoint(config, spec, entry):
    stage, _ = stage_request(spec)
    if stage is None:
        return
    relative = entry.relative_to(Path(spec["code_dir"]).resolve()).as_posix()
    if uses_torch_recap(spec):
        validate_recap_torch_selection(spec.get("auto_config"), relative)
        return
    policy = {"scripts/train.py", "scripts/train_multi_dataset.py", "scripts/train_multi_dataset_hybrid.py"}
    single_value = {"scripts/train_value.py", "scripts/acp/train_value.py"}
    value = single_value | {"scripts/train_value_multi_dataset.py", "scripts/acp/train_value_multi_dataset.py"}
    if (relative in {"scripts/train_advantage.py", "scripts/train_multi_dataset_hybrid.py"}
            or stage == "value" and relative in policy
            or stage == "offline_rl" and relative in value
            or (stage == STAGE_PROGRESS) != (relative == STAGE_PROGRESS_ENTRY)):
        raise ValueError("RECAP_STAGE_ENTRYPOINT_MISMATCH: 训练入口与本次 ReCap 阶段不匹配")
    if stage == "value" and hasattr(config.data, "datasets") and relative in single_value:
        raise ValueError("RECAP_STAGE_ENTRYPOINT_MISMATCH: 多数据集价值训练必须使用混采入口")


def _mapping_columns(mapping):
    columns = set()
    pending = [mapping]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, str) and value:
            # LeRobot injects task text from task_index / tasks metadata.
            if value != "task":
                columns.add(value)
        else:
            raise ValueError("RECAP_DATA_SCHEMA_UNSUPPORTED: 数据映射需要明确的列名")
    return sorted(columns)


def _scalar(value):
    for _ in range(4):
        if isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        else:
            break
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("RECAP_LABEL_INVALID: 价值目标和优势标签必须是有限标量")
    return value


def _parquet_evidence(path, required, supervised, *, stage, value_min, value_max):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValueError("RECAP_DATA_SCHEMA_UNAVAILABLE: 训练环境需要 PyArrow 检查阶段数据列") from exc
    files = sorted((Path(path) / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("RECAP_DATA_SCHEMA_INVALID: 数据集没有 parquet 文件")
    # Check evenly spaced files including both ends, with a bounded read of the
    # supervision columns. The full file manifest is frozen by training_plan.
    indices = sorted({round(i * (len(files) - 1) / min(15, len(files) - 1))
                      for i in range(min(16, len(files)))}) if len(files) > 1 else [0]
    rows = []
    summary = {key: {"min": None, "max": None, "count": 0} for key in supervised}
    deadline = time.monotonic() + 15
    for index in indices:
        if time.monotonic() > deadline:
            raise ValueError("RECAP_DATA_SCAN_LIMIT: 阶段数据列检查超时")
        file = files[index]
        source = None
        try:
            source = parquet.ParquetFile(file)
            columns = sorted(source.schema_arrow.names)
            missing = sorted((set(required) | set(supervised)) - set(columns))
            if missing:
                raise ValueError(f"RECAP_DATA_SCHEMA_INVALID: {file} 缺少阶段数据列：{', '.join(missing)}")
            batch = next(iter(source.iter_batches(batch_size=64, columns=supervised)), None)
            if batch is None or batch.num_rows <= 0:
                raise ValueError(f"RECAP_DATA_SCHEMA_INVALID: {file} 没有可检查的训练样本")
            values = batch.to_pydict()
            for key in supervised:
                for raw in values[key]:
                    number = _scalar(raw)
                    if key == "is_positive" and number not in (0, 1):
                        raise ValueError(f"RECAP_LABEL_INVALID: {file} 的 is_positive 必须为 0 或 1；请先完成离线优势标注")
                    if key == "stage_progress_gt" and not 0 <= number <= 1:
                        raise ValueError(f"RECAP_LABEL_INVALID: {file} 的 stage_progress_gt 必须位于 [0, 1]")
                    if key == "value_target" and value_min is not None and value_max is not None and not value_min <= number <= value_max:
                        raise ValueError(f"RECAP_LABEL_INVALID: {file} 的 value_target 超出 [{value_min}, {value_max}]")
                    measured = summary[key]
                    measured["min"] = number if measured["min"] is None else min(measured["min"], number)
                    measured["max"] = number if measured["max"] is None else max(measured["max"], number)
                    measured["count"] += 1
            rows.append({"path": str(file.relative_to(path)), "sample_rows": batch.num_rows, "columns": columns})
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"RECAP_DATA_SCHEMA_INVALID: 无法读取阶段数据列 {file}: {exc}") from exc
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()
    if "is_positive" in summary:
        summary["is_positive"]["binary"] = True
    return {"parquet_files": len(files), "checked_files": rows,
            "supervision_summary": summary,
            "scope": "bounded parquet schema and first 64 supervision rows from up to 16 evenly spaced files"}


_SUPERVISION_CONTEXT = {"scope": "recap_supervision"}
_STAGE_PROGRESS_CONTEXT = {"scope": "recap_stage_progress_supervision"}


def prepare_source_supervision(data, spec, info):
    """Measure optional ReCap labels only during the dataset's basic validation."""
    features = info.get("features", {})
    supervised = [key for key in ("value_target", "is_positive") if key in features]
    progress = RECAP_SUPERVISION[STAGE_PROGRESS]

    def measure(required, columns, context):
        def validate():
            return {"schema_version": 1, "status": "verified", "path": data["path"],
                    **_parquet_evidence(data["path"], required, columns, stage="source",
                                        value_min=None, value_max=None)}
        return get_or_validate("source", data, spec, validate, context=context)

    result = measure(supervised, supervised, _SUPERVISION_CONTEXT) if supervised else None
    if progress in features:
        stage_result = measure([progress, *STAGE_PROGRESS_ID_COLUMNS], [progress], _STAGE_PROGRESS_CONTEXT)
        result = ({**result, "supervision_summary": {**result["supervision_summary"], **stage_result["supervision_summary"]}}
                  if result else stage_result)
    return result


def _base_validation_required(dataset):
    raise ValueError("DATASET_BASE_VALIDATION_REQUIRED: 数据集 " + str(dataset.get("path", ""))[:512]
                     + " 的 ReCap 监督基础校验证据缺失或已失效，请先在数据集配置中执行基础校验，再重新提交训练")


def require_cached_supervision(dataset, spec):
    """Read an existing source-bound label receipt; never scan or load on a miss."""
    progress = spec.get("training_stage") == STAGE_PROGRESS
    context = _STAGE_PROGRESS_CONTEXT if progress else _SUPERVISION_CONTEXT
    evidence = lookup_cached("source", dataset, spec, context=context)
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
            or evidence.get("status") != "verified" or evidence.get("path") != dataset.get("path")):
        _base_validation_required(dataset)
    checked = evidence.get("checked_files")
    summary = evidence.get("supervision_summary")
    if (type(evidence.get("parquet_files")) is not int or not isinstance(checked, list)
            or not 1 <= len(checked) <= min(16, evidence["parquet_files"])
            or not isinstance(summary, dict) or not summary
            or set(summary) - ({"stage_progress_gt"} if progress else {"value_target", "is_positive"})):
        _base_validation_required(dataset)
    count = 0
    for row in checked:
        if not isinstance(row, dict):
            _base_validation_required(dataset)
        path = row.get("path")
        columns = row.get("columns")
        if (not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
                or not path.startswith("data/") or not path.endswith(".parquet")
                or type(row.get("sample_rows")) is not int or not 1 <= row["sample_rows"] <= 64
                or not isinstance(columns, list) or not columns
                or any(not isinstance(column, str) or not column for column in columns)
                or len(set(columns)) != len(columns) or not set(summary) <= set(columns)):
            _base_validation_required(dataset)
        count += row["sample_rows"]
    for key, measured in summary.items():
        if (not isinstance(measured, dict) or type(measured.get("count")) is not int or measured["count"] != count
                or any(type(measured.get(bound)) not in (int, float) or not math.isfinite(measured[bound])
                       for bound in ("min", "max")) or measured["min"] > measured["max"]
                or (key == "is_positive" and (measured.get("binary") is not True
                                              or measured["min"] not in (0, 1) or measured["max"] not in (0, 1)))
                or (key == "stage_progress_gt" and not 0 <= measured["min"] <= measured["max"] <= 1)):
            _base_validation_required(dataset)
    return evidence


def _value_runtime_cache(spec):
    """Built-in Value entries override the environment with ~/.cache/jax.

    Probe that real path too, conservatively for custom Value entries, without
    changing ownership or permissions and without replacing existing contents.
    """
    cache = None
    probe = None
    try:
        cache = _managed_path(Path("~/.cache/jax").expanduser(), spec, label="价值训练 JAX 缓存")
        cache.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix=".vla-recap-cache-probe-", dir=cache)
        probe = Path(filename)
        with os.fdopen(descriptor, "w+b") as stream:
            stream.write(b"vla-recap-cache-check")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            if stream.read() != b"vla-recap-cache-check":
                raise ValueError("TRAINING_CACHE_UNAVAILABLE: 价值训练 JAX 缓存读写核对失败")
        return {"jax_compilation_cache_dir": str(cache),
                "reason": "built-in Value entries use HOME/.cache/jax; conservatively checked for custom Value entries"}
    except Exception as exc:
        if str(exc).startswith("TRAINING_CACHE_"):
            raise
        code = "TRAINING_CACHE_UNWRITABLE" if isinstance(exc, (PermissionError, OSError)) else "TRAINING_CACHE_UNAVAILABLE"
        raise ValueError(f"{code}: 无法读写价值训练 JAX 缓存 {cache or '~/.cache/jax'}: {exc}") from exc
    finally:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except OSError as exc:
                raise ValueError(f"TRAINING_CACHE_UNWRITABLE: 无法清理价值训练 JAX 缓存探针 {probe}: {exc}") from exc


def inspect_stage(config, spec, factories, datasets):
    stage, requested = stage_request(spec)
    if stage is None:
        return None
    # Recheck the actual resolved objects even when a caller did not use the
    # normal resolver, and record only effective, stable values.
    apply_stage(config, spec)
    if stage == "value" and any(getattr(config, key) != value for key, value in requested.items()):
        raise ValueError("RECAP_STAGE_CONFIGURATION_MISMATCH: 实际价值验证参数与本次请求不一致")
    if stage == "offline_rl" and "acp_dropout" in requested and any(
            factory.acp_dropout != requested["acp_dropout"] for factory in factories):
        raise ValueError("RECAP_STAGE_CONFIGURATION_MISMATCH: 实际 ACP 条件 dropout 与本次请求不一致")
    if stage == STAGE_PROGRESS and any(getattr(config.model, key) != value for key, value in requested.items()):
        raise ValueError("RECAP_STAGE_CONFIGURATION_MISMATCH: 实际阶段进度参数与本次请求不一致")
    minimum = getattr(config.model, "value_min", -1.0)
    maximum = getattr(config.model, "value_max", 0.0)
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)) or not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 价值目标范围无效")
    rows = []
    for factory, dataset in zip(factories, datasets):
        if stage == "offline_rl" and (isinstance(factory.acp_dropout, bool) or
                not isinstance(factory.acp_dropout, (int, float)) or not math.isfinite(factory.acp_dropout) or
                not 0 <= factory.acp_dropout <= 1):
            raise ValueError("RECAP_STAGE_RECIPE_MISMATCH: 配方中的 ACP 条件 dropout 必须在 0 到 1 之间")
        mapping = _factory_mapping(factory)
        required = _mapping_columns(mapping)
        required_supervision = RECAP_SUPERVISION[stage]
        if mapping.get(required_supervision) != required_supervision:
            raise ValueError("RECAP_DATA_SCHEMA_UNSUPPORTED: 阶段监督字段必须保留标准列名")
        if stage == STAGE_PROGRESS:
            required = sorted(set(required) | set(STAGE_PROGRESS_ID_COLUMNS))
            supervised = [required_supervision]
        else:
            supervised = [key for key in ("value_target", "is_positive") if key in required]
        cached = require_cached_supervision(dataset, spec)
        summary = cached["supervision_summary"]
        if not set(supervised) <= set(summary):
            _base_validation_required(dataset)
        for checked in cached["checked_files"]:
            missing = sorted(set(required) - set(checked["columns"]))
            if missing:
                raise ValueError("RECAP_DATA_SCHEMA_INVALID: " + checked["path"]
                                 + " 的基础校验证据缺少阶段数据列：" + ", ".join(missing))
        if "value_target" in supervised and not minimum <= summary["value_target"]["min"] <= summary["value_target"]["max"] <= maximum:
            raise ValueError(f"RECAP_LABEL_INVALID: {dataset['path']} 的 value_target 超出 [{minimum}, {maximum}]")
        row = {"path": dataset["path"], "required_columns": required,
               "supervision_columns": supervised,
               "parquet_files": cached["parquet_files"], "checked_files": cached["checked_files"],
               "supervision_summary": {key: summary[key] for key in supervised},
               "scope": "reused bounded supervision evidence from dataset basic validation; no Parquet rows read at submission"}
        if stage == "offline_rl":
            row["acp_dropout"] = factory.acp_dropout
        rows.append(row)
    result = {"stage": stage, "model_type": type(config.model).__name__,
              "loader_type": type(config.weight_loader).__name__, "datasets": rows}
    if stage == "value":
        result["value_range"] = [minimum, maximum]
        result["validation"] = {key: getattr(config, key) for key in ("val_ratio", "eval_interval", "eval_batches")}
    if uses_torch_recap(spec):
        result.update(backend="pytorch", loader_type=RECAP_TORCH_LOADER,
                      source_sha256=recap_torch_source_hashes(spec, stage),
                      runtime_versions=recap_torch_runtime_versions())
    elif stage in {"value", STAGE_PROGRESS}:
        result["runtime_cache"] = _value_runtime_cache(spec)
    if stage == STAGE_PROGRESS:
        result["annotation"] = {"output": STAGE_PROGRESS_DATASET,
                                **{key: getattr(config.model, key) for key in STAGE_PROGRESS_KEYS}}
        if not uses_torch_recap(spec):
            result["source_sha256"] = stage_progress_source_hashes(spec)
    return result


def stage_progress_source_hashes(spec):
    candidates = [(f"src/openpi/{module}", f"openpi/{module}") for module in STAGE_PROGRESS_MODULES]
    candidates.append((STAGE_PROGRESS_SUPPORT_ENTRY,))
    return _recap_source_hashes(spec, candidates)


def recap_torch_source_hashes(spec, stage):
    return _recap_source_hashes(spec, [(f"src/openpi/{module}", f"openpi/{module}")
                                      for module in recap_torch_module_paths(stage)])


def recap_torch_runtime_versions():
    """Use the same patched Transformers check as PI0Pytorch, without creating a model."""
    try:
        versions = {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "transformers")}
        siglip = importlib.import_module("transformers.models.siglip.check")
        if siglip.check_whether_transformers_replace_is_installed_correctly() is not True:
            raise ValueError("Transformers patch check did not pass")
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError("RECAP_TORCH_RUNTIME_UNAVAILABLE: 请在所选训练环境安装 PyTorch、safetensors 和 OpenPI 的 Transformers 补丁") from exc
    return versions


def _recap_source_hashes(spec, candidates):
    root = Path(spec["code_dir"]).resolve(strict=True)
    result = {}
    for paths in candidates:
        source = next((root / name for name in paths if (root / name).is_file()), None)
        if source is None or not source.resolve().is_relative_to(root) or source.stat().st_size > 1024 * 1024:
            raise ValueError(f"RECAP_STAGE_SOURCE_UNAVAILABLE: 无法核验 ReCap 源码 {paths[0]}")
        result[source.relative_to(root).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result
