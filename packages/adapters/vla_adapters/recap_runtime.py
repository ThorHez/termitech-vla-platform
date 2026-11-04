"""ReCap stage checks in the registered interpreter, without model creation.

Value training consumes targets; ACP consumes previously annotated labels.  The
annotation program mutates datasets and is deliberately not run by this adapter.
"""

import dataclasses
import math
import os
import tempfile
import time
from pathlib import Path

if __package__:
    from .composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from .cache_contract import _managed_path
    from .dataset_validation_cache import get_or_validate, lookup_cached
    from .recap import validate_training_stage
else:
    from composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from cache_contract import _managed_path
    from dataset_validation_cache import get_or_validate, lookup_cached
    from recap import validate_training_stage


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
    if model_type == "Pi0ValueConfig" or any(
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
    multi = hasattr(config.data, "datasets")
    if stage == "value":
        return "scripts/train_value_multi_dataset.py" if multi else "scripts/train_value.py"
    return "scripts/train_multi_dataset.py" if multi else "scripts/train.py"


def validate_stage_entrypoint(config, spec, entry):
    stage, _ = stage_request(spec)
    if stage is None:
        return
    relative = entry.relative_to(Path(spec["code_dir"]).resolve()).as_posix()
    policy = {"scripts/train.py", "scripts/train_multi_dataset.py", "scripts/train_multi_dataset_hybrid.py"}
    value = {"scripts/train_value.py", "scripts/train_value_multi_dataset.py"}
    if relative in {"scripts/train_advantage.py", "scripts/train_multi_dataset_hybrid.py"} or stage == "value" and relative in policy or stage == "offline_rl" and relative in value:
        raise ValueError("RECAP_STAGE_ENTRYPOINT_MISMATCH: 训练入口与本次 ReCap 阶段不匹配")
    if stage == "value" and hasattr(config.data, "datasets") and relative == "scripts/train_value.py":
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


def prepare_source_supervision(data, spec, info):
    """Measure optional ReCap labels only during the dataset's basic validation."""
    features = info.get("features", {})
    supervised = [key for key in ("value_target", "is_positive") if key in features]
    if not supervised:
        return None

    def validate():
        return {"schema_version": 1, "status": "verified", "path": data["path"],
                **_parquet_evidence(data["path"], supervised, supervised, stage="source",
                                    value_min=None, value_max=None)}

    return get_or_validate("source", data, spec, validate, context=_SUPERVISION_CONTEXT)


def _base_validation_required(dataset):
    raise ValueError("DATASET_BASE_VALIDATION_REQUIRED: 数据集 " + str(dataset.get("path", ""))[:512]
                     + " 的 ReCap 监督基础校验证据缺失或已失效，请先在数据集配置中执行基础校验，再重新提交训练")


def require_cached_supervision(dataset, spec):
    """Read an existing source-bound label receipt; never scan or load on a miss."""
    evidence = lookup_cached("source", dataset, spec, context=_SUPERVISION_CONTEXT)
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
            or evidence.get("status") != "verified" or evidence.get("path") != dataset.get("path")):
        _base_validation_required(dataset)
    checked = evidence.get("checked_files")
    summary = evidence.get("supervision_summary")
    if (type(evidence.get("parquet_files")) is not int or not isinstance(checked, list)
            or not 1 <= len(checked) <= min(16, evidence["parquet_files"])
            or not isinstance(summary, dict) or not summary or set(summary) - {"value_target", "is_positive"}):
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
                                              or measured["min"] not in (0, 1) or measured["max"] not in (0, 1)))):
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
        required_supervision = "value_target" if stage == "value" else "is_positive"
        if mapping.get(required_supervision) != required_supervision:
            raise ValueError("RECAP_DATA_SCHEMA_UNSUPPORTED: 阶段监督字段必须保留标准列名")
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
        result["runtime_cache"] = _value_runtime_cache(spec)
    return result
