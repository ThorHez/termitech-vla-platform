"""Bounded local contract checks, executed only by the training interpreter.

This inspects configuration, metadata, file manifests and bounded decoded
samples. It neither initializes CUDA nor loads model tensors, constructs a
batched dataloader, or performs an SGD step.
"""

import ast
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path

if __package__:
    from .automatic_config import normalize_auto_config
    from .generated_config import generated_training_config
    from .composition_spec import COMPOSABLE_PROCESSOR, SLOTS, composition_digest, normalize_composition, validate_composition_profile_input, verify_composition_metadata
    from .cache_contract import inspect_training_cache
    from .data_processors import processor_identity, validate_processor_instance
    from .dataset_bindings import validate_binding_assets
    from .dataset_contracts import verify_registered_dataset_profile
    from .dataset_fps import resolve_registered_fps
    from .dataset_timestamps import inspect_dataset_timestamps
    from .dataset_samples import inspect_composed_samples_isolated
    from .entrypoints import read_entrypoint, validate_entrypoint
    from .recap_runtime import default_entrypoint, inspect_stage, stage_request, validate_stage_entrypoint
else:
    from automatic_config import normalize_auto_config
    from generated_config import generated_training_config
    from composition_spec import COMPOSABLE_PROCESSOR, SLOTS, composition_digest, normalize_composition, validate_composition_profile_input, verify_composition_metadata
    from cache_contract import inspect_training_cache
    from data_processors import processor_identity, validate_processor_instance
    from dataset_bindings import validate_binding_assets
    from dataset_contracts import verify_registered_dataset_profile
    from dataset_fps import resolve_registered_fps
    from dataset_timestamps import inspect_dataset_timestamps
    from dataset_samples import inspect_composed_samples_isolated
    from entrypoints import read_entrypoint, validate_entrypoint
    from recap_runtime import default_entrypoint, inspect_stage, stage_request, validate_stage_entrypoint


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_path(value, spec):
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError("INVALID_PATH: 训练输入必须使用本机绝对路径")
    path = Path(value).resolve(strict=True)
    roots = spec.get("allowed_roots")
    if roots is not None and (not roots or not any(path.is_relative_to(Path(root).resolve()) for root in roots)):
        raise ValueError(f"PATH_OUTSIDE_ROOTS: 路径超出 Agent 受管目录：{path}")
    return path


def require_lerobot_source(factory):
    """Alternative loader resources cannot bypass the registered LeRobot pair."""
    for owner in (factory, getattr(factory, "base_config", None)):
        for key in ("rlds_data_dir", "filter_dict_path"):
            value = owner.get(key) if isinstance(owner, dict) else getattr(owner, key, None)
            if value is not None and value != "":
                raise ValueError(f"RECIPE_INPUT_SOURCE_UNSUPPORTED: {key} 不能绕过本次 LeRobot 数据资产绑定；请使用兼容的数据处理配方")


def read_json(path, spec, *, limit=8 * 1024 * 1024):
    path = local_path(path, spec)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"INPUT_METADATA_INVALID: 元数据必须是受限大小的普通文件：{path}")
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"INPUT_METADATA_INVALID: 元数据文件过大：{path}")
    value = json.loads(content, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {v}")))
    if not isinstance(value, dict):
        raise ValueError(f"INPUT_METADATA_INVALID: 元数据应为 JSON 对象：{path}")
    return value, hashlib.sha256(content).hexdigest()


def file_manifest(directory, spec, *, max_entries=20000, timeout=10):
    """Hash path/size/mtime manifests, not many gigabytes of tensor contents."""
    root = local_path(directory, spec)
    if not root.is_dir():
        raise ValueError(f"INPUT_FORMAT: 应为本机目录：{root}")
    deadline = time.monotonic() + timeout
    rows = []
    pending = [(root, 0)]
    count = 0
    while pending:
        current, depth = pending.pop()
        if depth > 16:
            raise ValueError("INPUT_SCAN_LIMIT: 输入目录层级过深")
        with os.scandir(current) as entries:
            for entry in entries:
                count += 1
                if count > max_entries or time.monotonic() > deadline:
                    raise ValueError("INPUT_SCAN_LIMIT: 输入文件过多或扫描超时")
                if entry.is_symlink():
                    raise ValueError(f"INPUT_SYMLINK: 输入文件清单包含符号链接：{entry.path}")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    rows.append([str(Path(entry.path).relative_to(root)), info.st_size, info.st_mtime_ns])
                else:
                    raise ValueError(f"INPUT_FORMAT: 输入目录包含特殊文件：{entry.path}")
    rows.sort()
    return rows, sum(row[1] for row in rows), fingerprint(rows)


def training_entrypoint(config, spec):
    code = local_path(spec["code_dir"], spec)
    selected = validate_entrypoint(spec.get("entrypoint"))
    multi = hasattr(config.data, "datasets")
    relative = selected or default_entrypoint(config, spec)
    candidate = code / relative
    try:
        entry = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"ENTRYPOINT_UNAVAILABLE: 项目内的训练入口不存在或不可访问：{relative}") from exc
    if not entry.is_relative_to(code):
        raise ValueError("ENTRYPOINT_OUTSIDE_CODE_DIR: 训练入口及其软链接目标必须位于登记的项目目录内")
    if not entry.is_file() or entry.suffix != ".py" or not os.access(entry, os.R_OK):
        raise ValueError("ENTRYPOINT_INVALID: 训练入口必须是项目内可读取的 .py 普通文件")
    validate_stage_entrypoint(config, spec, entry)
    if multi and entry == (code / "scripts/train.py").resolve():
        raise ValueError("ENTRYPOINT_RECIPE_MISMATCH: 此配方包含多个数据集，请选择 scripts/train_multi_dataset.py 或兼容的混采入口")
    tree = ast.parse(read_entrypoint(entry))
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    init = any(isinstance(node, (ast.FunctionDef, ast.Assign, ast.ImportFrom)) and
               (getattr(node, "name", None) == "init_wandb" or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "init_wandb" for t in node.targets) or
                isinstance(node, ast.ImportFrom) and any((a.asname or a.name) == "init_wandb" for a in node.names))
               for node in tree.body)
    if main is None or not init:
        raise ValueError("ENTRYPOINT_UNSUPPORTED: 训练入口需要 main(config) 和 init_wandb 接口")
    arguments = main.args
    positional = arguments.posonlyargs + arguments.args
    if ((not positional and arguments.vararg is None)
            or len(positional) - len(arguments.defaults) > 1
            or any(default is None for default in arguments.kw_defaults)):
        raise ValueError("ENTRYPOINT_UNSUPPORTED: main 必须能接收一个 config 位置参数，且不能要求其他必填参数")
    return entry


def registered_task_features(path, info, spec, registered_profile, *, selected_composition=None):
    """Prove LeRobot v2's task_index -> task mapping from its real task file."""
    declared = registered_profile.get("feature_keys", []) if isinstance(registered_profile, dict) else []
    automatic_task = False
    if selected_composition is not None:
        composition = normalize_composition(selected_composition)
        declared = [*declared, composition["prompt_key"]]
        claimed = {row["key"] for row in composition["state_fields"] + composition["cameras"]}
        claimed.update([composition["action_key"], *composition["aux_keys"]])
        automatic_task = not composition["prompt_key"] and not any(
            key == "task" or key.startswith("task/") for key in claimed)
    elif isinstance(registered_profile, dict) and isinstance(registered_profile.get("processing"), dict):
        composition = normalize_composition(registered_profile["processing"])
        declared = [*declared, composition["prompt_key"]]
        claimed = {row["key"] for row in composition["state_fields"] + composition["cameras"]}
        claimed.update([composition["action_key"], *composition["aux_keys"]])
        automatic_task = not composition["prompt_key"] and not any(
            key == "task" or key.startswith("task/") for key in claimed)
    task_declared = "task" in declared
    features = info.get("features", {})
    if not (task_declared or automatic_task) or "task" in features:
        return {}
    if not isinstance(features.get("task_index"), dict):
        return {}
    try:
        source = local_path(path / "meta/tasks.jsonl", spec)
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            stat_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(stat_info.st_mode) or stat_info.st_size > 1024 * 1024:
                raise ValueError("tasks.jsonl must be a bounded regular file")
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("tasks.jsonl exceeds size limit")
        lines = content.decode("utf-8").splitlines()
        if not 1 <= len(lines) <= 10000:
            raise ValueError("tasks.jsonl row count is invalid")
        indices = set()
        for line in lines:
            if len(line) > 8192:
                raise ValueError("task row exceeds size limit")
            row = json.loads(line)
            task, index = row.get("task"), row.get("task_index")
            if (not isinstance(row, dict) or type(index) is not int or index < 0 or index in indices
                    or not isinstance(task, str) or not 1 <= len(task) <= 2000 or not task.strip()
                    or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in task)):
                raise ValueError("task mapping row is invalid")
            indices.add(index)
        return {"derived_features": ["task"], "task_metadata_sha256": hashlib.sha256(content).hexdigest()}
    except FileNotFoundError as exc:
        if automatic_task and not task_declared:
            return {}
        raise ValueError("DATASET_PROFILE_UNVERIFIABLE: 登记了派生 task 字段，但无法从真实 meta/tasks.jsonl 核实 task_index 到任务文本的映射；首版暂不支持 tasks.parquet") from exc
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError, RecursionError) as exc:
        raise ValueError("DATASET_PROFILE_UNVERIFIABLE: 登记了派生 task 字段，但无法从真实 meta/tasks.jsonl 核实 task_index 到任务文本的映射；首版暂不支持 tasks.parquet") from exc


def dataset_contract(factory, spec, registered_profile=None, *, model_config=None, binding=None, checks=None):
    path = local_path(factory.repo_id, spec)
    info, metadata_sha = read_json(path / "meta" / "info.json", spec)
    for key in ("total_frames", "total_episodes"):
        if type(info.get(key)) is not int or info[key] <= 0:
            raise ValueError(f"DATASET_FORMAT: {path} 的 {key} 必须大于零")
    features = info.get("features")
    composition = normalize_composition(factory.composition) if type(factory).__name__ == COMPOSABLE_PROCESSOR else None
    action_key = composition["action_key"] if composition else "actions"
    if not isinstance(features, dict) or not isinstance(features.get(action_key), dict):
        raise ValueError(f"DATASET_FORMAT: {path} 缺少 actions 特征元数据")
    actions = features[action_key].get("shape")
    if not isinstance(actions, list) or not actions or any(type(v) is not int or v <= 0 for v in actions):
        raise ValueError(f"DATASET_FORMAT: {path} 的动作形状无效")
    derived = (registered_task_features(path, info, spec, registered_profile, selected_composition=composition)
               if registered_profile is not None or composition is not None else {})
    warnings = []
    declared = ({"registered_profile": verify_registered_dataset_profile(registered_profile, info,
                  derived_features=derived.get("derived_features", ()), selected_composition=composition,
                  warnings=warnings)} if registered_profile is not None else {})
    warnings = [{**warning, "dataset_path": str(path),
                 "message": f"数据集 {path}：{warning['message']}"} for warning in warnings]
    if checks is not None:
        # Preserve completed advisory checks even if a later contract for this
        # same dataset raises; the following hard failure remains authoritative.
        checks.extend({"name": "dataset_fps", "ok": True, "severity": "warning",
                       "code": warning["code"], "message": warning["message"]} for warning in warnings)
    composed = {}
    if composition is not None:
        processor = binding.get("processor") if isinstance(binding, dict) else None
        expected = (processor.get("parameters", {}).get("composition")
                    if isinstance(processor, dict) and processor.get("name") == COMPOSABLE_PROCESSOR else None)
        if binding is None and isinstance(registered_profile, dict):
            # Compatibility for callers holding only a historical frozen asset.
            # Normal run planning always supplies the explicit dataset binding.
            expected = registered_profile.get("processing")
        if expected is None or composition_digest(expected) != composition_digest(composition):
            raise ValueError("COMPOSITION_BINDING_MISMATCH: 实际组合处理配置与本次冻结的训练数据绑定不一致")
        validate_composition_profile_input(registered_profile, composition)
        model_type = type(model_config).__name__
        if model_type not in {"Pi0Config", "Pi0GripperConfig", "Pi0ValueConfig"}:
            raise ValueError("COMPOSITION_MODEL_UNSUPPORTED: 组合处理首版支持标准 Pi0/Pi05、Value 和 Gripper 模型，MEM/Hybrid 使用原处理器")
        horizon, action_dim = getattr(model_config, "action_horizon", None), getattr(model_config, "action_dim", None)
        if type(horizon) is not int or horizon <= 0 or type(action_dim) is not int or action_dim <= 0:
            raise ValueError("COMPOSITION_MODEL_UNSUPPORTED: 模型动作维度和预测长度无效")
        verify_composition_metadata(composition, info, derived_features=derived.get("derived_features", ()), action_horizon=horizon)
        if sum(field["width"] for field in composition["action_layout"]) > action_dim:
            raise ValueError("COMPOSITION_MODEL_UNSUPPORTED: 数据动作宽度超过模型 action_dim")
        if (getattr(model_config, "pi05", False) is not True and
                composition["state_history"] * sum(field["width"] for field in composition["state_fields"]) > action_dim):
            raise ValueError("COMPOSITION_MODEL_UNSUPPORTED: Pi0 展平状态宽度超过模型 action_dim，禁止截断")
        stage, _ = stage_request(spec)
        stage = stage or "sft"
        if getattr(factory, "training_stage", None) != stage or (model_type == "Pi0ValueConfig") != (stage == "value"):
            raise ValueError("COMPOSITION_STAGE_UNSUPPORTED: 组合处理的模型/阶段与任务不一致")
        required_aux = {"value": "value_target", "offline_rl": "is_positive"}.get(stage)
        if required_aux and required_aux not in composition["aux_keys"]:
            raise ValueError("COMPOSITION_STAGE_UNSUPPORTED: 组合处理缺少阶段监督字段 " + required_aux)
        describe = getattr(factory, "processing_contract", None)
        if not callable(describe):
            raise ValueError("COMPOSITION_SOURCE_UNAVAILABLE: 训练处理器缺少轻量预检接口，请更新训练代码")
        runtime = describe(model_config)
        depth_keys = list(getattr(model_config, "depth_keys", None) or ())
        image_keys = getattr(model_config, "image_keys", None)
        image_keys = list(image_keys) if image_keys is not None else [key for key in SLOTS if key not in depth_keys]
        expected_runtime = {"training_stage": stage, "model_type": model_type,
            "pi05": getattr(model_config, "pi05", False) is True, "action_dim": action_dim, "action_horizon": horizon,
            "raw_state_dim": sum(field["width"] for field in composition["state_fields"]),
            "raw_action_dim": sum(field["width"] for field in composition["action_layout"]),
            "state_history": composition["state_history"], "normalization": composition["normalization"],
            "action_loss_mask": [1.] * sum(field["width"] for field in composition["action_layout"]) +
                                [0.] * (action_dim - sum(field["width"] for field in composition["action_layout"])),
            "acp_dropout": float(getattr(factory, "acp_dropout", .1)),
            "image_keys": image_keys, "depth_keys": depth_keys,
            "action_sequence_keys": [action_key] if len(actions) == 1 else [], "action_metadata_sha256": metadata_sha}
        if not isinstance(runtime, dict) or fingerprint(runtime) != fingerprint(expected_runtime):
            raise ValueError("COMPOSITION_RUNTIME_MISMATCH: 组合处理预检的模型/动作采样或元信息与登记证据不一致，请重新预检")
        source = next((Path(spec["code_dir"]) / relative for relative in
                       ("src/openpi/training/composed_processing.py", "openpi/training/composed_processing.py")
                       if (Path(spec["code_dir"]) / relative).is_file()), None)
        if source is None:
            raise ValueError("COMPOSITION_SOURCE_UNAVAILABLE: 训练代码缺少组合处理实现，请更新训练代码")
        source = local_path(source, spec)
        root = Path(spec["code_dir"]).resolve(strict=True)
        if not source.is_relative_to(root) or source.stat().st_size > 1024 * 1024:
            raise ValueError("COMPOSITION_SOURCE_UNAVAILABLE: 组合处理实现超出登记代码范围或大小限制")
        contents = source.read_bytes()
        if len(contents) > 1024 * 1024:
            raise ValueError("COMPOSITION_SOURCE_UNAVAILABLE: 组合处理实现超过读取上限")
        composed = {"composition": {"schema_version": 1, "spec": composition,
                    "digest": composition_digest(composition), **runtime,
                    "source": {"path": source.relative_to(root).as_posix(), "sha256": hashlib.sha256(contents).hexdigest()}}}
    rows, data_bytes, manifest = file_manifest(path / "data", spec)
    if not any(row[0].endswith(".parquet") and row[1] > 0 for row in rows):
        raise ValueError(f"DATASET_FORMAT: {path} 缺少非空 parquet 数据文件")
    video_keys = [key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"]
    video_evidence = {}
    if video_keys:
        video_rows, video_bytes, video_manifest = file_manifest(path / "videos", spec)
        available = {"videos/" + row[0] for row in video_rows if row[1] > 0}
        pattern = info.get("video_path")
        chunk_size = info.get("chunks_size", 1000)
        if not isinstance(pattern, str) or type(chunk_size) is not int or chunk_size <= 0 or info["total_episodes"] * len(video_keys) > 20000:
            raise ValueError("DATASET_VIDEO_UNSUPPORTED: 视频路径模板或数量超出预检范围")
        try:
            for key in video_keys:
                for episode in range(info["total_episodes"]):
                    relative = pattern.format(video_key=key, episode_index=episode, episode_chunk=episode // chunk_size)
                    candidate = local_path(path / relative, spec)
                    if not candidate.is_relative_to(path / "videos") or relative not in available:
                        raise ValueError(f"DATASET_VIDEO_MISSING: 缺少数据集视频：{relative}")
        except (KeyError, IndexError) as exc:
            raise ValueError("DATASET_VIDEO_UNSUPPORTED: 无法解析数据集视频路径模板") from exc
        video_evidence = {"video_bytes": video_bytes, "video_manifest_digest": video_manifest}
    return {"path": str(path), "metadata_sha256": metadata_sha, "frames": info["total_frames"],
            "episodes": info["total_episodes"], "factory": type(factory).__name__,
            "effective_fps": registered_profile["fps"] if isinstance(registered_profile, dict) and "fps" in registered_profile else info.get("fps"),
            "fps_source": "registered" if isinstance(registered_profile, dict) and "fps" in registered_profile else "metadata",
            "actions_shape": actions, "data_bytes": data_bytes, "data_manifest_digest": manifest,
            **({"warnings": warnings} if warnings else {}),
            **video_evidence, **derived, **declared, **composed}


def check_composed_mixture(data_factory, datasets):
    compositions = [data["composition"] for data in datasets if "composition" in data]
    if not compositions or not hasattr(data_factory, "datasets"):
        return
    if getattr(data_factory, "use_merged_norm_stats", None) is not False:
        raise ValueError("COMPOSITION_MIXTURE_UNSUPPORTED: 组合数据集必须分别使用所绑定的归一化统计，不能跨布局合并")
    target = getattr(data_factory, "state_pad_dim", None)
    for composition in compositions:
        actual = (composition["state_history"] * composition["raw_state_dim"]
                  if composition["pi05"] else composition["action_dim"])
        if (type(target) is not int or target < actual or
                not composition["pi05"] and target != composition["action_dim"]):
            raise ValueError(f"COMPOSITION_STATE_PADDING_INVALID: 配方 data.state_pad_dim={target} 与组合输出状态宽度 {actual} 不兼容；Pi05 至少需要 {actual}，Pi0 必须等于模型 action_dim")


def norm_contract(factory, spec):
    assets = factory.assets
    asset_dir = assets.get("assets_dir") if isinstance(assets, dict) else getattr(assets, "assets_dir", None)
    asset_id = assets.get("asset_id") if isinstance(assets, dict) else getattr(assets, "asset_id", None)
    if not asset_dir or not isinstance(asset_id, str):
        raise ValueError("NORM_MISSING: 每个数据集必须显式指定本机归一化统计目录")
    norm_path = local_path(Path(asset_dir) / asset_id / "norm_stats.json", spec)
    document, sha = read_json(norm_path, spec)
    stats = document.get("norm_stats")
    composition = normalize_composition(factory.composition) if type(factory).__name__ == COMPOSABLE_PROCESSOR else None
    if not isinstance(stats, dict) or not all(key in stats for key in ("state", "actions")):
        raise ValueError(f"NORM_FORMAT: {norm_path} 必须包含 state 和 actions 统计")
    for key in ("state", "actions"):
        values = stats[key]
        if not isinstance(values, dict) or not all(values.get(k) is not None for k in ("mean", "std")):
            raise ValueError(f"NORM_FORMAT: {norm_path} 缺少 {key} mean/std")
        lengths = []
        for field in ("mean", "std", "q01", "q99", "min", "max"):
            array = values.get(field)
            if array is None:
                continue
            if (not isinstance(array, list) or not array or len(array) > 4096 or
                    any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in array)):
                raise ValueError(f"NORM_FORMAT: {norm_path} 的 {key}.{field} 应为有限数值数组")
            if field == "std" and any(v < 0 for v in array):
                raise ValueError(f"NORM_FORMAT: {norm_path} 的标准差不能为负数")
            lengths.append(len(array))
        if len(set(lengths)) != 1:
            raise ValueError(f"NORM_FORMAT: {norm_path} 的 {key} 统计维度不一致")
        if composition is not None:
            expected = sum(field["width"] for field in composition["state_fields" if key == "state" else "action_layout"])
            if lengths[0] != expected:
                raise ValueError(f"COMPOSITION_NORM_MISMATCH: {key} 归一化统计宽度与组合拼接顺序的总宽度不一致")
        base = getattr(factory, "base_config", None)
        if (getattr(base, "use_quantile_norm", False) or composition is not None and composition["normalization"] == "quantile") and not all(values.get(k) is not None for k in ("q01", "q99")):
            raise ValueError(f"NORM_FORMAT: {norm_path} 缺少分位数归一化统计")
    return {"norm_path": str(norm_path), "norm_sha256": sha}


def model_contract(spec):
    checkpoint = local_path(spec["inputs"]["model"]["path"], spec)
    params = checkpoint / "params" if (checkpoint / "params").is_dir() else checkpoint
    params = local_path(params, spec)
    rows, parameter_bytes, manifest = file_manifest(params, spec)
    names = {row[0] for row in rows}
    ocdbt = any(name == "manifest.ocdbt" or name.endswith("/manifest.ocdbt") for name in names)
    tensor_data = any(row[1] > 0 and (row[0].startswith("d/") or
        re.fullmatch(r"ocdbt\.process_[^/]+/d/.+", row[0])) for row in rows)
    # Older Orbax uses Zarr directories with array metadata and numeric chunks.
    zarr_arrays = [str(Path(name).parent) for name in names if name.endswith("/.zarray")]
    zarr = any(row[1] > 0 and str(Path(row[0]).parent) in zarr_arrays and
               re.fullmatch(r"\d+(?:\.\d+)*", Path(row[0]).name) for row in rows)
    if "_METADATA" not in names or not (ocdbt and tensor_data or zarr):
        raise ValueError("MODEL_FORMAT: 模型需要完整的 Orbax params 元数据、数组清单和非空权重分片")
    metadata = []
    for name in ("_METADATA", "_sharding"):
        if (params / name).is_file():
            _, sha = read_json(params / name, spec)
            metadata.append({"path": str(params / name), "sha256": sha})
    total_bytes = parameter_bytes
    if params != checkpoint:
        _, total_bytes, _ = file_manifest(checkpoint, spec)
    return {"path": str(params), "checkpoint_path": str(checkpoint), "metadata": metadata,
            "parameter_bytes": parameter_bytes, "file_manifest_digest": manifest}, total_bytes


def storage_estimate(config, model, reference_bytes):
    # Existing train_state provides the best local estimate. A params-only input
    # needs a conservative allowance for optimizer state and EMA as well.
    parameter_bytes = model["parameter_bytes"]
    checkpoint_bytes = max(reference_bytes, parameter_bytes * 5)
    interval = getattr(config, "save_interval", 1000)
    keep_period = getattr(config, "keep_period", None)
    if type(interval) is not int or interval <= 0:
        raise ValueError("RECIPE_UNSUPPORTED: save_interval 必须是正整数")
    last = config.num_train_steps - 1
    saves = last // interval + (1 if last % interval else 0)
    saves = max(saves, 1)
    pinned = last // math.lcm(interval, keep_period) if type(keep_period) is int and keep_period > 0 else 0
    simultaneous = min(saves, pinned + 2)
    required = math.ceil(checkpoint_bytes * simultaneous * 1.1) + 1024 ** 3
    return {"required_bytes": required, "reference_checkpoint_bytes": reference_bytes,
            "model_params_bytes": parameter_bytes, "estimated_checkpoint_bytes": checkpoint_bytes,
            "checkpoint_copies": simultaneous, "headroom_bytes": 1024 ** 3}


def inspect_training_config(config, spec, *, checks, fps_runtime=None):
    """Return deterministic evidence shared by preflight and actual launch."""
    def passed(name, message):
        checks.append({"name": name, "ok": True, "message": message})

    bindings = validate_binding_assets(spec.get("dataset_bindings"), spec["inputs"])
    children = getattr(config.data, "datasets", None)
    multi = children is not None
    if multi and bindings is None:
        raise ValueError("DATASET_BINDINGS_REQUIRED: 多数据集配方必须逐组选择数据集和归一化资产，不能使用配方中的默认目录；请重新配置数据绑定并预检")
    factories = children if multi else [config.data]
    if not isinstance(factories, (list, tuple)) or not 1 <= len(factories) <= 64:
        raise ValueError("RECIPE_UNSUPPORTED: 配方的数据集数量无效")
    weights = getattr(config.data, "weights", None) if multi else None
    if weights is not None and (len(weights) != len(factories) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights) or not any(weights)):
        raise ValueError("RECIPE_UNSUPPORTED: 混采权重需要非负有限数值，并与数据集数量一致")
    if multi and (type(getattr(config.data, "state_pad_dim", None)) is not int or config.data.state_pad_dim <= 0):
        raise ValueError("RECIPE_UNSUPPORTED: 多数据集配方需要正整数 state_pad_dim")
    if spec["gpu_count"] <= 0 or config.batch_size % spec["gpu_count"]:
        raise ValueError("INVALID_BATCH_SIZE: batch_size 必须整除申请的 GPU 数量")
    if bindings is not None and (len(factories) != len(bindings) or (multi and weights != [row["weight"] for row in bindings])):
        raise ValueError("DATASET_BINDINGS_INVALID: 实际训练的数据集数量或权重与本次绑定不一致")
    roles = [(f"dataset_{i}", f"norm_{i}") for i in range(len(bindings))] if bindings is not None else [("dataset", "norm")]
    for factory, (dataset_role, _) in zip(factories, roles):
        require_lerobot_source(factory)
        if local_path(factory.repo_id, spec) != local_path(spec["inputs"][dataset_role]["path"], spec):
            raise ValueError("DATASET_BINDINGS_INVALID: 实际训练的数据目录与所选资产不一致")
    passed("recipe_configuration", f"已解析配方 {spec['recipe']}，{len(factories)} 个数据集；batch {config.batch_size} / {spec['gpu_count']} GPU")
    datasets = []
    for index, (factory, (dataset_role, _)) in enumerate(zip(factories, roles)):
        data = dataset_contract(factory, spec, spec["inputs"][dataset_role].get("dataset_profile"),
                                model_config=getattr(config, "model", None),
                                binding=bindings[index] if bindings is not None else None, checks=checks)
        datasets.append(data)
    check_composed_mixture(config.data, datasets)
    passed("data_contract", "已核对所有数据集的 LeRobot 元数据与 parquet 文件：" + "、".join(d["path"] for d in datasets))
    for data in datasets:
        try:
            data["timestamp_contract"] = inspect_dataset_timestamps(data, spec)
        except ValueError as exc:
            checks.append({"name": "dataset_timestamps", "ok": False, "message": str(exc)[:2000]})
            raise
    passed("dataset_timestamps", "已按本次生效 FPS 和训练加载器容差，逐帧核对所有数据集的真实时间戳")
    for data, factory in zip(datasets, factories):
        data.update(norm_contract(factory, spec))
    for i, (data, (dataset_role, norm_role)) in enumerate(zip(datasets, roles)):
        norm = local_path(spec["inputs"][norm_role]["path"], spec)
        data["norm_asset_path"] = str(norm)
        norm = norm / "norm_stats.json" if norm.is_dir() else norm
        if data["norm_path"] != str(norm.resolve(strict=True)):
            raise ValueError("DATASET_BINDINGS_INVALID: 实际训练的数据与归一化参数配对不一致")
        data["dataset_asset_id"] = spec["inputs"][dataset_role].get("id")
        data["norm_asset_id"] = spec["inputs"][norm_role].get("id")
        binding = bindings[i] if bindings is not None else {"recipe_index": 0}
        if "processor" in binding:
            data["processor"] = validate_processor_instance(factories[i], binding["processor"])
            data["processor_identity"] = processor_identity(factories[i], spec["code_dir"], binding["processor"]["name"])
        else:
            data["recipe_index"] = binding["recipe_index"]
    passed("norm_contract", "所有数据集归一化统计可读，数值有限且维度一致")
    sampled = 0
    for data, factory in zip(datasets, factories):
        try:
            evidence = inspect_composed_samples_isolated(factory, config, spec, data)
            expected_status = "verified" if "composition" in data else "not_applicable"
            if not isinstance(evidence, dict) or evidence.get("status") != expected_status:
                raise ValueError("DATASET_SAMPLE_CHECK_UNAVAILABLE: 真实样本检查未返回完整证据")
            if expected_status == "verified":
                # Decoder reads must not reuse evidence for files or transforms
                # changed while the bounded sample subprocess was running.
                path = Path(data["path"])
                _, metadata_sha = read_json(path / "meta/info.json", spec)
                _, _, manifest = file_manifest(path / "data", spec)
                source = Path(spec["code_dir"]) / data["composition"]["source"]["path"]
                unchanged = (metadata_sha == data["metadata_sha256"]
                             and manifest == data["data_manifest_digest"]
                             and hashlib.sha256(source.read_bytes()).hexdigest() == data["composition"]["source"]["sha256"])
                if "video_manifest_digest" in data:
                    unchanged = unchanged and file_manifest(path / "videos", spec)[2] == data["video_manifest_digest"]
                current_norm = norm_contract(factory, spec)
                unchanged = unchanged and current_norm == {key: data[key] for key in current_norm}
                if not unchanged:
                    raise ValueError("DATASET_SAMPLE_INVALID: 样本检查期间数据、归一化参数或处理源码已改变，请重新预检")
            data["sample_probe"] = evidence
            sampled += expected_status == "verified"
        except ValueError as exc:
            checks.append({"name": "dataset_samples", "ok": False, "message": str(exc)[:2000]})
            raise
    passed("dataset_samples", (f"已对 {sampled} 组组合数据执行有限真实样本解码、重排、数据处理和模型视觉输入校验；未加载模型或执行训练"
                               if sampled else "本配方未使用组合处理器，组合样本检查不适用；其他检查仍按各自范围执行"))
    stage, stage_config = stage_request(spec)
    stage_contract = inspect_stage(config, spec, factories, datasets)
    if stage_contract is not None:
        passed("recap_stage", "已核对 ReCap 阶段、模型和实际 parquet 监督字段：" + stage)
    model, reference = model_contract(spec)
    loader_path = getattr(config.weight_loader, "params_path", config.weight_loader)
    if not isinstance(loader_path, (str, Path)) or Path(loader_path).resolve() != Path(model["path"]):
        raise ValueError("MODEL_BINDING_MISMATCH: 实际初始化权重目录与所选模型资产不一致")
    model["asset_id"] = spec["inputs"]["model"].get("id")
    passed("model_checkpoint", f"Orbax 权重元数据与文件清单有效：{model['path']}")
    entry = training_entrypoint(config, spec)
    passed("training_entrypoint", str(entry))
    roots = spec.get("allowed_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("INVALID_PATH: 自动预检缺少 Agent 受管目录")
    cache = inspect_training_cache(datasets, spec)
    passed("training_cache", f"数据集缓存可写，已核对 {len(cache['builders'])} 组 parquet builder 文件锁：{cache['datasets_cache']}")
    return {"recipe": spec["recipe"], "input_authority": "registered_assets", "recipe_definition_digest": (spec.get("recipe_definition") or {}).get("digest"),
            "dataset_fps_policy": resolve_registered_fps(spec),
            "dataset_fps_runtime": fps_runtime if fps_runtime and fps_runtime.get("installed") is True else None,
            "generated_config": generated_training_config(config, spec, entrypoint=entry),
            **({"auto_config": normalize_auto_config(spec["auto_config"])} if spec.get("auto_config") is not None else {}),
            **({"training_stage": stage, "stage_config": stage_config, "recap_stage_contract": stage_contract}
               if stage is not None else {}),
            **({"dataset_bindings": bindings} if bindings is not None else {}),
            "requested_entrypoint": validate_entrypoint(spec.get("entrypoint")),
            "requested_inputs": spec.get("requested_inputs", {role: asset["path"] for role, asset in spec["inputs"].items()}),
            "inputs": {role: str(local_path(asset["path"], spec)) for role, asset in spec["inputs"].items()},
            "parameters": spec["parameters"], "gpu_count": spec["gpu_count"], "entrypoint": str(entry),
            "entrypoint_sha256": hashlib.sha256(read_entrypoint(entry)).hexdigest(),
            "adapter_sha256": fingerprint({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                            for name in ("openpi_entry.py", "training_plan.py", "automatic_config.py", "generated_config.py", "recipe_definitions.py", "runtime_options.py", "entrypoints.py", "dataset_bindings.py", "dataset_contracts.py", "dataset_fps.py", "dataset_timestamps.py", "dataset_samples.py", "composition_spec.py", "data_processors.py", "cache_contract.py", "recap.py", "recap_runtime.py")}),
            "datasets": datasets, "weights": [row["weight"] for row in bindings] if bindings is not None else weights,
            "state_pad_dim": getattr(config.data, "state_pad_dim", None),
            "use_merged_norm_stats": bool(getattr(config.data, "use_merged_norm_stats", False)),
            "model": model, "storage": storage_estimate(config, model, reference), "training_cache": cache,
            "evidence_scope": "metadata hashes, file path/size/mtime manifests, all source timestamp/episode rows, and bounded decoded samples for composable processing; model tensors and training steps are not loaded"}
