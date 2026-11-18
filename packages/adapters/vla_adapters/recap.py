"""Public ReCap stage configuration, shared across control and Agent boundaries."""

import copy
import math
from pathlib import PurePosixPath
import re

if __package__:
    from .composition_spec import STAGE_PROGRESS_COLUMN, stage_progress_source_key
else:
    from composition_spec import STAGE_PROGRESS_COLUMN, stage_progress_source_key


STAGE_PROGRESS = "stage_progress"
STAGE_ANNOTATION = "stage_annotation"
VALUE_ANNOTATION = "value_annotation"
ANNOTATION_STAGES = {STAGE_ANNOTATION, VALUE_ANNOTATION}
STAGE_ADVANTAGE_STAGES = {STAGE_PROGRESS, STAGE_ANNOTATION}
STAGE_PROGRESS_ENTRY = "scripts/acp/train_stage_progress.py"
STAGE_ANNOTATION_ENTRY = "scripts/acp/lerobot_stage_progress_infer_pytorch.py"
VALUE_ANNOTATION_ENTRY = "scripts/acp/lerobot_value_infer_pytorch.py"
VALUE_ANNOTATION_DATASET = "value_advantage_dataset"
VALUE_ANNOTATION_COLUMNS = ("predicted_value", "advantage", "is_positive")
VALUE_ANNOTATION_DEFAULTS = {"n_step": 50, "positive_ratio": 0.3}
RECAP_TORCH_ENTRY = "scripts/acp/train_recap_pytorch.py"
ACP_STAGE = "offline_rl"
ACP_ENTRYPOINT = "scripts/train.py"
ACP_MULTI_ENTRYPOINT = "scripts/train_multi_dataset.py"
ACP_WEIGHT_LOADERS = frozenset({"CheckpointWeightLoader", "CheckpointWeightLoaderWithGripperHead",
    "CheckpointWeightLoaderWithDiscreteHead", "CheckpointWeightLoaderIgnoreDiscreteHead", "CheckpointWeightLoaderIgnoreGripperHead"})
RECAP_TORCH_LOADER = "safetensors"
RECAP_TORCH_MODULES = ("models_pytorch/recap.py", "models_pytorch/pi0_pytorch.py",
                       "models_pytorch/gemma_pytorch.py", "models_pytorch/preprocessing_pytorch.py",
                       "training/recap_pytorch.py")
STAGE_PROGRESS_DATASET = "stage_advantage_dataset"
STAGE_PROGRESS_MODEL = "Pi0StageProgressConfig"
STAGE_PROGRESS_LOADER = "CheckpointWeightLoaderWithStageProgressHead"
STAGE_PROGRESS_KEYS = ("n_step", "positive_ratio", "stage_count")
STAGE_PROGRESS_ID_COLUMNS = ("episode_index", "frame_index", "index", "task_index")
STAGE_PROGRESS_OUTPUT_COLUMNS = ("absolute_value", "absolute_advantage", "relative_advantage", "is_positive")
STAGE_PROGRESS_MODULES = ("models/pi0_stage_progress.py", "training/stage_progress.py",
                          "training/stage_progress_labels.py", "training/weight_loaders.py")
STAGE_PROGRESS_SUPPORT_ENTRY = "scripts/acp/train_value.py"
RECAP_SUPERVISION = {"value": "value_target", VALUE_ANNOTATION: "value_target", "offline_rl": "is_positive",
                     STAGE_PROGRESS: STAGE_PROGRESS_COLUMN, STAGE_ANNOTATION: STAGE_PROGRESS_COLUMN}
ESTIMATOR_MODELS = {"value": "Pi0ValueConfig", VALUE_ANNOTATION: "Pi0ValueConfig",
                    STAGE_PROGRESS: STAGE_PROGRESS_MODEL, STAGE_ANNOTATION: STAGE_PROGRESS_MODEL}

ESTIMATOR_VALIDATION_PARAMETERS = {
    "val_ratio": {"type": "number", "label": "验证集比例", "default": 0.1, "min": 0, "max": 0.9},
    "eval_interval": {"type": "integer", "label": "验证间隔（步）", "default": 1000, "min": 1, "max": 1000000},
    "eval_batches": {"type": "integer", "label": "每次验证批数", "default": 10, "min": 1, "max": 10000},
}


STAGES = (
    {"id": "value", "name": "价值函数训练", "runtime_adapter": "openpi-sft", "backend": "pytorch",
     "description": "训练价值模型。数据必须已经包含 value_target；产出的价值模型用于后续离线优势标注。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": RECAP_TORCH_ENTRY, "default_multi_entrypoint": RECAP_TORCH_ENTRY,
     "recipe_prefixes": ["pi0_value_"],
     "parameters": ESTIMATOR_VALIDATION_PARAMETERS},
    {"id": VALUE_ANNOTATION, "name": "价值优势标注", "runtime_adapter": "openpi-sft", "backend": "pytorch",
     "description": "用已训练的价值估计器预测逐帧回报，从 value_target 推导奖励，按任务生成 is_positive 数据。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": VALUE_ANNOTATION_ENTRY, "default_multi_entrypoint": VALUE_ANNOTATION_ENTRY,
     "recipe_prefixes": ["pi0_value_"],
     "parameters": {
         "n_step": {"type": "integer", "label": "优势窗口（帧）", "default": VALUE_ANNOTATION_DEFAULTS["n_step"], "min": 1, "max": 1000},
         "positive_ratio": {"type": "number", "label": "每任务正例比例", "default": VALUE_ANNOTATION_DEFAULTS["positive_ratio"], "min": .001, "max": 1},
     }},
    {"id": ACP_STAGE, "name": "离线强化学习（ACP 策略训练）", "runtime_adapter": "openpi-sft", "backend": "jax",
     "description": "使用已完成离线优势标注的数据和 Orbax 策略权重，在 JAX 中训练 ACP 策略。数据必须包含 is_positive。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": ACP_ENTRYPOINT, "default_multi_entrypoint": ACP_MULTI_ENTRYPOINT,
     "recipe_prefixes": ["pi05_acp_"],
     "parameters": {
         "acp_dropout": {"type": "number", "label": "优势条件丢弃概率", "default": 0.1, "min": 0, "max": 1},
     }},
    {"id": STAGE_PROGRESS, "name": "阶段进度估计器训练", "runtime_adapter": "openpi-sft", "backend": "pytorch",
     "description": "选择逐帧阶段进度列训练进度差估计器，输出模型检查点。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": RECAP_TORCH_ENTRY, "default_multi_entrypoint": RECAP_TORCH_ENTRY,
     "recipe_prefixes": ["pi05_stage_progress_"],
     "parameters": {**ESTIMATOR_VALIDATION_PARAMETERS,
                    "val_ratio": {**ESTIMATOR_VALIDATION_PARAMETERS["val_ratio"], "default": 0.0}}},
    {"id": STAGE_ANNOTATION, "name": "Stage Advantage 优势标注", "runtime_adapter": "openpi-sft", "backend": "pytorch",
     "description": "使用已训练的阶段进度估计器，生成带 is_positive 的派生数据集。",
     "required_inputs": ["dataset", "model", "norm"],
     "default_entrypoint": STAGE_ANNOTATION_ENTRY, "default_multi_entrypoint": STAGE_ANNOTATION_ENTRY,
     "recipe_prefixes": ["pi05_stage_progress_"],
     "parameters": {
         "n_step": {"type": "integer", "label": "优势窗口（帧）", "default": 50, "min": 1, "max": 1000},
         "positive_ratio": {"type": "number", "label": "每阶段正例比例", "default": 0.3, "min": 0.001, "max": 1},
         "stage_count": {"type": "integer", "label": "阶段数", "default": 2, "min": 1, "max": 100},
     }},
)


def recap_stages():
    return copy.deepcopy(list(STAGES))


def runtime_adapter_for(template_id):
    return "openpi-sft" if template_id in {"recap", "openpi-norm"} else template_id


def uses_torch_recap(spec):
    return spec.get("template_id") == "recap" and spec.get("recap_backend") == "pytorch"


def recap_backend(stage):
    return "jax" if stage == ACP_STAGE else "pytorch"


def uses_jax_acp(spec):
    return (spec.get("template_id") == "recap" and spec.get("training_stage") == ACP_STAGE
            and spec.get("recap_backend") == "jax")


def recap_entrypoint_options(stage, backend, dataset_count):
    if stage == ACP_STAGE and backend == "jax":
        return (ACP_MULTI_ENTRYPOINT,) if dataset_count > 1 else (ACP_ENTRYPOINT, ACP_MULTI_ENTRYPOINT)
    return (recap_torch_entrypoint(stage),)


def recap_torch_module_paths(stage):
    stage_sources = ("training/stage_progress.py",) if stage in STAGE_ADVANTAGE_STAGES else ()
    if stage == STAGE_ANNOTATION:
        stage_sources += ("training/stage_progress_labels.py",)
    if stage == VALUE_ANNOTATION:
        stage_sources = ("training/recap_advantages.py", "training/stage_progress_labels.py")
    return (*RECAP_TORCH_MODULES, *stage_sources)


def recap_model_role(stage):
    return {"value": "value_estimator", STAGE_PROGRESS: "stage_progress_estimator", "offline_rl": "policy"}.get(stage)


def recap_data_stage(stage):
    return {STAGE_ANNOTATION: STAGE_PROGRESS, VALUE_ANNOTATION: "value"}.get(stage, stage)


def recap_torch_entrypoint(stage):
    return {STAGE_ANNOTATION: STAGE_ANNOTATION_ENTRY, VALUE_ANNOTATION: VALUE_ANNOTATION_ENTRY}.get(stage, RECAP_TORCH_ENTRY)


def validate_recap_torch_selection(auto_config, entrypoint, stage=None):
    if auto_config and auto_config.get("finetune_mode") == "lora":
        raise ValueError("RECAP_TORCH_LORA_UNSUPPORTED: PyTorch ReCap 支持全量微调，请选择 full")
    expected = recap_torch_entrypoint(stage)
    if entrypoint not in (None, expected):
        raise ValueError(f"RECAP_STAGE_ENTRYPOINT_MISMATCH: PyTorch ReCap 使用固定入口 {expected}")


def validate_recap_selection(auto_config, entrypoint, stage, backend, dataset_count):
    if stage == VALUE_ANNOTATION and dataset_count != 1:
        raise ValueError("RECAP_STAGE_DATASET_COUNT: 价值优势标注需要绑定一组数据集和归一化参数")
    if backend == "pytorch":
        validate_recap_torch_selection(auto_config, entrypoint, stage)
        return
    if backend != "jax" or stage != ACP_STAGE:
        raise ValueError("RECAP_STAGE_RUNTIME_UNAVAILABLE: ReCap 阶段需要与平台匹配的训练后端")
    if auto_config and auto_config.get("finetune_mode") == "lora":
        raise ValueError("RECAP_STAGE_MODEL_UNSUPPORTED: ACP 自动配置请使用全量微调")
    if entrypoint is not None and entrypoint not in recap_entrypoint_options(stage, backend, dataset_count):
        raise ValueError("RECAP_STAGE_ENTRYPOINT_MISMATCH: ACP 使用 JAX 单数据集或混采训练入口")


def recap_model_matches(stage, model_type):
    expected = ESTIMATOR_MODELS.get(stage)
    return model_type == expected if expected else model_type not in ESTIMATOR_MODELS.values()


def validate_stage_progress_selection(stage, bindings, auto_config):
    if stage not in STAGE_ADVANTAGE_STAGES:
        return
    if not bindings or len(bindings) != 1:
        raise ValueError("RECAP_STAGE_DATASET_COUNT: Stage Advantage 需要绑定一组数据集和归一化参数")
    if auto_config is not None and (auto_config["model_type"] != "pi05" or auto_config["finetune_mode"] != "full"):
        raise ValueError("RECAP_STAGE_MODEL_UNSUPPORTED: Stage Advantage 自动配置需要 Pi0.5 全量微调")
    processor = bindings[0].get("processor")
    if auto_config is not None and (not isinstance(processor, dict) or processor.get("name") != "ComposableDataConfig"):
        raise ValueError("RECAP_STAGE_PROCESSOR_REQUIRED: Stage Advantage 自动配置需要组合字段处理器")


def recap_contract_valid(stage, requested, contract, datasets, *, require_pytorch=False, expected_backend=None, bindings=None):
    if not isinstance(contract, dict) or contract.get("stage") != stage or stage not in RECAP_SUPERVISION:
        return False
    model_type, loader_type, rows = contract.get("model_type"), contract.get("loader_type"), contract.get("datasets")
    if (not isinstance(model_type, str) or not model_type or not recap_model_matches(stage, model_type)
            or not isinstance(loader_type, str) or not loader_type or not isinstance(datasets, list)
            or not isinstance(rows, list) or len(rows) != len(datasets)):
        return False
    for index, (row, dataset) in enumerate(zip(rows, datasets)):
        if not isinstance(row, dict) or not isinstance(dataset, dict) or row.get("path") != dataset.get("path"):
            return False
        columns = row.get("required_columns")
        required = {RECAP_SUPERVISION[stage]}
        if stage in STAGE_ADVANTAGE_STAGES:
            binding = bindings[index] if isinstance(bindings, list) and index < len(bindings) else {}
            progress_key = _frozen_progress_key(row, dataset, binding)
            if progress_key is None:
                return False
            required = {progress_key, *STAGE_PROGRESS_ID_COLUMNS}
        elif stage == VALUE_ANNOTATION:
            required.update(STAGE_PROGRESS_ID_COLUMNS)
        if not isinstance(columns, list) or any(not isinstance(column, str) or not column for column in columns) or not required.issubset(columns):
            return False
    pytorch = contract.get("backend") == "pytorch"
    if (require_pytorch and not pytorch) or (expected_backend is not None and contract.get("backend") != expected_backend):
        return False
    sources = contract.get("source_sha256")
    if pytorch:
        runtime = contract.get("runtime_versions")
        if (stage == "offline_rl" and model_type not in {"Pi0Config", "Pi0GripperConfig"}
                or loader_type != RECAP_TORCH_LOADER or not _source_hashes_match(sources, {
                "openpi/" + name for name in recap_torch_module_paths(stage)
            }) or not isinstance(runtime, dict) or set(runtime) != {"torch", "safetensors", "transformers"}
                or any(not isinstance(version, str) or not 1 <= len(version) <= 128 for version in runtime.values())):
            return False
    if contract.get("backend") == "jax" and (stage != ACP_STAGE or model_type not in {"Pi0Config", "Pi0GripperConfig"}
            or loader_type not in ACP_WEIGHT_LOADERS):
        return False
    if stage == STAGE_PROGRESS and pytorch:
        if len(rows) != 1 or contract.get("operation") != "train" or "annotation" in contract:
            return False
        validation = contract.get("validation")
        if validation is None:
            return not requested  # Legacy previews had no validation settings.
        if not isinstance(validation, dict) or set(validation) != set(ESTIMATOR_VALIDATION_PARAMETERS):
            return False
        try:
            _, clean = validate_training_stage("recap", stage, validation)
        except ValueError:
            return False
        return all(clean.get(key) == value for key, value in (requested or {}).items())
    if stage not in ANNOTATION_STAGES and stage != STAGE_PROGRESS:
        return True
    annotation = contract.get("annotation")
    settings_keys = VALUE_ANNOTATION_DEFAULTS if stage == VALUE_ANNOTATION else STAGE_PROGRESS_KEYS
    dataset_dir = VALUE_ANNOTATION_DATASET if stage == VALUE_ANNOTATION else STAGE_PROGRESS_DATASET
    if (len(rows) != 1 or not isinstance(annotation, dict)
            or set(annotation) != {"output", "source_manifest_digest", *settings_keys}
            or annotation["output"] != dataset_dir
            or not _is_sha256(annotation["source_manifest_digest"])
            or not pytorch and (loader_type != STAGE_PROGRESS_LOADER or not _source_hashes_match(sources, {
                STAGE_PROGRESS_SUPPORT_ENTRY, *("openpi/" + name for name in STAGE_PROGRESS_MODULES)
            }))):
        return False
    try:
        settings = {key: annotation[key] for key in settings_keys}
        validate_training_stage("recap", VALUE_ANNOTATION if stage == VALUE_ANNOTATION else STAGE_ANNOTATION, settings)
        return all(settings.get(key) == value for key, value in (requested or {}).items())
    except (ValueError, AttributeError):
        return False


def _frozen_progress_key(row, dataset, binding):
    composed = dataset.get("composition", {})
    spec = composed.get("spec", {}) if isinstance(composed, dict) else {}
    if not isinstance(spec, dict):
        return None
    key = spec.get("stage_progress_key", STAGE_PROGRESS_COLUMN)
    processor = binding.get("processor") if isinstance(binding, dict) else None
    if isinstance(processor, dict) and processor.get("name") == "ComposableDataConfig":
        parameters = processor.get("parameters")
        expected = parameters.get("composition") if isinstance(parameters, dict) else None
        if not isinstance(expected, dict) or expected.get("stage_progress_key", STAGE_PROGRESS_COLUMN) != key:
            return None
    if key == STAGE_PROGRESS_COLUMN:
        return key if row.get("stage_progress_key", key) == key else None
    try:
        if stage_progress_source_key(key) != key:
            return None
    except ValueError:
        return None
    aux = spec.get("aux_keys")
    if (row.get("stage_progress_key") != key or composed.get("stage_progress_key") != key
            or not isinstance(aux, list) or STAGE_PROGRESS_COLUMN not in aux
            or row.get("supervision_columns") != [key]):
        return None
    summary = row.get("supervision_summary", {})
    measured = summary.get(key) if isinstance(summary, dict) else None
    if (not isinstance(measured, dict) or type(measured.get("count")) is not int or measured["count"] <= 0
            or any(type(measured.get(bound)) not in (int, float) for bound in ("min", "max"))
            or not 0 <= measured["min"] <= measured["max"] <= 1):
        return None
    return key


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_hashes_match(sources, expected):
    return (isinstance(sources, dict) and len(sources) == len(expected)
            and all(isinstance(name, str) and _is_sha256(value) for name, value in sources.items())
            and {name.removeprefix("src/") for name in sources} == expected)


def stage_progress_dataset_profile(source):
    return _annotation_dataset_profile(source, STAGE_PROGRESS_OUTPUT_COLUMNS)


def value_annotation_dataset_profile(source):
    return _annotation_dataset_profile(source, VALUE_ANNOTATION_COLUMNS)


def _annotation_dataset_profile(source, columns):
    if not isinstance(source, dict):
        return None
    result = copy.deepcopy(source)
    keys = result.get("feature_keys")
    if keys:
        extra = [key for key in columns if key not in keys]
        # An empty declaration leaves LeRobot metadata authoritative when the
        # enriched declaration would exceed the DatasetProfile field limit.
        result["feature_keys"] = [*keys, *extra] if len(keys) + len(extra) <= 256 else []
    return result


def valid_stage_progress_receipt(receipt, annotation, frame_count):
    if (not isinstance(receipt, dict) or not isinstance(annotation, dict)
            or set(receipt) != {"schema_version", "method", "score", "frame_count", "positive_count", "stages", *STAGE_PROGRESS_KEYS}
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["method"] != "kai0_stage_progress" or receipt["score"] != "absolute_advantage"
            or type(receipt["n_step"]) is not int or type(receipt["stage_count"]) is not int
            or type(receipt["positive_ratio"]) not in (int, float)
            or any(receipt[key] != annotation.get(key) for key in STAGE_PROGRESS_KEYS)
            or type(receipt["frame_count"]) is not int or not 1 <= receipt["frame_count"] == frame_count
            or type(receipt["positive_count"]) is not int or not 0 < receipt["positive_count"] <= frame_count
            or not isinstance(receipt["stages"], list) or len(receipt["stages"]) != annotation["stage_count"]):
        return False
    for stage, row in enumerate(receipt["stages"]):
        if (not isinstance(row, dict) or set(row) != {"stage", "frame_count", "positive_count", "threshold"}
                or type(row["stage"]) is not int or row["stage"] != stage
                or type(row["frame_count"]) is not int or not 0 <= row["frame_count"] <= frame_count
                or type(row["positive_count"]) is not int or not 0 <= row["positive_count"] <= row["frame_count"]
                or (row["threshold"] is not None if not row["frame_count"] else
                    type(row["threshold"]) not in (int, float) or not -1 <= row["threshold"] <= 1)):
            return False
    return all(sum(row[key] for row in receipt["stages"]) == receipt[key] for key in ("frame_count", "positive_count"))


def valid_value_annotation_receipt(receipt, annotation, frame_count, checkpoint):
    if (not isinstance(receipt, dict) or not isinstance(annotation, dict)
            or set(receipt) != {"schema_version", "backend", "method", "checkpoint", "n_step", "positive_ratio",
                               "frame_count", "positive_count", "thresholds"}
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["backend"] != "pytorch" or receipt["method"] != "n_step_value"
            or receipt["checkpoint"] != checkpoint
            or type(receipt["n_step"]) is not int or type(receipt["positive_ratio"]) not in (int, float)
            or any(receipt[key] != annotation.get(key) for key in VALUE_ANNOTATION_DEFAULTS)
            or type(receipt["frame_count"]) is not int or not 1 <= receipt["frame_count"] == frame_count
            or type(receipt["positive_count"]) is not int or not 0 < receipt["positive_count"] <= frame_count
            or not isinstance(receipt["thresholds"], dict) or not 1 <= len(receipt["thresholds"]) <= frame_count):
        return False
    return all(isinstance(task, str) and re.fullmatch(r"0|[1-9][0-9]{0,18}", task)
               and (type(threshold) is int or type(threshold) is float and math.isfinite(threshold))
               for task, threshold in receipt["thresholds"].items())


def stage_progress_outputs_match(outputs, output_dir, input_ids, plan):
    if (not isinstance(outputs, list) or len(outputs) != 2 or not isinstance(plan, dict)
            or not isinstance(output_dir, str) or not isinstance(input_ids, dict)):
        return False
    models = [row for row in outputs if isinstance(row, dict) and row.get("kind") == "MODEL"]
    datasets = [row for row in outputs if isinstance(row, dict) and row.get("kind") == "DATASET"]
    if len(models) != 1 or len(datasets) != 1:
        return False
    dataset, model = datasets[0], models[0]
    metadata = dataset.get("metadata")
    source = plan.get("datasets")
    contract = plan.get("recap_stage_contract")
    annotation = contract.get("annotation") if isinstance(contract, dict) else None
    version, recipe = model.get("version"), plan.get("recipe")
    return (isinstance(metadata, dict) and dataset.get("path") == str(PurePosixPath(output_dir) / STAGE_PROGRESS_DATASET)
            and isinstance(version, str) and version.isascii() and version.isdigit() and dataset.get("version") == version
            and isinstance(recipe, str) and model.get("path") == str(PurePosixPath(output_dir) / "checkpoints" / recipe / "platform" / version)
            and metadata.get("source_dataset_asset_id") == input_ids.get("dataset_0") is not None
            and metadata.get("source_norm_asset_id") == input_ids.get("norm_0") is not None
            and metadata.get("scorer_checkpoint_path") == model.get("path")
            and _is_sha256(metadata.get("annotation_sha256"))
            and isinstance(annotation, dict) and _is_sha256(annotation.get("source_manifest_digest"))
            and metadata.get("source_manifest_digest") == annotation["source_manifest_digest"]
            and isinstance(source, list) and len(source) == 1 and isinstance(source[0], dict)
            and valid_stage_progress_receipt(metadata.get("annotation"), annotation, source[0].get("frames")))


def stage_annotation_outputs_match(outputs, output_dir, input_ids, plan):
    return _annotation_outputs_match(outputs, output_dir, input_ids, plan, stage=STAGE_ANNOTATION)


def value_annotation_outputs_match(outputs, output_dir, input_ids, plan):
    return _annotation_outputs_match(outputs, output_dir, input_ids, plan, stage=VALUE_ANNOTATION)


def _annotation_outputs_match(outputs, output_dir, input_ids, plan, *, stage):
    if (not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict)
            or not isinstance(output_dir, str) or not isinstance(input_ids, dict) or not isinstance(plan, dict)):
        return False
    dataset = outputs[0]
    metadata = dataset.get("metadata")
    sources, model, contract = plan.get("datasets"), plan.get("model"), plan.get("recap_stage_contract")
    annotation = contract.get("annotation") if isinstance(contract, dict) else None
    scorer = model.get("scorer_checkpoint") if isinstance(model, dict) else None
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict) or not isinstance(metadata, dict):
        return False
    receipt = metadata.get("annotation")
    frames = sources[0].get("frames")
    if stage == VALUE_ANNOTATION:
        output, registration = VALUE_ANNOTATION_DATASET, "recap_value_annotation"
        valid_receipt = isinstance(model, dict) and valid_value_annotation_receipt(receipt, annotation, frames, model.get("checkpoint_path"))
    else:
        output, registration = STAGE_PROGRESS_DATASET, "recap_stage_annotation"
        valid_receipt = valid_stage_progress_receipt(receipt, annotation, frames)
    return (dataset.get("kind") == "DATASET" and isinstance(metadata, dict)
            and dataset.get("path") == str(PurePosixPath(output_dir) / output)
            and metadata.get("format") == "lerobot" and metadata.get("registration_source") == registration
            and metadata.get("source_dataset_asset_id") == input_ids.get("dataset_0") is not None
            and metadata.get("source_norm_asset_id") == input_ids.get("norm_0") is not None
            and metadata.get("scorer_model_asset_id") == input_ids.get("model") is not None
            and isinstance(scorer, dict) and type(scorer.get("step")) is int
            and dataset.get("version") == str(scorer["step"])
            and metadata.get("scorer_checkpoint_path") == model.get("checkpoint_path") is not None
            and _is_sha256(metadata.get("annotation_sha256"))
            and isinstance(annotation, dict) and _is_sha256(annotation.get("source_manifest_digest"))
            and metadata.get("source_manifest_digest") == annotation["source_manifest_digest"]
            and valid_receipt)


def validate_training_stage(template_id, training_stage=None, stage_config=None):
    if template_id != "recap":
        if training_stage is not None or stage_config is not None:
            raise ValueError("TRAINING_STAGE_UNSUPPORTED: 只有 ReCap 模板支持分阶段配置")
        return None, None
    if training_stage is None:
        raise ValueError("RECAP_STAGE_REQUIRED: 请选择 ReCap 训练阶段")
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
