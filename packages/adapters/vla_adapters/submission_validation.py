"""Submission checks reuse registered source facts without opening a dataset.

This evidence proves source validation and the current metadata/configuration
contract separately. It does not claim that the current transforms were run on
decoded samples. Their runtime checks still execute in the training loader.
"""

import hashlib
import json
import math
import re

if __package__:
    from .dataset_validation_cache import lookup_cached
else:
    from dataset_validation_cache import lookup_cached


MODE = "registered_source"
SCOPE = "registered_source_and_configuration"
BASE_VALIDATION_REQUIRED = "DATASET_BASE_VALIDATION_REQUIRED"


def base_validation_asset_key(dataset):
    return dataset.get("dataset_asset_id") or dataset["path"]


def record_base_validation_failure(failures, name, dataset, error):
    message = str(error)
    if message.partition(":")[0] != BASE_VALIDATION_REQUIRED:
        return False
    asset_key = base_validation_asset_key(dataset)
    if asset_key not in failures:
        check = {"name": name, "ok": False, "code": BASE_VALIDATION_REQUIRED, "message": message[:2000]}
        if dataset.get("dataset_asset_id"):
            check["dataset_asset_id"] = dataset["dataset_asset_id"]
        failures[asset_key] = check
    return True


def require_base_validation_results(failures, checks):
    if not failures:
        return
    checks.extend(failures.values())
    if len(failures) == 1:
        raise ValueError(next(iter(failures.values()))["message"])
    raise ValueError(f"{BASE_VALIDATION_REQUIRED}: {len(failures)} 个数据集的基础校验缺失或已失效，请刷新校验后重新预检。")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_samples(source, data):
    """Validate only bounded shape/type evidence, never decode source arrays."""
    if (not isinstance(source, dict) or type(source.get("schema_version")) is not int
            or source["schema_version"] != 1 or source.get("status") != "verified"
            or not isinstance(data.get("path"), str) or not data["path"].startswith("/")
            or source.get("path") != data.get("path")):
        raise ValueError("missing or mismatched registered source evidence")
    raw = source.get("raw_samples")
    frames = data.get("frames")
    if not isinstance(raw, dict) or type(frames) is not int or frames <= 0:
        raise ValueError("missing registered raw sample evidence")
    indices = sorted({0, frames // 2, frames - 1})
    rows = raw.get("samples")
    if (raw.get("selection") != "first_middle_last"
            or type(raw.get("checked_samples")) is not int or raw["checked_samples"] != len(indices)
            or raw.get("sample_indices") != indices
            or any(type(index) is not int for index in raw["sample_indices"])
            or not isinstance(rows, list) or len(rows) != len(indices)):
        raise ValueError("incomplete registered source sample coverage")
    composition = (data.get("composition") or {}).get("spec")
    required = set()
    if composition is not None:
        if not isinstance(composition, dict):
            raise ValueError("invalid submission composition")
        required = {row["key"] for row in composition["state_fields"] + composition["cameras"]}
        required.update([composition["action_key"], *composition["aux_keys"]])
    for index, row in zip(indices, rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index
                or not isinstance(row.get("fields"), dict) or not row["fields"]
                or not required <= row["fields"].keys()):
            raise ValueError("registered source samples do not cover the selected fields")
        for key, field in row["fields"].items():
            if (not isinstance(key, str) or not key or not isinstance(field, dict)
                    or not isinstance(field.get("shape"), list) or len(field["shape"]) > 5
                    or any(type(width) is not int or width <= 0 for width in field["shape"])
                    or not isinstance(field.get("dtype"), str)
                    or not re.fullmatch(r"(?:bool|u?int(?:8|16|32|64)|float(?:16|32|64|128))", field["dtype"])):
                raise ValueError("invalid registered source field summary")
    return raw


def require_cached_dataset_source(data, spec):
    source = lookup_cached("source", data, spec, context={"scope": "dataset_source"})
    try:
        _source_samples(source, data)
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError("DATASET_BASE_VALIDATION_REQUIRED: 数据集 " + data["path"] +
                         " 缺少匹配的基础校验结果，或数据、登记配置、校验环境已变化；"
                         "请先在数据集详情完成基础校验，再提交训练。提交阶段不会重新加载或解码数据。") from exc
    return source


def submission_evidence(data, source):
    _source_samples(source, data)
    timestamps = data.get("timestamp_contract")
    fps = data.get("effective_fps")
    if (not isinstance(timestamps, dict) or type(timestamps.get("schema_version")) is not int
            or timestamps["schema_version"] != 1 or timestamps.get("status") != "verified"
            or type(data.get("episodes")) is not int or not 0 < data["episodes"] <= data["frames"]
            or type(timestamps.get("checked_rows")) is not int or timestamps["checked_rows"] != data["frames"]
            or type(timestamps.get("checked_episodes")) is not int or timestamps["checked_episodes"] != data["episodes"]
            or type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 1000
            or type(timestamps.get("effective_fps")) not in (int, float) or timestamps["effective_fps"] != fps
            or any(not isinstance(data.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", data[key])
                   or timestamps.get(key) != data[key] for key in ("metadata_sha256", "data_manifest_digest"))):
        raise ValueError("invalid or mismatched registered timestamp evidence")
    return {"schema_version": 1, "scope": SCOPE, "status": "verified", "path": data["path"],
            "metadata_sha256": data["metadata_sha256"], "data_manifest_digest": data["data_manifest_digest"],
            "timestamp_contract_sha256": _digest(data["timestamp_contract"]),
            "source_evidence": source, "source_evidence_sha256": _digest(source),
            "composition_sha256": _digest(data.get("composition")),
            "image_processing_sha256": _digest(data.get("image_processing")),
            "processor_identity_sha256": _digest(data.get("processor_identity"))}


def validate_submission_evidence(data):
    if not isinstance(data, dict):
        raise ValueError("invalid submission dataset")
    evidence = data.get("submission_validation")
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence["schema_version"] != 1):
        raise ValueError("missing registered source submission evidence")
    try:
        expected = submission_evidence(data, evidence.get("source_evidence"))
    except (KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError("invalid submission source or configuration evidence") from exc
    if evidence != expected:
        raise ValueError("submission source or configuration evidence differs from the requested dataset")
