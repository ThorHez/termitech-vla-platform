"""Dataset-registration checks, executed by the registered CPU interpreter.

This prepares the same local LeRobot/HF cache used by training. It validates
source data, not a future model's composition, horizon, or normalization choice.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import signal
import sys

if __package__:
    from .cache_contract import configure_cache
    from .dataset_fps import install_registered_fps
    from .dataset_timestamps import inspect_dataset_timestamps
    from .dataset_validation_cache import cache_key, get_or_validate
    from .training_plan import file_manifest, fingerprint, local_path, read_json
    from .recap_runtime import prepare_source_supervision
else:
    from cache_contract import configure_cache
    from dataset_fps import install_registered_fps
    from dataset_timestamps import inspect_dataset_timestamps
    from dataset_validation_cache import cache_key, get_or_validate
    from training_plan import file_manifest, fingerprint, local_path, read_json
    from recap_runtime import prepare_source_supervision


PREFIX = "VLA_DATASET_VALIDATION:"
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_FEATURES = 256


def _raw_samples(path, info):
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(str(path))
    count = len(dataset)
    if type(count) is not int or count != info["total_frames"]:
        raise ValueError("DATASET_SOURCE_INVALID: 实际加载器帧数与数据集元信息不一致")
    indices = sorted({0, count // 2, count - 1})
    rows = []
    for index in indices:
        sample = dataset[index]
        if not isinstance(sample, dict):
            raise ValueError("DATASET_SOURCE_INVALID: 原始加载器未返回字段字典")
        fields = {}
        for key, feature in info["features"].items():
            if feature.get("dtype") in {"string", "str"} or key == "task":
                continue  # Never persist task text or another raw data value.
            if key not in sample:
                raise ValueError(f"DATASET_SOURCE_INVALID: 样本 {index} 缺少原始字段 {key}")
            value = sample[key]
            if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
                raise ValueError("DATASET_SOURCE_INVALID: 原始数据校验只允许 CPU 张量")
            array = np.asarray(value)
            if (array.ndim > 5 or not array.size or array.nbytes > MAX_ARRAY_BYTES
                    or array.dtype.kind not in "biuf" or not np.all(np.isfinite(array))):
                raise ValueError(f"DATASET_SOURCE_INVALID: 样本 {index} 的字段 {key} 不是有效有限数值数组")
            if feature.get("dtype") in {"image", "video"}:
                if array.ndim not in (3, 4):
                    raise ValueError(f"DATASET_SOURCE_INVALID: 样本 {index} 的图像字段 {key} 解码形状无效")
            else:
                expected = tuple(feature.get("shape") or ())
                if array.shape != expected and not (expected == (1,) and array.shape == ()):
                    raise ValueError(f"DATASET_SOURCE_INVALID: 样本 {index} 的字段 {key} 与声明形状不一致")
            fields[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        rows.append({"index": index, "fields": fields})
    return {"schema_version": 1, "status": "verified", "path": str(path),
            "raw_samples": {"selection": "first_middle_last", "checked_samples": len(rows),
                            "sample_indices": indices, "samples": rows}}


def validate_source(spec):
    configure_cache(spec)
    install_registered_fps(spec)
    path = local_path(spec["path"], spec)
    info, info_sha = read_json(path / "meta/info.json", spec)
    features = info.get("features")
    if (not isinstance(features, dict) or not 1 <= len(features) <= MAX_FEATURES
            or any(not isinstance(key, str) or not key or len(key) > 256 or not isinstance(value, dict)
                   for key, value in features.items())):
        raise ValueError("DATASET_SOURCE_INVALID: 数据集字段元信息无效")
    if any(type(info.get(key)) is not int or info[key] <= 0 for key in ("total_frames", "total_episodes")):
        raise ValueError("DATASET_SOURCE_INVALID: 数据集帧数和 episode 数量必须大于零")
    profile = spec.get("registered_profile") or {}
    if not isinstance(profile, dict):
        raise ValueError("DATASET_SOURCE_INVALID: 登记的数据描述必须为对象")
    _, _, manifest = file_manifest(path / "data", spec)
    data = {"path": str(path), "metadata_sha256": info_sha, "frames": info["total_frames"],
            "episodes": info["total_episodes"], "data_manifest_digest": manifest,
            "effective_fps": profile.get("fps", info.get("fps")), "registered_profile": profile}
    if any(feature.get("dtype") == "video" for feature in features.values()):
        video_rows, _, video_manifest = file_manifest(path / "videos", spec)
        data["video_manifest_digest"] = video_manifest
        available = {"videos/" + row[0] for row in video_rows if row[1] > 0}
        keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
        pattern, chunks = info.get("video_path"), info.get("chunks_size", 1000)
        if (not isinstance(pattern, str) or type(chunks) is not int or chunks <= 0
                or info["total_episodes"] * len(keys) > 20000):
            raise ValueError("DATASET_SOURCE_INVALID: 视频路径模板或数量无效")
        for key in keys:
            for episode in range(info["total_episodes"]):
                relative = pattern.format(video_key=key, episode_index=episode, episode_chunk=episode // chunks)
                video = local_path(path / relative, spec)
                if not video.is_relative_to(path / "videos") or relative not in available:
                    raise ValueError("DATASET_SOURCE_INVALID: 视频路径逃逸或文件缺失")
    timestamps = inspect_dataset_timestamps(data, spec)
    supervision = prepare_source_supervision(data, spec, info)
    context = {"scope": "dataset_source"}
    sampled = get_or_validate("source", data, spec, lambda: _raw_samples(path, info), context=context)
    source_id = cache_key("source", data, spec, context=context)
    # The audit identifier remains meaningful when caching is disabled. It is
    # never used as a substitute for the cache's stronger reuse fingerprint.
    if source_id is None:
        source_id = fingerprint({"source": data, "timestamp_runtime": timestamps["runtime"]})
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    loader = Path(inspect.getsourcefile(LeRobotDataset)).resolve(strict=True)
    runtime = timestamps["runtime"]
    checks = [
        {"name": "source_metadata", "ok": True, "message": "数据集元信息、帧数及数据文件可读"},
        {"name": "dataset_timestamps", "ok": True, "message": "已全量检查时间戳、episode 顺序及本次生效 FPS"},
        {"name": "raw_samples", "ok": True, "message": "已读取首、中、末原始样本，检查解码、字段形状和有限数值"},
        {"name": "dataset_cache", "ok": True, "message": "原始数据加载缓存已准备，可供后续训练复用"},
    ]
    return {"schema_version": 1, "scope": "dataset_source", "path": str(path),
            "profile_id": spec["profile_id"], "source_fingerprint": source_id,
            "checks": checks, "timestamp_contract": timestamps, "raw_samples": sampled["raw_samples"],
            **({"recap_supervision": supervision} if supervision is not None else {}),
            "runtime": {"python_executable": sys.executable, "python_version": platform.python_version(),
                        "loader_path": str(loader), "loader_sha256": hashlib.sha256(loader.read_bytes()).hexdigest(),
                        "validator_path": runtime["validator_path"], "validator_sha256": runtime["validator_sha256"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", required=True)
    args = parser.parse_args()
    # Independent expiry protects this child if its Agent parent disappears.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(299)
    try:
        result = validate_source(json.loads(args.spec_json))
    except Exception as error:
        message = str(error).splitlines()[0][:1800]
        code = message.split(":", 1)[0]
        if not code.replace("_", "").isalnum() or code.upper() != code:
            code = "DATASET_VALIDATION_FAILED"
        result = {"error": {"code": code, "message": message}}
    finally:
        signal.alarm(0)
    print(PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 2 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
