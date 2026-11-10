"""Declarative camera processing shared by registration and training.

Only source metadata is inspected here. This module never imports a training
runtime, decodes an image, or changes the registered dataset on disk.
"""

import re


TARGET_IMAGE_SIZE = (224, 224)
RESIZE_MODES = frozenset({"pad", "center_crop", "stretch"})
AUGMENTATION_CODES = {"default": 0, "none": 1, "color": 2, "crop_rotate_color": 3}
AUGMENTATIONS = frozenset(AUGMENTATION_CODES)


def normalize_image_processing(value):
    if (not isinstance(value, dict) or set(value) - {"resize_mode", "augmentation"}
            or not isinstance(value.get("resize_mode"), str) or value["resize_mode"] not in RESIZE_MODES):
        raise ValueError("IMAGE_PROCESSING_INVALID: 请选择有效的图像缩放方式")
    policy = {"resize_mode": value["resize_mode"]}
    if "augmentation" in value:
        choice = value["augmentation"]
        if not isinstance(choice, str) or choice not in AUGMENTATIONS:
            raise ValueError("IMAGE_PROCESSING_INVALID: 历史图像增强方式无效")
        policy["augmentation"] = choice
    return policy


def composition_augmentation_codes(cameras, source_policies):
    """Resolve a frozen composition choice before falling back to old assets."""
    codes = {}
    for camera in cameras:
        legacy = source_policies.get(camera["key"], {}).get("augmentation")
        choice = camera.get("augmentation", legacy)
        if choice is not None:
            codes[camera["slot"]] = AUGMENTATION_CODES[choice]
    return codes


def image_dimensions(feature, kind="rgb"):
    """Return an unambiguous single-frame H/W, never guessing a channel axis."""
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if (not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 5
            or any(type(n) is not int or not 1 <= n <= 2**31 - 1 for n in shape)):
        return None
    frame = shape[1:] if len(shape) == 4 else shape
    if kind == "depth" and len(frame) == 2:
        return tuple(frame)
    if len(frame) != 3:
        return None
    channel = 1 if kind == "depth" else 3
    first, last = frame[0] == channel, frame[-1] == channel
    if first == last:
        return None
    return tuple(frame[1:]) if first else tuple(frame[:2])


def validate_profile_image_processing(profile, features, *, require_selection=True):
    """Validate saved policies against real source facts and return keyed policies.

    ``features`` accepts a LeRobot metadata feature dictionary or the bounded
    inspection DTO's feature list. All codec image/video fields are covered,
    plus explicitly registered numeric camera tensors. An unknown or ambiguous
    shape requires an explicit processing choice just as a non-224 image does;
    it is never reported as an already compatible 224 image.
    """
    if isinstance(features, list):
        if any(not isinstance(row, dict) or not isinstance(row.get("key"), str) for row in features):
            raise ValueError("DATASET_IMAGE_METADATA_INVALID: 图像字段元信息无效，请重新读取")
        by_key = {row["key"]: row for row in features}
        if len(by_key) != len(features):
            raise ValueError("DATASET_IMAGE_METADATA_INVALID: 图像字段元信息重复，请重新读取")
    elif isinstance(features, dict):
        by_key = features
    else:
        raise ValueError("DATASET_IMAGE_METADATA_INVALID: 缺少图像字段元信息，请重新读取")
    cameras = profile.get("cameras", []) if isinstance(profile, dict) else []
    if not isinstance(cameras, list):
        raise ValueError("IMAGE_PROCESSING_INVALID: 数据集视觉输入无效")
    declared = {}
    policies = {}
    for camera in cameras:
        if not isinstance(camera, dict) or not isinstance(camera.get("key"), str) or camera.get("kind") not in {"rgb", "depth"}:
            raise ValueError("IMAGE_PROCESSING_INVALID: 数据集视觉输入无效")
        key = camera["key"]
        if key in declared:
            raise ValueError("IMAGE_PROCESSING_INVALID: 数据集视觉字段不能重复")
        declared[key] = camera
        policy = camera.get("image_processing")
        if policy is not None:
            policy = normalize_image_processing(policy)
            if camera["kind"] == "depth" and policy.get("augmentation", "default") not in {"default", "none"}:
                raise ValueError("IMAGE_PROCESSING_INVALID: 深度图不支持 RGB 颜色或裁剪旋转颜色增强")
            policies[key] = policy
        if key not in by_key or not isinstance(by_key[key], dict):
            raise ValueError("DATASET_IMAGE_METADATA_INVALID: 登记的视觉字段不存在：" + key)
        dtype = by_key[key].get("dtype")
        if (dtype not in {"image", "video"} and
                (not isinstance(dtype, str) or re.fullmatch(r"(?:u?int(?:8|16|32|64)?|float(?:16|32|64)?|double|bfloat16)", dtype) is None)):
            raise ValueError("DATASET_IMAGE_METADATA_INVALID: 登记的视觉字段不是图像或数值张量：" + key)
    relevant = {key for key, feature in by_key.items()
                if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}} | set(declared)
    for key in sorted(relevant):
        camera = declared.get(key)
        kind = camera["kind"] if camera else ("depth" if re.search(r"(?:^|[._/])depth(?:$|[._/])", key, re.I) else "rgb")
        dimensions = image_dimensions(by_key[key], kind)
        if require_selection and dimensions != TARGET_IMAGE_SIZE and key not in policies:
            size = f"{dimensions[0]}×{dimensions[1]}" if dimensions else "尺寸未知或布局不明确"
            raise ValueError(f"IMAGE_PROCESSING_REQUIRED: 视觉字段 {key}（{size}）需要在数据集特征中明确选择处理到 224×224 的方式")
    return policies
