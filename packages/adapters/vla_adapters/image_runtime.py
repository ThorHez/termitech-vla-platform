"""Apply frozen camera policies at the registered training boundaries.

Contract resolution only reads metadata and declarations. Pixel transforms run
when the native training loader requests a sample, never during submission or
numeric normalization. The factory hook retains the original dataclass identity
and also covers temporary ``dataclasses.replace`` copies used by shared stats.
"""

import ast
import dataclasses
import functools
import math
from pathlib import Path
import re

if __package__:
    from .composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from .image_policy import (AUGMENTATION_CODES, composition_augmentation_codes, normalize_image_processing,
                               validate_profile_image_processing)
else:
    from composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from image_policy import (AUGMENTATION_CODES, composition_augmentation_codes, normalize_image_processing,
                              validate_profile_image_processing)


POLICY_VERSION = 1
_HOOK = "_vla_camera_policy_create_v1"


def _fail(message, code="IMAGE_PROCESSING_UNSUPPORTED"):
    raise ValueError(code + ": " + message)


def _flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return {}
        if isinstance(item, str):
            result[prefix + key] = item
        elif isinstance(item, dict):
            result.update(_flatten(item, prefix + key + "."))
        else:
            return {}
    return result


def _static_view_mapping(factory, code_dir):
    """Prove supported literal Repack + standard parser mappings without calls.

    Arbitrary processor Python is not a camera-provenance interface. Refuse it
    rather than borrowing a slot from its name or guessing from image contents.
    """
    if __package__:
        from .dataset_contracts import _class_node, _name, _repack_mapping
    else:
        from dataset_contracts import _class_node, _name, _repack_mapping
    try:
        root = Path(code_dir).resolve(strict=True)
        nodes = [_class_node(cls, root) for cls in type(factory).__mro__ if cls is not object]
        nodes = [node for node in nodes if node is not None]
        create = next((item for node in nodes for item in node.body
                       if isinstance(item, ast.FunctionDef) and item.name == "create"), None)
        if create is None:
            return None
        mapping = _repack_mapping(nodes, create)
        # Aloha declares its frozen Repack group as a dataclass field.
        if mapping is None:
            group = getattr(factory, "repack_transforms", None)
            transforms = getattr(group, "inputs", ())
            if len(transforms) == 1 and type(transforms[0]).__name__ == "RepackTransform":
                mapping = _flatten(getattr(transforms[0], "structure", None))
        if not mapping:
            return None
        calls = {_name(node.func) for node in ast.walk(create) if isinstance(node, ast.Call)}
        if type(factory).__module__ == "openpi.training.config" and "AlohaInputs" in calls:
            outputs = {"images.cam_high": "base_0_rgb", "images.cam_left_wrist": "left_wrist_0_rgb",
                       "images.cam_right_wrist": "right_wrist_0_rgb"}
        elif type(factory).__module__ == "openpi.training.config" and "DroidInputs" in calls:
            outputs = {"observation/exterior_image_1_left": "base_0_rgb",
                       "observation/wrist_image_left": "left_wrist_0_rgb"}
        else:
            # A direct canonical Repack is usable only when the returned data
            # group is visibly empty. Other transforms could remap the images.
            empty = False
            for statement in create.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                value = statement.value
                if (any(isinstance(target, ast.Name) and target.id == "data_transforms" for target in targets)
                        and isinstance(value, ast.Call) and _name(value.func) == "Group" and not value.args
                        and all(keyword.arg in {"inputs", "outputs"} and isinstance(keyword.value, (ast.List, ast.Tuple))
                                and not keyword.value.elts for keyword in value.keywords)):
                    empty = True
            if not empty:
                return None
            outputs = {key: key[len("image."):] for key in mapping if key.startswith("image.")}
        result = {}
        for output, slot in outputs.items():
            if output in mapping:
                result.setdefault(mapping[output], []).append(slot)
        return result or None
    except (OSError, TypeError, ValueError, StopIteration):
        return None


def image_processing_contract(factory, profile, features, *, code_dir, model_config=None):
    """Resolve source transforms and per-view augmentation without decoding data."""
    if not isinstance(features, dict):
        _fail("LeRobot meta/info.json 缺少 features", "DATASET_FORMAT")
    policies = validate_profile_image_processing(profile, features)
    composition = (normalize_composition(factory.composition)
                   if type(factory).__name__ == COMPOSABLE_PROCESSOR else None)
    if not policies and (composition is None or not any("augmentation" in camera for camera in composition["cameras"])):
        return None
    cameras = {camera["key"]: camera for camera in (profile or {}).get("cameras", [])}
    spatial = []
    routes = {}
    uniform = None
    if composition is not None:
        selected = {camera["key"]: (index, camera) for index, camera in enumerate(composition["cameras"])}
        for key, policy in policies.items():
            if key not in selected:
                continue
            index, camera = selected[key]
            spatial.append({"key": "_composition_camera_" + str(index), "source_key": key,
                            "layout": camera["layout"], "kind": camera["kind"],
                            "history": camera["time_index"] is not None,
                            "fill": camera["depth_min"] if camera["kind"] == "depth" else 0,
                            "resize_mode": policy["resize_mode"]})
        routes = composition_augmentation_codes(composition["cameras"], policies)
        insertion = "after_repack"
    else:
        if any(getattr(owner, "video_frame_config", None) is not None
               for owner in (factory, getattr(factory, "base_config", None))):
            _fail("独立视频/历史处理器会展开原始相机字段；图像策略需要显式组合或受支持的单帧处理器",
                  "IMAGE_PROCESSING_MAPPING_UNSUPPORTED")
        # Native LeRobot image/video RGB decoders emit CHW/TCHW regardless of
        # the source storage layout. Numeric/depth arrays need an explicit
        # composition to establish their channel axis and black depth value.
        for key, policy in policies.items():
            camera, feature = cameras[key], features[key]
            if camera["kind"] != "rgb" or feature.get("dtype") not in {"image", "video"}:
                _fail("独立处理器的深度/数值图像策略需要显式组合处理以确定布局", "IMAGE_PROCESSING_MAPPING_UNSUPPORTED")
            spatial.append({"key": key, "source_key": key, "layout": "chw", "kind": "rgb", "fill": 0, "history": False,
                            "resize_mode": policy["resize_mode"]})
        # An unlisted 224 camera is still a real input with default processing.
        # Omitting it from the profile must never turn a partial override into
        # a dataset-wide augmentation override.
        def source_kind(key, feature):
            if key in cameras:
                return cameras[key]["kind"]
            detail = feature.get("info") if isinstance(feature.get("info"), dict) else {}
            return ("depth" if feature.get("semantic") == "depth" or detail.get("is_depth") is True
                    or re.search(r"(?:^|[._/])depth(?:$|[._/])", key, re.I) else "rgb")
        rgb_keys = {key for key, feature in features.items() if isinstance(feature, dict)
                    and feature.get("dtype") in {"image", "video"} and source_kind(key, feature) == "rgb"}
        rgb_keys.update(key for key, camera in cameras.items() if camera["kind"] == "rgb")
        codes = {key: AUGMENTATION_CODES[policies.get(key, {}).get("augmentation", "default")]
                 for key in sorted(rgb_keys)}
        if len(set(codes.values())) == 1:
            uniform = next(iter(codes.values()))
        else:
            mapping = _static_view_mapping(factory, code_dir)
            if mapping is None or any(key not in mapping for key, code in codes.items() if code):
                _fail("该独立处理器无法证明原始相机到模型视角的增强映射；请使用显式组合处理", "IMAGE_PROCESSING_MAPPING_UNSUPPORTED")
            for key, code in codes.items():
                for slot in mapping.get(key, ()):
                    if slot in routes and routes[slot] != code:
                        _fail("多个相机映射到同一模型视角且增强选择不同", "IMAGE_PROCESSING_MAPPING_UNSUPPORTED")
                    routes[slot] = code
        insertion = "before_repack"
    return {"schema_version": POLICY_VERSION, "target_size": [224, 224],
            "source_policies": {key: normalize_image_processing(value) for key, value in sorted(policies.items())},
            "unused_source_keys": sorted(set(policies) - {camera["source_key"] for camera in spatial}),
            "spatial_insertion": insertion, "spatial": spatial,
            "depth_keys": (tuple(model_config.depth_keys) if getattr(model_config, "depth_keys", None) is not None else None),
            "augmentation_codes": routes, "uniform_augmentation_code": uniform}


def _get(data, key):
    if key in data:
        return data[key]
    value = data
    for part in key.split("."):
        value = value[part]
    return value


def _put(data, key, value):
    result = dict(data)
    if key in result:
        result[key] = value
        return result
    current = result
    parts = key.split(".")
    for part in parts[:-1]:
        current[part] = dict(current[part])
        current = current[part]
    current[parts[-1]] = value
    return result


def resize_source_image(value, *, layout, kind, resize_mode, fill=0, history=False):
    """Resize every frame, retaining source layout, dtype and numeric range."""
    import numpy as np
    from PIL import Image

    image = np.asarray(value)
    if image.dtype.kind not in "uif" or image.ndim not in {2, 3, 4} or not np.all(np.isfinite(image)):
        _fail("图像必须是有限的 HW/CHW/HWC 或带时间轴数组", "IMAGE_PROCESSING_DATA_INVALID")
    if image.dtype.kind == "f" and kind == "rgb" and (np.any(image < 0) or np.any(image > 1)):
        _fail("源 RGB 浮点图像必须位于 0..1；禁止重复归一化", "IMAGE_PROCESSING_DATA_INVALID")
    if kind == "rgb" and image.dtype != np.uint8 and image.dtype.kind != "f":
        _fail("源 RGB 图像只支持 uint8 或 0..1 浮点", "IMAGE_PROCESSING_DATA_INVALID")
    planar = image.ndim == 2 or kind == "depth" and image.ndim == 3 and history
    if planar:
        if kind != "depth":
            _fail("RGB 图像缺少三通道轴", "IMAGE_PROCESSING_DATA_INVALID")
        hwc = image[..., None]
    elif layout == "chw":
        hwc = np.moveaxis(image, -3, -1)
    elif layout == "hwc":
        hwc = image
    else:
        _fail("图像布局必须明确为 CHW 或 HWC", "IMAGE_PROCESSING_DATA_INVALID")
    channels = 3 if kind == "rgb" else 1
    if hwc.shape[-1] != channels or min(hwc.shape[-3:-1]) <= 0 or max(hwc.shape[-3:-1]) > 4096:
        _fail("图像形状与声明的通道轴不一致，或空间尺寸超过4096", "IMAGE_PROCESSING_DATA_INVALID")
    prefix, height, width = hwc.shape[:-3], *hwc.shape[-3:-1]
    if resize_mode == "stretch":
        resized_h = resized_w = 224
    elif resize_mode == "pad":
        # Match OpenPI's division order. Multiplying 360 by (224 / 640)
        # truncates 125.99999999999999 to 125 instead of the intended 126.
        ratio = max(width / 224, height / 224)
        resized_h, resized_w = max(1, int(height / ratio)), max(1, int(width / ratio))
    elif resize_mode == "center_crop":
        resized_h, resized_w = ((math.ceil(height * 224 / width), 224) if height >= width
                                else (224, math.ceil(width * 224 / height)))
    else:
        _fail("未知图像缩放方式", "IMAGE_PROCESSING_DATA_INVALID")
    frames = hwc.reshape((-1, height, width, channels))
    converted = []
    for frame in frames:
        # Float-mode PIL preserves LeRobot's 0..1 floats and raw depth units;
        # augmentation remains the model's native augmax operation on device.
        resized = np.stack([np.asarray(Image.fromarray(frame[..., channel].astype(np.float32)).resize(
            (resized_w, resized_h), resample=Image.Resampling.BILINEAR)) for channel in range(channels)], axis=-1)
        if resize_mode == "pad":
            target = np.full((224, 224, channels), fill, dtype=np.float32)
            top, left = (224 - resized_h) // 2, (224 - resized_w) // 2
            target[top:top + resized_h, left:left + resized_w] = resized
        else:
            top, left = (resized_h - 224) // 2, (resized_w - 224) // 2
            target = resized[top:top + 224, left:left + 224]
        if image.dtype.kind in "ui":
            limits = np.iinfo(image.dtype)
            target = np.rint(target).clip(limits.min, limits.max)
        converted.append(target.astype(image.dtype))
    result = np.stack(converted).reshape((*prefix, 224, 224, channels))
    if planar:
        return result[..., 0]
    return np.moveaxis(result, -1, -3) if layout == "chw" else result


@dataclasses.dataclass(frozen=True)
class SourceImagePolicies:
    cameras: tuple

    def __call__(self, data):
        result = data
        for camera in self.cameras:
            options = {key: camera[key] for key in ("layout", "kind", "resize_mode", "fill", "history")}
            try:
                value = resize_source_image(_get(result, camera["key"]), **options)
                result = _put(result, camera["key"], value)
            except (KeyError, TypeError) as exc:
                raise ValueError("IMAGE_PROCESSING_DATA_INVALID: 无法读取原始相机字段 " + camera["source_key"]) from exc
        return result


@dataclasses.dataclass(frozen=True)
class ImageAugmentationRouting:
    codes: dict
    uniform_code: int | None = None
    depth_keys: tuple | None = None

    def __call__(self, data):
        import numpy as np
        images = data.get("image")
        if not isinstance(images, dict) or not images:
            _fail("处理器没有产生模型图像字典", "IMAGE_PROCESSING_DATA_INVALID")
        if self.uniform_code is None and set(self.codes) - set(images):
            _fail("处理器没有产生声明的模型相机槽", "IMAGE_PROCESSING_MAPPING_UNSUPPORTED")
        depth = set(self.depth_keys) if self.depth_keys is not None else {key for key in images if "depth" in key}
        codes = {key: np.asarray((0 if key in depth else self.uniform_code) if self.uniform_code is not None
                                else self.codes.get(key, 0), dtype=np.int32) for key in images}
        return {**data, "image_augmentation": codes}


def _uses_augmentation(policy):
    return policy["uniform_augmentation_code"] not in (None, 0) or any(policy["augmentation_codes"].values())


def _apply_data_config(data, policy):
    changes = {}
    if policy["spatial"]:
        repack = data.repack_transforms
        spatial = SourceImagePolicies(tuple(policy["spatial"]))
        inputs = ([spatial, *repack.inputs] if policy["spatial_insertion"] == "before_repack"
                  else [*repack.inputs, spatial])
        changes["repack_transforms"] = dataclasses.replace(repack, inputs=inputs)
    if _uses_augmentation(policy):
        model = data.model_transforms
        routing = ImageAugmentationRouting(policy["augmentation_codes"], policy["uniform_augmentation_code"],
                                           policy.get("depth_keys"))
        changes["model_transforms"] = dataclasses.replace(model, inputs=[*model.inputs, routing])
    return dataclasses.replace(data, **changes) if changes else data


def _register_factory(factory, policy, config_module):
    cls = type(factory)
    if "base_config" not in {field.name for field in dataclasses.fields(factory)}:
        _fail("处理器缺少可复制的 base_config 图像策略绑定接口")
    base = getattr(factory, "base_config", None)
    base = config_module.DataConfig() if base is None else dataclasses.replace(base)
    # Temporary shared-norm factories shallow-copy base_config. Bind the
    # policy to that run-owned object, not a global path: two selected rows may
    # use the same data directory with different processors or image policies.
    factory = dataclasses.replace(factory, base_config=base)
    registry = getattr(cls.create, _HOOK, None)
    if isinstance(registry, dict):
        registry[id(base)] = (base, policy, _apply_data_config)
        return factory
    registry = {id(base): (base, policy, _apply_data_config)}
    original = cls.create

    @functools.wraps(original)
    def create(instance, *args, **kwargs):
        data = original(instance, *args, **kwargs)
        selected = registry.get(id(getattr(instance, "base_config", None)))
        return selected[2](data, selected[1]) if selected is not None else data

    setattr(create, _HOOK, registry)
    cls.create = create
    return factory


def install_image_processing(config, spec, config_module):
    """Install source-bound training transforms after all stage/factory merges."""
    if __package__:
        from .training_plan import read_json
    else:
        from training_plan import read_json
    datasets = getattr(config.data, "datasets", None)
    factories = list(datasets) if datasets is not None else [config.data]
    roles = ([f"dataset_{index}" for index in range(len(factories))]
             if spec.get("dataset_bindings") is not None else ["dataset"])
    resolved = []
    for index, (factory, role) in enumerate(zip(factories, roles)):
        asset = spec["inputs"][role]
        # Reuse metadata only. No factory or loader is called in this resolver.
        info, _ = read_json(Path(asset["path"]) / "meta/info.json", spec)
        # resolve_config is also a public asset-binding helper. Historical
        # callers can bind their deliberately minimal metadata before the real
        # dataset_contract rejects it; no selected image policy is executed.
        if not isinstance(info.get("features"), dict) and asset.get("dataset_profile") is None:
            continue
        policy = image_processing_contract(factory, asset.get("dataset_profile"), info.get("features"),
            code_dir=spec.get("code_dir", ""), model_config=getattr(config, "model", None))
        if policy is not None and (policy["spatial"] or _uses_augmentation(policy)):
            resolved.append((index, policy))
    if any(_uses_augmentation(policy) for _, policy in resolved):
        from openpi.models import model
        if getattr(model, "IMAGE_AUGMENTATION_POLICY_VERSION", None) != POLICY_VERSION:
            _fail("当前 OpenPI 环境不支持逐样本图像增强策略；请升级训练代码后重新预检", "IMAGE_PROCESSING_RUNTIME_UPGRADE_REQUIRED")
    model_config = getattr(config, "model", None)
    if (resolved and datasets is None and type(config.data).__name__ == COMPOSABLE_PROCESSOR
            and type(model_config).__name__ == "Pi0Config" and hasattr(model_config, "fixed_image_augmentation")
            and spec.get("template_id", "openpi-sft") == "openpi-sft"
            and spec.get("entrypoint") in (None, "scripts/train.py")):
        index, policy = resolved[0]
        if _uses_augmentation(policy):
            # This entrypoint uses one composed source, which always produces
            # the declared camera slots. Freeze its policies before JAX traces.
            fixed = tuple(sorted(policy["augmentation_codes"].items()))
            config = dataclasses.replace(config, model=dataclasses.replace(model_config, fixed_image_augmentation=fixed))
            resolved = [(index, {**policy, "augmentation_codes": {}})] if policy["spatial"] else []
    for index, policy in resolved:
        factories[index] = _register_factory(factories[index], policy, config_module)
    if resolved:
        data = (dataclasses.replace(config.data, datasets=factories)
                if datasets is not None else factories[0])
        config = dataclasses.replace(config, data=data)
    return config


def prepare_recap_worker(config, *, spec):
    """Restore process-local FPS and image bindings in a spawned Torch rank."""
    import openpi.training.config as config_module

    if __package__:
        from .dataset_fps import install_registered_fps
    else:
        from dataset_fps import install_registered_fps

    install_registered_fps(spec)
    return install_image_processing(config, spec, config_module)
