"""Bounded, real decoded samples through native repack/data transforms only.

The caller supplies a CPU-only, offline subprocess with a hard timeout and binds
source manifests before/after this check. This module never builds a model, a
worker dataloader, or tokenizer/model transforms. Evidence contains shapes and
source digests, never pixels or task text.
"""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


SAMPLE_SECONDS = 60
MAX_SOURCE_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
LOADER_MODULE = "openpi.training.data_loader"
FACTORY_MODULE = "openpi.training.config"
CHILD_TIMEOUT_SECONDS = 70
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 256 * 1024


def _fail(message, code="DATASET_SAMPLE_INVALID"):
    raise ValueError(code + ": " + message[:1800])


def _remaining(deadline):
    if time.monotonic() > deadline:
        _fail("真实解码样本检查超过 60 秒预算；请检查数据读取后重新预检", "DATASET_SAMPLE_CHECK_LIMIT")


def _source(function, spec):
    """Bind the actual imported implementation, not a user-supplied module name."""
    filename = inspect.getsourcefile(function)
    if not filename:
        _fail("无法确认 native 样本接口的实现来源", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
    path, root = Path(filename).resolve(strict=True), Path(spec["code_dir"]).resolve(strict=True)
    if not path.is_relative_to(root):
        _fail("样本接口或加载器不属于当前登记的 OpenPI 项目", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_SOURCE_BYTES:
            _fail("样本接口源码必须是受限大小的普通文件", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        contents = stream.read(MAX_SOURCE_BYTES + 1)
    if len(contents) > MAX_SOURCE_BYTES:
        _fail("样本接口源码超过大小限制", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(contents).hexdigest()}


def _field(data, key):
    # LeRobot features normally use literal dotted keys; retain native nested
    # dictionary support without evaluating a path expression.
    if key in data:
        return data[key]
    value = data
    for part in key.split("."):
        value = value[part]
    return value


def _array(value, np, *, numeric=False):
    if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
        _fail("样本检查只能读取 CPU 张量")
    value = np.asarray(value)
    if (value.ndim > 5 or not value.size or value.nbytes > MAX_ARRAY_BYTES
            or value.dtype.kind not in "biuf"):
        _fail("样本数组必须是受限大小的非空数值数组")
    if numeric and not np.all(np.isfinite(value)):
        _fail("处理后的样本包含非有限数值")
    return value, {"shape": list(value.shape), "dtype": str(value.dtype)}


def _camera_evidence(raw, cameras, np):
    evidence = []
    for camera in cameras:
        try:
            _, observed = _array(_field(raw, camera["key"]), np)
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"无法读取 camera={camera['key']}，layout={camera['layout']}：{str(exc)[:300]}")
        evidence.append({"key": camera["key"], "slot": camera["slot"], "kind": camera["kind"],
                         "layout": camera["layout"], "time_index": camera["time_index"], **observed})
    return evidence


def _model_view_evidence(data, model_config, np):
    """Check selected views once in the CPU sample probe, outside the train JIT.

    Samples are unbatched HWC images or THWC histories. Match the model's view
    selection and frame-validity rules without importing JAX or model code.
    """
    images, masks = data.get("image"), data.get("image_mask")
    if (not isinstance(images, dict) or not isinstance(masks, dict)
            or any(not isinstance(key, str) or not key for key in images)):
        _fail("模型视觉预检需要图像与掩码字典")
    available = tuple(images)
    depth = getattr(model_config, "depth_keys", None)
    depth = [key for key in available if "depth" in key] if depth is None else depth
    if not isinstance(depth, (tuple, list)):
        _fail("模型 image_keys/depth_keys 必须是无重复的视觉槽列表")
    image_keys = getattr(model_config, "image_keys", None)
    image_keys = [key for key in available if key not in depth] if image_keys is None else image_keys
    for keys in (image_keys, depth):
        if (not isinstance(keys, (tuple, list)) or len(keys) > 64
                or any(not isinstance(key, str) or not key or len(key) > 256 for key in keys)
                or len(set(keys)) != len(keys)):
            _fail("模型 image_keys/depth_keys 必须是无重复的视觉槽列表")
    selected = [*image_keys, *depth]
    if not selected or set(image_keys) & set(depth):
        _fail("模型 RGB/深度视觉槽不可交叉或全部为空")
    allow_empty = getattr(model_config, "allow_all_masked_images", False)
    if type(allow_empty) is not bool:
        _fail("模型 allow_all_masked_images 必须是布尔值")
    frames = data.get("frame_valid_mask")
    if frames is not None and not isinstance(frames, dict):
        _fail("历史图像有效帧掩码必须是字典")
    valid_slots = []
    for key in selected:
        # A selected view missing from the sample is padded and masked by the
        # model; another selected view must be valid unless ablation is explicit.
        if key not in images:
            continue
        image, _ = _array(images[key], np, numeric=True)
        if image.ndim not in (3, 4) or image.shape[-1] != 3:
            _fail("模型视觉预检需要 HWC 图像或 THWC 历史图像：" + key)
        mask = np.asarray(masks.get(key, True))
        if mask.shape != () or mask.dtype.kind != "b":
            _fail("样本图像掩码必须是标量布尔值：" + key)
        valid = bool(mask)
        if frames is not None and image.ndim == 4:
            frame_mask = np.asarray(frames.get(key, np.full(image.shape[0], valid, dtype=bool)))
            if frame_mask.shape != (image.shape[0],) or frame_mask.dtype.kind != "b":
                _fail("历史图像有效帧掩码与时间维度不一致：" + key)
            valid = valid and bool(np.any(frame_mask))
        if valid:
            valid_slots.append(key)
    if not valid_slots and not allow_empty:
        _fail("模型所选视觉输入均无有效图像；请检查 image_keys/depth_keys、image_mask 和 frame_valid_mask")
    return {"checked": True, "selected_slots": selected, "valid_slots": valid_slots,
            "allow_all_masked_images": allow_empty}


def _output_evidence(data, composition, horizon, np, model_config):
    state, state_summary = _array(data["state"], np, numeric=True)
    actions, action_summary = _array(data["actions"], np, numeric=True)
    state_shape = (composition["state_history"], sum(field["width"] for field in composition["state_fields"]))
    action_shape = (horizon, sum(field["width"] for field in composition["action_layout"]))
    if state.shape != state_shape or actions.shape != action_shape:
        _fail("native 数据处理输出的状态/动作形状与本次组合不一致")
    images, masks = data["image"], data["image_mask"]
    if not isinstance(images, dict) or not isinstance(masks, dict) or set(images) != set(masks):
        _fail("native 数据处理必须返回一致的图像和图像掩码")
    active = {camera["slot"] for camera in composition["cameras"]}
    if not active <= set(images):
        _fail("native 数据处理缺少已配置的视觉输入")
    image_summary, mask_summary = {}, {}
    for slot in sorted(images):
        image, summary = _array(images[slot], np, numeric=True)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            _fail("native 数据处理图像必须是 HWC 三通道 uint8：" + slot)
        mask = np.asarray(masks[slot])
        if mask.shape != () or mask.dtype.kind != "b" or bool(mask) != (slot in active):
            _fail("native 数据处理图像掩码与本次视觉配置不一致：" + slot)
        image_summary[slot], mask_summary[slot] = summary, bool(mask)
    return {"state": state_summary, "actions": action_summary,
            "image": image_summary, "image_mask": mask_summary,
            "model_views": _model_view_evidence(data, model_config, np)}


def inspect_composed_samples(factory, config, spec, dataset_evidence):
    """Require positive native decoded-sample evidence for the known composition.

    First/middle/last global records are deterministic and bounded. They detect
    decoder/layout mismatches, but do not establish that every frame decodes.
    Legacy processors explicitly make no sampled-processing claim here.
    """
    factory_type = type(factory)
    if factory_type.__name__ != "ComposableDataConfig" or factory_type.__module__ != FACTORY_MODULE:
        return {"schema_version": 1, "status": "not_applicable", "checked_samples": 0,
                "scope": "no decoded-sample processing claim for this processor"}
    deadline = time.monotonic() + SAMPLE_SECONDS
    path, index, observed = dataset_evidence.get("path", ""), None, []
    try:
        if __package__:
            from .composition_spec import normalize_composition
        else:
            from composition_spec import normalize_composition
        composition = dataset_evidence.get("composition", {}).get("spec")
        if not isinstance(composition, dict) or not composition.get("cameras"):
            _fail("组合数据集缺少已核验的处理布局")
        # Do not permit a different factory composition/path to reuse evidence.
        if normalize_composition(factory.composition) != composition:
            _fail("native 处理布局与本次已核验的组合配置不一致")
        if not Path(path).is_absolute() or not Path(factory.repo_id).is_absolute():
            _fail("native 样本检查只能读取登记的本机绝对路径")
        expected = Path(path).resolve(strict=True)
        if Path(factory.repo_id).resolve(strict=True) != expected:
            _fail("native 样本处理器改变了登记的数据集路径")
        hook = getattr(factory, "processing_sample_config", None)
        if not callable(hook):
            _fail("训练代码缺少真实样本检查接口，请更新 OpenPI 项目后重新预检", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        hook_source = _source(hook, spec)
        _remaining(deadline)
        data_config = hook(config.assets_dirs, config.model)
        _remaining(deadline)
        if not Path(data_config.repo_id).is_absolute() or Path(data_config.repo_id).resolve(strict=True) != expected:
            _fail("native 样本配置改变了登记的数据集路径")
        if tuple(getattr(data_config.model_transforms, "inputs", ())) or tuple(getattr(data_config.model_transforms, "outputs", ())):
            _fail("native 样本接口必须排除 tokenizer/model transforms", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        sequence_keys = dataset_evidence["composition"]["action_sequence_keys"]
        if list(data_config.action_sequence_keys) != sequence_keys:
            _fail("native 动作序列采样配置与已核验的布局不一致")
        transforms = tuple(data_config.repack_transforms.inputs) + tuple(data_config.data_transforms.inputs)
        if not transforms or not all(callable(transform) for transform in transforms):
            _fail("native 样本接口缺少实际 repack/data transforms", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        loader = importlib.import_module(LOADER_MODULE).create_torch_dataset
        loader_source = _source(loader, spec)
        _remaining(deadline)
        horizon = config.model.action_horizon
        if type(horizon) is not int or horizon <= 0:
            _fail("训练动作预测长度必须是正整数")
        raw_dataset = loader(data_config, horizon, config.model)
        _remaining(deadline)
        length = len(raw_dataset)
        if type(length) is not int or length <= 0 or length != dataset_evidence.get("frames"):
            _fail("真实加载器数据集为空，或帧数与已核验元信息不一致")
        np = importlib.import_module("numpy")
        indices = sorted({0, length // 2, length - 1})
        samples = []
        for index in indices:
            observed = []
            _remaining(deadline)
            data = raw_dataset[index]
            _remaining(deadline)
            if not isinstance(data, dict):
                _fail("真实加载器未返回样本字典")
            observed = _camera_evidence(data, composition["cameras"], np)
            for transform in transforms:
                _remaining(deadline)
                data = transform(data)
                _remaining(deadline)
            output = _output_evidence(data, composition, horizon, np, config.model)
            _remaining(deadline)
            samples.append({"index": index, "raw_cameras": observed, "output": output})
        if not samples:
            _fail("未取得任何真实解码样本", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        if hook_source != _source(hook, spec) or loader_source != _source(loader, spec):
            _fail("样本检查过程中 native 接口源码已改变，请重新预检")
        _remaining(deadline)
        return {"schema_version": 1, "status": "verified", "path": str(expected),
                "scope": "bounded decoded samples through native repack and data transforms; no model transforms",
                "selection": "first_middle_last", "checked_samples": len(samples), "sample_indices": indices,
                "action_horizon": horizon, "action_sequence_keys": sequence_keys,
                "runtime": {"hook": hook_source, "loader": loader_source}, "samples": samples}
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("DATASET_SAMPLE_CHECK_"):
            raise
        camera_context = "; ".join(
            f"camera={camera['key']}, actual_shape={camera['shape']}, layout={camera['layout']}, time_index={camera['time_index']}"
            for camera in observed)
        # Native exception strings can contain arbitrary data: retain only a
        # bounded diagnostic, never repr(sample), arrays, or task content.
        message = str(exc).splitlines()[0][:500]
        _fail(f"数据集 {path}，sample_index={index}：{message}。{camera_context}")


def _stop_child(process):
    """Always remove decoder descendants, including after a successful probe."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait(timeout=5)


def inspect_composed_samples_isolated(factory, config, spec, dataset_evidence):
    """Decode exclusively in a fresh interpreter, preserving training workers.

    Loading video codecs in the eventual training parent can poison subsequent
    worker creation. Only JSON crosses this boundary; factories/models/decoders
    are reconstructed in the disposable CPU child and are never pickled.
    """
    factory_type = type(factory)
    if factory_type.__name__ != "ComposableDataConfig" or factory_type.__module__ != FACTORY_MODULE:
        return {"schema_version": 1, "status": "not_applicable", "checked_samples": 0,
                "scope": "no decoded-sample processing claim for this processor"}
    children = getattr(config.data, "datasets", None)
    factories = [config.data] if children is None else children
    if not isinstance(factories, (tuple, list)) or not 1 <= len(factories) <= 64:
        _fail("父进程的数据集绑定数量无效")
    matches = [index for index, candidate in enumerate(factories) if candidate is factory]
    if len(matches) != 1:
        _fail("父进程不能确认当前处理器的唯一绑定索引")
    factory_index = matches[0]
    try:
        request = json.dumps({"schema_version": 1, "spec": spec, "dataset_evidence": dataset_evidence,
                              "factory_index": factory_index},
                             ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        _fail("样本子进程请求不能编码为有限 JSON：" + str(exc)[:200])
    if len(request) > MAX_REQUEST_BYTES:
        _fail("样本子进程请求超过大小限制", "DATASET_SAMPLE_CHECK_LIMIT")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                       WANDB_MODE="disabled", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = None
    try:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--sample-probe"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=spec["code_dir"], env=environment, start_new_session=True)
        stdout, stderr = process.communicate(request, timeout=CHILD_TIMEOUT_SECONDS)
        if len(stdout) > MAX_RESPONSE_BYTES or len(stderr) > MAX_RESPONSE_BYTES:
            _fail("样本子进程输出超过大小限制", "DATASET_SAMPLE_CHECK_LIMIT")
        try:
            response = json.loads(stdout)
        except (ValueError, UnicodeDecodeError) as exc:
            _fail(f"样本解码子进程未返回有效 JSON（退出码 {process.returncode}），请检查解码器或训练环境",
                  "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        if not isinstance(response, dict) or response.get("schema_version") != 1 or type(response.get("ok")) is not bool:
            _fail("样本子进程返回不支持的结果结构", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        if not response["ok"]:
            message = response.get("error")
            if not isinstance(message, str) or not message.startswith("DATASET_SAMPLE_"):
                message = "DATASET_SAMPLE_INVALID: 样本解码子进程未完成验证"
            raise ValueError(message[:2000])
        result = response.get("result")
        if (process.returncode != 0 or not isinstance(result, dict) or result.get("schema_version") != 1
                or result.get("status") != "verified" or type(result.get("checked_samples")) is not int
                or not 1 <= result["checked_samples"] <= 3 or not isinstance(result.get("samples"), list)
                or len(result["samples"]) != result["checked_samples"]
                or result.get("path") != str(Path(dataset_evidence["path"]).resolve(strict=True))
                or result.get("action_horizon") != config.model.action_horizon
                or result.get("action_sequence_keys") != dataset_evidence["composition"]["action_sequence_keys"]):
            _fail("样本子进程未提供与本次绑定一致的正向验证结果", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
        return result
    except subprocess.TimeoutExpired:
        _fail("真实解码样本子进程超过 70 秒硬期限", "DATASET_SAMPLE_CHECK_LIMIT")
    except (OSError, subprocess.SubprocessError) as exc:
        _fail("无法执行独立样本检查子进程：" + str(exc)[:300], "DATASET_SAMPLE_CHECK_UNAVAILABLE")
    finally:
        if process is not None:
            _stop_child(process)


def _child_result(request):
    """Resolve the exact binding index, permitting repeated dataset paths."""
    if (not isinstance(request, dict) or set(request) != {"schema_version", "spec", "dataset_evidence", "factory_index"}
            or request["schema_version"] != 1 or not isinstance(request["spec"], dict)
            or not isinstance(request["dataset_evidence"], dict)
            or type(request["factory_index"]) is not int or not 0 <= request["factory_index"] < 64):
        _fail("样本子进程请求结构无效")
    spec, evidence = request["spec"], request["dataset_evidence"]
    # package and standalone Agent bundle execution both use this same resolver.
    entry = importlib.import_module((__package__ + "." if __package__ else "") + "openpi_entry")
    config, _ = entry.prepare_config(spec)
    factories = getattr(config.data, "datasets", None)
    factories = [config.data] if factories is None else factories
    if not isinstance(factories, (list, tuple)) or not 1 <= len(factories) <= 64:
        _fail("子进程重新解析的数据集绑定数量无效")
    target = Path(evidence["path"])
    if not target.is_absolute():
        _fail("样本子进程需要已登记的绝对数据集路径")
    target = target.resolve(strict=True)
    index = request["factory_index"]
    if index >= len(factories):
        _fail("子进程重新解析的处理器绑定索引超出范围")
    selected = factories[index]
    if not getattr(selected, "repo_id", None) or Path(selected.repo_id).resolve(strict=True) != target:
        _fail("子进程重新解析的处理器绑定路径与本次数据集不一致")
    result = inspect_composed_samples(selected, config, spec, evidence)
    if result.get("status") != "verified":
        _fail("子进程解析出的处理器不支持真实组合样本检查", "DATASET_SAMPLE_CHECK_UNAVAILABLE")
    return result


def _child_main():
    # The outer Agent probe may die before our parent's finally block runs.
    # A separate process group therefore also needs an independent hard expiry.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(CHILD_TIMEOUT_SECONDS - 1)
    # Keep the IPC descriptor separate, then discard Python *and native codec*
    # diagnostics at FD level. Only a bounded JSON envelope can reach the parent;
    # noisy libraries cannot fill communicate()'s stdout/stderr memory buffers.
    reply = os.dup(sys.stdout.fileno())
    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        try:
            payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
            if len(payload) > MAX_REQUEST_BYTES:
                _fail("样本子进程请求超过大小限制", "DATASET_SAMPLE_CHECK_LIMIT")
            response = {"schema_version": 1, "ok": True, "result": _child_result(json.loads(payload))}
        except Exception as exc:
            message = str(exc).splitlines()[0][:1800]
            if not message.startswith("DATASET_SAMPLE_"):
                message = "DATASET_SAMPLE_INVALID: " + message
            response = {"schema_version": 1, "ok": False, "error": message}
        output = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if len(output) > MAX_RESPONSE_BYTES:
            output = b'{"schema_version":1,"ok":false,"error":"DATASET_SAMPLE_CHECK_LIMIT: response too large"}'
        pending = memoryview(output)
        while pending:
            pending = pending[os.write(reply, pending):]
        return 0 if response["ok"] else 2
    finally:
        signal.alarm(0)
        os.close(reply)


if __name__ == "__main__":
    if sys.argv[1:] != ["--sample-probe"]:
        raise SystemExit(2)
    raise SystemExit(_child_main())
