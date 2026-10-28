"""Observation-only serving for explicitly declared composable data contracts.

Training's ComposedInputs consumes future actions and supervision. Serving uses
the same declared state/image layouts without creating training targets. The
normalization and model transforms still come from the resolved training config.
"""

from copy import deepcopy
import dataclasses

from .composition_spec import SLOTS, normalize_composition


def _fail(message):
    raise ValueError("POLICY_OBSERVATION_INVALID: " + message)


def _field(data, key):
    if key in data:
        return data[key]
    current = data
    for part in key.split("/"):
        if not isinstance(current, dict) or part not in current:
            _fail("缺少观测字段：" + key)
        current = current[part]
    return current


@dataclasses.dataclass(frozen=True)
class ComposedInferenceInputs:
    composition: dict
    positive_condition: bool = False

    def __call__(self, data):
        import numpy as np

        if not isinstance(data, dict):
            _fail("观测必须是对象")
        spec = normalize_composition(self.composition)

        def numeric(value, key):
            try:
                result = np.asarray(value)
            except (ValueError, TypeError) as exc:
                _fail("无法读取数值字段：" + key)
            if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
                _fail("数值字段需要有限实数：" + key)
            return result

        states = []
        for field in spec["state_fields"]:
            value = numeric(_field(data, field["key"]), field["key"])
            if spec["state_history"] == 1 and value.shape == (field["width"],):
                value = value.reshape(1, field["width"])
            if value.shape != (spec["state_history"], field["width"]):
                _fail("状态形状与声明的历史长度或宽度不一致：" + field["key"])
            if field["kind"] == "gripper" and spec["gripper_range"] is not None:
                lower, upper = spec["gripper_range"]
                if np.any(value < lower) or np.any(value > upper):
                    _fail("夹爪状态超出声明范围：" + field["key"])
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype(np.float32)
            if not np.isfinite(value).all():
                _fail("状态数值超出 float32 范围：" + field["key"])
            states.append(value)

        images, masks = {}, {}
        for camera in spec["cameras"]:
            image = np.asarray(_field(data, camera["key"]))
            time_index = camera["time_index"]
            if time_index is not None:
                dimensions = {4} if camera["kind"] == "rgb" else {3, 4}
                if image.ndim not in dimensions or not -image.shape[0] <= time_index < image.shape[0]:
                    _fail("历史图像缺少有效的显式时间索引：" + camera["key"])
                image = image[time_index]
            if camera["kind"] == "rgb":
                channel = 0 if camera["layout"] == "chw" else -1
                if image.ndim != 3 or image.shape[channel] != 3:
                    _fail("RGB 图像必须匹配声明的 CHW/HWC 布局：" + camera["key"])
                if image.dtype.kind in "iu" and np.all((image >= 0) & (image <= 255)):
                    image = image.astype(np.uint8)
                elif image.dtype.kind == "f" and np.isfinite(image).all() and np.all((image >= 0) & (image <= 1)):
                    image = (image * 255).astype(np.uint8)
                else:
                    _fail("RGB 需要 0..255 整数或 0..1 有限浮点数：" + camera["key"])
                if camera["layout"] == "chw":
                    image = np.transpose(image, (1, 2, 0))
            else:
                if image.ndim == 3:
                    channel = 0 if camera["layout"] == "chw" else -1
                    if image.shape[channel] != 1:
                        _fail("深度图只能有一个显式通道：" + camera["key"])
                    image = np.squeeze(image, axis=channel)
                if image.ndim != 2:
                    _fail("深度图需要 HW 或声明的单通道布局：" + camera["key"])
                image = numeric(image, camera["key"])
                scaled = np.clip((image - camera["depth_min"]) / (camera["depth_max"] - camera["depth_min"]), 0, 1)
                image = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)
            if not 1 <= image.shape[0] <= 4096 or not 1 <= image.shape[1] <= 4096:
                _fail("图像尺寸必须位于 1..4096：" + camera["key"])
            images[camera["slot"]], masks[camera["slot"]] = image, np.True_
        reference = next(iter(images.values()))
        for slot in SLOTS:
            if slot not in images:
                images[slot], masks[slot] = np.zeros_like(reference), np.False_

        claimed = {row["key"] for row in spec["state_fields"] + spec["cameras"]}
        claimed.update([spec["action_key"], *spec["aux_keys"]])
        prompt = None
        if spec["prompt_key"]:
            try:
                prompt = _field(data, spec["prompt_key"])
            except ValueError:
                # create_trained_policy injects its configured default into
                # prompt before this transform, regardless of dataset key.
                prompt = data.get("prompt")
        else:
            for key in ("task", "prompt"):
                if any(source == key or source.startswith(key + "/") for source in claimed):
                    continue
                candidate = data.get(key)
                if candidate is not None:
                    if not isinstance(candidate, str):
                        _fail("任务字段必须是文本：" + key)
                    if candidate.strip():
                        prompt = candidate
                        break
            prompt = prompt or ""
        if isinstance(prompt, np.ndarray) and prompt.shape == ():
            prompt = prompt.item()
        if not isinstance(prompt, str) or spec["prompt_key"] and not prompt.strip():
            _fail("任务字段需要非空文本：" + spec["prompt_key"])
        result = {"state": np.concatenate(states, axis=-1), "image": images, "image_mask": masks, "prompt": prompt}
        if self.positive_condition:
            # ACP inference predicts positive actions unless the caller chooses
            # a negative condition. This is conditioning, not a fabricated label.
            condition = np.asarray(data.get("is_positive", True))
            if condition.size != 1 or condition.dtype.kind not in "biuf" or condition.reshape(-1)[0] not in (0, 1):
                _fail("ACP 推理条件 is_positive 需要单个 0/1 值")
            result["is_positive"] = condition.astype(np.bool_).reshape(())
        return result


@dataclasses.dataclass(frozen=True)
class ComposedActionOutputs:
    width: int
    state_width: int | None = None
    state_history: int = 1

    def __call__(self, data):
        import numpy as np

        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < self.width:
            raise ValueError("POLICY_ACTION_INVALID: 输出动作形状小于声明的真实动作布局")
        result = {**data, "actions": actions[:, :self.width]}
        if self.state_width is not None and "state" in data:
            state = np.asarray(data["state"]).reshape(-1)
            count = self.state_width * self.state_history
            if state.size < count:
                raise ValueError("POLICY_ACTION_INVALID: 输出状态小于声明的观测历史宽度")
            result["state"] = state[:count].reshape(self.state_history, self.state_width)
        return result


@dataclasses.dataclass(frozen=True)
class _InferenceDataFactory:
    original: object

    def create(self, assets_dirs, model_config):
        original = self.original.create(assets_dirs, model_config)
        inputs = list(original.data_transforms.inputs)
        composition = [item for item in inputs if type(item).__name__ == "ComposedInputs" and hasattr(item, "composition")]
        if not composition:
            return original
        if len(composition) != 1:
            raise ValueError("POLICY_CONFIG_UNSUPPORTED: 组合推理需要唯一的观测处理合同")
        marker = composition[0]
        spec = normalize_composition(marker.composition)
        model_inputs, positive = [], False
        for transform in original.model_transforms.inputs:
            if type(transform).__name__ == "ACPConditionPrompt":
                if not dataclasses.is_dataclass(transform):
                    raise ValueError("POLICY_CONFIG_UNSUPPORTED: 无法固定 ACP 推理条件")
                transform = dataclasses.replace(transform, dropout=0.0, default_positive=True)
                positive = True
            model_inputs.append(transform)
        adapted_inputs = [ComposedInferenceInputs(deepcopy(spec), positive) if item is marker else item for item in inputs]
        width = sum(item["width"] for item in spec["action_layout"])
        state_width = sum(item["width"] for item in spec["state_fields"])
        return dataclasses.replace(original,
            data_transforms=dataclasses.replace(original.data_transforms, inputs=adapted_inputs),
            # create_trained_policy runs model outputs before Unnormalize. Trim
            # model padding here so the original raw-width masks and norm stats
            # remain valid, and restore state history before its normalization.
            model_transforms=dataclasses.replace(original.model_transforms, inputs=model_inputs,
                outputs=[*original.model_transforms.outputs,
                         ComposedActionOutputs(width, state_width, spec["state_history"])]))


def compile_inference_config(train_config):
    """Preserve the resolved model and weights; adapt only data.create for serving.

    The wrapper defers factory creation to the policy loader, preserving the
    project's own norm sources, padding, tokenizers and model output transforms.
    Non-composable data configs are returned unchanged by the wrapper.
    """
    return dataclasses.replace(train_config, data=_InferenceDataFactory(train_config.data))
