"""Serving consumes observations without future actions or training labels."""

from copy import deepcopy
import ast
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")

from vla_adapters.policy_inference import ComposedActionOutputs, ComposedInferenceInputs, compile_inference_config


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "tests/fixtures/composed-processing-v1.json").read_text())["valid"][0]


def observation(spec):
    sample = {field["key"]: np.full((spec["state_history"], field["width"]), .5, dtype=np.float32)
              for field in spec["state_fields"]}
    for camera in spec["cameras"]:
        sample[camera["key"]] = np.arange(18, dtype=np.uint8).reshape((3, 2, 3) if camera["layout"] == "chw" else (2, 3, 3))
    sample[spec["prompt_key"] or "prompt"] = "pick the object"
    return sample


@pytest.fixture
def training_core(monkeypatch):
    source = ROOT / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    loader = importlib.util.spec_from_file_location("policy_inference_training_core", source)
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, module)
    loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_observations_match_training_preprocessing_without_target_actions(training_core, layout):
    spec = deepcopy(SPEC)
    spec["cameras"][0]["layout"] = layout
    raw = observation(spec)
    width = sum(field["width"] for field in spec["action_layout"])
    training = {**raw, spec["action_key"]: np.zeros((16, width), dtype=np.float32)}
    expected = training_core.ComposedInputs(spec, 16, 32)(training_core.ComposedRepack(spec)(training))
    actual = ComposedInferenceInputs(spec)(raw)
    assert "actions" not in actual
    assert "actions" not in raw
    np.testing.assert_array_equal(actual["state"], expected["state"])
    for slot in expected["image"]:
        np.testing.assert_array_equal(actual["image"][slot], expected["image"][slot])
        assert actual["image_mask"][slot] == expected["image_mask"][slot]
    assert actual["prompt"] == expected["prompt"]


def test_json_rgb_is_converted_by_declared_kind_and_depth_retains_precision():
    spec = deepcopy(SPEC)
    raw = observation(spec)
    camera = spec["cameras"][0]
    raw[camera["key"]] = raw[camera["key"]].tolist()
    result = ComposedInferenceInputs(spec)(raw)
    assert result["image"][camera["slot"]].dtype == np.uint8
    camera.update(kind="depth", layout="hwc", depth_min=0, depth_max=2)
    raw[camera["key"]] = [[0., .5], [1., 2.]]
    result = ComposedInferenceInputs(spec)(raw)
    np.testing.assert_array_equal(result["image"][camera["slot"]][:, :, 0], [[0, 63], [127, 255]])


def test_history_nested_source_paths_missing_fields_and_invalid_shapes():
    spec = deepcopy(SPEC)
    spec["state_history"] = 2
    spec["state_fields"][0]["key"] = "observation/joints"
    raw = observation(spec)
    raw["observation"] = {"joints": raw.pop("observation/joints")}
    result = ComposedInferenceInputs(spec)(raw)
    assert result["state"].shape == (2, 7)
    del raw["observation"]["joints"]
    with pytest.raises(ValueError, match="缺少观测字段"):
        ComposedInferenceInputs(spec)(raw)
    raw = observation(spec)
    raw["observation/joints"] = np.zeros((6,))
    with pytest.raises(ValueError, match="状态形状"):
        ComposedInferenceInputs(spec)(raw)


def test_action_supervision_is_unused_and_acp_condition_is_explicit():
    spec = deepcopy(SPEC)
    spec["aux_keys"] = ["is_positive", "value_target"]
    raw = observation(spec)
    raw["actions"] = "not a future action target"
    result = ComposedInferenceInputs(spec, positive_condition=True)(raw)
    assert result["is_positive"].item() is True
    assert "actions" not in result and "value_target" not in result
    raw["is_positive"] = 0
    assert ComposedInferenceInputs(spec, True)(raw)["is_positive"].item() is False
    raw["is_positive"] = 2
    with pytest.raises(ValueError, match="is_positive"):
        ComposedInferenceInputs(spec, True)(raw)


def test_rejects_nonfinite_and_wrong_image_or_gripper_units():
    spec = deepcopy(SPEC)
    for key, value, message in (
        (spec["state_fields"][0]["key"], np.full((6,), np.nan), "有限实数"),
        (spec["state_fields"][1]["key"], [2.], "夹爪状态"),
        (spec["cameras"][0]["key"], np.zeros((4, 2, 3)), "CHW/HWC"),
    ):
        raw = observation(spec)
        raw[key] = value
        with pytest.raises(ValueError, match=message):
            ComposedInferenceInputs(spec)(raw)


@dataclasses.dataclass(frozen=True)
class Group:
    inputs: tuple = ()
    outputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class ACPConditionPrompt:
    dropout: float = .5
    default_positive: bool = False


@dataclasses.dataclass(frozen=True)
class DataConfig:
    data_transforms: Group
    model_transforms: Group
    normalize_masks: object


@dataclasses.dataclass(frozen=True)
class Factory:
    config: DataConfig

    def create(self, *args):
        return self.config


@dataclasses.dataclass(frozen=True)
class Config:
    data: Factory
    model: object


def test_compilation_preserves_model_norms_and_existing_transforms(training_core):
    original_output = object()
    model_output = object()
    normalizer = object()
    original = DataConfig(Group((training_core.ComposedInputs(SPEC, 16, 32),), (original_output,)),
                          Group((ACPConditionPrompt(),), (model_output,)), normalizer)
    config = Config(Factory(original), object())
    compiled = compile_inference_config(config)
    data = compiled.data.create(None, None)
    assert compiled.model is config.model
    assert data.normalize_masks is normalizer
    assert data.model_transforms.outputs[0] is model_output
    assert data.data_transforms.outputs[0] is original_output
    assert data.model_transforms.inputs[0].dropout == 0
    assert isinstance(data.data_transforms.inputs[0], ComposedInferenceInputs)
    assert data.data_transforms.inputs[0].positive_condition
    assert original.model_transforms.inputs[0].dropout == .5
    assert isinstance(data.model_transforms.outputs[-1], ComposedActionOutputs)
    output = data.model_transforms.outputs[-1]({"actions": np.ones((16, 32)), "state": np.ones((32,))})
    assert output["actions"].shape == (16, 10)
    assert output["state"].shape == (1, 7)


def test_non_composition_preserves_its_complete_data_config():
    original = DataConfig(Group((object(),), (object(),)), Group(), object())
    config = Config(Factory(original), object())
    assert compile_inference_config(config).data.create(None, None) is original


def test_trim_and_history_restore_precede_real_quantile_unnormalization():
    # Exercise the checked-out OpenPI math without importing its JAX/Flax
    # runtime into the CPU-only control-plane environment.
    tree = ast.parse((ROOT / "third_party/openpi-umi/src/openpi/transforms.py").read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Unnormalize")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "_unnormalize_quantile")
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "real_openpi_unnormalize", "exec"), namespace)
    unnormalize = namespace["_unnormalize_quantile"]

    @dataclasses.dataclass
    class Stats:
        q01: object
        q99: object
        min: object = None
        max: object = None

    output = ComposedActionOutputs(10, state_width=7, state_history=2)(
        {"actions": np.zeros((16, 32)), "state": np.zeros((32,))})
    for key, width, shape in (("actions", 10, (16, 10)), ("state", 7, (2, 7))):
        stats = Stats(np.zeros(width), np.full(width, 2.))
        restored = unnormalize(None, output[key], stats, norm_mask=tuple([True] * width))
        assert restored.shape == shape
        np.testing.assert_allclose(restored, 1.)
