"""Real pixel geometry and per-source augmentation routing through the adapter."""
from copy import deepcopy
import dataclasses
import importlib.util
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from vla_adapters import image_runtime as runtime
from vla_adapters.image_runtime import (ImageAugmentationRouting, SourceImagePolicies,
    image_processing_contract, install_image_processing, resize_source_image)
from vla_adapters.policy_inference import InferenceImagePolicies, compile_inference_config
from vla_adapters.portable_policy import _Codec
from vla_adapters import submission_validation
from test_submission_validation import dataset, source_summary  # noqa: F401
from test_training_plan import training  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "tests/fixtures/composed-processing-v1.json").read_text())["valid"][0]


def pixels():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    image[:, 16:48, 1] = 255
    image[:, 48:, 2] = 255
    return image


@pytest.mark.parametrize("mode", ["pad", "center_crop", "stretch"])
def test_real_pixels_distinguish_resize_choices(mode):
    raw = pixels()
    actual = resize_source_image(raw, layout="hwc", kind="rgb", resize_mode=mode)
    assert actual.shape == (224, 224, 3) and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[112, 112], [0, 255, 0])
    if mode == "pad":
        assert not actual[:56].any() and not actual[168:].any()
        np.testing.assert_array_equal(actual[112, 8], [255, 0, 0])
    elif mode == "stretch":
        np.testing.assert_array_equal(actual[8, 8], [255, 0, 0])
        np.testing.assert_array_equal(actual[8, -8], [0, 0, 255])
    else:
        assert not actual[:, 4:-4, 0].any() and not actual[:, 4:-4, 2].any()
        assert actual[:, 4:-4, 1].min() == 255
    np.testing.assert_array_equal(raw, pixels())


@pytest.mark.parametrize("layout,sequence", [("chw", False), ("hwc", False), ("chw", True), ("hwc", True)])
def test_layout_time_axis_dtype_and_float_range_survive(layout, sequence):
    raw = pixels().astype(np.float32) / 255
    if layout == "chw":
        raw = raw.transpose(2, 0, 1)
    if sequence:
        raw = np.stack((raw, raw * .5))
    actual = resize_source_image(raw, layout=layout, kind="rgb", resize_mode="stretch")
    shape = (3, 224, 224) if layout == "chw" else (224, 224, 3)
    assert actual.shape == ((2, *shape) if sequence else shape)
    assert actual.dtype == np.float32 and actual.min() == 0 and actual.max() == 1
    if sequence:
        np.testing.assert_allclose(actual[1], actual[0] * .5)


def test_depth_history_preserves_units_and_declared_black_value():
    raw = np.full((1, 24, 48), 1500, dtype=np.uint16)
    actual = resize_source_image(raw, layout="hwc", kind="depth", resize_mode="pad", fill=1000, history=True)
    assert actual.shape == (1, 224, 224) and actual.dtype == np.uint16
    assert np.all(actual[:, :56] == 1000) and np.all(actual[:, 56:168] == 1500)


def test_360_by_640_pad_matches_openpi_126_content_rows_and_49_black_rows():
    actual = resize_source_image(np.full((360, 640, 3), 255, dtype=np.uint8),
        layout="hwc", kind="rgb", resize_mode="pad")
    assert np.all(actual[:49] == 0) and np.all(actual[175:] == 0)
    assert np.all(actual[49:175] == 255)


@pytest.mark.parametrize("raw,layout", [(np.zeros((4, 12, 16)), "chw"),
    (np.full((3, 12, 16), -1., dtype=np.float32), "chw"),
    (np.ones((12, 16, 3), dtype=np.uint16), "hwc"),
    (np.ones((12, 16, 3), dtype=np.uint8), "unknown")])
def test_invalid_source_layout_or_normalization_is_rejected(raw, layout):
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_DATA_INVALID"):
        resize_source_image(raw, layout=layout, kind="rgb", resize_mode="pad")


@dataclasses.dataclass(frozen=True)
class Group:
    inputs: tuple = ()
    outputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repack_transforms: Group = dataclasses.field(default_factory=Group)
    data_transforms: Group = dataclasses.field(default_factory=Group)
    model_transforms: Group = dataclasses.field(default_factory=Group)


@dataclasses.dataclass(frozen=True)
class RepackTransform:
    structure: dict
    def __call__(self, data):
        return {key: data[value] for key, value in self.structure.items()}


@dataclasses.dataclass(frozen=True)
class ToImage:
    def __call__(self, data):
        return {"image": {"base_0_rgb": np.moveaxis(data["frame"], 0, -1)},
                "image_mask": {"base_0_rgb": np.True_}}


@dataclasses.dataclass(frozen=True)
class Factory:
    repo_id: str
    base_config: DataConfig | None = None
    def create(self, *_args):
        return dataclasses.replace(self.base_config or DataConfig(),
            repack_transforms=Group((RepackTransform({"frame": "source.camera"}),)),
            data_transforms=Group((ToImage(),)))


@dataclasses.dataclass(frozen=True)
class Config:
    data: object
    model: object = None


@dataclasses.dataclass(frozen=True)
class Multi:
    datasets: tuple


def profile(augmentation="none", mode="stretch"):
    return {"cameras": [{"name": "front", "key": "source.camera", "kind": "rgb",
                         "image_processing": {"resize_mode": mode, "augmentation": augmentation}}]}


def execute(data, raw):
    for group in (data.repack_transforms, data.data_transforms, data.model_transforms):
        for transform in group.inputs:
            raw = transform(raw)
    return raw


def spawn_execute(args):
    return execute(*args)


def model_capability(monkeypatch, version=1):
    package = ModuleType("openpi.models")
    package.model = SimpleNamespace(IMAGE_AUGMENTATION_POLICY_VERSION=version)
    monkeypatch.setitem(sys.modules, "openpi.models", package)


def test_same_directory_mixture_and_factory_copies_keep_independent_routes(tmp_path, monkeypatch):
    model_capability(monkeypatch)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": {
        "source.camera": {"dtype": "image", "shape": [3, 32, 64]}}}))
    original = Factory(str(tmp_path))
    config = Config(Multi((original, original)))
    spec = {"inputs": {f"dataset_{i}": {"path": str(tmp_path), "dataset_profile": profile(aug)}
                       for i, aug in enumerate(("none", "color"))},
            "dataset_bindings": [{}, {}], "code_dir": str(tmp_path), "allowed_roots": [str(tmp_path)]}
    selected = install_image_processing(config, spec, SimpleNamespace(DataConfig=DataConfig))
    assert original.base_config is None
    children = [dataclasses.replace(child) for child in selected.data.datasets]
    assert children[0].base_config is not children[1].base_config
    for child, code in zip(children, (1, 2)):
        data = child.create(None, None)
        assert isinstance(data.repack_transforms.inputs[0], SourceImagePolicies)
        assert isinstance(data.model_transforms.inputs[-1], ImageAugmentationRouting)
        data = pickle.loads(pickle.dumps(data))
        result = execute(data, {"source.camera": pixels().transpose(2, 0, 1)})
        assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
        assert result["image_mask"]["base_0_rgb"].dtype == np.bool_
        assert result["image_augmentation"]["base_0_rgb"].item() == code


def test_old_runtime_rejects_nondefault_augmentation_but_keeps_default(tmp_path, monkeypatch):
    model_capability(monkeypatch, None)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": {
        "source.camera": {"dtype": "image", "shape": [3, 32, 64]}}}))
    config = Config(Factory(str(tmp_path)))
    spec = {"inputs": {"dataset": {"path": str(tmp_path), "dataset_profile": profile()}},
            "code_dir": str(tmp_path), "allowed_roots": [str(tmp_path)]}
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_RUNTIME_UPGRADE_REQUIRED"):
        install_image_processing(config, spec, SimpleNamespace(DataConfig=DataConfig))
    spec["inputs"]["dataset"]["dataset_profile"] = profile("default")
    selected = install_image_processing(config, spec, SimpleNamespace(DataConfig=DataConfig))
    assert not selected.data.create(None, None).model_transforms.inputs


def test_concrete_image_transforms_execute_in_native_spawn_worker(tmp_path):
    factory = Factory(str(tmp_path))
    policy = image_processing_contract(factory, profile(),
        {"source.camera": {"dtype": "image", "shape": [3, 32, 64]}}, code_dir=str(tmp_path))
    selected = runtime._register_factory(factory, policy, SimpleNamespace(DataConfig=DataConfig))
    arguments = (selected.create(None, None), {"source.camera": pixels().transpose(2, 0, 1)})
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        result = pool.apply(spawn_execute, (arguments,))
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image_augmentation"]["base_0_rgb"].item() == 1


def test_package_and_standalone_imports_share_existing_factory_hook(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(runtime.__file__).parent))
    loader = importlib.util.spec_from_file_location("image_runtime", runtime.__file__)
    standalone = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, standalone)
    loader.loader.exec_module(standalone)
    factory = Factory(str(tmp_path))
    features = {"source.camera": {"dtype": "image", "shape": [3, 32, 64]}}
    one = runtime._register_factory(factory,
        image_processing_contract(factory, profile("none"), features, code_dir=str(tmp_path)), SimpleNamespace(DataConfig=DataConfig))
    two = standalone._register_factory(factory,
        standalone.image_processing_contract(factory, profile("color"), features, code_dir=str(tmp_path)), SimpleNamespace(DataConfig=DataConfig))
    for selected, code in ((one, 1), (two, 2)):
        result = execute(selected.create(None, None), {"source.camera": pixels().transpose(2, 0, 1)})
        assert result["image_augmentation"]["base_0_rgb"].item() == code


def test_unlisted_224_camera_keeps_default_rather_than_dataset_wide_override(tmp_path):
    features = {"source.camera": {"dtype": "image", "shape": [3, 360, 640]},
                "unlisted.wrist": {"dtype": "image", "shape": [3, 224, 224]}}
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_MAPPING_UNSUPPORTED"):
        image_processing_contract(Factory(str(tmp_path)), profile(), features, code_dir=str(tmp_path))


def test_uniform_rgb_routes_do_not_override_undeclared_or_explicit_depth():
    raw = {"image": {"base_0_rgb": pixels(), "wrist_depth": pixels(), "right_wrist_0_rgb": pixels()},
           "image_mask": {"base_0_rgb": np.True_}}
    named = ImageAugmentationRouting({}, 2)(raw)
    assert named["image_augmentation"]["base_0_rgb"] == 2
    assert named["image_augmentation"]["wrist_depth"] == 0
    explicit = ImageAugmentationRouting({}, 3, ("right_wrist_0_rgb",))(raw)
    assert explicit["image_augmentation"]["right_wrist_0_rgb"] == 0
    assert explicit["image_mask"] is raw["image_mask"]


def test_dynamic_video_frame_expansion_is_rejected_before_execution(tmp_path):
    factory = SimpleNamespace(video_frame_config={"history": 6})
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_MAPPING_UNSUPPORTED"):
        image_processing_contract(factory, profile(),
            {"source.camera": {"dtype": "image", "shape": [3, 224, 224]}}, code_dir=str(tmp_path))


@pytest.fixture
def core(monkeypatch):
    path = ROOT / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    spec = importlib.util.spec_from_file_location("image_runtime_composed_core", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout,history", [("chw", False), ("hwc", False), ("hwc", True)])
def test_composed_geometry_runs_after_exact_decode_boundary(core, layout, history, tmp_path):
    composition = deepcopy(SPEC)
    camera = composition["cameras"][0]
    camera.update(layout=layout, time_index=1 if history else None)
    shape = [3, 32, 64] if layout == "chw" else [32, 64, 3]
    features = {camera["key"]: {"dtype": "image", "shape": ([2, *shape] if history else shape)}}
    source_profile = {"cameras": [{"name": "wrist", "key": camera["key"], "kind": "rgb",
        "image_processing": {"resize_mode": "center_crop", "augmentation": "none"}}]}
    factory = type("ComposableDataConfig", (), {})()
    factory.composition = composition
    policy = image_processing_contract(factory, source_profile, features, code_dir=str(tmp_path))
    assert policy["spatial_insertion"] == "after_repack"
    raw = {f["key"]: np.zeros((f["width"],), dtype=np.float32) for f in composition["state_fields"]}
    raw[composition["action_key"]] = np.zeros((16, 10), dtype=np.float32)
    raw[composition["prompt_key"]] = "pick cup"
    decoded = pixels().transpose(2, 0, 1)
    raw[camera["key"]] = np.stack((decoded, decoded)) if history else decoded
    repack = core.LeRobotComposedRepack(composition, core._lerobot_image_layouts(composition, {"features": features}))
    data = runtime._apply_data_config(DataConfig(Group((repack,)), Group((core.ComposedInputs(composition, 16, 32),))), policy)
    assert data.repack_transforms.inputs[0] is repack
    result = execute(data, raw)
    assert result["image"][camera["slot"]].shape == (224, 224, 3)
    assert result["image_augmentation"][camera["slot"]].item() == 1
    assert not result["image"][camera["slot"]][:, 4:-4, 0].any()
    assert raw[camera["key"]].shape[-2:] == (32, 64)


def test_serving_retains_geometry_without_training_augmentation(core):
    composition = deepcopy(SPEC)
    camera = composition["cameras"][0]
    camera.update(layout="hwc", augmentation="crop_rotate_color")
    policy = {"spatial": [{"key": "_composition_camera_0", "source_key": camera["key"], "layout": "hwc",
                          "kind": "rgb", "fill": 0, "resize_mode": "center_crop", "history": False}],
              "spatial_insertion": "after_repack", "augmentation_codes": {camera["slot"]: 3},
              "uniform_augmentation_code": None}
    data = runtime._apply_data_config(DataConfig(Group(), Group((core.ComposedInputs(composition, 16, 32),))), policy)
    class FrozenFactory:
        def create(self, *_):
            return data
    compiled = compile_inference_config(Config(FrozenFactory())).data.create(None, None)
    transform = compiled.data_transforms.inputs[0]
    assert isinstance(transform, InferenceImagePolicies)
    assert not any(isinstance(item, ImageAugmentationRouting) for item in compiled.model_transforms.inputs)
    assert "augmentation" not in compiled.data_transforms.inputs[1].composition["cameras"][0]
    assert camera["augmentation"] == "crop_rotate_color"
    raw = {f["key"]: np.zeros((f["width"],), dtype=np.float32) for f in composition["state_fields"]}
    raw[camera["key"]] = pixels()
    raw[composition["prompt_key"]] = "pick cup"
    actual = raw
    for transform in compiled.data_transforms.inputs:
        actual = transform(actual)
    expected = resize_source_image(pixels(), layout="hwc", kind="rgb", resize_mode="center_crop")
    np.testing.assert_array_equal(actual["image"][camera["slot"]], expected)
    transform = compiled.data_transforms.inputs[0]
    restored = _Codec().decode(_Codec().encode(transform))
    np.testing.assert_array_equal(restored(raw)[camera["key"]], expected)


@pytest.mark.parametrize("selected,legacy,code", [(None, "none", 1), ("default", "none", 0), ("color", "none", 2)])
def test_composition_choice_overrides_legacy_only_when_explicit(tmp_path, selected, legacy, code):
    composition = deepcopy(SPEC)
    camera = composition["cameras"][0]
    if selected is not None:
        camera["augmentation"] = selected
    source_profile = profile(legacy, "pad")
    source_profile["cameras"][0]["key"] = camera["key"]
    factory = type("ComposableDataConfig", (), {"composition": composition})()
    policy = image_processing_contract(factory, source_profile,
        {camera["key"]: {"dtype": "image", "shape": [3, 360, 640]}}, code_dir=tmp_path)
    assert policy["augmentation_codes"] == {camera["slot"]: code}
    assert policy["spatial"][0]["resize_mode"] == "pad"
    assert policy["source_policies"][camera["key"]]["augmentation"] == legacy


@pytest.mark.parametrize("supports_fixed,entrypoint,use_fixed", [
    (False, None, False),
    (True, None, True),
    (True, "scripts/train_pytorch.py", False),
])
def test_training_binds_224_camera_augmentation_without_installing_geometry(
    tmp_path, monkeypatch, supports_fixed, entrypoint, use_fixed,
):
    @dataclasses.dataclass(frozen=True)
    class ComposableDataConfig:
        composition: dict
        base_config: DataConfig | None = None

        def create(self, *_args):
            return self.base_config or DataConfig()

    @dataclasses.dataclass(frozen=True)
    class Pi0Config:
        fixed_image_augmentation: tuple = ()

    composition = deepcopy(SPEC)
    camera = composition["cameras"][0]
    camera["augmentation"] = "crop_rotate_color"
    features = {camera["key"]: {"dtype": "image", "shape": [3, 224, 224]}}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": features}))
    config = Config(ComposableDataConfig(composition), Pi0Config() if supports_fixed else None)
    spec = {"inputs": {"dataset": {"path": str(tmp_path)}},
            "code_dir": str(tmp_path), "allowed_roots": [str(tmp_path)], "entrypoint": entrypoint}
    model_capability(monkeypatch)
    bound = install_image_processing(config, spec, SimpleNamespace(DataConfig=DataConfig))
    data = bound.data.create(None, None)
    assert not data.repack_transforms.inputs
    image = np.ones((224, 224, 3), dtype=np.uint8)
    sample = execute(data, {"image": {camera["slot"]: image}})
    assert sample["image"][camera["slot"]] is image
    if use_fixed:
        assert bound.model.fixed_image_augmentation == ((camera["slot"], 3),)
        assert not data.model_transforms.inputs
        assert "image_augmentation" not in sample
    else:
        assert len(data.model_transforms.inputs) == 1
        assert sample["image_augmentation"][camera["slot"]].item() == 3
    model_capability(monkeypatch, None)
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_RUNTIME_UPGRADE_REQUIRED"):
        install_image_processing(config, spec, SimpleNamespace(DataConfig=DataConfig))


def test_independent_serving_translates_repack_field_and_hwc_layout(tmp_path):
    factory = Factory(str(tmp_path))
    policy = image_processing_contract(factory, profile("none", "pad"),
        {"source.camera": {"dtype": "image", "shape": [3, 32, 64]}}, code_dir=str(tmp_path))
    factory = runtime._register_factory(factory, policy, SimpleNamespace(DataConfig=DataConfig))
    compiled = compile_inference_config(Config(factory)).data.create(None, None)
    transform = compiled.data_transforms.inputs[0]
    assert isinstance(transform, InferenceImagePolicies) and transform.cameras[0]["key"] == "frame"
    np.testing.assert_array_equal(transform({"frame": pixels()})["frame"],
        resize_source_image(pixels(), layout="hwc", kind="rgb", resize_mode="pad"))
    assert not compiled.model_transforms.inputs


def test_json_integer_observations_retain_geometry_and_reject_invalid_pixels():
    transform = InferenceImagePolicies(({"key": "camera", "layout": "hwc", "kind": "rgb",
        "resize_mode": "center_crop", "fill": 0, "history": False},))
    actual = transform({"camera": pixels().tolist()})["camera"]
    np.testing.assert_array_equal(actual,
        resize_source_image(pixels(), layout="hwc", kind="rgb", resize_mode="center_crop"))
    for value in (-1, 256, float("nan"), float("inf")):
        raw = pixels().astype(np.float64 if isinstance(value, float) else np.int64)
        raw[0, 0, 0] = value
        with pytest.raises(ValueError, match="POLICY_OBSERVATION_INVALID|IMAGE_PROCESSING_DATA_INVALID"):
            transform({"camera": raw.tolist()})


def test_geometry_change_invalidates_submission_configuration_evidence(dataset):
    dataset["image_processing"] = {"source_policies": {"camera": {"resize_mode": "pad", "augmentation": "none"}}}
    dataset["submission_validation"] = submission_validation.submission_evidence(dataset, source_summary(dataset))
    submission_validation.validate_submission_evidence(dataset)
    dataset["image_processing"]["source_policies"]["camera"]["resize_mode"] = "stretch"
    with pytest.raises(ValueError, match="differs"):
        submission_validation.validate_submission_evidence(dataset)


def test_numeric_normalization_ignores_non224_policy_requirement(training, monkeypatch):
    from vla_adapters import training_plan
    base, spec, _, _ = training
    factory = base.data.datasets[0]
    path = Path(factory.repo_id) / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["camera"] = {"dtype": "image", "shape": [3, 480, 640]}
    path.write_text(json.dumps(info))
    monkeypatch.setattr(runtime, "resize_source_image", lambda *_args, **_kwargs: pytest.fail("norm must not decode images"))
    contract = training_plan.dataset_contract(factory, spec, norm_stats=True)
    assert "image_processing" not in contract
    with pytest.raises(ValueError, match="IMAGE_PROCESSING_REQUIRED"):
        training_plan.dataset_contract(factory, spec)
