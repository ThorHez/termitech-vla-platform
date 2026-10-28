"""Factory-installed LeRobot codec boundaries preserve declared source layouts."""

from copy import deepcopy
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from vla_adapters.composition_spec import verify_composition_metadata


ROOT = Path(__file__).resolve().parents[2]
CAMERA = "observation.images.wrist"
SLOT = "left_wrist_0_rgb"
SPEC = json.loads((ROOT / "tests/fixtures/composed-processing-v1.json").read_text())["valid"][0]


@pytest.fixture
def composed(monkeypatch):
    source = ROOT / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    loader = importlib.util.spec_from_file_location("lerobot_layout_factory_test", source)
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, module)
    loader.loader.exec_module(module)

    @dataclasses.dataclass
    class Pi0Config:
        action_dim: int = 32
        action_horizon: int = 16
        max_token_len: int = 32
        pi05: bool = True
        discrete_state_input: bool = False
        image_keys: tuple | None = None
        depth_keys: tuple | None = None

    class Pi0GripperConfig(Pi0Config):
        pass

    class Pi0ValueConfig(Pi0Config):
        pass

    @dataclasses.dataclass(frozen=True)
    class Group:
        inputs: tuple = ()
        outputs: tuple = ()

    class ModelTransform:
        def __init__(self, *args, **kwargs):
            pass

    package = types.ModuleType("openpi");package.__path__ = []
    models = types.ModuleType("openpi.models");models.__path__ = []
    transforms = types.ModuleType("openpi.transforms");transforms.Group = Group
    for name in ("InjectDefaultPrompt", "ResizeImages", "TokenizePrompt", "PadActionsOnly", "FlattenState", "KeepModelKeys"):
        setattr(transforms, name, ModelTransform)
    monkeypatch.setitem(sys.modules, "openpi", package)
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms)
    package.models, package.transforms = models, transforms
    for name, value in (("pi0_config", Pi0Config), ("pi0_gripper", Pi0GripperConfig), ("pi0_value", Pi0ValueConfig),
                        ("tokenizer", ModelTransform)):
        child = types.ModuleType("openpi.models." + name)
        setattr(child, "PaligemmaTokenizer" if name == "tokenizer" else value.__name__, value)
        setattr(models, name, child)
        monkeypatch.setitem(sys.modules, child.__name__, child)
    return module, Pi0Config


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str
    repack_transforms: object = None
    data_transforms: object = None
    model_transforms: object = None
    normalize_masks: object = None
    use_quantile_norm: bool = False
    action_sequence_keys: tuple = ()
    action_loss_mask: tuple = ()
    prompt_from_task: bool = False


def spec(layout="hwc", *, time_index=None, kind="rgb"):
    value = deepcopy(SPEC)
    value["prompt_key"] = ""
    value["gripper_range"] = None
    value["cameras"][0].update(layout=layout, time_index=time_index, kind=kind)
    return value


def write_metadata(repo, composition, shape, dtype="image"):
    features = {field["key"]: {"dtype": "float32", "shape": [field["width"]]}
                for field in composition["state_fields"]}
    features[composition["action_key"]] = {"dtype": "float32", "shape": [16, 10]}
    features[CAMERA] = {"dtype": dtype, "shape": shape}
    source = repo / "meta/info.json"
    source.parent.mkdir(exist_ok=True)
    source.write_text(json.dumps({"features": features}))
    return {"features": features}, source


def build(composed, repo, composition, shape, dtype="image", *, change_metadata=False):
    module, model_type = composed
    document, source = write_metadata(repo, composition, shape, dtype)
    before = source.read_bytes()

    def create_base(*_):
        if change_metadata:
            source.write_text(json.dumps({**document, "changed": True}))
        return DataConfig(str(repo))

    factory = types.SimpleNamespace(repo_id=str(repo), composition=composition, training_stage="sft", acp_dropout=0.0,
                                    create_base_config=create_base)
    model = model_type(depth_keys=(SLOT,)) if composition["cameras"][0]["kind"] == "depth" else model_type()
    config = module.compose_data_config(factory, repo, model)
    assert source.read_bytes() == before
    return config, document


def sample(composition, image):
    return {**{field["key"]: np.zeros(field["width"], dtype=np.float32)
               for field in composition["state_fields"]},
            composition["action_key"]: np.zeros((16, 10), dtype=np.float32), CAMERA: image}


def model_inputs(config, data):
    for transform in config.repack_transforms.inputs:
        data = transform(data)
    for transform in config.data_transforms.inputs:
        data = transform(data)
    return data


@pytest.mark.parametrize("dtype", ["image", "video"])
@pytest.mark.parametrize("layout,shape", [("hwc", [7, 11, 3]), ("chw", [3, 7, 11])])
def test_factory_reconciles_lerobot_decoded_chw_with_both_source_layouts(composed, tmp_path, dtype, layout, shape):
    composition = spec(layout)
    config, document = build(composed, tmp_path, composition, shape, dtype)
    verify_composition_metadata(composition, document, action_horizon=16)
    image = np.arange(3 * 7 * 11, dtype=np.uint8).reshape(3, 7, 11)
    original = image.copy()
    data = sample(composition, image)
    result = model_inputs(config, data)
    assert result["image"][SLOT].shape == (7, 11, 3)
    np.testing.assert_array_equal(result["image"][SLOT], image.transpose(1, 2, 0))
    np.testing.assert_array_equal(image, original)
    assert data[CAMERA] is image and image.shape == (3, 7, 11)
    # A reused source sample is not progressively transposed by repeated reads.
    np.testing.assert_array_equal(model_inputs(config, data)["image"][SLOT], result["image"][SLOT])


def test_factory_accepts_lerobot_float_decoding_without_changing_pixel_values(composed, tmp_path):
    composition = spec("hwc")
    config, _ = build(composed, tmp_path, composition, [7, 11, 3])
    rgb = np.zeros((3, 7, 11), dtype=np.float32)
    rgb[0], rgb[1], rgb[2] = 0.0, 0.5, 1.0
    image = model_inputs(config, sample(composition, rgb))["image"][SLOT]
    np.testing.assert_array_equal(image[0, 0], [0, 127, 255])


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_decoder_layout_adaptation_does_not_relax_rgb_value_validation(composed, tmp_path, value):
    composition = spec("hwc")
    config, _ = build(composed, tmp_path, composition, [7, 11, 3])
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID.*RGB"):
        model_inputs(config, sample(composition, np.full((3, 7, 11), value, dtype=np.float32)))


@pytest.mark.parametrize("layout,shape", [("hwc", [2, 7, 11, 3]), ("chw", [2, 3, 7, 11])])
def test_history_decoding_keeps_time_axis_until_explicit_frame_selection(composed, tmp_path, layout, shape):
    composition = spec(layout, time_index=-1)
    config, document = build(composed, tmp_path, composition, shape)
    verify_composition_metadata(composition, document, action_horizon=16)
    frames = np.stack([np.full((3, 7, 11), 10, dtype=np.uint8), np.full((3, 7, 11), 90, dtype=np.uint8)])
    repacked = config.repack_transforms.inputs[0](sample(composition, frames))
    assert repacked["_composition_camera_0"].shape == ((2, 7, 11, 3) if layout == "hwc" else (2, 3, 7, 11))
    output = config.data_transforms.inputs[0](repacked)
    np.testing.assert_array_equal(output["image"][SLOT], np.full((7, 11, 3), 90, dtype=np.uint8))


@pytest.mark.parametrize("shape,index", [([2, 7, 11, 3], None), ([2, 7, 11, 3], 2), ([7, 11, 3], 0)])
def test_factory_never_drops_history_or_invents_a_time_axis(composed, tmp_path, shape, index):
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID"):
        build(composed, tmp_path, spec(time_index=index), shape)


@pytest.mark.parametrize("shape", [(7, 11, 3), (4, 7, 11), (3, 8, 11), (1, 3, 7, 11), (7, 11)])
def test_wrong_decoded_rank_channels_size_or_already_hwc_are_rejected(composed, tmp_path, shape):
    composition = spec("hwc")
    config, _ = build(composed, tmp_path, composition, [7, 11, 3])
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID.*CHW/TCHW"):
        model_inputs(config, sample(composition, np.zeros(shape, dtype=np.uint8)))


@pytest.mark.parametrize("shape", [[7, 11, 4], [3, 7, 11], [7, 11], [1, 2, 7, 11, 3]])
def test_declared_source_layout_remains_strict_in_metadata_and_factory(composed, tmp_path, shape):
    composition = spec("hwc")
    document, _ = write_metadata(tmp_path, composition, shape)
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID|COMPOSITION_INVALID"):
        build(composed, tmp_path, composition, shape)
    with pytest.raises(ValueError):
        verify_composition_metadata(composition, document, action_horizon=16)


def test_numeric_rgb_is_not_assumed_to_use_lerobot_image_decoder(composed, tmp_path):
    composition = spec("hwc")
    config, _ = build(composed, tmp_path, composition, [7, 11, 3], dtype="float32")
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    repacked = config.repack_transforms.inputs[0](sample(composition, image))
    assert repacked["_composition_camera_0"] is image
    assert model_inputs(config, sample(composition, image))["image"][SLOT].shape == (7, 11, 3)
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID"):
        model_inputs(config, sample(composition, image.transpose(2, 0, 1)))


def test_depth_image_is_not_forced_through_rgb_transposition(composed, tmp_path):
    composition = spec("hwc", kind="depth")
    config, _ = build(composed, tmp_path, composition, [7, 11, 1], dtype="image")
    depth = np.full((7, 11, 1), 0.5, dtype=np.float32)
    repacked = config.repack_transforms.inputs[0](sample(composition, depth))
    assert repacked["_composition_camera_0"] is depth
    np.testing.assert_array_equal(model_inputs(config, sample(composition, depth))["image"][SLOT],
                                  np.full((7, 11, 3), 127, dtype=np.uint8))


def test_direct_repack_array_contract_is_unchanged(composed):
    module, _ = composed
    composition = spec("hwc")
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    repacked = module.ComposedRepack(composition)(sample(composition, image))
    assert repacked["_composition_camera_0"] is image
    assert module.ComposedInputs(composition, 16, 32)(repacked)["image"][SLOT].shape == (7, 11, 3)
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID"):
        module.ComposedInputs(composition, 16, 32)(module.ComposedRepack(composition)(sample(composition, image.transpose(2, 0, 1))))


def test_factory_binds_action_and_image_metadata_to_the_same_snapshot(composed, tmp_path):
    with pytest.raises(ValueError, match="COMPOSITION_DATA_INVALID.*元信息已改变"):
        build(composed, tmp_path, spec("hwc"), [7, 11, 3], change_metadata=True)
