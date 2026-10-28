"""Portable policy JSON restores observations without source training paths."""

from copy import deepcopy
import dataclasses
import enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters.portable_policy import (
    MAX_BYTES, _Codec, export_portable_policy, load_portable_policy, portable_digest,
    validate_portable_policy,
)
from vla_adapters.policy_inference import ComposedInferenceInputs


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    for name in ("openpi.models.pi0_config", "openpi.models.model", "openpi.models.tokenizer", "openpi.transforms"):
        module = ModuleType(name)
        path = root / "src" / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned " + name + "\n")
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
    Pi0Config = dataclasses.make_dataclass("Pi0Config", [("action_dim", int, 32), ("action_horizon", int, 50),
        ("image_keys", tuple, ("base_0_rgb",)), ("pi05", bool, True)], frozen=True)
    Pi0Config.__module__ = "openpi.models.pi0_config"
    sys.modules[Pi0Config.__module__].Pi0Config = Pi0Config
    Group = dataclasses.make_dataclass("Group", [("inputs", tuple, ()), ("outputs", tuple, ())], frozen=True)
    Group.__module__ = "openpi.transforms"
    sys.modules[Group.__module__].Group = Group
    ModelType = enum.Enum("ModelType", {"PI05": "pi05"}, module="openpi.models.model")
    sys.modules[ModelType.__module__].ModelType = ModelType
    class PaligemmaTokenizer:
        def __init__(self, max_len=48):
            self._max_len = max_len
            self._tokenizer = SimpleNamespace(serialized_model_proto=lambda: b"fixed vocabulary")
    PaligemmaTokenizer.__module__ = "openpi.models.tokenizer"
    sys.modules[PaligemmaTokenizer.__module__].PaligemmaTokenizer = PaligemmaTokenizer
    return root, Pi0Config, Group, PaligemmaTokenizer


@dataclasses.dataclass(frozen=True)
class Data:
    asset_id: str
    data_transforms: object
    model_transforms: object
    use_quantile_norm: bool = True
    normalize_masks: object = None
    robot_type: str = "ARM=1,G=0,H=0"


@dataclasses.dataclass(frozen=True)
class Factory:
    value: Data
    unavailable_training_path: str

    def create(self, *_args):
        assert not Path(self.unavailable_training_path).exists()
        return self.value


@dataclasses.dataclass(frozen=True)
class Train:
    model: object
    data: object
    policy_metadata: dict
    assets_dirs: tuple = ()


def configuration(runtime):
    root, Model, Group, _ = runtime
    data = Data(".", Group((ComposedInferenceInputs({"schema_version": 1}),), ()), Group(),
                normalize_masks={"state": (True, False), "actions": (True, True)})
    return Train(Model(), Factory(data, "/missing/source-machine/training-dataset"), {"task": "折叠衣服"})


def test_roundtrip_has_no_training_path_and_keeps_model_contract(runtime):
    root, Model, Group, _ = runtime
    original = configuration(runtime)
    exported = export_portable_policy(original, root)
    encoded = json.dumps(exported)
    assert "/missing/source-machine" not in encoded and "pickle" not in encoded
    restored = load_portable_policy(json.loads(encoded), portable_digest(exported), root)
    assert restored.model == original.model
    assert dataclasses.asdict(restored.data.create(None, None)) == dataclasses.asdict(original.data.value)
    assert restored.assets_dirs == ()
    assert restored.policy_metadata == {"task": "折叠衣服"}


def test_changed_source_target_adapter_runtime_and_payload_are_rejected(runtime, monkeypatch):
    root, *_ = runtime
    exported = export_portable_policy(configuration(runtime), root)
    invalid = deepcopy(exported)
    invalid["data"]["asset_id"] = "/source/norm"
    with pytest.raises(ValueError, match="digest"):
        load_portable_policy(invalid, portable_digest(exported), root)
    for field, value in (("adapter_files", {}), ("runtime_versions", {}), ("code_files", {})):
        invalid = dict(exported, **{field: value})
        with pytest.raises(ValueError, match="differs|differ"):
            load_portable_policy(invalid, portable_digest(invalid), root)
    file = root / "src/openpi/models/pi0_config.py"
    file.write_text("# changed model behavior\n")
    with pytest.raises(ValueError, match="source differs"):
        load_portable_policy(exported, portable_digest(exported), root)


def test_rejects_unknown_constructor_opaque_callable_and_path_literals(runtime):
    root, *_ = runtime
    for value in (lambda: None, Path("/source/data"), object()):
        with pytest.raises(ValueError, match="Cannot safely export"):
            _Codec().encode(value)
    with pytest.raises(ValueError, match="Unsupported constructor"):
        _Codec().decode({"kind": "dataclass", "module": "os", "name": "system", "fields": {}})
    with pytest.raises(ValueError, match="Unknown literal"):
        _Codec().decode({"kind": "exec", "source": "raise AssertionError()"})


def test_target_dataclass_schema_and_import_location_are_pinned(runtime):
    root, *_ = runtime
    exported = export_portable_policy(configuration(runtime), root)
    changed = deepcopy(exported)
    changed["model"]["fields"].pop("action_horizon")
    with pytest.raises(ValueError, match="dataclass fields differ"):
        load_portable_policy(changed, portable_digest(changed), root)
    module = sys.modules["openpi.models.pi0_config"]
    module.__file__ = "/another/openpi/models/pi0_config.py"
    with pytest.raises(ValueError, match="outside its verified checkout"):
        load_portable_policy(exported, portable_digest(exported), root)


def test_tokenizer_has_same_max_length_and_actual_vocabulary(runtime):
    root, _, _, Tokenizer = runtime
    codec = _Codec(root)
    exported = codec.encode(Tokenizer(200))
    assert exported["max_len"] == 200
    restored = codec.decode(exported)
    assert restored._max_len == 200
    exported["vocabulary_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="vocabulary differs"):
        codec.decode(exported)


def test_literal_bounds_finite_values_and_normalization_escape(runtime):
    root, *_ = runtime
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            _Codec().encode(value)
    with pytest.raises(ValueError, match="too long"):
        _Codec().encode("a" * 20000)
    with pytest.raises(ValueError, match="digest or size"):
        validate_portable_policy({"oversized": "a" * MAX_BYTES}, "0" * 64)
    exported = export_portable_policy(configuration(runtime), root)
    exported["data"]["asset_id"] = "../../training/norm"
    with pytest.raises(ValueError, match="normalization"):
        load_portable_policy(exported, portable_digest(exported), root)


def test_source_inventory_rejects_symlinks(runtime, tmp_path):
    root, *_ = runtime
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n")
    (root / "src/openpi/escaped.py").symlink_to(outside)
    with pytest.raises(ValueError, match="regular files"):
        export_portable_policy(configuration(runtime), root)


def test_current_composition_survives_portable_roundtrip_without_future_actions(runtime, monkeypatch):
    runtime_numpy = os.environ.get("VLA_TEST_NUMPY_PATH")
    if runtime_numpy:
        monkeypatch.syspath_prepend(runtime_numpy)
    np = pytest.importorskip("numpy")
    from vla_adapters.policy_inference import compile_inference_config
    project = Path(__file__).resolve().parents[2]
    path = project / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    loader = importlib.util.spec_from_file_location("portable_training_composition", path)
    training = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, training)
    loader.loader.exec_module(training)
    spec = json.loads((project / "tests/fixtures/composed-processing-v1.json").read_text())["valid"][0]
    root, Model, Group, _ = runtime
    data = Data(".", Group((training.ComposedInputs(spec, 16, 32),), ()), Group(),
                normalize_masks={"state": (True,) * 7, "actions": (True,) * 10})
    original = Train(Model(), Factory(data, "/unavailable/dataset"), {"robot": "test"})
    exported = export_portable_policy(original, root)
    assert "ComposedInputs" not in json.dumps(exported)
    restored = load_portable_policy(exported, portable_digest(exported), root)
    observed = {field["key"]: np.full((spec["state_history"], field["width"]), .5, dtype=np.float32)
                for field in spec["state_fields"]}
    for camera in spec["cameras"]:
        shape = (3, 2, 3) if camera["layout"] == "chw" else (2, 3, 3)
        observed[camera["key"]] = np.arange(18, dtype=np.uint8).reshape(shape).tolist()
    observed[spec["prompt_key"] or "prompt"] = "pick"
    source_data = compile_inference_config(original).data.create(None, None)
    target_data = restored.data.create(None, None)
    expected = source_data.data_transforms.inputs[0](observed)
    actual = target_data.data_transforms.inputs[0](observed)
    assert "actions" not in actual
    np.testing.assert_array_equal(actual["state"], expected["state"])
    for slot, expected_image in expected["image"].items():
        np.testing.assert_array_equal(actual["image"][slot], expected_image)
    assert target_data.normalize_masks == source_data.normalize_masks
    actions = target_data.model_transforms.outputs[-1]({"actions": np.ones((16, 32)), "state": np.ones((32,))})
    assert actions["actions"].shape == (16, 10) and actions["state"].shape == (1, 7)
