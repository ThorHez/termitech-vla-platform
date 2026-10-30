"""Decoded-sample checks execute real NumPy composition, without model imports."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import dataset_samples as check


np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
CAMERA = "observation.images.wrist"
SLOT = "left_wrist_0_rgb"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    composition = json.loads((ROOT / "tests/fixtures/composed-processing-v1.json").read_text())["valid"][0]
    composition.update(prompt_key="", gripper_range=None)
    composition["cameras"][0]["layout"] = "hwc"
    native_source = ROOT / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    copied = tmp_path / "composed_processing.py"
    copied.write_bytes(native_source.read_bytes())
    loader = importlib.util.spec_from_file_location("sample_probe_native", copied)
    native = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, native)
    loader.loader.exec_module(native)
    source = tmp_path / "data_loader.py"
    source.write_text('''
def processing_sample_config(self, assets_dirs, model):
    self.hook_calls.append((assets_dirs, model))
    return self.probe_data
def create_torch_dataset(data_config, horizon, model):
    calls.append((data_config, horizon, model))
    return dataset
def create_data_loader(*args, **kwargs):
    raise AssertionError("must not construct a worker dataloader")
''')
    loader = importlib.util.spec_from_file_location(check.LOADER_MODULE, source)
    runtime = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, runtime)
    loader.loader.exec_module(runtime)
    runtime.calls = []
    repo = tmp_path / "dataset"
    repo.mkdir()

    class Dataset:
        def __init__(self):
            self.indices = []
            self.length = 7
            self.image = np.zeros((3, 7, 11), np.uint8)

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            self.indices.append(index)
            return {**{field["key"]: np.zeros(field["width"], np.float32)
                       for field in composition["state_fields"]},
                    "actions": np.zeros((16, 10), np.float32), CAMERA: self.image.copy()}

    runtime.dataset = Dataset()
    repack = native.LeRobotComposedRepack(composition, (
        native._LeRobotImageLayout(0, CAMERA, (3, 7, 11), "hwc"),))
    inputs = native.ComposedInputs(composition, 16, 32)
    group = lambda values=(): SimpleNamespace(inputs=tuple(values), outputs=())
    data_config = SimpleNamespace(repo_id=str(repo), repack_transforms=group([repack]),
        data_transforms=group([inputs]), model_transforms=group(), action_sequence_keys=())
    factory_type = type("ComposableDataConfig", (), {"__module__": check.FACTORY_MODULE,
        "processing_sample_config": runtime.processing_sample_config})
    factory = factory_type()
    factory.composition, factory.repo_id = composition, str(repo)
    factory.probe_data, factory.hook_calls = data_config, []
    model = SimpleNamespace(action_horizon=16)
    config = SimpleNamespace(assets_dirs=tmp_path, model=model, data=factory)
    spec = {"code_dir": str(tmp_path), "allowed_roots": [str(tmp_path)]}
    evidence = {"path": str(repo), "frames": 7,
                "composition": {"spec": deepcopy(composition), "action_sequence_keys": []}}
    return SimpleNamespace(factory=factory, config=config, spec=spec, evidence=evidence,
                           runtime=runtime, native=native, source=source, data_config=data_config)


def inspect_sample(setup):
    return check.inspect_composed_samples(setup.factory, setup.config, setup.spec, setup.evidence)


def test_native_decode_repack_and_data_transform_have_stable_shape_evidence(setup):
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in Path(setup.spec["code_dir"]).rglob("*") if path.is_file()}
    result = inspect_sample(setup)
    assert result == inspect_sample(setup)
    assert result["status"] == "verified" and result["sample_indices"] == [0, 3, 6]
    assert result["checked_samples"] == 3 and result["action_horizon"] == 16
    assert setup.runtime.dataset.indices == [0, 3, 6, 0, 3, 6]
    sample = result["samples"][0]
    assert sample["raw_cameras"][0]["shape"] == [3, 7, 11]
    assert sample["raw_cameras"][0]["layout"] == "hwc"
    assert sample["output"]["image"][SLOT] == {"shape": [7, 11, 3], "dtype": "uint8"}
    assert sample["output"]["state"] == {"shape": [1, 7], "dtype": "float32"}
    assert sample["output"]["actions"]["shape"] == [16, 10]
    assert sample["output"]["model_views"] == {"checked": True,
        "selected_slots": list(setup.native.SLOTS), "valid_slots": [SLOT], "allow_all_masked_images": False}
    assert result["runtime"]["loader"]["sha256"] == hashlib.sha256(setup.source.read_bytes()).hexdigest()
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path(setup.spec["code_dir"]).rglob("*") if path.is_file()}
    assert not any("pixel" in key for key in sample)


def test_real_sample_preflight_checks_the_selected_model_views(setup):
    setup.config.model.image_keys = (SLOT,)
    setup.config.model.depth_keys = ()
    result = inspect_sample(setup)
    assert all(sample["output"]["model_views"] == {"checked": True,
        "selected_slots": [SLOT], "valid_slots": [SLOT], "allow_all_masked_images": False}
        for sample in result["samples"])


def test_preflight_rejects_model_selecting_only_an_inactive_view(setup):
    setup.config.model.image_keys = ("right_wrist_0_rgb",)
    setup.config.model.depth_keys = ()
    with pytest.raises(ValueError, match="模型所选视觉输入均无有效图像"):
        inspect_sample(setup)


def model_view_data(*, history=None, mask=True, frames=None):
    shape = (7, 11, 3) if history is None else (history, 7, 11, 3)
    data = {"image": {"rgb": np.zeros(shape, np.uint8)}, "image_mask": {"rgb": np.bool_(mask)}}
    if frames is not None:
        data["frame_valid_mask"] = {"rgb": np.asarray(frames)}
    return data


def test_model_view_probe_rejects_all_masked_samples_and_honors_explicit_ablation():
    data = model_view_data(mask=False)
    with pytest.raises(ValueError, match="均无有效图像"):
        check._model_view_evidence(data, SimpleNamespace(), np)
    result = check._model_view_evidence(data, SimpleNamespace(allow_all_masked_images=True), np)
    assert result == {"checked": True, "selected_slots": ["rgb"], "valid_slots": [],
                      "allow_all_masked_images": True}


def test_depth_only_model_view_is_valid_without_requiring_rgb():
    data = {"image": {"depth": np.zeros((7, 11, 3), np.uint8)}, "image_mask": {"depth": True}}
    model = SimpleNamespace(image_keys=(), depth_keys=("depth",))
    assert check._model_view_evidence(data, model, np)["valid_slots"] == ["depth"]
    assert check._model_view_evidence(data, SimpleNamespace(), np)["selected_slots"] == ["depth"]


@pytest.mark.parametrize("image_keys,depth_keys", [(("rgb", "rgb"), ()), (("rgb",), ("rgb",)),
                                                   ((), ()), (("rgb",), ("depth", "depth")),
                                                   (("rgb",), 3)])
def test_model_view_probe_rejects_duplicate_overlapping_or_empty_selection(image_keys, depth_keys):
    with pytest.raises(ValueError, match="视觉槽"):
        check._model_view_evidence(model_view_data(), SimpleNamespace(image_keys=image_keys, depth_keys=depth_keys), np)


def test_history_view_requires_a_valid_frame_and_honors_image_mask():
    data = model_view_data(history=3, frames=[False, True, False])
    assert check._model_view_evidence(data, SimpleNamespace(), np)["valid_slots"] == ["rgb"]
    data["frame_valid_mask"]["rgb"][:] = False
    with pytest.raises(ValueError, match="均无有效图像"):
        check._model_view_evidence(data, SimpleNamespace(), np)
    data["frame_valid_mask"]["rgb"][:] = True
    data["image_mask"]["rgb"] = False
    with pytest.raises(ValueError, match="均无有效图像"):
        check._model_view_evidence(data, SimpleNamespace(), np)


def test_history_view_without_explicit_frame_mask_inherits_image_validity():
    data = model_view_data(history=3)
    data["frame_valid_mask"] = {}
    assert check._model_view_evidence(data, SimpleNamespace(), np)["valid_slots"] == ["rgb"]


@pytest.mark.parametrize("frames", [[True, False], [1, 1, 1], [[True, True, True]]])
def test_history_view_preflight_rejects_malformed_frame_masks(frames):
    with pytest.raises(ValueError, match="有效帧掩码"):
        check._model_view_evidence(model_view_data(history=3, frames=frames), SimpleNamespace(), np)


def test_missing_selected_view_is_masked_while_another_valid_view_remains():
    model = SimpleNamespace(image_keys=("missing", "rgb"), depth_keys=())
    assert check._model_view_evidence(model_view_data(), model, np) == {
        "checked": True, "selected_slots": ["missing", "rgb"], "valid_slots": ["rgb"],
        "allow_all_masked_images": False}


def test_legacy_processor_is_explicitly_not_applicable_without_importing_openpi(monkeypatch):
    monkeypatch.setattr(check.importlib, "import_module", lambda *_: pytest.fail("unexpected import"))
    result = check.inspect_composed_samples(SimpleNamespace(), None, {}, {})
    assert result["status"] == "not_applicable" and result["checked_samples"] == 0


def test_old_unadapted_hwc_repack_is_blocked_with_actual_shape_and_camera(setup):
    setup.data_config.repack_transforms.inputs = (setup.native.ComposedRepack(setup.factory.composition),)
    with pytest.raises(ValueError, match="DATASET_SAMPLE_INVALID") as error:
        inspect_sample(setup)
    message = str(error.value)
    assert "sample_index=0" in message and CAMERA in message and "actual_shape=[3, 7, 11]" in message
    assert "layout=hwc" in message and setup.evidence["path"] in message


@pytest.mark.parametrize("shape", [(4, 7, 11), (2, 3, 7, 11)])
def test_bad_channels_or_unselected_image_history_fail_before_training(setup, shape):
    setup.runtime.dataset.image = np.zeros(shape, np.uint8)
    with pytest.raises(ValueError, match="DATASET_SAMPLE_INVALID"):
        inspect_sample(setup)


def test_missing_native_hook_is_a_hard_unavailable_error(setup, monkeypatch):
    monkeypatch.delattr(type(setup.factory), "processing_sample_config")
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_UNAVAILABLE"):
        inspect_sample(setup)


def test_native_hook_cannot_include_or_execute_model_transforms(setup):
    def model_transform(_):
        pytest.fail("must never call model transforms")
    setup.data_config.model_transforms.inputs = (model_transform,)
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_UNAVAILABLE"):
        inspect_sample(setup)
    assert setup.runtime.calls == []


@pytest.mark.parametrize("length", [0, 8])
def test_empty_or_changed_dataset_cannot_be_reported_verified(setup, length):
    setup.runtime.dataset.length = length
    with pytest.raises(ValueError, match="DATASET_SAMPLE_INVALID"):
        inspect_sample(setup)


def test_decode_error_cannot_be_skipped_or_downgraded(setup, monkeypatch):
    def read(_self, _index):
        raise OSError("video decoder unavailable")
    monkeypatch.setattr(type(setup.runtime.dataset), "__getitem__", read)
    with pytest.raises(ValueError, match="video decoder unavailable"):
        inspect_sample(setup)


def test_later_sample_failure_does_not_turn_partial_success_into_verification(setup, monkeypatch):
    original = type(setup.runtime.dataset).__getitem__

    def read(self, index):
        sample = original(self, index)
        if index == 3:
            sample["actions"] = np.zeros((15, 10), np.float32)
        return sample

    monkeypatch.setattr(type(setup.runtime.dataset), "__getitem__", read)
    with pytest.raises(ValueError, match="sample_index=3"):
        inspect_sample(setup)
    assert setup.runtime.dataset.indices == [0, 3]


def test_native_source_change_during_decode_requires_another_preflight(setup, monkeypatch):
    original = type(setup.runtime.dataset).__getitem__

    def read(self, index):
        setup.source.write_text(setup.source.read_text() + "\n# changed\n")
        return original(self, index)

    monkeypatch.setattr(type(setup.runtime.dataset), "__getitem__", read)
    with pytest.raises(ValueError, match="源码已改变"):
        inspect_sample(setup)


def test_budget_exhaustion_blocks_before_constructing_dataset(setup, monkeypatch):
    ticks = iter([0, check.SAMPLE_SECONDS + 1])
    monkeypatch.setattr(check.time, "monotonic", lambda: next(ticks, check.SAMPLE_SECONDS + 1))
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_LIMIT"):
        inspect_sample(setup)
    assert setup.runtime.calls == []


def test_native_source_outside_bound_project_is_rejected(setup):
    setup.spec["code_dir"] = setup.evidence["path"]
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_UNAVAILABLE"):
        inspect_sample(setup)


def test_path_and_sampling_cannot_change_between_contract_and_native_hook(setup, tmp_path):
    setup.data_config.repo_id = str(tmp_path)
    with pytest.raises(ValueError, match="登记的数据集路径"):
        inspect_sample(setup)
    setup.data_config.repo_id = setup.evidence["path"]
    setup.data_config.action_sequence_keys = ("actions",)
    with pytest.raises(ValueError, match="动作序列采样配置"):
        inspect_sample(setup)


@pytest.fixture
def isolated_process(setup, monkeypatch):
    receipt = inspect_sample(setup)
    state = SimpleNamespace(response={"schema_version": 1, "ok": True, "result": receipt},
                            returncode=0, timeout=False, arguments=None, input=None, stopped=[], waits=[])

    class Process:
        pid = 123456

        def __init__(self, arguments, **options):
            state.arguments = (arguments, options)
            self.returncode = state.returncode

        def communicate(self, value, timeout):
            state.input, state.communicate_timeout = value, timeout
            if state.timeout:
                raise subprocess.TimeoutExpired("sample probe", timeout)
            output = state.response if isinstance(state.response, bytes) else json.dumps(state.response).encode()
            return output, b""

        def wait(self, timeout):
            state.waits.append(timeout)
            return self.returncode

    monkeypatch.setattr(check.subprocess, "Popen", Process)
    monkeypatch.setattr(check.os, "killpg", lambda pid, sig: state.stopped.append((pid, sig)))
    monkeypatch.setattr(check, "inspect_composed_samples", lambda *_: pytest.fail("decoder ran in training parent"))
    return state


def test_isolated_probe_serializes_only_json_and_preserves_training_parent_environment(setup, isolated_process, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-training")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("WANDB_MODE", "online")
    before = os.environ.copy()
    result = check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)
    assert result["status"] == "verified" and os.environ == before
    command, options = isolated_process.arguments
    assert command == [sys.executable, str(Path(check.__file__).resolve()), "--sample-probe"]
    assert options["start_new_session"] is True and options["cwd"] == setup.spec["code_dir"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "" and options["env"]["JAX_PLATFORMS"] == "cpu"
    assert options["env"]["HF_HUB_OFFLINE"] == "1" and options["env"]["WANDB_MODE"] == "disabled"
    assert isolated_process.communicate_timeout == 70
    payload = json.loads(isolated_process.input)
    assert payload == {"schema_version": 1, "spec": setup.spec, "dataset_evidence": setup.evidence, "factory_index": 0}
    assert isolated_process.stopped == [(123456, signal.SIGKILL)] and isolated_process.waits == [5]


def test_isolated_legacy_processor_does_not_spawn_or_import_decoder(monkeypatch):
    monkeypatch.setattr(check.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unnecessary process"))
    assert check.inspect_composed_samples_isolated(SimpleNamespace(), None, {}, {})["status"] == "not_applicable"


def test_isolated_timeout_kills_entire_decoder_process_group(setup, isolated_process):
    isolated_process.timeout = True
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_LIMIT"):
        check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)
    assert isolated_process.stopped == [(123456, signal.SIGKILL)]


@pytest.mark.parametrize("reply", [b"", b"invalid json", b"x" * (check.MAX_RESPONSE_BYTES + 1),
                                  {"schema_version": 1, "ok": True, "result": {"status": "not_applicable"}}])
def test_isolated_missing_oversized_or_incomplete_receipts_never_pass(setup, isolated_process, reply):
    isolated_process.response = reply
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_"):
        check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)
    assert isolated_process.stopped


def test_isolated_native_failure_keeps_actionable_camera_context(setup, isolated_process):
    isolated_process.response = {"schema_version": 1, "ok": False,
        "error": "DATASET_SAMPLE_INVALID: camera=wrist, actual_shape=[3,7,11], layout=hwc"}
    isolated_process.returncode = 2
    with pytest.raises(ValueError, match=r"actual_shape=\[3,7,11\]"):
        check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)


def test_isolated_crashed_decoder_is_not_reported_as_verified(setup, isolated_process):
    isolated_process.returncode = -11
    with pytest.raises(ValueError, match="DATASET_SAMPLE_CHECK_UNAVAILABLE"):
        check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)


@pytest.mark.parametrize("index", [0, 1])
def test_child_resolver_rejects_changed_path_or_out_of_range_binding(setup, monkeypatch, index):
    entry = ModuleType("vla_adapters.openpi_entry")
    factories = [SimpleNamespace(repo_id=setup.spec["code_dir"])]
    config = SimpleNamespace(data=SimpleNamespace(datasets=factories))
    entry.prepare_config = lambda spec: (config, None)
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    with pytest.raises(ValueError, match="绑定索引|绑定路径"):
        check._child_result({"schema_version": 1, "spec": setup.spec, "dataset_evidence": setup.evidence, "factory_index": index})


def test_child_resolves_full_spec_and_invokes_native_check_after_reconstruction(setup, monkeypatch):
    entry = ModuleType("vla_adapters.openpi_entry")
    received = []

    def prepare(spec):
        received.append(spec)
        return setup.config, None

    entry.prepare_config = prepare
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    result = check._child_result({"schema_version": 1, "spec": setup.spec, "dataset_evidence": setup.evidence, "factory_index": 0})
    assert result["status"] == "verified" and received == [setup.spec]
    assert setup.runtime.dataset.indices == [0, 3, 6]


def test_parent_uses_factory_identity_index_when_two_bindings_share_the_same_path(setup, isolated_process):
    other = type(setup.factory)()
    vars(other).update(vars(setup.factory))
    setup.config.data = SimpleNamespace(datasets=[other, setup.factory])
    check.inspect_composed_samples_isolated(setup.factory, setup.config, setup.spec, setup.evidence)
    assert json.loads(isolated_process.input)["factory_index"] == 1


def test_child_index_selects_correct_composition_among_repeated_dataset_paths(setup, monkeypatch):
    other = type(setup.factory)()
    vars(other).update(vars(setup.factory))
    other.composition = deepcopy(other.composition)
    other.composition["robot_model"] = "other binding"
    setup.config.data = SimpleNamespace(datasets=[other, setup.factory])
    entry = ModuleType("vla_adapters.openpi_entry")
    entry.prepare_config = lambda spec: (setup.config, None)
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    request = {"schema_version": 1, "spec": setup.spec, "dataset_evidence": setup.evidence, "factory_index": 1}
    assert check._child_result(request)["status"] == "verified"
    with pytest.raises(ValueError, match="组合配置不一致"):
        check._child_result({**request, "factory_index": 0})


@pytest.mark.parametrize("index", [None, True, -1, 64, "1"])
def test_child_binding_index_schema_is_strict(setup, index):
    with pytest.raises(ValueError, match="请求结构无效"):
        check._child_result({"schema_version": 1, "spec": setup.spec, "dataset_evidence": setup.evidence,
                             "factory_index": index})


def test_standalone_child_protocol_rejects_invalid_request_without_unstructured_output():
    result = subprocess.run([sys.executable, str(Path(check.__file__).resolve()), "--sample-probe"],
                            input=b"{}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 2 and result.stderr == b""
    response = json.loads(result.stdout)
    assert response["ok"] is False and response["error"].startswith("DATASET_SAMPLE_INVALID:")


def test_child_discards_python_and_native_library_noise_before_bounded_json_response():
    script = '''
import os, sys
from vla_adapters import dataset_samples as sample
def noisy(request):
    print("python decoder log" * 10000)
    os.write(1, b"native decoder log" * 10000)
    os.write(2, b"native stderr log" * 10000)
    return {"status": "verified"}
sample._child_result = noisy
sys.exit(sample._child_main())
'''
    result = subprocess.run([sys.executable, "-c", script], input=b"{}", stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 0 and result.stderr == b"" and len(result.stdout) < 200
    assert json.loads(result.stdout)["result"] == {"status": "verified"}


def test_child_sets_its_own_default_alarm_independent_of_parent_cleanup():
    script = '''
import sys
from vla_adapters import dataset_samples as sample
armed = []
sample.signal.signal = lambda number, handler: armed.append([number, handler])
sample.signal.alarm = lambda seconds: armed.append(seconds)
sample._child_result = lambda request: {"guards": armed.copy()}
sys.exit(sample._child_main())
'''
    result = subprocess.run([sys.executable, "-c", script], input=b"{}", stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 0 and result.stderr == b""
    assert json.loads(result.stdout)["result"]["guards"] == [[signal.SIGALRM, signal.SIG_DFL], 69]
