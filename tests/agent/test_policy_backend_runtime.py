"""Backend/device gates without allocating GPU memory or loading model weights."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vla_agent.policy_deployment_worker import backend_runtime_options, validate_backend_model, verify_loaded_backend


@pytest.fixture
def manifest(tmp_path):
    import vla_protocol
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    return {"checkpoint_path": str(tmp_path), "inference_backend": "pytorch", "resolved_backend": "pytorch",
            "gpu_uuid": "GPU-selected", "protocol_root": str(Path(vla_protocol.__file__).parent.parent)}


def test_pytorch_gpu_uses_selected_visible_device_and_never_falls_back_to_cpu(manifest, monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert backend_runtime_options(manifest) == {"pytorch_device": "cuda:0"}
    cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="POLICY_BACKEND_GPU_UNAVAILABLE"):
        backend_runtime_options(manifest)
    manifest["gpu_uuid"] = None
    assert backend_runtime_options(manifest) == {"pytorch_device": "cpu"}


def test_changed_checkpoint_format_is_rejected_before_runtime_import(manifest, monkeypatch):
    (Path(manifest["checkpoint_path"]) / "model.safetensors").unlink()
    (Path(manifest["checkpoint_path"]) / "params").mkdir()
    manifest["inference_backend"] = "auto"
    monkeypatch.setattr("vla_agent.policy_deployment_worker.importlib.import_module",
                        lambda *_: pytest.fail("changed format must not import a framework"))
    with pytest.raises(ValueError, match="POLICY_BACKEND_MISMATCH"):
        backend_runtime_options(manifest)


def test_missing_framework_is_reported_as_runtime_error(manifest, monkeypatch):
    def unavailable(_name):
        raise ImportError("CUDA wheel unavailable")
    monkeypatch.setattr("vla_agent.policy_deployment_worker.importlib.import_module", unavailable)
    with pytest.raises(ValueError, match="POLICY_RUNTIME_UNAVAILABLE.*pytorch"):
        backend_runtime_options(manifest)


def test_jax_requires_framework_gpu_visibility_not_only_nvml(manifest, monkeypatch):
    root = Path(manifest["checkpoint_path"])
    (root / "model.safetensors").unlink()
    (root / "params").mkdir()
    manifest.update(inference_backend="jax", resolved_backend="jax")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda backend: []))
    with pytest.raises(ValueError, match="POLICY_BACKEND_GPU_UNAVAILABLE"):
        backend_runtime_options(manifest)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda backend: [object()]))
    assert backend_runtime_options(manifest) == {}


@pytest.mark.parametrize("family,dimension,accepted", [
    ("Pi0Config", 32, True), ("Pi0Config", 14, False),
    ("Pi0FASTConfig", 32, False), ("Pi0MemoryConfig", 32, False), ("Pi0ValueConfig", 32, False),
])
def test_pytorch_cannot_silently_load_another_model_family(manifest, family, dimension, accepted):
    model = type(family, (), {"__module__": "openpi.models.pi0_config", "action_dim": dimension})()
    train = SimpleNamespace(model=model)
    if accepted:
        validate_backend_model(manifest, train)
    else:
        with pytest.raises(ValueError, match="POLICY_BACKEND_MODEL_UNSUPPORTED"):
            validate_backend_model(manifest, train)
    validate_backend_model({"resolved_backend": "jax"}, train)


def test_legacy_service_keeps_its_original_process_contract():
    assert backend_runtime_options({}) == {}


def test_loaded_policy_is_checked_before_claiming_a_framework_or_device(manifest):
    for policy in [SimpleNamespace(), SimpleNamespace(_is_pytorch_model=False)]:
        with pytest.raises(ValueError, match="POLICY_BACKEND_MISMATCH"):
            verify_loaded_backend(manifest, policy)
    with pytest.raises(ValueError, match="POLICY_BACKEND_GPU_UNAVAILABLE"):
        verify_loaded_backend(manifest, SimpleNamespace(_is_pytorch_model=True, _pytorch_device="cpu"))
    verify_loaded_backend(manifest, SimpleNamespace(_is_pytorch_model=True, _pytorch_device="cuda:0"))
