"""Checkpoint framework contracts are checked without reading model weights."""

from pathlib import Path

import pytest

from vla_protocol.policy_backend import normalize_inference_backend, resolve_checkpoint_backend


def test_normalizes_only_exact_supported_frameworks():
    assert normalize_inference_backend() == "auto"
    for backend in ("auto", "jax", "pytorch"):
        assert normalize_inference_backend(backend) == backend


@pytest.mark.parametrize("value", [None, True, 0, "", "torch", "JAX", "PyTorch", " jax", "jax ", [], {}])
def test_rejects_unrecognized_framework_values(value):
    with pytest.raises(ValueError, match="^POLICY_BACKEND_INVALID:"):
        normalize_inference_backend(value)


@pytest.mark.parametrize("backend", ["jax", "pytorch"])
def test_resolves_single_format_without_reading_weights(tmp_path, monkeypatch, backend):
    if backend == "jax":
        (tmp_path / "params").mkdir()
    else:
        (tmp_path / "model.safetensors").write_bytes(b"not a loaded model")
    def no_weight_reads(*_args, **_kwargs):
        raise AssertionError("framework selection must not read model contents")
    monkeypatch.setattr(Path, "read_bytes", no_weight_reads)
    assert resolve_checkpoint_backend(tmp_path) == backend
    assert resolve_checkpoint_backend(str(tmp_path), backend) == backend
    other = "pytorch" if backend == "jax" else "jax"
    with pytest.raises(ValueError, match="^POLICY_BACKEND_MISMATCH:"):
        resolve_checkpoint_backend(tmp_path, other)


def test_mixed_weights_require_explicit_torch_and_never_claim_jax(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="^POLICY_BACKEND_AMBIGUOUS:"):
        resolve_checkpoint_backend(tmp_path)
    assert resolve_checkpoint_backend(tmp_path, "pytorch") == "pytorch"
    with pytest.raises(ValueError, match="^POLICY_BACKEND_MISMATCH:.*分开放置"):
        resolve_checkpoint_backend(tmp_path, "jax")


@pytest.mark.parametrize("backend", ["auto", "jax", "pytorch"])
def test_empty_or_wrong_marker_kinds_are_invalid_checkpoints(tmp_path, backend):
    for path in (tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError, match="^POLICY_CHECKPOINT_INVALID:"):
            resolve_checkpoint_backend(path, backend)
    (tmp_path / "params").write_text("not a directory")
    (tmp_path / "model.safetensors").mkdir()
    with pytest.raises(ValueError, match="^POLICY_CHECKPOINT_INVALID:"):
        resolve_checkpoint_backend(tmp_path, backend)


@pytest.mark.parametrize("value", [None, 1, True])
def test_invalid_checkpoint_path_types_have_stable_errors(value):
    with pytest.raises(ValueError, match="^POLICY_CHECKPOINT_INVALID:"):
        resolve_checkpoint_backend(value)


def test_symbolic_weight_markers_are_not_accepted_as_regular_weights(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "params").mkdir()
    (real / "model.safetensors").write_bytes(b"fixture")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "params").symlink_to(real / "params", target_is_directory=True)
    (linked / "model.safetensors").symlink_to(real / "model.safetensors")
    with pytest.raises(ValueError, match="^POLICY_CHECKPOINT_INVALID:"):
        resolve_checkpoint_backend(linked)


def test_jax_rejects_any_torch_marker_that_would_override_the_loader(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "model.safetensors").mkdir()
    with pytest.raises(ValueError, match="^POLICY_BACKEND_MISMATCH:"):
        resolve_checkpoint_backend(tmp_path, "jax")
    with pytest.raises(ValueError, match="^POLICY_CHECKPOINT_INVALID:"):
        resolve_checkpoint_backend(tmp_path)
