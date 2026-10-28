"""Explicit inference framework selection without importing a model runtime.

This contract follows the registered OpenPI loader: ``model.safetensors``
selects PyTorch whenever it exists, otherwise ``params/`` selects JAX. It
does not convert weights or infer a framework from a machine's GPU model.
"""

from pathlib import Path


def normalize_inference_backend(value="auto"):
    if type(value) is not str or value not in {"auto", "jax", "pytorch"}:
        raise ValueError("POLICY_BACKEND_INVALID: 推理框架必须是 auto、jax 或 pytorch")
    return value


def resolve_checkpoint_backend(checkpoint_path, requested="auto"):
    """Resolve regular checkpoint markers, leaving weight contents untouched.

    A mixed directory requires an explicit selection. Explicit JAX cannot
    accept a PyTorch marker because the current OpenPI loader would then run
    PyTorch regardless of the requested framework. Callers remain responsible
    for checking managed roots, normalization assets and weight integrity.
    """
    requested = normalize_inference_backend(requested)
    try:
        checkpoint = Path(checkpoint_path)
    except (TypeError, ValueError):
        raise ValueError("POLICY_CHECKPOINT_INVALID: 权重路径必须是 checkpoint 目录") from None
    if not checkpoint.is_dir():
        raise ValueError("POLICY_CHECKPOINT_INVALID: 权重路径必须是 checkpoint 目录")
    params = checkpoint / "params"
    tensors = checkpoint / "model.safetensors"
    has_jax = params.is_dir() and not params.is_symlink()
    has_pytorch = tensors.is_file() and not tensors.is_symlink()
    if not has_jax and not has_pytorch:
        raise ValueError("POLICY_CHECKPOINT_INVALID: checkpoint 需要 params/ 目录或 model.safetensors 常规文件")
    if requested == "auto":
        if not has_pytorch and (tensors.exists() or tensors.is_symlink()):
            raise ValueError("POLICY_CHECKPOINT_INVALID: model.safetensors 必须是常规文件且不能是符号链接")
        if has_jax and has_pytorch:
            raise ValueError("POLICY_BACKEND_AMBIGUOUS: checkpoint 同时包含 JAX 和 PyTorch 权重，请明确选择推理框架")
        return "pytorch" if has_pytorch else "jax"
    if requested == "pytorch":
        if has_pytorch:
            return "pytorch"
        raise ValueError("POLICY_BACKEND_MISMATCH: 已选择 PyTorch，但 checkpoint 缺少 model.safetensors；请使用对应框架的权重")
    # Even a non-regular PyTorch marker would make the upstream loader choose
    # its PyTorch branch. Never claim that this directory will load as JAX.
    if tensors.exists() or tensors.is_symlink():
        raise ValueError("POLICY_BACKEND_MISMATCH: 已选择 JAX，但 checkpoint 包含 model.safetensors；请将两种框架的权重分开放置")
    if has_jax:
        return "jax"
    raise ValueError("POLICY_BACKEND_MISMATCH: 已选择 JAX，但 checkpoint 缺少 params/ 目录；请使用对应框架的权重")
