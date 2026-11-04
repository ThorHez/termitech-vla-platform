"""Explicit shared-training admission and process-local JAX allocator settings.

The allocator pool is not hardware isolation or a cap on all CUDA/library
allocations. Existing platform GPU locks and resource guards remain mandatory.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR

from vla_protocol.training_gpu_policy import normalize_training_gpu_policy

from .safety import AgentError


def _invalid(message):
    raise AgentError("TRAINING_GPU_POLICY_INVALID", message)


def requested_training_gpus(payload):
    selected = payload.get("requested_gpu_uuids")
    if selected is None:
        selected = (payload.get("code_expectation") or {}).get("requested_gpu_uuids")
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 256
            or any(not isinstance(value, str) or not value or len(value) > 200
                   or any(character.isspace() or ord(character) < 32 for character in value)
                   for value in selected) or len(set(selected)) != len(selected)):
        _invalid("共享训练必须明确选择具体 GPU，不能自动分配或使用重复设备")
    return selected


def validate_training_gpu_policy(payload, profile, *, require_frozen=False):
    try:
        policy = normalize_training_gpu_policy(payload.get("training_gpu_policy"))
        frozen = normalize_training_gpu_policy((payload.get("code_expectation") or {}).get("training_gpu_policy"))
    except (TypeError, ValueError) as exc:
        _invalid(str(exc))
    if require_frozen and policy != frozen:
        _invalid("启动 GPU 使用策略与已确认预检不一致，请重新预检")
    if policy is None:
        return None
    count = payload.get("gpu_count", len(payload.get("gpu_uuids", [])))
    if (payload.get("template_id") not in {"openpi-sft", "recap"}
            or profile.get("adapter") != "openpi-sft" or type(count) is not int or count < 1):
        _invalid("共享显存仅支持申请 GPU 的 OpenPI 训练，不支持 CPU 统计或链路测试")
    selected = requested_training_gpus(payload)
    if len(selected) != count:
        _invalid("共享训练所选 GPU 数量与申请数量不一致")
    if require_frozen:
        expected = (payload.get("code_expectation") or {}).get("requested_gpu_uuids")
        if selected != expected or payload.get("gpu_uuids") != selected:
            _invalid("共享训练的实际分配必须与已确认的具体 GPU 顺序一致")
    return policy


def training_gpu_evidence(policy, selected, devices):
    return {"policy": dict(policy), "gpu_uuids": list(selected), "devices": deepcopy(devices)}


def shared_training_environment(environment, policy, devices):
    """Set one conservative BFC fraction for every assigned physical device."""
    try:
        policy = normalize_training_gpu_policy(policy)
    except (TypeError, ValueError) as exc:
        _invalid(str(exc))
    if policy is None:
        return environment
    if not isinstance(devices, list) or not devices:
        _invalid("共享训练缺少启动前的实际 GPU 显存核验")
    try:
        total = max(Decimal(str(row["memory_total_mb"])) for row in devices)
        fraction = (Decimal(policy["memory_mb"]) / total).quantize(Decimal("0.000001"), rounding=ROUND_FLOOR)
        if not fraction.is_finite() or not 0 < fraction < 1:
            raise ValueError()
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        _invalid("无法根据实际显存容量生成共享训练分配比例")
    result = dict(environment)
    for name in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                 "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
                 "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB", "TF_FORCE_UNIFIED_MEMORY",
                 "TF_GPU_ALLOCATOR", "TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC",
                 "JAX_PJRT_CLIENT_CREATE_OPTIONS"):
        result.pop(name, None)
    result.update(XLA_CLIENT_MEM_FRACTION=str(fraction), XLA_PYTHON_CLIENT_ALLOCATOR="bfc",
                  XLA_PYTHON_CLIENT_PREALLOCATE="false")
    return result
