"""Explicit training coexistence with external processes, separate from inference.

The per-card budget is an admission and allocator setting, not hardware memory
isolation. Platform allocations and user reservations remain separate blockers.
"""

import math


TRAINING_SHARED_GPU_CAPABILITY = "TRAINING_SHARED_GPU_V1"
TRAINING_GPU_MEMORY_RESERVE_MB = 1024
MIN_TRAINING_GPU_MEMORY_MB = 256
MAX_TRAINING_GPU_MEMORY_MB = 1024 * 1024


def normalize_training_gpu_policy(value):
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("TRAINING_GPU_POLICY_INVALID: GPU 使用策略必须为对象")
    if value == {"mode": "exclusive"}:
        return None
    if set(value) != {"mode", "memory_mb"} or value.get("mode") != "shared":
        raise ValueError("TRAINING_GPU_POLICY_INVALID: 共享训练需明确 mode=shared 和每卡显存预算")
    memory = value["memory_mb"]
    if type(memory) is not int or not MIN_TRAINING_GPU_MEMORY_MB <= memory <= MAX_TRAINING_GPU_MEMORY_MB:
        raise ValueError("TRAINING_GPU_POLICY_INVALID: 每卡显存预算必须为 256 至 1048576 MiB 的整数")
    return {"mode": "shared", "memory_mb": memory}


def _number(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _error(code, message):
    return {"code": code, "message": message}


def training_gpu_error(gpu, policy=None):
    """Read-only admission against a fresh, complete telemetry snapshot."""
    try:
        policy = normalize_training_gpu_policy(policy)
    except (ValueError, TypeError) as error:
        return _error("TRAINING_GPU_POLICY_INVALID", str(error))
    if not isinstance(gpu, dict):
        return _error("GPU_TELEMETRY_UNAVAILABLE", "无法读取所选 GPU 状态")
    if gpu.get("reserved_by"):
        return _error("GPU_BUSY", "GPU 已被平台训练或推理任务分配")
    processes = gpu.get("processes")
    total, used, utilization = (gpu.get(name) for name in
                                ("memory_total_mb", "memory_used_mb", "utilization_pct"))
    if (not isinstance(processes, list) or not _number(total) or total <= 0
            or not _number(used) or not 0 <= used <= total
            or not _number(utilization) or not 0 <= utilization <= 100):
        return _error("GPU_TELEMETRY_UNAVAILABLE", "GPU 进程、显存或利用率信息缺失或无效")
    if policy is None:
        if processes or utilization > 5 or used > max(128, total * .02):
            return _error("GPU_BUSY", "独占训练要求 GPU 无现有进程、利用率不超过 5%，显存接近空闲")
        return None
    free = gpu.get("memory_free_mb")
    if not _number(free) or not 0 <= free <= total or free + used > total + 1:
        return _error("GPU_TELEMETRY_UNAVAILABLE", "共享训练需要可靠的实际剩余显存，请等待 Agent 更新")
    for process in processes:
        if (not isinstance(process, dict) or type(process.get("pid")) is not int or process["pid"] <= 0
                or type(process.get("kind")) is not str
                or process["kind"] not in {"compute", "graphics", "compute_graphics"}
                or process.get("ownership") is not None and (
                    type(process["ownership"]) is not str or process["ownership"] not in {"external", "platform"})):
            return _error("GPU_TELEMETRY_UNAVAILABLE", "无法确认 GPU 现有进程，暂不能共享训练")
        if process.get("ownership") == "platform":
            return _error("GPU_BUSY", "GPU 存在平台任务，不允许同时运行另一训练")
    required = policy["memory_mb"] + TRAINING_GPU_MEMORY_RESERVE_MB
    if free < required:
        return _error("GPU_MEMORY_INSUFFICIENT",
                      f"实际剩余显存 {free / 1024:.2f} GiB 不足以容纳每卡训练预算 "
                      f"{policy['memory_mb'] / 1024:.2f} GiB 和 1 GiB 余量")
    return None
