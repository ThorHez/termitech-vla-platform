"""Inference GPU admission, independent of training's exclusive scheduling.

Memory budgets are admission estimates, not hardware isolation or a promise
about the model's eventual peak allocation. Shared mode permits graphics-only
processes; compute and unknown processes remain blockers.
"""

import math


SHARED_GPU_CAPABILITY = "POLICY_DEPLOY_SHARED_GPU_V1"
GPU_MEMORY_RESERVE_MB = 1024
MIN_GPU_MEMORY_MB = 256
MAX_GPU_MEMORY_MB = 1024 * 1024


def normalize_gpu_policy(mode="exclusive", memory_mb=None):
    if type(mode) is not str or mode not in {"exclusive", "shared"}:
        raise ValueError("GPU 模式必须是 exclusive 或 shared")
    if mode == "exclusive":
        if memory_mb is not None:
            raise ValueError("独占模式无需填写共享显存预算")
        return {"mode": "exclusive", "memory_mb": None, "reserve_mb": 0}
    if type(memory_mb) is not int or not MIN_GPU_MEMORY_MB <= memory_mb <= MAX_GPU_MEMORY_MB:
        raise ValueError(f"共享显存预算必须是 {MIN_GPU_MEMORY_MB} 至 {MAX_GPU_MEMORY_MB} MiB 的整数")
    return {"mode": "shared", "memory_mb": memory_mb, "reserve_mb": GPU_MEMORY_RESERVE_MB}


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _error(code, message):
    return {"code": code, "message": message}


def gpu_deployment_error(gpu, mode="exclusive", memory_mb=None):
    """Return a specific admission blocker, or None for the observed snapshot."""
    try:
        policy = normalize_gpu_policy(mode, memory_mb)
    except (TypeError, ValueError) as exc:
        return _error("GPU_POLICY_INVALID", str(exc))
    if not isinstance(gpu, dict):
        return _error("GPU_TELEMETRY_UNAVAILABLE", "无法读取所选 GPU 的状态")
    if gpu.get("reserved_by"):
        return _error("GPU_BUSY", "GPU 已被平台训练或部署预约")
    processes = gpu.get("processes")
    if not isinstance(processes, list):
        return _error("GPU_TELEMETRY_UNAVAILABLE", "无法确认 GPU 进程，等待 Agent 更新状态")
    total, used = gpu.get("memory_total_mb"), gpu.get("memory_used_mb")
    if (not _number(total) or total <= 0 or not _number(used) or used < 0 or used > total):
        return _error("GPU_TELEMETRY_UNAVAILABLE", "GPU 显存信息缺失或无效")
    if mode == "exclusive":
        if processes:
            return _error("GPU_BUSY", "独占模式要求 GPU 没有现有进程")
        utilization = gpu.get("utilization_pct")
        if not _number(utilization) or not 0 <= utilization <= 100:
            return _error("GPU_TELEMETRY_UNAVAILABLE", "GPU 利用率信息缺失或无效")
        if utilization > 5 or used > max(128, total * .02):
            return _error("GPU_BUSY", "独占模式要求利用率不超过 5%，已用显存不超过总量的 2% 或 128 MiB")
        return None
    free = gpu.get("memory_free_mb")
    if not _number(free) or not 0 <= free <= total or free + used > total + 1:
        return _error("GPU_TELEMETRY_UNAVAILABLE", "缺少可靠的剩余显存，请更新 Agent 后重试")
    for process in processes:
        if (not isinstance(process, dict) or type(process.get("pid")) is not int or process["pid"] <= 0
                or process.get("kind") not in {"graphics", "compute", "compute_graphics", "unknown"}):
            return _error("GPU_TELEMETRY_UNAVAILABLE", "无法确认 GPU 进程类型，请更新 Agent 后重试")
        if process["kind"] == "unknown":
            return _error("GPU_TELEMETRY_UNAVAILABLE", "GPU 存在类型未知的进程，暂不能共享部署")
        if process["kind"] != "graphics" or process.get("ownership") == "platform":
            return _error("GPU_BUSY", "GPU 存在计算进程或平台任务；共享模式仅允许图形桌面进程")
    required = policy["memory_mb"] + policy["reserve_mb"]
    if free < required:
        return _error("GPU_MEMORY_INSUFFICIENT",
                      f"剩余显存 {free / 1024:.2f} GiB，不足以容纳推理预算 {memory_mb / 1024:.2f} GiB"
                      f" 和预留余量 {GPU_MEMORY_RESERVE_MB / 1024:.0f} GiB")
    return None
