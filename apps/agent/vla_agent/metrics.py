from __future__ import annotations

import os
import psutil

from .safety import AgentError, identity_matches


def gpu_inventory() -> tuple[list[dict], dict]:
    try:
        import pynvml as nvml
    except ImportError:
        return [], {"available": False, "reason": "NVML Python binding is not installed"}
    try:
        nvml.nvmlInit()
    except Exception as exc:
        return [], {"available": False, "reason": type(exc).__name__}
    rows = []
    try:
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            def read(call, default=None):
                try:
                    return call()
                except nvml.NVMLError:
                    return default
            uuid = read(lambda: nvml.nvmlDeviceGetUUID(handle))
            name = read(lambda: nvml.nvmlDeviceGetName(handle))
            memory = read(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
            utilization = read(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
            compute = read(lambda: nvml.nvmlDeviceGetComputeRunningProcesses(handle))
            graphics = read(lambda: nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            processes = None
            if compute is not None and graphics is not None:
                by_pid = {}
                for kind, entries in (("compute", compute), ("graphics", graphics)):
                    for process in entries:
                        used = getattr(process, "usedGpuMemory", None)
                        unknown = used is None or used >= (1 << 60)
                        memory_mb = None if unknown else round(used / 2**20, 2)
                        previous = by_pid.get(process.pid)
                        if previous is None:
                            by_pid[process.pid] = {"pid": process.pid, "memory_mb": memory_mb, "kind": kind}
                        else:
                            if previous["kind"] != kind:
                                previous["kind"] = "compute_graphics"
                            values = [value for value in (previous["memory_mb"], memory_mb) if value is not None]
                            previous["memory_mb"] = max(values) if values else None
                processes = list(by_pid.values())
            rows.append({"uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                         "name": name.decode() if isinstance(name, bytes) else name,
                         "memory_total_mb": round(memory.total / 2**20, 2) if memory else None,
                         "memory_used_mb": round(memory.used / 2**20, 2) if memory else None,
                         "memory_free_mb": round(memory.free / 2**20, 2) if memory else None,
                         "utilization_pct": utilization.gpu if utilization else None,
                         "processes": processes})
        return rows, {"available": True, "reason": None}
    except Exception as exc:
        return rows, {"available": False, "reason": type(exc).__name__}
    finally:
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass


def system_metrics(output_root: str) -> dict:
    errors = {}
    def sample(name, callback):
        try:
            return callback()
        except (psutil.Error, OSError) as exc:
            errors[name] = type(exc).__name__
            return None
    memory = sample("memory", psutil.virtual_memory)
    disk = sample("disk", lambda: psutil.disk_usage(output_root))
    return {"cpu_percent": sample("cpu", lambda: psutil.cpu_percent(interval=None)),
            "memory_total_bytes": memory.total if memory else None,
            "memory_used_bytes": memory.used if memory else None, "memory_percent": memory.percent if memory else None,
            "disk_total_bytes": disk.total if disk else None, "disk_free_bytes": disk.free if disk else None,
            "load_average": sample("load", lambda: list(os.getloadavg())) if hasattr(os, "getloadavg") else None,
            "collection_errors": errors}


def process_metrics(identity: dict) -> dict | None:
    if not identity_matches(identity):
        return None
    try:
        parent = psutil.Process(identity["pid"])
        descendants_complete = True
        try:
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = [parent]
            descendants_complete = False
        rss = 0
        cpu_seconds = 0
        pids = []
        complete = descendants_complete
        for process in processes:
            pids.append(process.pid)
            try:
                rss += process.memory_info().rss
                times = process.cpu_times()
                cpu_seconds += times.user + times.system
            except (psutil.Error, OSError):
                complete = False
                continue
        return {"pid": identity["pid"], "process_ids": pids, "rss_bytes": rss if complete else None,
                "cpu_seconds": cpu_seconds if complete else None,
                "descendants_complete": descendants_complete}
    except (psutil.Error, OSError):
        return None


def check_gpus(requested: list[str], *, inventory: tuple[list[dict], dict] | None = None):
    if not requested:
        return
    if len(requested) != len(set(requested)):
        raise AgentError("GPU_UNAVAILABLE", "Duplicate GPU UUIDs")
    rows, availability = inventory if inventory is not None else gpu_inventory()
    if not availability["available"]:
        raise AgentError("GPU_UNAVAILABLE", "Cannot verify GPU availability through NVML")
    by_uuid = {row["uuid"]: row for row in rows}
    for uuid in requested:
        row = by_uuid.get(uuid)
        if row is None:
            raise AgentError("GPU_UNAVAILABLE", f"GPU is not present: {uuid}")
        if row["processes"] is None:
            raise AgentError("GPU_UNAVAILABLE", f"Cannot verify processes on GPU: {uuid}")
        if row["processes"]:
            raise AgentError("GPU_BUSY", f"GPU has an existing process: {uuid}")


def check_deployment_gpu(uuid: str, mode="exclusive", memory_mb=None, *, inventory=None):
    """Inference admission is separate from the training Runner's exclusivity."""
    from vla_protocol.gpu_policy import gpu_deployment_error, normalize_gpu_policy

    try:
        policy = normalize_gpu_policy(mode, memory_mb)
    except (TypeError, ValueError) as exc:
        raise AgentError("GPU_POLICY_INVALID", str(exc)) from exc
    rows, availability = inventory if inventory is not None else gpu_inventory()
    if availability.get("available") is not True:
        raise AgentError("GPU_TELEMETRY_UNAVAILABLE", "无法通过 NVML 确认部署 GPU 的当前状态")
    row = next((item for item in rows if item.get("uuid") == uuid), None)
    if row is None:
        raise AgentError("GPU_UNAVAILABLE", "所选部署 GPU 不存在")
    error = gpu_deployment_error(row, mode, memory_mb)
    if error is not None:
        raise AgentError(error["code"], error["message"])
    return policy
