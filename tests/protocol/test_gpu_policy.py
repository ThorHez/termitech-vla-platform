import copy

import pytest

from vla_protocol.gpu_policy import gpu_deployment_error, normalize_gpu_policy


def desktop():
    return {"uuid": "GPU-desktop", "memory_total_mb": 24564., "memory_used_mb": 1460.44,
            "memory_free_mb": 22580., "utilization_pct": 17.,
            "processes": [{"pid": 2728, "kind": "graphics", "memory_mb": 209.84}]}


def test_desktop_graphics_are_eligible_only_in_explicit_shared_mode():
    gpu = desktop()
    assert gpu_deployment_error(gpu, "shared", 16384) is None
    assert gpu_deployment_error(gpu)["code"] == "GPU_BUSY"
    assert normalize_gpu_policy("shared", 16384) == {"mode": "shared", "memory_mb": 16384, "reserve_mb": 1024}


def test_reserve_boundary_and_nvml_free_rather_than_total_minus_used():
    gpu = desktop()
    gpu["memory_free_mb"] = 16384 + 1024
    assert gpu_deployment_error(gpu, "shared", 16384) is None
    gpu["memory_free_mb"] -= .01
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == "GPU_MEMORY_INSUFFICIENT"
    gpu["memory_used_mb"] = 100
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == "GPU_MEMORY_INSUFFICIENT"


@pytest.mark.parametrize("kind,code", [("compute", "GPU_BUSY"), ("compute_graphics", "GPU_BUSY"),
                                      ("unknown", "GPU_TELEMETRY_UNAVAILABLE"),
                                      (None, "GPU_TELEMETRY_UNAVAILABLE")])
def test_compute_and_untyped_processes_block_sharing(kind, code):
    gpu = desktop()
    gpu["processes"][0]["kind"] = kind
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == code


@pytest.mark.parametrize("value", [None, 255, 0, -1, 16384., True, "16384", 1048577])
def test_budget_requires_a_bounded_explicit_integer(value):
    with pytest.raises(ValueError):
        normalize_gpu_policy("shared", value)
    assert gpu_deployment_error(desktop(), "shared", value)["code"] == "GPU_POLICY_INVALID"


@pytest.mark.parametrize("field,value", [("memory_total_mb", float("nan")), ("memory_used_mb", None),
                                        ("memory_free_mb", None), ("memory_free_mb", float("inf")),
                                        ("memory_free_mb", -1), ("memory_free_mb", 25000),
                                        ("processes", None)])
def test_unreliable_telemetry_never_implies_available(field, value):
    gpu = desktop()
    gpu[field] = value
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == "GPU_TELEMETRY_UNAVAILABLE"


def test_platform_reservations_and_process_ownership_remain_blockers():
    gpu = desktop()
    gpu["reserved_by"] = "another-platform-deployment"
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == "GPU_BUSY"
    del gpu["reserved_by"]
    gpu["processes"][0]["ownership"] = "platform"
    assert gpu_deployment_error(gpu, "shared", 16384)["code"] == "GPU_BUSY"


def test_shared_does_not_apply_training_utilization_limit_or_modify_snapshot():
    gpu = desktop()
    gpu["utilization_pct"] = 90
    before = copy.deepcopy(gpu)
    assert gpu_deployment_error(gpu, "shared", 16384) is None
    assert gpu == before
    with pytest.raises(ValueError):
        normalize_gpu_policy("exclusive", 16384)
