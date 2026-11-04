import copy

import pytest

from vla_protocol.training_gpu_policy import normalize_training_gpu_policy, training_gpu_error


POLICY = {"mode": "shared", "memory_mb": 51200}


def occupied(**changes):
    return {"uuid": "GPU-7", "memory_total_mb": 81920, "memory_used_mb": 20480,
            "memory_free_mb": 61440, "utilization_pct": 0,
            "processes": [{"pid": 546536, "kind": "compute", "ownership": "external"}], **changes}


def test_explicit_external_compute_sharing_does_not_relax_exclusive_or_inference():
    from vla_protocol.gpu_policy import gpu_deployment_error
    row = occupied()
    before = copy.deepcopy(row)
    assert training_gpu_error(row, POLICY) is None
    assert training_gpu_error(row)["code"] == "GPU_BUSY"
    assert gpu_deployment_error(row, "shared", 51200)["code"] == "GPU_BUSY"
    assert row == before
    assert normalize_training_gpu_policy(None) is None
    assert normalize_training_gpu_policy({"mode": "exclusive"}) is None
    assert normalize_training_gpu_policy(POLICY) == POLICY


@pytest.mark.parametrize("value", [True, "shared", [], {}, {"mode": "shared"},
    {"mode": "shared", "memory_mb": True}, {"mode": "shared", "memory_mb": 255},
    {"mode": "shared", "memory_mb": 1048577}, {"mode": "shared", "memory_mb": 8192.0},
    {"mode": "exclusive", "memory_mb": 8192}, {**POLICY, "allow_all": True}])
def test_shared_budget_is_strict_and_unknown_policy_fields_are_rejected(value):
    with pytest.raises(ValueError, match="TRAINING_GPU_POLICY_INVALID"):
        normalize_training_gpu_policy(value)
    assert training_gpu_error(occupied(), value)["code"] == "TRAINING_GPU_POLICY_INVALID"


@pytest.mark.parametrize("kind", ["compute", "graphics", "compute_graphics"])
def test_known_external_process_classes_can_coexist(kind):
    assert training_gpu_error(occupied(processes=[{"pid": 12, "kind": kind}]), POLICY) is None


@pytest.mark.parametrize("process", [{"pid": 3, "kind": "unknown"}, {"pid": True, "kind": "compute"},
    {"pid": 0, "kind": "compute"}, {"pid": 3}, {"pid": 3, "kind": []},
    {"pid": 3, "kind": "compute", "ownership": []},
    {"pid": 3, "kind": "compute", "ownership": "unclassified"}])
def test_unknown_or_invalid_processes_do_not_become_shared_capacity(process):
    assert training_gpu_error(occupied(processes=[process]), POLICY)["code"] == "GPU_TELEMETRY_UNAVAILABLE"


def test_platform_allocations_and_processes_still_block_shared_training():
    assert training_gpu_error(occupied(reserved_by="another-attempt"), POLICY)["code"] == "GPU_BUSY"
    assert training_gpu_error(occupied(processes=[{"pid": 3, "kind": "compute", "ownership": "platform"}]), POLICY)["code"] == "GPU_BUSY"


@pytest.mark.parametrize("field,value", [("memory_free_mb", None), ("memory_free_mb", float("inf")),
    ("memory_free_mb", -1), ("memory_free_mb", 70000), ("memory_total_mb", float("nan")),
    ("memory_total_mb", 10**4000),
    ("memory_used_mb", True), ("utilization_pct", None), ("utilization_pct", float("nan")),
    ("utilization_pct", 101), ("processes", None)])
def test_unreliable_memory_utilization_or_process_telemetry_fails_closed(field, value):
    assert training_gpu_error(occupied(**{field: value}), POLICY)["code"] == "GPU_TELEMETRY_UNAVAILABLE"


def test_real_free_memory_includes_one_gib_margin_and_never_uses_total_minus_used():
    assert training_gpu_error(occupied(memory_free_mb=52224), POLICY) is None
    row = occupied(memory_used_mb=1, memory_free_mb=52223.9)
    assert training_gpu_error(row, POLICY)["code"] == "GPU_MEMORY_INSUFFICIENT"
    assert training_gpu_error(occupied(utilization_pct=100), POLICY) is None
