"""Inference sharing needs classified graphics processes and actual free memory."""

from types import ModuleType, SimpleNamespace
import sys

import pytest

from vla_agent.metrics import check_deployment_gpu, check_gpus, gpu_inventory
from vla_agent.safety import AgentError


def _graphics_gpu(**changes):
    return {"uuid": "GPU-fixture", "memory_total_mb": 24564, "memory_used_mb": 1460,
            "memory_free_mb": 22600, "utilization_pct": 18,
            "processes": [{"pid": 100, "kind": "graphics", "memory_mb": 400}], **changes}


def test_nvml_keeps_actual_free_memory_and_merged_process_classes(monkeypatch):
    module = ModuleType('pynvml')
    module.NVMLError = type('NVMLError', (Exception,), {})
    module.nvmlInit = module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetCount = lambda: 1
    module.nvmlDeviceGetHandleByIndex = lambda _: object()
    module.nvmlDeviceGetUUID = lambda _: b'GPU-fixture'
    module.nvmlDeviceGetName = lambda _: b'Graphics fixture'
    module.nvmlDeviceGetMemoryInfo = lambda _: SimpleNamespace(total=24 * 2**30, used=2**30, free=22 * 2**30)
    module.nvmlDeviceGetUtilizationRates = lambda _: SimpleNamespace(gpu=18)
    module.nvmlDeviceGetComputeRunningProcesses = lambda _: [SimpleNamespace(pid=1, usedGpuMemory=100 * 2**20), SimpleNamespace(pid=3, usedGpuMemory=200 * 2**20)]
    module.nvmlDeviceGetGraphicsRunningProcesses = lambda _: [SimpleNamespace(pid=2, usedGpuMemory=300 * 2**20), SimpleNamespace(pid=3, usedGpuMemory=400 * 2**20)]
    monkeypatch.setitem(sys.modules, 'pynvml', module)
    rows, available = gpu_inventory()
    assert available['available'] is True
    row = rows[0]
    assert row['memory_free_mb'] == 22 * 1024
    assert row['memory_total_mb'] - row['memory_used_mb'] != row['memory_free_mb']
    assert {item['pid']: (item['kind'], item['memory_mb']) for item in row['processes']} == {
        1: ('compute', 100), 2: ('graphics', 300), 3: ('compute_graphics', 400)}
    module.nvmlDeviceGetGraphicsRunningProcesses = lambda _: None
    assert gpu_inventory()[0][0]['processes'] is None


def test_shared_admission_does_not_relax_training_exclusivity():
    row = _graphics_gpu()
    inventory = ([row], {'available': True})
    assert check_deployment_gpu('GPU-fixture', 'shared', 16384, inventory=inventory) == {
        'mode': 'shared', 'memory_mb': 16384, 'reserve_mb': 1024}
    with pytest.raises(AgentError) as error:
        check_gpus(['GPU-fixture'], inventory=inventory)
    assert error.value.code == 'GPU_BUSY'
    with pytest.raises(AgentError) as error:
        check_deployment_gpu('GPU-fixture', inventory=inventory)
    assert error.value.code == 'GPU_BUSY'


@pytest.mark.parametrize('changes,code', [
    ({'processes': [{'pid': 100, 'kind': 'compute'}]}, 'GPU_BUSY'),
    ({'processes': [{'pid': 100, 'kind': 'compute_graphics'}]}, 'GPU_BUSY'),
    ({'processes': [{'pid': 100, 'kind': 'unknown'}]}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': [{'pid': 100}]}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': None}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'memory_free_mb': None}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'memory_free_mb': 17407}, 'GPU_MEMORY_INSUFFICIENT'),
])
def test_shared_admission_fails_closed_for_compute_unknown_or_low_free_memory(changes, code):
    with pytest.raises(AgentError) as error:
        check_deployment_gpu('GPU-fixture', 'shared', 16384,
                             inventory=([_graphics_gpu(**changes)], {'available': True}))
    assert error.value.code == code


def test_shared_memory_budget_boundary_and_unknown_gpu():
    inventory = ([_graphics_gpu(memory_free_mb=17408)], {'available': True})
    check_deployment_gpu('GPU-fixture', 'shared', 16384, inventory=inventory)
    for gpu, availability, code in [('GPU-missing', True, 'GPU_UNAVAILABLE'),
                                    ('GPU-fixture', False, 'GPU_TELEMETRY_UNAVAILABLE')]:
        with pytest.raises(AgentError) as error:
            check_deployment_gpu(gpu, 'shared', 16384, inventory=(inventory[0], {'available': availability}))
        assert error.value.code == code
    with pytest.raises(AgentError) as error:
        check_deployment_gpu('GPU-fixture', 'shared', True, inventory=inventory)
    assert error.value.code == 'GPU_POLICY_INVALID'


def test_export_forces_cpu_and_never_checks_gpu(monkeypatch):
    from vla_agent.policy_export import _export_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-never-use')
    monkeypatch.setenv('JAX_PLATFORMS', 'cuda')
    monkeypatch.setenv('JAX_PLATFORM_NAME', 'gpu')
    monkeypatch.setattr('vla_agent.metrics.gpu_inventory', lambda: pytest.fail('Export must not inspect GPU'))
    env = _export_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['JAX_PLATFORMS'] == env['JAX_PLATFORM_NAME'] == 'cpu'
