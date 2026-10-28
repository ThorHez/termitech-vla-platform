"""Shared inference admission remains separate from training exclusivity."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import inspect, select
from vla_protocol.gpu_policy import normalize_gpu_policy

from vla_control.models import Command, Node
from vla_control.policy_deployment_models import PolicyDeployment
from vla_control.scheduler import gpu_available

from .conftest import ADMIN
from .test_policy_deployments import ack, action, command, create, get, running
from .test_recipe_details import register


SHARED = {"gpu_mode": "shared", "gpu_memory_mb": 16384, "gpu_uuid": "GPU-1"}


def graphics_gpu(**changes):
    return {"uuid": "GPU-1", "name": "desktop 4090", "memory_total_mb": 24564,
            "memory_used_mb": 1460, "memory_free_mb": 22600, "utilization_pct": 18,
            "processes": [{"pid": 100, "kind": "graphics", "memory_mb": 210},
                          {"pid": 200, "kind": "graphics", "memory_mb": 400}], **changes}


def shared_node(platform, *, gpu=None, capabilities=None):
    node = register(platform)
    response = platform.client.post('/agent/v1/heartbeat', headers=node['headers'], json={
        'node_id': node['node_id'], 'gpus': [graphics_gpu() if gpu is None else gpu],
        'metrics': {'capabilities': ['POLICY_DEPLOY_SHARED_GPU_V1'] if capabilities is None else capabilities},
        'active_attempts': []})
    assert response.status_code == 200, response.text
    return node


def test_desktop_graphics_allows_explicit_shared_inference_but_remains_busy_for_training(platform):
    node = shared_node(platform)
    assert not gpu_available(graphics_gpu())
    exclusive = create(platform, node)
    assert exclusive.status_code == 409 and exclusive.json()['error']['code'] == 'POLICY_DEPLOY_GPU_UNAVAILABLE'
    response = create(platform, node, **SHARED)
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment['gpu_mode'] == 'shared' and deployment['gpu_memory_mb'] == 16384
    payload = command(platform, node, 'POLICY_DEPLOY_START')['payload']
    assert payload['gpu_mode'] == 'shared' and payload['gpu_memory_mb'] == 16384
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource['gpus'][0]['memory_free_mb'] == 22600
    assert resource['gpus'][0]['processes'][0]['kind'] == 'graphics'
    assert resource['gpus'][0]['reserved_by'] == 'deployment:' + deployment['id']
    assert create(platform, node, port=8001, **SHARED).json()['error']['code'] == 'POLICY_DEPLOY_GPU_UNAVAILABLE'


@pytest.mark.parametrize('changes,code', [
    ({'memory_free_mb': 16384 + 1023}, 'GPU_MEMORY_INSUFFICIENT'),
    ({'memory_free_mb': None}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'memory_free_mb': 25000}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': None}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': [{'pid': 100, 'kind': 'compute'}]}, 'GPU_BUSY'),
    ({'processes': [{'pid': 100, 'kind': 'compute_graphics'}]}, 'GPU_BUSY'),
    ({'processes': [{'pid': 100, 'kind': 'unknown'}]}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': [{'pid': 100}]}, 'GPU_TELEMETRY_UNAVAILABLE'),
    ({'processes': [{'pid': 100, 'kind': 'graphics', 'ownership': 'platform'}]}, 'GPU_BUSY'),
])
def test_shared_gpu_rejects_insufficient_or_unreliable_free_memory_and_non_graphics(platform, changes, code):
    node = shared_node(platform, gpu=graphics_gpu(**changes))
    response = create(platform, node, **SHARED)
    assert response.status_code == 409, response.text
    assert response.json()['error']['code'] == 'POLICY_DEPLOY_' + code
    assert platform.commands(node) == []


def test_shared_budget_uses_real_free_memory_including_driver_reservation(platform):
    # total-used suggests 23 GiB, but NVML's real free count includes driver-reserved memory.
    node = shared_node(platform, gpu=graphics_gpu(memory_used_mb=1024, memory_free_mb=17000))
    response = create(platform, node, **SHARED)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'POLICY_DEPLOY_GPU_MEMORY_INSUFFICIENT'


def test_shared_budget_accepts_exact_headroom_and_ignores_graphics_utilization_threshold(platform):
    node = shared_node(platform, gpu=graphics_gpu(memory_free_mb=17408, utilization_pct=95))
    assert create(platform, node, **SHARED).status_code == 202


@pytest.mark.parametrize('capabilities', [[], ['POLICY_DEPLOY_START'], 'POLICY_DEPLOY_SHARED_GPU_V1'])
def test_shared_requires_explicit_agent_support(platform, capabilities):
    node = shared_node(platform, capabilities=capabilities)
    response = create(platform, node, **SHARED)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'POLICY_DEPLOY_SHARED_GPU_UNSUPPORTED'
    assert platform.commands(node) == []


@pytest.mark.parametrize('changes', [
    {'gpu_mode': 'other'}, {'gpu_memory_mb': None}, {'gpu_memory_mb': True},
    {'gpu_memory_mb': 255}, {'gpu_memory_mb': 1048577}, {'gpu_memory_mb': 16384.0},
    {'gpu_memory_mb': '16384'}, {'gpu_mode': 'exclusive', 'gpu_memory_mb': 16384},
])
def test_gpu_mode_and_budget_are_strictly_validated(platform, changes):
    node = shared_node(platform)
    response = create(platform, node, **{**SHARED, **changes})
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


def test_shared_without_a_gpu_does_not_fall_back_to_cpu(platform):
    node = shared_node(platform)
    with platform.app.state.db.write() as session:
        session.get(Node, node['node_id']).gpus = []
    response = create(platform, node, gpu_mode='shared', gpu_memory_mb=16384)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'POLICY_DEPLOY_GPU_UNAVAILABLE'


@pytest.mark.parametrize('gpu_policy', [None, {'mode': 'exclusive', 'memory_mb': None, 'reserve_mb': 0},
    {'mode': 'shared', 'memory_mb': 8192, 'reserve_mb': 1024},
    {'mode': 'shared', 'memory_mb': 16384.0, 'reserve_mb': 1024},
    {'mode': 'shared', 'memory_mb': 16384, 'reserve_mb': 1024.0}])
@pytest.mark.parametrize('state', ['STARTING', 'RUNNING'])
def test_shared_receipt_must_confirm_frozen_mode_and_budget(platform, gpu_policy, state):
    node = shared_node(platform)
    deployment = create(platform, node, **SHARED).json()
    receipt = running(state=state)
    if gpu_policy is not None:
        receipt['gpu_policy'] = gpu_policy
    ack(platform, node, command(platform, node, 'POLICY_DEPLOY_START'), receipt)
    current = get(platform, deployment)
    assert current['state'] == 'UNKNOWN'
    assert current['error']['code'] == 'POLICY_DEPLOY_GPU_POLICY_MISMATCH'
    assert create(platform, node, port=8001, **SHARED).status_code == 409


def test_shared_receipt_survives_status_and_stop_releases_the_reservation(platform):
    node = shared_node(platform)
    deployment = create(platform, node, **SHARED).json()
    policy = normalize_gpu_policy('shared', 16384)
    receipt = running(gpu_policy=policy)
    cmd = command(platform, node, 'POLICY_DEPLOY_START')
    ack(platform, node, cmd, receipt)
    assert get(platform, deployment)['state'] == 'RUNNING'
    with platform.app.state.db.read() as session:
        saved = session.get(Command, cmd['id'])
        assert saved.result['gpu_policy'] == policy
    assert action(platform, deployment, 'refresh').status_code == 202
    ack(platform, node, command(platform, node, 'POLICY_DEPLOY_STATUS'), receipt)
    assert get(platform, deployment)['state'] == 'RUNNING'
    assert action(platform, deployment, 'stop').status_code == 202
    ack(platform, node, command(platform, node, 'POLICY_DEPLOY_STOP'), {'state': 'STOPPED'})
    assert get(platform, deployment)['state'] == 'STOPPED'
    assert create(platform, node, **SHARED).status_code == 202


def test_concurrent_shared_requests_still_reserve_the_whole_gpu_for_one_platform_service(platform):
    node = shared_node(platform)
    barrier = Barrier(2)
    def submit(port):
        barrier.wait(timeout=5)
        return create(platform, node, port=port, **SHARED).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(submit, [8000, 8001]))
    assert sorted(statuses) == [202, 409]


def test_additive_shared_policy_migration_preserves_legacy_records(platform):
    node = register(platform)
    deployment = create(platform, node).json()
    database = platform.app.state.db
    with database.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE policy_deployments DROP COLUMN gpu_mode')
        connection.exec_driver_sql('ALTER TABLE policy_deployments DROP COLUMN gpu_memory_mb')
    database.initialize()
    columns = {row['name'] for row in inspect(database.engine).get_columns('policy_deployments')}
    assert {'gpu_mode', 'gpu_memory_mb'} <= columns
    current = get(platform, deployment)
    assert current == deployment
    with database.read() as session:
        rows = list(session.scalars(select(PolicyDeployment)))
        assert len(rows) == 1 and rows[0].gpu_mode == 'exclusive' and rows[0].gpu_memory_mb is None
