"""An explicit GPU request is a binding, never a preference or fallback list."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import inspect, select

from vla_control.common import now
from vla_control.db import Database
from vla_control.models import Allocation, Node, Run
from vla_control.policy_deployment_models import PolicyDeployment

from .conftest import ADMIN, confirmed_run_request, gpu, snapshot


OMIT = object()


def preview_request(platform, node, *, selected=OMIT, count=1):
    body = {"name": "specific GPU training", "node_id": node["node_id"], "profile_id": "default",
            "template_id": "smoke", "inputs": {}, "parameters": {"steps": 3}, "gpu_count": count}
    if selected is not OMIT:
        body["requested_gpu_uuids"] = selected
    return platform.client.post('/api/v1/run-previews', headers=ADMIN, json=body)


def ready_preview(platform, node, *, selected=OMIT, count=1):
    response = preview_request(platform, node, selected=selected, count=count)
    assert response.status_code == 202, response.text
    identifier = response.json()['id']
    command = next(item for item in platform.commands(node)
                   if item['type'] == 'PREFLIGHT' and item['payload']['preview_id'] == identifier)
    # An unchanged Agent has no new field or GPU-choice support to negotiate.
    result = {'checks': [{'name': 'local inputs and runtime', 'ok': True}],
              'code_snapshot': snapshot(recipe=command['payload']['recipe'])}
    ack = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node['headers'],
        json={'node_id': node['node_id'], 'status': 'APPLIED', 'result': result})
    assert ack.status_code == 200, ack.text
    preview = platform.client.get(f'/api/v1/run-previews/{identifier}', headers=ADMIN).json()
    assert preview['status'] == 'READY', preview
    return preview


def submit(platform, preview, **extra):
    return platform.client.post('/api/v1/runs', headers={**ADMIN, 'Idempotency-Key': preview['id']},
                                json={**confirmed_run_request(preview), **extra})


def selected_run(platform, node, *, selected=OMIT, count=1):
    response = submit(platform, ready_preview(platform, node, selected=selected, count=count))
    assert response.status_code == 201, response.text
    return response.json()


def starts(platform, node):
    return [item for item in platform.commands(node) if item['type'] == 'START']


@pytest.mark.parametrize('selection,count', [
    ([], 0), ([], 1), (['GPU-1', 'GPU-1'], 2), (['GPU-1'], 2),
    (['GPU-1', 'GPU-2'], 1), ([''], 1), ([' GPU-1'], 1), (['GPU-1\n'], 1),
    (['GPU-1,GPU-2'], 1), ([True], 1), ([1], 1), ('GPU-1', 1),
    (['GPU-other-node'], 1), ([None], 1), (['x' * 201], 1),
])
def test_invalid_selection_cannot_reach_the_agent(platform, selection, count):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    response = preview_request(platform, node, selected=selection, count=count)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


def test_gpu_identity_is_bound_to_the_selected_node(platform):
    first = platform.register(gpus=[gpu('GPU-first')])
    second = platform.register(name='second', gpus=[gpu('GPU-second')])
    response = preview_request(platform, first, selected=['GPU-second'])
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'REQUESTED_GPU_NOT_FOUND'
    assert platform.commands(first) == [] and platform.commands(second) == []


def test_preview_run_and_start_preserve_exact_selection_and_device_order(platform):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2'), gpu('GPU-3')])
    preview = ready_preview(platform, node, selected=['GPU-3', 'GPU-1'], count=2)
    assert preview['request']['requested_gpu_uuids'] == ['GPU-3', 'GPU-1']
    response = submit(platform, preview)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run['requested_gpu_uuids'] == ['GPU-3', 'GPU-1']
    assert run['code_expectation']['requested_gpu_uuids'] == ['GPU-3', 'GPU-1']
    assert platform.schedule() == 1
    assert starts(platform, node)[0]['payload']['gpu_uuids'] == ['GPU-3', 'GPU-1']
    detail = platform.get_run(run)
    assert detail['attempts'][0]['gpu_uuids'] == ['GPU-3', 'GPU-1']
    assert next(item for item in platform.client.get('/api/v1/runs', headers=ADMIN).json()
                if item['id'] == run['id'])['requested_gpu_uuids'] == ['GPU-3', 'GPU-1']


@pytest.mark.parametrize('selection', [OMIT, None])
def test_omitted_or_null_selection_preserves_legacy_automatic_assignment(platform, selection):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    preview = ready_preview(platform, node, selected=selection)
    assert 'requested_gpu_uuids' not in preview['request']
    response = submit(platform, preview)
    assert response.status_code == 201
    run = response.json()
    assert run['requested_gpu_uuids'] is None
    assert 'requested_gpu_uuids' not in run['code_expectation']
    assert platform.schedule() == 1
    assert starts(platform, node)[0]['payload']['gpu_uuids'] == ['GPU-1']


@pytest.mark.parametrize('busy', [
    {'processes': [{'pid': 123, 'memory_mb': 100}]}, {'processes': None},
    {'utilization_pct': 90}, {'memory_used_mb': None},
])
def test_busy_or_unknown_requested_gpu_waits_without_using_other_free_cards(platform, busy):
    node = platform.register(gpus=[gpu('GPU-1', **busy), gpu('GPU-2')])
    run = selected_run(platform, node, selected=['GPU-1'])
    assert platform.schedule() == 0
    queued = platform.get_run(run)
    assert queued['state'] == 'QUEUED' and queued['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'
    assert queued['attempts'] == [] and starts(platform, node) == []
    platform.heartbeat(node, gpus=[gpu('GPU-1'), gpu('GPU-2')])
    assert platform.schedule() == 1
    assert starts(platform, node)[0]['payload']['gpu_uuids'] == ['GPU-1']


def test_a_missing_requested_gpu_waits_while_other_training_can_use_remaining_devices(platform):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    pinned = selected_run(platform, node, selected=['GPU-2'])
    automatic = selected_run(platform, node)
    platform.heartbeat(node, gpus=[gpu('GPU-1')])
    assert platform.schedule() == 1
    assert platform.get_run(pinned)['queue_reason'] == 'REQUESTED_GPU_MISSING'
    assert platform.get_run(pinned)['attempts'] == []
    assert platform.get_run(automatic)['attempts'][0]['gpu_uuids'] == ['GPU-1']


def test_manual_and_automatic_jobs_reserve_disjoint_actual_devices(platform):
    node = platform.register(gpus=[gpu(f'GPU-{i}') for i in range(1, 5)])
    pinned = selected_run(platform, node, selected=['GPU-3', 'GPU-1'], count=2)
    automatic = selected_run(platform, node)
    waiting = selected_run(platform, node, selected=['GPU-3'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: platform.schedule(), range(2))) == 2
    assert platform.get_run(pinned)['attempts'][0]['gpu_uuids'] == ['GPU-3', 'GPU-1']
    assert platform.get_run(automatic)['attempts'][0]['gpu_uuids'] == ['GPU-2']
    assert platform.get_run(waiting)['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'
    with platform.app.state.db.read() as session:
        assert {row.gpu_uuid for row in session.scalars(select(Allocation))} == {'GPU-1', 'GPU-2', 'GPU-3'}


@pytest.mark.parametrize('mode', ['exclusive', 'shared'])
def test_policy_deployment_reservations_block_exact_training_selection(platform, mode):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    with platform.app.state.db.write() as session:
        session.add(PolicyDeployment(id='policy-reservation', name='inference', node_id=node['node_id'],
            profile_id='default', checkpoint_path='/machine/checkpoint', config_name='fixture', port=8000,
            gpu_uuid='GPU-1', gpu_mode=mode, gpu_memory_mb=1024 if mode == 'shared' else None,
            state='RUNNING', created_at=now(), updated_at=now()))
    run = selected_run(platform, node, selected=['GPU-1'])
    assert platform.schedule() == 0
    assert platform.get_run(run)['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'


def test_submission_cannot_replace_gpu_choice_without_a_new_preflight(platform):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    preview = ready_preview(platform, node, selected=['GPU-2'])
    rejected = submit(platform, preview, requested_gpu_uuids=['GPU-1'])
    assert rejected.status_code == 422
    response = submit(platform, preview)
    assert response.status_code == 201
    assert response.json()['requested_gpu_uuids'] == ['GPU-2']


def test_submission_rechecks_device_membership_after_preflight(platform):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    preview = ready_preview(platform, node, selected=['GPU-2'])
    platform.heartbeat(node, gpus=[gpu('GPU-1')])
    response = submit(platform, preview)
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'REQUESTED_GPU_NOT_FOUND'
    assert starts(platform, node) == []


@pytest.mark.parametrize('changed_selection', [['GPU-1'], None])
def test_scheduler_rejects_device_choice_drift_from_the_frozen_request(platform, changed_selection):
    node = platform.register(gpus=[gpu('GPU-1'), gpu('GPU-2')])
    run = selected_run(platform, node, selected=['GPU-2'])
    with platform.app.state.db.write() as session:
        session.get(Run, run['id']).requested_gpu_uuids = changed_selection
    assert platform.schedule() == 0
    assert platform.get_run(run)['queue_reason'] == 'REQUESTED_GPU_INVALID'
    assert starts(platform, node) == []


def test_additive_gpu_selection_migration_preserves_legacy_runs_and_is_concurrent_safe(platform):
    node = platform.register()
    run = platform.run(node)
    database = platform.app.state.db
    with database.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE runs DROP COLUMN requested_gpu_uuids')
    def upgrade(_):
        separate = Database(platform.settings.database_url)
        try:
            separate.initialize()
        finally:
            separate.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(upgrade, range(2)))
    database.initialize()
    with database.read() as session:
        record = session.get(Run, run['id'])
        assert record.requested_gpu_uuids is None
        assert record.name == run['name'] and record.state == run['state']
        assert record.parameters == run['parameters'] and record.code_expectation == run['code_expectation']
    column = next(column for column in inspect(database.engine).get_columns('runs')
                  if column['name'] == 'requested_gpu_uuids')
    assert column['nullable']
    assert platform.schedule() == 1
    assert starts(platform, node)[0]['payload']['gpu_uuids'] == ['GPU-1']
