"""Complete statistics publish local defaults without overwriting later edits."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from vla_control.models import Asset, Run
from vla_control.norm_outputs import BINDING_EXPECTATIONS, reconcile_norm_outputs

from .conftest import ADMIN, confirmed_run_request
from .test_norm_tasks import acknowledge, batch_outputs, pending_batch_norm, start_batch_norm


def listed(platform, asset_id):
    return next(row for row in platform.client.get('/api/v1/assets', headers=ADMIN).json() if row['id'] == asset_id)


def norm(platform, node, name='Old statistics'):
    response = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'name': name, 'kind': 'NORM', 'node_id': node['node_id'], 'path': '/machine/' + name.replace(' ', '-')})
    assert response.status_code == 201, response.text
    return response.json()


def bind(platform, asset_id, norm_id):
    response = platform.client.put(f'/api/v1/assets/{asset_id}/norm-binding', headers=ADMIN, json={'norm_asset_id': norm_id})
    assert response.status_code == 200, response.text


def start_pending(platform, pending):
    node, request, preview, command, code, checks = pending
    ready = acknowledge(platform, node, preview, command, code, checks)
    assert ready['status'] == 'READY', ready
    response = platform.client.post('/api/v1/runs', headers={**ADMIN, 'Idempotency-Key': preview['id']},
                                   json=confirmed_run_request(ready))
    assert response.status_code == 201, response.text
    run = response.json()
    assert platform.schedule() == 1
    attempt = platform.start(node, run)
    return node, request, run, attempt, '/machine/output/' + attempt['id']


def succeed(platform, node, request, run, attempt, output_dir):
    outputs = batch_outputs(request, output_dir)
    for output in outputs:
        output['name'] = 'auto_pi05_full_h50_d32_sft norm_stats'
    response = platform.emit(node, attempt['id'], 2, 'SUCCEEDED', {'exit_code': 0, 'outputs': outputs})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed['state'] == 'SUCCEEDED', completed
    return completed


def test_each_complete_output_is_named_and_bound_only_to_its_frozen_source(platform):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    expected = {row['dataset_asset_id']: None for row in request['norm_datasets']}
    assert run['code_expectation'][BINDING_EXPECTATIONS] == expected
    # A display-name edit is irrelevant to input/config identity and the final
    # controller label follows the real source, never a neighboring table row.
    first = request['norm_datasets'][0]['dataset_asset_id']
    with platform.app.state.db.write() as session:
        session.get(Asset, first).name = '\u202e共享动作 收集数据'
    completed = succeed(platform, node, request, run, attempt, output_dir)
    for output in completed['outputs']:
        source_id = output['metadata']['source_dataset_asset_id']
        source = listed(platform, source_id)
        assert source['norm_asset_id'] == output['id']
        assert output['name'] == ('共享动作 收集数据' if source_id == first else source['name']) + ' norm_stats'
        assert output['metadata']['dataset_binding'] == {'status': 'BOUND'}


@pytest.mark.parametrize('changed', [False, True, 'clear'])
def test_success_replaces_unchanged_old_default_and_preserves_a_later_manual_binding(platform, changed):
    pending = pending_batch_norm(platform)
    node, request, *_ = pending
    source_id = request['norm_datasets'][0]['dataset_asset_id']
    old = norm(platform, node)
    bind(platform, source_id, old['id'])
    node, request, run, attempt, output_dir = start_pending(platform, pending)
    assert run['code_expectation'][BINDING_EXPECTATIONS][source_id] == old['id']
    manual = norm(platform, node, 'Manual statistics')
    if changed:
        bind(platform, source_id, None if changed == 'clear' else manual['id'])
    completed = succeed(platform, node, request, run, attempt, output_dir)
    output = next(row for row in completed['outputs'] if row['metadata']['source_dataset_asset_id'] == source_id)
    if changed:
        assert listed(platform, source_id)['norm_asset_id'] == (None if changed == 'clear' else manual['id'])
        assert output['metadata']['dataset_binding'] == {'status': 'SKIPPED', 'reason': 'BINDING_CHANGED'}
    else:
        assert listed(platform, source_id)['norm_asset_id'] == output['id']
        assert output['metadata']['dataset_binding'] == {'status': 'BOUND'}
    assert listed(platform, old['id'])['state'] == 'REGISTERED'


def test_simultaneous_manual_binding_and_terminal_event_keep_the_manual_selection(platform):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    source_id = request['norm_datasets'][0]['dataset_asset_id']
    manual = norm(platform, node, 'Manual selection')
    barrier = Barrier(2)

    def select_manual():
        barrier.wait(timeout=5)
        bind(platform, source_id, manual['id'])

    def complete():
        barrier.wait(timeout=5)
        return succeed(platform, node, request, run, attempt, output_dir)

    with ThreadPoolExecutor(max_workers=2) as workers:
        changed = workers.submit(select_manual)
        completed = workers.submit(complete)
        changed.result(timeout=10)
        assert completed.result(timeout=10)['state'] == 'SUCCEEDED'
    assert listed(platform, source_id)['norm_asset_id'] == manual['id']


@pytest.mark.parametrize('event', ['ARTIFACT', 'FAILED', 'partial_success'])
def test_partial_or_failed_statistics_never_publish_dataset_bindings(platform, event):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    outputs = batch_outputs(request, output_dir)[:1]
    response = platform.emit(node, attempt['id'], 2, 'SUCCEEDED' if event == 'partial_success' else event,
        {'exit_code': 1 if event == 'FAILED' else 0, 'outputs': outputs})
    assert response.status_code == 200, response.text
    assert platform.get_run(run)['outputs'] == []
    assert all(listed(platform, row['dataset_asset_id'])['norm_asset_id'] is None for row in request['norm_datasets'])


@pytest.mark.parametrize('change', ['unselected', 'other_node', 'source_kind'])
def test_norm_source_must_be_a_same_node_dataset_selected_by_this_run(platform, change):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    outputs = batch_outputs(request, output_dir)
    if change == 'source_kind':
        with platform.app.state.db.write() as session:
            session.get(Asset, request['norm_datasets'][0]['dataset_asset_id']).kind = 'MODEL'
    else:
        owner = platform.register(name='other-machine') if change == 'other_node' else node
        source = platform.client.post('/api/v1/assets', headers=ADMIN, json={
            'name': 'Unselected source', 'node_id': owner['node_id'], 'kind': 'DATASET', 'path': '/machine/unselected'}).json()
        outputs[0]['metadata']['source_dataset_asset_id'] = source['id']
    response = platform.emit(node, attempt['id'], 2, 'SUCCEEDED', {'exit_code': 0, 'outputs': outputs})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed['state'] == 'FAILED' and completed['outputs'] == []
    assert all(listed(platform, row['dataset_asset_id'])['norm_asset_id'] is None for row in request['norm_datasets'])


def test_unavailable_source_keeps_valid_output_without_changing_its_default(platform):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    source_id = request['norm_datasets'][0]['dataset_asset_id']
    with platform.app.state.db.write() as session:
        session.get(Asset, source_id).state = 'UNAVAILABLE'
    completed = succeed(platform, node, request, run, attempt, output_dir)
    output = next(row for row in completed['outputs'] if row['metadata']['source_dataset_asset_id'] == source_id)
    assert listed(platform, source_id)['norm_asset_id'] is None
    assert output['metadata']['dataset_binding'] == {'status': 'SKIPPED', 'reason': 'DATASET_ASSET_UNAVAILABLE'}


@pytest.mark.parametrize('occupied', [False, True])
def test_historical_reconciliation_only_fills_empty_bindings_and_is_idempotent(platform, occupied):
    node, request, run, attempt, _, output_dir = start_batch_norm(platform)
    completed = succeed(platform, node, request, run, attempt, output_dir)
    manual = norm(platform, node, 'Historical manual choice')
    with platform.app.state.db.write() as session:
        stored = session.get(Run, run['id'])
        stored.code_expectation = {key: value for key, value in stored.code_expectation.items() if key != BINDING_EXPECTATIONS}
        for row in request['norm_datasets']:
            session.get(Asset, row['dataset_asset_id']).norm_asset_id = manual['id'] if occupied else None
        for output in completed['outputs']:
            session.get(Asset, output['id']).name = 'auto_pi05_full_h50_d32_sft norm_stats'
        first = reconcile_norm_outputs(session, stored)
        second = reconcile_norm_outputs(session, stored)
        assert first == second
    after = platform.get_run(run)
    assert {row['id'] for row in after['outputs']} == {row['id'] for row in completed['outputs']}
    for output in after['outputs']:
        source = listed(platform, output['metadata']['source_dataset_asset_id'])
        assert output['name'] == source['name'] + ' norm_stats'
        assert source['norm_asset_id'] == (manual['id'] if occupied else output['id'])
        assert output['metadata']['dataset_binding']['status'] == ('SKIPPED' if occupied else 'BOUND')
