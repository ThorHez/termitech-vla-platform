"""Completed models retain the statistics actually saved in their checkpoint."""

from copy import deepcopy

import pytest

from vla_control.models import Asset

from .conftest import ADMIN
from .test_incremental_outputs import checkpoint, running


def shared_checkpoint(attempt, **changes):
    output = checkpoint(attempt)
    output['metadata'] = {'format': 'openpi-orbax', 'step': 100,
        'norm_stats': {'path': output['path'] + '/assets/shared/norm_stats.json',
                       'sha256': 'a' * 64, 'normalization_method': 'quantile', 'shared': True}}
    output['metadata']['norm_stats'].update(changes)
    return output


def complete(platform, node, attempt, output, *, seq=2, event='SUCCEEDED'):
    response = platform.emit(node, attempt['id'], seq, event,
                             {'exit_code': 0 if event == 'SUCCEEDED' else 1, 'outputs': [output]})
    assert response.status_code == 200, response.text


def test_completed_model_binds_its_own_statistics_without_rebinding_dataset_defaults(platform):
    node, run, attempt = running(platform)
    old = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': node['node_id'], 'kind': 'NORM', 'name': 'Old input norm', 'path': '/machine/old-norm'}).json()
    dataset = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': node['node_id'], 'kind': 'DATASET', 'name': 'Dataset', 'path': '/machine/data',
        'norm_asset_id': old['id']}).json()
    output = shared_checkpoint(attempt)
    complete(platform, node, attempt, output)
    result = platform.get_run(run)
    assert result['state'] == 'SUCCEEDED'
    model = next(item for item in result['outputs'] if item['kind'] == 'MODEL')
    norm = next(item for item in result['outputs'] if item['kind'] == 'NORM')
    assert model['norm_asset_id'] == norm['id'] != old['id']
    assert norm['path'] == output['metadata']['norm_stats']['path']
    assert norm['node_id'] == model['node_id'] == node['node_id']
    assert norm['origin_attempt_id'] == model['origin_attempt_id'] == attempt['id']
    assert norm['metadata']['sha256'] == 'a' * 64
    assert norm['metadata']['normalization_method'] == 'quantile'
    assert norm['metadata']['model_checkpoint_path'] == model['path']
    assets = platform.client.get('/api/v1/assets', headers=ADMIN).json()
    assert next(item for item in assets if item['id'] == dataset['id'])['norm_asset_id'] == old['id']
    assert next(item for item in assets if item['id'] == model['id'])['norm_asset_id'] == norm['id']
    complete(platform, node, attempt, output, seq=3)
    assert {item['id'] for item in platform.get_run(run)['outputs']} == {model['id'], norm['id']}


def test_discovered_checkpoint_is_not_bound_until_successful_final_manifest(platform):
    node, run, attempt = running(platform)
    output = shared_checkpoint(attempt)
    complete(platform, node, attempt, output, event='ARTIFACT')
    before = platform.get_run(run)
    assert before['state'] == 'RUNNING' and len(before['outputs']) == 1
    model_id = before['outputs'][0]['id']
    assert before['outputs'][0]['norm_asset_id'] is None
    complete(platform, node, attempt, output, seq=3)
    after = platform.get_run(run)
    model = next(item for item in after['outputs'] if item['id'] == model_id)
    assert model['norm_asset_id'] == next(item['id'] for item in after['outputs'] if item['kind'] == 'NORM')


@pytest.mark.parametrize('event', ['FAILED', 'CANCELLED'])
def test_unsuccessful_attempt_does_not_register_or_bind_model_norms(platform, event):
    node, run, attempt = running(platform)
    complete(platform, node, attempt, shared_checkpoint(attempt), event=event)
    assert platform.get_run(run)['outputs'] == []


@pytest.mark.parametrize('change', [
    {'path': '/machine/input/norm_stats.json'},
    {'path': '/machine/output/other-attempt/checkpoints/100/assets/shared/norm_stats.json'},
    {'path': '/machine/output/../outside/norm_stats.json'},
    {'sha256': 'invalid'}, {'sha256': None}, {'shared': False}, {'shared': 1},
    {'normalization_method': 'unknown'}, {'normalization_method': []},
])
def test_invalid_or_foreign_statistics_reject_terminal_manifest_without_partial_registration(platform, change):
    node, run, attempt = running(platform)
    output = shared_checkpoint(attempt, **change)
    complete(platform, node, attempt, output)
    result = platform.get_run(run)
    assert result['state'] == 'FAILED' and result['error']['code'] == 'MODEL_NORM_INVALID'
    assert result['outputs'] == []


def test_model_cannot_bind_a_neighboring_checkpoint_statistics(platform):
    node, run, attempt = running(platform)
    output = shared_checkpoint(attempt)
    output['metadata']['norm_stats']['path'] = output['path'].replace('/100', '/200') + '/assets/shared/norm_stats.json'
    complete(platform, node, attempt, output)
    result = platform.get_run(run)
    assert result['state'] == 'FAILED' and result['error']['code'] == 'MODEL_NORM_INVALID'


def test_legacy_model_without_statistics_descriptor_retains_existing_behavior(platform):
    node, run, attempt = running(platform)
    complete(platform, node, attempt, checkpoint(attempt))
    result = platform.get_run(run)
    assert result['state'] == 'SUCCEEDED' and len(result['outputs']) == 1
    assert result['outputs'][0]['norm_asset_id'] is None


def test_conflicting_descriptors_for_one_checkpoint_fail_before_registering_any_asset(platform):
    node, run, attempt = running(platform)
    first = shared_checkpoint(attempt)
    second = deepcopy(first)
    second['metadata']['norm_stats']['sha256'] = 'b' * 64
    response = platform.emit(node, attempt['id'], 2, 'SUCCEEDED',
                             {'exit_code': 0, 'outputs': [first, second]})
    assert response.status_code == 200, response.text
    result = platform.get_run(run)
    assert result['state'] == 'FAILED' and result['error']['code'] == 'MODEL_NORM_INVALID'
    assert result['outputs'] == []


def test_completion_does_not_revive_an_archived_discovered_norm_or_bind_it_to_a_model(platform):
    node, run, attempt = running(platform)
    output = shared_checkpoint(attempt)
    candidate = {'kind': 'NORM', 'name': 'Discovered norm', 'version': output['version'],
                 'path': output['metadata']['norm_stats']['path'], 'metadata': {}}
    response = platform.emit(node, attempt['id'], 2, 'ARTIFACT', {'outputs': [output, candidate]})
    assert response.status_code == 200, response.text
    prior = platform.get_run(run)
    norm = next(item for item in prior['outputs'] if item['kind'] == 'NORM')
    archived = platform.client.post(f"/api/v1/assets/{norm['id']}/archive", headers=ADMIN)
    assert archived.status_code == 200, archived.text
    complete(platform, node, attempt, output, seq=3)
    result = platform.get_run(run)
    assert result['state'] == 'SUCCEEDED'
    assert next(item for item in result['outputs'] if item['kind'] == 'NORM')['state'] == 'ARCHIVED'
    model = next(item for item in result['outputs'] if item['kind'] == 'MODEL')
    assert model['norm_asset_id'] is None
    assert model['metadata']['norm_binding'] == {'status': 'SKIPPED', 'reason': 'NORM_ASSET_ARCHIVED'}


def test_checkpoint_removal_marks_only_its_bound_embedded_norm_unavailable(platform):
    node, run, attempt = running(platform)
    output = shared_checkpoint(attempt)
    embedded = {'kind': 'NORM', 'name': 'Checkpoint norm', 'version': output['version'],
                'path': output['metadata']['norm_stats']['path'], 'metadata': {}}
    response = platform.emit(node, attempt['id'], 2, 'ARTIFACT', {'outputs': [output, embedded]})
    assert response.status_code == 200, response.text
    prior = platform.get_run(run)
    model = next(item for item in prior['outputs'] if item['kind'] == 'MODEL')
    norm = next(item for item in prior['outputs'] if item['kind'] == 'NORM')
    unrelated = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': node['node_id'], 'kind': 'NORM', 'name': 'Dataset norm', 'path': '/machine/norm'}).json()
    # The same retention reducer handles output scans regardless of whether
    # the model association was restored before a scan or just published.
    with platform.app.state.db.write() as session:
        session.get(Asset, model['id']).norm_asset_id = norm['id']
    response = platform.emit(node, attempt['id'], 3, 'ARTIFACT', {'removed_paths': [output['path']]})
    assert response.status_code == 200, response.text
    outputs = platform.get_run(run)['outputs']
    assert all(item['state'] == 'UNAVAILABLE' for item in outputs)
    assets = platform.client.get('/api/v1/assets', headers=ADMIN).json()
    assert next(item for item in assets if item['id'] == unrelated['id'])['state'] == 'REGISTERED'


def test_bound_model_norm_cannot_be_archived_until_model_is_archived(platform):
    node, run, attempt = running(platform)
    complete(platform, node, attempt, shared_checkpoint(attempt))
    outputs = platform.get_run(run)['outputs']
    model = next(item for item in outputs if item['kind'] == 'MODEL')
    norm = next(item for item in outputs if item['kind'] == 'NORM')
    response = platform.client.post(f"/api/v1/assets/{norm['id']}/archive", headers=ADMIN)
    assert response.status_code == 409 and response.json()['error']['code'] == 'MODEL_NORM_IN_USE'
    response = platform.client.post(f"/api/v1/assets/{model['id']}/archive", headers=ADMIN)
    assert response.status_code == 200, response.text
    response = platform.client.post(f"/api/v1/assets/{norm['id']}/archive", headers=ADMIN)
    assert response.status_code == 200, response.text
