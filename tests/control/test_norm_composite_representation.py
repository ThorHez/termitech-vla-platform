"""Opaque 14D vectors can be measured without inventing Joint/EEF semantics."""

from copy import deepcopy

from vla_protocol import canonical_digest

from .conftest import ADMIN, confirmed_run_request, inspected_profile_payload
from .test_composed_asset_processing import description
from .test_multi_dataset_control import configured
from .test_norm_tasks import acknowledge, batch_outputs, pending_batch_norm
from .test_task_recipe import request_preview


def pending_composite_norm(platform):
    node, request, _, _, code, checks = pending_batch_norm(platform, auto_config={}, composed=True)
    request = deepcopy(request)
    profile = description()
    profile.pop('processing')
    profile.update(action_representation='unknown', action_mode='unknown', rotation_representation='unknown', state_dim=14, action_dim=14,
        feature_keys=['observation.state', 'actions', profile['cameras'][0]['key']], field_annotations=[
            {'key': 'observation.state', 'role': 'state', 'representation': 'composite',
             'value_mode': 'absolute', 'component': 'other', 'arm': 'shared', 'shape': [14], 'dtype': 'float32', 'names': []},
            {'key': 'actions', 'role': 'action', 'representation': 'composite',
             'value_mode': 'absolute', 'component': 'other', 'arm': 'shared', 'shape': [14],
             'dtype': 'float32', 'names': [], 'horizon': 1}])
    for row in request['norm_datasets']:
        response = platform.client.patch(f"/api/v1/assets/{row['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                         json={'dataset_profile': profile})
        assert response.status_code == 200, response.text
        spec = row['processor']['parameters']['composition']
        spec.update(state_representation='composite', action_representation='composite',
            state_fields=[{'key': 'observation.state', 'width': 14, 'arm': 'shared', 'kind': 'other', 'normalize': True}],
            state_history=1, action_key='actions',
            action_layout=[{'kind': 'shared', 'arm': 'shared', 'width': 14, 'normalize': True}],
            action_mode='', rotation_representation='', position_unit='', joint_unit='', coordinate_frame='',
            gripper_range=None, cameras=[], prompt_key='', aux_keys=[])
    response = platform.client.post('/api/v1/run-previews', headers=ADMIN, json=request)
    assert response.status_code == 202, response.text
    preview = response.json()
    command = next(item for item in platform.commands(node) if item['type'] == 'PREFLIGHT'
                   and item['payload']['preview_id'] == preview['id'])
    rows = command['payload']['norm_datasets']
    code['norm_datasets'] = deepcopy(rows)
    plan = code['automatic_preflight']['plan']
    plan['norm_datasets'] = deepcopy(rows)
    for row, dataset in zip(rows, plan['datasets']):
        spec = row['processor']['parameters']['composition']
        dataset['processor'] = deepcopy(row['processor'])
        dataset['sample_shapes'] = {'state': [14], 'actions': [50, 14]}
        dataset['actions_shape'] = [14]
        dataset['numeric_io'].update(state_columns=['observation.state'], action_column='actions',
            temporal_columns=['episode_index', 'frame_index', 'timestamp'], action_sequence='episode_clamp',
            columns=['observation.state', 'actions', 'episode_index', 'frame_index', 'timestamp'])
        dataset['composition'].update(spec=deepcopy(spec), digest=canonical_digest(spec),
            raw_state_dim=14, raw_action_dim=14, action_horizon=50, normalization=spec['normalization'],
            action_sequence_keys=['actions'], action_loss_mask=[1.] * 14 + [0.] * 18)
    for dataset, projection in zip(plan['datasets'], plan['training_cache']['datasets']):
        projection['columns'] = list(dataset['numeric_io']['columns'])
    code['automatic_preflight']['plan_digest'] = canonical_digest(plan)
    return node, request, preview, command, code, checks, profile


def test_norm_preserves_composite_source_semantics_through_preview_confirmation_and_start(platform):
    node, request, preview, command, code, checks, profile = pending_composite_norm(platform)
    assert profile['action_representation'] == 'unknown' and 'processing' not in profile
    assert all(row['representation'] == 'composite' for row in profile['field_annotations'])
    for value in command['payload']['inputs'].values():
        assert value['dataset_profile'] == profile
    checked = acknowledge(platform, node, preview, command, code, checks)
    assert checked['status'] == 'READY', checked
    response = platform.client.post('/api/v1/runs', headers={**ADMIN, 'Idempotency-Key': preview['id']},
                                   json=confirmed_run_request(checked))
    assert response.status_code == 201, response.text
    run = response.json()
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item['type'] == 'START')
    for row in start['payload']['norm_datasets']:
        spec = row['processor']['parameters']['composition']
        assert spec['state_representation'] == spec['action_representation'] == 'composite'
        assert spec['action_layout'] == [{'kind': 'shared', 'arm': 'shared', 'width': 14, 'normalize': True}]
        assert spec['cameras'] == []
    assert start['payload']['confirmed_config_sha256'] == code['automatic_preflight']['plan']['generated_config']['sha256']
    attempt = platform.start(node, run)
    response = platform.emit(node, attempt['id'], 2, 'SUCCEEDED', {'exit_code': 0,
        'outputs': batch_outputs(request, '/machine/output/' + attempt['id'])})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed['state'] == 'SUCCEEDED', completed
    assert len(completed['outputs']) == 2


def test_norm_composite_action_cannot_hide_regular_action_chunks_inside_shared_vector(platform):
    _, request, *_ = pending_composite_norm(platform)
    request['norm_datasets'][0]['processor']['parameters']['composition']['action_layout'] = [
        {'kind': 'shared', 'arm': 'shared', 'width': 8, 'normalize': True},
        {'kind': 'joint', 'arm': 'left', 'width': 6, 'normalize': True}]
    response = platform.client.post('/api/v1/run-previews', headers=ADMIN, json=request)
    assert response.status_code == 422, response.text


def test_sft_inherits_composite_fields_as_shared_raw_actions_and_keeps_training_vision(platform):
    _, request, *_, profile = pending_composite_norm(platform)
    node, inputs, rows = configured(platform)
    processor = deepcopy(request['norm_datasets'][0]['processor'])
    processor['parameters']['composition']['cameras'] = description()['processing']['cameras']
    rows = rows[:1]
    rows[0].pop('recipe_index')
    rows[0]['processor'] = processor
    response = platform.client.patch(f"/api/v1/assets/{rows[0]['dataset_asset_id']}/dataset-profile", headers=ADMIN,
                                     json=inspected_profile_payload(platform, rows[0]['dataset_asset_id'], profile))
    assert response.status_code == 200, response.text
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 202, response.text
    command = next(item for item in platform.commands(node) if item['type'] == 'PREFLIGHT')
    composition = command['payload']['dataset_bindings'][0]['processor']['parameters']['composition']
    assert composition['state_representation'] == composition['action_representation'] == 'composite'
    assert composition['action_layout'] == [{'kind': 'shared', 'arm': 'shared', 'width': 14, 'normalize': True}]
    assert composition['cameras'] == description()['processing']['cameras']
    frozen = command['payload']['inputs']['dataset_0']['dataset_profile']
    assert frozen['action_representation'] == 'unknown'
    assert all(row['representation'] == 'composite' for row in frozen['field_annotations'])
