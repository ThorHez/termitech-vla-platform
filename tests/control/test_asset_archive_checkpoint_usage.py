"""Catalogue archival preserves active checkpoint consumers across accounts."""

from copy import deepcopy

import pytest
from sqlalchemy import func, select

from vla_control.checkpoint_transfer_models import CheckpointTransfer, TRANSFER_ACTIVE
from vla_control.common import now, uid
from vla_control.dataset_validations import INTERNAL_KEY
from vla_control.models import Asset, Command
from vla_control.policy_deployment_models import PolicyDeployment, RESOURCE_STATES

from .test_multiuser_task_permissions import team  # noqa: F401
from .test_policy_deployments import add_run


CHECKPOINT = '/machine/checkpoints/10000'


def register(team, kind, path):
    _, node, users, clients = team
    response = clients['alice'].post('/api/v1/assets', json={
        'node_id': node['node_id'], 'name': 'my registered input', 'kind': kind, 'path': path,
        'metadata': {'annotation': 'retain this declaration'}})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result['registered_by_user_id'] == users['alice']['id']
    return result


def reference(team, consumer, state, *, owner='bob', other_node=False, path=CHECKPOINT):
    platform, node, users, _ = team
    other = platform.register(name='other-machine')
    source, target = (other, node) if other_node else (node, other)
    stamp, identifier = now(), uid()
    with platform.app.state.db.write() as session:
        if consumer == 'policy':
            row = PolicyDeployment(id=identifier, name='synthetic policy', node_id=source['node_id'],
                profile_id='default', owner_user_id=users[owner]['id'], execution_user_id=users[owner]['id'],
                checkpoint_path=path, config_name='fixture', port=8000, gpu_uuid=None, state=state,
                pid=7123 if state in RESOURCE_STATES else None, logs='', checks=[], created_at=stamp, updated_at=stamp)
        else:
            run = add_run(session, source)
            row = CheckpointTransfer(id=identifier, name='synthetic export/copy', source_node_id=source['node_id'],
                source_run_id=run.id, source_checkpoint_path=path, target_node_id=target['node_id'],
                target_profile_id='default', target_directory='/machine/exported', state=state,
                owner_user_id=users[owner]['id'], execution_user_id=users[owner]['id'],
                binding_snapshot={}, created_at=stamp, updated_at=stamp)
        session.add(row)
    return type(row), identifier


def snapshot(team, asset, consumer):
    with team[0].app.state.db.read() as session:
        row = session.get(Asset, asset['id'])
        active = session.get(*consumer)
        metadata = {key: deepcopy(value) for key, value in row.metadata_json.items() if key != INTERNAL_KEY}
        return (row.state, metadata, row.registered_by_user_id,
                active.state, active.owner_user_id,
                session.scalar(select(func.count()).select_from(Command)))


@pytest.mark.parametrize('kind,suffix', [('MODEL', ''), ('NORM', '/assets/shared/norm_stats.json')])
@pytest.mark.parametrize('consumer,state', [('policy', state) for state in sorted(RESOURCE_STATES)] +
                                         [('transfer', state) for state in sorted(TRANSFER_ACTIVE)])
def test_own_registered_asset_cannot_be_hidden_while_another_accounts_consumer_is_active(team, kind, suffix, consumer, state):
    asset = register(team, kind, CHECKPOINT + suffix)
    active = reference(team, consumer, state)
    before = snapshot(team, asset, active)
    for user in ('alice', 'team-admin'):
        response = team[3][user].post('/api/v1/assets/' + asset['id'] + '/archive')
        assert response.status_code == 409 and response.json()['error']['code'] == 'ASSET_IN_USE', response.text
        assert snapshot(team, asset, active) == before
    with team[0].app.state.db.write() as session:
        row = session.get(*active)
        row.state = 'STOPPED' if consumer == 'policy' else 'SUCCEEDED'
        if consumer == 'policy':
            row.pid = None
    response = team[3]['alice'].post('/api/v1/assets/' + asset['id'] + '/archive')
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'ARCHIVED'


@pytest.mark.parametrize('path', [CHECKPOINT + '/params', CHECKPOINT + '/model.safetensors',
    '/machine//checkpoints/./10000/'])
def test_explicit_model_weight_locations_and_safe_posix_spellings_match_the_active_checkpoint(team, path):
    asset = register(team, 'MODEL', path)
    active = reference(team, 'policy', 'RUNNING', owner='alice')
    before = snapshot(team, asset, active)
    response = team[3]['alice'].post('/api/v1/assets/' + asset['id'] + '/archive')
    assert response.status_code == 409 and response.json()['error']['code'] == 'ASSET_IN_USE'
    assert snapshot(team, asset, active) == before


@pytest.mark.parametrize('kind,path,consumer,state,other_node', [
    ('MODEL', '/machine', 'policy', 'RUNNING', False),
    ('MODEL', CHECKPOINT + '/params/unrelated', 'policy', 'RUNNING', False),
    ('MODEL', CHECKPOINT + '-other', 'transfer', 'EXPORTING', False),
    ('NORM', '/machine', 'policy', 'RUNNING', False),
    ('NORM', CHECKPOINT, 'policy', 'RUNNING', False),
    ('NORM', CHECKPOINT + '/assets-other/norm_stats.json', 'policy', 'RUNNING', False),
    ('MODEL', CHECKPOINT, 'policy', 'RUNNING', True),
    ('NORM', CHECKPOINT + '/assets/shared/norm_stats.json', 'transfer', 'EXPORTING', True),
    ('MODEL', CHECKPOINT, 'policy', 'FAILED', False),
    ('NORM', CHECKPOINT + '/assets/shared/norm_stats.json', 'transfer', 'INTERRUPTED', False),
    ('DATASET', CHECKPOINT, 'policy', 'RUNNING', False),
])
def test_unrelated_ancestors_children_other_machines_and_terminal_consumers_do_not_block_archival(team, kind, path, consumer, state, other_node):
    asset = register(team, kind, path)
    active = reference(team, consumer, state, other_node=other_node)
    before = snapshot(team, asset, active)
    response = team[3]['alice'].post('/api/v1/assets/' + asset['id'] + '/archive')
    assert response.status_code == 200, response.text
    after = snapshot(team, asset, active)
    assert after[0] == 'ARCHIVED' and after[1:] == before[1:]


def test_norm_bound_by_another_accounts_registered_model_keeps_existing_model_norm_guard(team):
    norm = register(team, 'NORM', '/machine/shared/norm_stats.json')
    _, node, users, clients = team
    response = clients['bob'].post('/api/v1/assets', json={
        'node_id': node['node_id'], 'name': 'foreign model', 'kind': 'MODEL', 'path': CHECKPOINT})
    assert response.status_code == 201, response.text
    model = response.json()
    with team[0].app.state.db.write() as session:
        session.get(Asset, model['id']).norm_asset_id = norm['id']
    response = clients['alice'].post('/api/v1/assets/' + norm['id'] + '/archive')
    assert response.status_code == 409 and response.json()['error']['code'] == 'MODEL_NORM_IN_USE'
    assert clients['bob'].post('/api/v1/assets/' + model['id'] + '/archive').status_code == 200
    assert clients['alice'].post('/api/v1/assets/' + norm['id'] + '/archive').status_code == 200
