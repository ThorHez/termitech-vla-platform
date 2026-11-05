"""Real human requests produce one attributed, bounded operation receipt."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.auth_models import AuditEvent
from .test_asset_transfers import fixture
from .test_asset_transfer_batches import BATCH, body as copy_batch_body, group
from .test_checkpoint_transfers import transfer
from .test_dataset_batches import PROFILE, batch, scan, submit
from .test_multiuser_task_permissions import PASSWORD, actor, team
from .test_resource_ownership import accounts, login
from .test_remote_deployment import PASSWORD as SSH_PASSWORD, remote


def ids(database):
    with database.read() as session:
        return set(session.scalars(select(AuditEvent.id)))


def added(database, previous):
    with database.read() as session:
        return list(session.scalars(select(AuditEvent).where(AuditEvent.id.not_in(previous))
                                   .order_by(AuditEvent.created_at, AuditEvent.id)))


def one(database, previous, action, actor_id, status):
    rows = added(database, previous)
    assert len(rows) == 1, [(row.action, row.status) for row in rows]
    row = rows[0]
    assert (row.action, row.actor_id, row.status) == (action, actor_id, status)
    assert row.details['operation_label'] and row.details['category'] and row.details['target_type']
    assert isinstance(row.details['request_id'], str) and row.details['duration_ms'] >= 0
    return row


@pytest.mark.parametrize('kind', ['DATASET', 'MODEL', 'NORM'])
def test_registration_success_foreign_denial_and_archive_have_correct_resource_and_actor_without_raw_metadata(team, kind):
    platform, node, users, clients = team
    database = platform.app.state.db
    secret = 'private-body-query-and-header-marker'
    before = ids(database)
    response = clients['alice'].post('/api/v1/assets?token=' + secret, headers={'X-Private-Audit-Fixture': secret}, json={
        'name': 'Alice registered ' + kind, 'kind': kind, 'node_id': node['node_id'], 'path': '/machine/' + kind,
        'metadata': {'password': secret, 'ssh_secret': secret, 'raw_source': secret}})
    assert response.status_code == 201, response.text
    asset = response.json()
    row = one(database, before, 'assets.register', users['alice']['id'], 201)
    assert row.details['target_id'] == asset['id'] and row.details['target_name'] == asset['name']
    assert row.details['node_id'] == node['node_id'] and row.details['asset_kind'] == kind
    assert secret not in json.dumps({'path': row.path, 'details': row.details})
    before = ids(database)
    denied = clients['bob'].post('/api/v1/assets/' + asset['id'] + '/archive')
    assert denied.status_code == 403
    row = one(database, before, 'assets.archive', users['bob']['id'], 403)
    assert row.details['target_id'] == asset['id'] and row.details['error_code'] == 'RESOURCE_ACCESS_DENIED'
    before = ids(database)
    assert clients['alice'].post('/api/v1/assets/' + asset['id'] + '/archive').status_code == 200
    row = one(database, before, 'assets.archive', users['alice']['id'], 200)
    assert row.details['target_id'] == asset['id'] and row.details['target_name'] == asset['name']


def test_training_preflight_submit_cancel_delete_are_human_actions_but_agent_protocol_and_get_polling_are_not(team):
    platform, node, users, clients = team
    database = platform.app.state.db
    before = ids(database)
    run = actor(team).run(node)
    rows = added(database, before)
    assert [(row.action, row.status) for row in rows] == [('runs.preflight', 202), ('runs.create', 201)]
    assert all(row.actor_id == users['alice']['id'] for row in rows)
    assert rows[1].details['target_id'] == run['id'] and rows[1].details['target_name'] == run['name']
    before = ids(database)
    for _ in range(3):
        assert clients['alice'].get('/api/v1/runs/' + run['id']).status_code == 200
        assert clients['alice'].get('/api/v1/assets').status_code == 200
        assert clients['alice'].get('/api/v1/auth/me').status_code == 200
        assert platform.commands(node) == []
    heartbeat = platform.client.post('/agent/v1/heartbeat', headers=node['headers'], json={
        'node_id': node['node_id'], 'gpus': [], 'metrics': {}, 'active_attempts': []})
    assert heartbeat.status_code == 200, heartbeat.text
    assert added(database, before) == []
    before = ids(database)
    assert clients['alice'].post('/api/v1/runs/' + run['id'] + '/cancel').status_code == 202
    one(database, before, 'runs.cancel', users['alice']['id'], 202)
    before = ids(database)
    assert clients['alice'].delete('/api/v1/runs/' + run['id']).status_code == 204
    row = one(database, before, 'runs.delete', users['alice']['id'], 204)
    assert row.details['target_id'] == run['id']


def test_private_drafts_and_templates_log_operations_but_not_their_configuration(team):
    platform, _, users, clients = team
    database, client = platform.app.state.db, clients['alice']
    secret = 'private-template-config-never-enters-audit'
    content = {'request': {'template_id': 'smoke', 'name': secret, 'parameters': {'secret': secret}}}
    before = ids(database)
    saved = client.put('/api/v1/training-draft?template_id=smoke', json={'expected_revision': 0, 'content': content})
    assert saved.status_code == 200, saved.text
    row = one(database, before, 'drafts.save', users['alice']['id'], 200)
    assert row.details.get('template_id') == 'smoke' and secret not in json.dumps(row.details)
    before = ids(database)
    cleared = client.request('DELETE', '/api/v1/training-draft?template_id=smoke', json={'expected_revision': 1})
    assert cleared.status_code == 200
    one(database, before, 'drafts.clear', users['alice']['id'], 200)
    before = ids(database)
    created = client.post('/api/v1/training-templates', json={'name': 'my reusable template', 'content': content})
    assert created.status_code == 201, created.text
    template = created.json()
    row = one(database, before, 'training_templates.create', users['alice']['id'], 201)
    assert row.details['target_name'] == template['name'] and row.details['target_id'] == template['id']
    assert secret not in json.dumps(row.details)
    before = ids(database)
    updated = client.put('/api/v1/training-templates/' + template['id'], json={
        'name': 'renamed template', 'description': template['description'], 'content': content, 'expected_revision': 1})
    assert updated.status_code == 200, updated.text
    one(database, before, 'training_templates.update', users['alice']['id'], 200)
    before = ids(database)
    assert client.request('DELETE', '/api/v1/training-templates/' + template['id'], json={'expected_revision': 2}).status_code == 200
    row = one(database, before, 'training_templates.delete', users['alice']['id'], 200)
    assert row.details['target_id'] == template['id'] and row.details['target_name'] == 'renamed template'


def test_batch_scan_registration_keeps_actor_counts_and_excludes_scan_contents(batch):
    app, _, _, _, _, authentication, _ = batch
    database = app.state.db
    before = ids(database)
    ready = scan(batch)
    one(database, before, 'datasets.scan', authentication['user']['id'], 202)
    before = ids(database)
    response = submit(batch, ready)
    assert response.status_code == 200, response.text
    row = one(database, before, 'datasets.register_batch', authentication['user']['id'], 200)
    assert row.details['target_id'] == ready['id']
    assert row.details['created_count'] == 2 and row.details['skipped_count'] == 0
    assert all(key not in row.details for key in ('entries', 'created', 'metadata', 'profile', 'raw_source'))


def test_single_asset_transfer_admission_and_cancellation_have_the_real_operator(fixture):
    platform, _, _, target, asset, _ = fixture
    users = accounts(platform)
    login(platform, 'alice')
    database = platform.app.state.db
    before = ids(database)
    response = platform.client.post('/api/v1/asset-transfers', json={
        'source_asset_id': asset['id'], 'target_node_id': target['node_id'],
        'target_directory': '/machine/copied', 'target_name': 'audit-copy'})
    assert response.status_code == 202, response.text
    transfer = response.json()
    row = one(database, before, 'transfers.create', users['alice'], 202)
    assert row.details['target_id'] == transfer['id'] and row.details['target_name'] == transfer['name']
    before = ids(database)
    assert platform.client.post('/api/v1/asset-transfers/' + transfer['id'] + '/cancel').status_code == 202
    one(database, before, 'transfers.cancel', users['alice'], 202)


def test_batch_asset_transfer_is_one_operation_without_child_response_payload(group):
    platform = group[0][0]
    before = ids(platform.app.state.db)
    response = group[1].post(BATCH, json=copy_batch_body(group))
    assert response.status_code == 202, response.text
    row = one(platform.app.state.db, before, 'transfers.create_batch', group[4], 202)
    assert row.details['target_type'] == 'asset_transfer_batch'
    assert row.details['count'] == len(response.json()['transfers'])
    related = row.details['related_ids'].split(',')
    assert len(related) <= 3 and len(row.details['related_ids']) <= 200
    assert related == [item['id'] for item in response.json()['transfers'][:3]]
    assert 'transfers' not in row.details and 'portable_config' not in row.details


def test_checkpoint_transfer_admission_is_one_operation_not_an_agent_export(transfer):
    platform, _, _, target, run, *_ = transfer
    users = accounts(platform)
    login(platform, 'alice')
    before = ids(platform.app.state.db)
    response = platform.client.post('/api/v1/checkpoint-transfers', json={
        'source_run_id': run.id, 'source_checkpoint_path': '/machine/output/owned/10000',
        'target_node_id': target['node_id'], 'target_profile_id': 'default', 'target_directory': '/machine/models'})
    assert response.status_code == 202, response.text
    row = one(platform.app.state.db, before, 'checkpoints.transfer', users['alice'], 202)
    assert row.details['target_id'] == response.json()['id']
    assert 'binding_snapshot' not in row.details and 'portable_config' not in row.details


def test_policy_deployment_admission_records_real_actor_and_resource_without_running_service(platform):
    from .test_recipe_details import register as register_node
    node = register_node(platform)
    users = accounts(platform)
    login(platform, 'alice')
    before = ids(platform.app.state.db)
    response = platform.client.post('/api/v1/policy-deployments', json={'name': 'Audited inference task',
        'node_id': node['node_id'], 'profile_id': 'default', 'checkpoint_path': '/machine/checkpoint',
        'config_name': 'pi0_base', 'port': 8000})
    assert response.status_code == 202, response.text
    deployment = response.json()
    row = one(platform.app.state.db, before, 'deployments.create', users['alice'], 202)
    assert row.details['target_id'] == deployment['id'] and row.details['target_name'] == deployment['name']
    before = ids(platform.app.state.db)
    assert platform.client.post('/api/v1/policy-deployments/' + deployment['id'] + '/stop').status_code == 202
    one(platform.app.state.db, before, 'deployments.stop', users['alice'], 202)


def test_personal_dataset_format_template_actions_do_not_log_the_format_payload(team):
    platform, _, users, clients = team
    client, database = clients['alice'], platform.app.state.db
    secret = 'private-format-declaration-not-an-audit-field'
    profile = {**deepcopy(PROFILE), 'task_description': secret}
    before = ids(database)
    response = client.post('/api/v1/dataset-format-templates', json={'name': 'My data format', 'profile': profile})
    assert response.status_code == 201, response.text
    template = response.json()
    row = one(database, before, 'dataset_templates.save', users['alice']['id'], 201)
    assert row.details['target_id'] == template['id'] and row.details['target_name'] == template['name']
    assert secret not in json.dumps(row.details)
    before = ids(database)
    response = client.put('/api/v1/dataset-format-templates/' + template['id'], json={
        'name': 'Updated format', 'profile': profile, 'expected_revision': 1})
    assert response.status_code == 200, response.text
    one(database, before, 'dataset_templates.update', users['alice']['id'], 200)
    before = ids(database)
    response = client.request('DELETE', '/api/v1/dataset-format-templates/' + template['id'], json={'expected_revision': 2})
    assert response.status_code == 200, response.text
    one(database, before, 'dataset_templates.delete', users['alice']['id'], 200)


def test_login_user_create_update_reset_and_logout_each_emit_exactly_one_audit_record(team):
    platform, _, users, clients = team
    database, admin = platform.app.state.db, clients['team-admin']
    assert admin.post('/api/v1/auth/reauth', json={'password': PASSWORD}).status_code == 200
    before = ids(database)
    response = admin.post('/api/v1/users', json={'username': 'audit-new-user', 'password': PASSWORD,
        'role': 'viewer', 'display_name': 'Audit fixture user'})
    assert response.status_code == 201, response.text
    target = response.json()
    row = one(database, before, 'users.create', users['team-admin']['id'], 201)
    assert row.details['target_id'] == target['id']
    before = ids(database)
    assert admin.patch('/api/v1/users/' + target['id'], json={'display_name': 'Renamed audit fixture'}).status_code == 200
    one(database, before, 'users.update', users['team-admin']['id'], 200)
    before = ids(database)
    assert admin.post('/api/v1/users/' + target['id'] + '/reset-password', json={'password': PASSWORD}).status_code == 200
    row = one(database, before, 'users.reset_password', users['team-admin']['id'], 200)
    assert PASSWORD not in json.dumps(row.details)
    before = ids(database)
    assert clients['alice'].post('/api/v1/auth/logout').status_code == 204
    one(database, before, 'auth.logout', users['alice']['id'], 204)
    before = ids(database)
    response = clients['alice'].post('/api/v1/auth/login', json={'username': 'alice', 'password': PASSWORD})
    assert response.status_code == 200
    one(database, before, 'auth.login', users['alice']['id'], 200)
    before = ids(database)
    clients['alice'].cookies.clear()
    bad_password = 'private-wrong-password-not-in-audit'
    response = clients['alice'].post('/api/v1/auth/login', json={'username': 'alice', 'password': bad_password})
    assert response.status_code == 401
    row = one(database, before, 'auth.login_failed', None, 401)
    assert bad_password not in json.dumps(row.details)


def test_registered_machine_rename_has_resource_id_name_and_no_raw_request_body(team):
    platform, node, users, clients = team
    before = ids(platform.app.state.db)
    response = clients['team-admin'].patch('/api/v1/nodes/' + node['node_id'], json={'name': 'Audit machine name'})
    assert response.status_code == 200
    row = one(platform.app.state.db, before, 'machines.rename', users['team-admin']['id'], 200)
    assert row.details['target_id'] == node['node_id'] and row.details['target_name'] == 'Audit machine name'


@pytest.mark.parametrize('endpoint,action', [('test', 'machines.ssh_test'), ('connect', 'machines.ssh_connect')])
def test_fake_ssh_admission_has_connection_identity_without_password_or_probe_payload(remote, endpoint, action):
    from .test_resource_ownership import PASSWORD as ACCOUNT_PASSWORD
    platform, _ = remote
    users = accounts(platform)
    login(platform, 'owner-admin')
    assert platform.client.post('/api/v1/auth/reauth', json={'password': ACCOUNT_PASSWORD}).status_code == 200
    body = {'host': '192.0.2.169', 'port': 22, 'username': 'root', 'password': SSH_PASSWORD}
    if endpoint == 'test':
        body['install_dir'] = '/data2/vla-agent'
    before = ids(platform.app.state.db)
    response = platform.client.post('/api/v1/ssh-connections/' + endpoint, json=body)
    assert response.status_code == 200, response.text
    row = one(platform.app.state.db, before, action, users['owner-admin'], 200)
    assert row.details['target_id'] == response.json()['connection_id']
    assert all(key not in row.details for key in ('password', 'encrypted_password', 'probe', 'probe_result', 'checks'))
    saved = json.dumps(row.details)
    assert SSH_PASSWORD not in saved and ACCOUNT_PASSWORD not in saved


def test_prefixed_asgi_mount_keeps_classification_actor_and_resource(platform):
    node = platform.register()
    user = platform.app.state.auth.create_user('prefix-admin', PASSWORD, 'admin')
    client = TestClient(platform.app, root_path='/platform', base_url='https://testserver')
    try:
        response = client.post('/platform/api/v1/auth/login', json={'username': 'prefix-admin', 'password': PASSWORD})
        assert response.status_code == 200, response.text
        client.headers['X-CSRF-Token'] = response.json()['csrf_token']
        before = ids(platform.app.state.db)
        response = client.post('/platform/api/v1/assets', json={
            'name': 'Prefixed resource', 'kind': 'MODEL', 'node_id': node['node_id'], 'path': '/machine/prefixed'})
        assert response.status_code == 201, response.text
        row = one(platform.app.state.db, before, 'assets.register', user['id'], 201)
        assert row.details['target_id'] == response.json()['id'] and row.details['target_name'] == 'Prefixed resource'
        assert row.path == '/api/v1/assets'
    finally:
        client.close()


def test_unhandled_business_failure_is_a_single_500_event_without_exception_or_request_secrets(team, monkeypatch):
    from vla_control import main
    platform, node, users, clients = team
    secret = 'private-failure-message-not-for-audit'

    def failed(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(main, 'queue_validation', failed)
    client = TestClient(platform.app, base_url='https://testserver', raise_server_exceptions=False)
    client.cookies.update(clients['alice'].cookies)
    client.headers['X-CSRF-Token'] = clients['alice'].headers['X-CSRF-Token']
    try:
        before = ids(platform.app.state.db)
        response = client.post('/api/v1/assets', json={'name': 'failed resource', 'kind': 'DATASET',
            'node_id': node['node_id'], 'path': '/machine/failed', 'metadata': {'secret': secret}})
        assert response.status_code == 500
        row = one(platform.app.state.db, before, 'assets.register', users['alice']['id'], 500)
        assert secret not in json.dumps(row.details)
    finally:
        client.close()
