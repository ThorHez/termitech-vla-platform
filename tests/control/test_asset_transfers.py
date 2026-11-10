"""Registered data copies keep ownership, SSH identity, and asset semantics."""

import hashlib
from copy import deepcopy
from pathlib import PurePosixPath
import re
import socket
import threading

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event, select
from vla_protocol import canonical_digest

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.asset_transfers import active_asset_transfers
from vla_control.checkpoint_transfer_transport import TransferError
from vla_control.common import APIError, now, offset_time, uid
from vla_control.models import Asset, Attempt, Command, Node, Run
from vla_control.remote_deployment import DeploymentManager
from vla_control.remote_models import RemoteConnection
from vla_control.ssh_remote import RemoteError
from vla_control.storage_configurations import _in_use

from .conftest import ADMIN, profile_inspection
from .test_checkpoint_transfers import SECRET, bind_ssh
from .test_dataset_profiles import profile
from .test_policy_deployments import add_run
from .test_recipe_details import register


BASE = '/api/v1/asset-transfers'


@pytest.fixture
def fixture(platform):
    source, target = register(platform, name='source'), register(platform, name='target')
    connections = [bind_ssh(platform, source, '192.0.2.10'), bind_ssh(platform, target, '192.0.2.20')]
    asset = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': source['node_id'], 'kind': 'DATASET', 'name': 'folding data',
        'path': '/machine/data/folding', 'version': 'v2',
        'dataset_inspection_id': profile_inspection(platform, source['node_id'], '/machine/data/folding', profile()),
        'metadata': {'dataset_profile': profile(), 'description': 'Retain task semantics',
                     'inspection': {'node_id': 'old-machine'}, 'dataset_asset_id': 'foreign-id'}}).json()
    assert asset.get('id'), asset
    manager = platform.app.state.asset_transfer_manager
    manager._launch = lambda _id: None
    with platform.app.state.db.write() as session:
        for item in (source, target):
            session.get(Node, item['node_id']).metrics = {'capabilities': ['STORAGE_CONFIGURE']}
    return platform, manager, source, target, asset, connections


def create(fixture, **changes):
    platform, _, _, target, asset, _ = fixture
    return platform.client.post(BASE, headers=ADMIN, json={
        'source_asset_id': asset['id'], 'target_node_id': target['node_id'],
        'target_directory': '/machine/copied', 'target_name': 'folding-copy', **changes})


def receipt(fixture, transfer_id):
    response = fixture[0].client.get(BASE + '/' + transfer_id, headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()


class FakeRemote:
    def __init__(self):
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True


def result_for(args, kwargs, *, file_name=None):
    target = str(PurePosixPath(args[3]) / kwargs['target_name'])
    files = [{'path': file_name or 'meta/info.json', 'size': 7,
              'sha256': hashlib.sha256(b'fixture').hexdigest()}]
    manifest = {'schema_version': 1, 'files': files, 'bytes_total': 7, 'files_total': 1}
    return {'target_path': target, 'asset_path': target + '/' + file_name if file_name else target,
            'manifest': manifest, 'manifest_sha256': canonical_digest(manifest),
            'bytes_total': 7, 'files_total': 1}


def run_copy(fixture, transfer_id, monkeypatch, transfer=None):
    platform, manager, *_ = fixture
    remotes = []

    def remote(_connection):
        value = FakeRemote()
        remotes.append(value)
        return value

    monkeypatch.setattr(manager.remote_manager, '_remote', remote)
    manager.transfer_function = transfer or (lambda *args, **kwargs: result_for(args, kwargs))
    manager._run(transfer_id)
    assert all(r.closed for r in remotes)
    return receipt(fixture, transfer_id)


def test_create_is_asset_bound_durable_and_does_not_queue_agent_commands(fixture):
    platform, _, source, target, asset, connections = fixture
    response = create(fixture, idempotency_key='copy-one')
    assert response.status_code == 202, response.text
    item = response.json()
    assert item['state'] == 'QUEUED' and item['source_asset_id'] == asset['id']
    assert item['source_kind'] == 'DATASET' and item['target_asset_id'] is None
    assert item['source_node_id'] == source['node_id'] and item['target_node_id'] == target['node_id']
    assert not item['byte_verified'] and item['training_preflight_required']
    listing = platform.client.get(BASE, headers=ADMIN)
    assert listing.json() == [item]
    assert SECRET not in listing.text and 'binding_snapshot' not in item
    with platform.app.state.db.read() as session:
        row = session.get(AssetTransfer, item['id'])
        assert row.binding_snapshot['source_connection']['id'] == connections[0].id
        assert list(session.scalars(select(Command).where(Command.status == 'PENDING'))) == []
        assert [r.id for r in active_asset_transfers(session, asset_id=asset['id'])] == [item['id']]


def test_idempotency_replays_same_request_and_rejects_changed_request(fixture):
    first = create(fixture, idempotency_key='stable-key').json()
    assert create(fixture, idempotency_key='stable-key').json() == first
    changed = create(fixture, idempotency_key='stable-key', target_name='other')
    assert changed.status_code == 409 and changed.json()['error']['code'] == 'IDEMPOTENCY_CONFLICT'
    assert len(fixture[0].client.get(BASE, headers=ADMIN).json()) == 1


def test_active_targets_cannot_overlap_and_terminal_targets_leave_transport_to_verify_ownership(fixture):
    platform = fixture[0]
    first = create(fixture).json()
    assert create(fixture).status_code == 409
    assert create(fixture, target_directory='/machine/copied/folding-copy', target_name='nested').status_code == 409
    assert create(fixture, target_directory='/machine', target_name='copied').status_code == 409
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, first['id']).state = 'FAILED'
    # A distinct transfer may be created, but the filesystem transport must
    # refuse a destination owned by the earlier job rather than overwriting it.
    assert create(fixture).status_code == 202


@pytest.mark.parametrize('changes', [
    {'target_name': '../escape'}, {'target_name': 'a/b'}, {'target_name': '.'},
    {'target_name': '..'}, {'target_name': 'a\\b'}, {'target_name': ' '},
    {'target_name': 'x\n'}, {'target_name': '数' * 100},
    {'target_directory': '/outside'}, {'target_directory': '/'},
    {'target_directory': '/machine/../outside'}, {'name': '\n'},
    {'idempotency_key': ' '}, {'source_path': '/machine/arbitrary'},
])
def test_untrusted_fields_paths_and_component_names_are_rejected(fixture, changes):
    assert create(fixture, **changes).status_code == 422


def test_source_must_be_registered_available_and_on_a_distinct_online_node(fixture):
    platform, _, source, target, asset, _ = fixture
    assert create(fixture, source_asset_id=uid()).status_code == 404
    assert create(fixture, target_node_id=source['node_id']).status_code == 422
    with platform.app.state.db.write() as session:
        session.get(Node, target['node_id']).last_seen = offset_time(-600)
    assert create(fixture).status_code == 409
    with platform.app.state.db.write() as session:
        session.get(Node, target['node_id']).last_seen = offset_time(0)
        session.get(Asset, asset['id']).state = 'ARCHIVED'
    assert create(fixture).status_code == 409


def test_ssh_credentials_are_checked_before_worker_launch_without_network(fixture):
    platform, _, _, _, _, connections = fixture
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, connections[0].id).encrypted_password = 'bad-encrypted-value'
    response = create(fixture)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'TRANSFER_SOURCE_SSH_CREDENTIAL_UNAVAILABLE'
    assert SECRET not in response.text and 'bad-encrypted-value' not in response.text
    assert platform.client.get(BASE, headers=ADMIN).json() == []


def test_same_physical_host_aliases_cannot_copy_into_the_source_tree(fixture):
    platform, _, _, _, _, connections = fixture
    with platform.app.state.db.write() as session:
        session.get(RemoteConnection, connections[1].id).fingerprint = connections[0].fingerprint
    response = create(fixture, target_directory='/machine/data/folding', target_name='recursive')
    assert response.status_code == 422 and response.json()['error']['code'] == 'ASSET_TRANSFER_PATH_OVERLAP'
    assert create(fixture, target_directory='/machine/independent', target_name='copy').status_code == 202


@pytest.mark.parametrize('kind,file_name', [('DATASET', None), ('MODEL', None), ('NORM', 'norm_stats.json')])
def test_success_registers_exact_copy_with_semantics_but_not_foreign_machine_bindings(fixture, monkeypatch, kind, file_name):
    platform, _, _, target, asset, _ = fixture
    with platform.app.state.db.write() as session:
        source = session.get(Asset, asset['id'])
        source.kind = kind
        if file_name:
            source.path = '/machine/norm/' + file_name
    task = create(fixture, name='User synchronization task').json()
    final = run_copy(fixture, task['id'], monkeypatch,
                     lambda *args, **kwargs: result_for(args, kwargs, file_name=file_name))
    assert final['state'] == 'SUCCEEDED' and final['byte_verified']
    assert final['bytes_total'] == final['bytes_transferred'] == 7
    assert final['files_total'] == final['files_transferred'] == 1 and final['progress_pct'] == 100
    target_asset = next(a for a in platform.client.get('/api/v1/assets', headers=ADMIN).json()
                        if a['id'] == final['target_asset_id'])
    assert target_asset['name'] == asset['name'] and target_asset['version'] == 'v2'
    assert target_asset['node_id'] == target['node_id'] and target_asset['kind'] == kind
    assert target_asset['path'] == final['asset_path'] and target_asset['state'] == 'REGISTERED'
    assert target_asset.get('origin_attempt_id') is None
    assert target_asset['metadata']['dataset_profile'] == asset['metadata']['dataset_profile']
    assert 'inspection' not in target_asset['metadata'] and 'dataset_asset_id' not in target_asset['metadata']
    assert target_asset['metadata']['asset_sync']['source_asset_id'] == asset['id']
    assert target_asset['metadata']['asset_sync']['training_preflight_required']
    assert final['target_path'] == '/machine/copied/folding-copy'
    assert final['asset_path'] == final['target_path'] + '/' + file_name if file_name else final['asset_path'] == final['target_path']
    assert platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 409


@pytest.mark.parametrize('change', ['target_path', 'asset_path', 'digest', 'total', 'unsafe_manifest'])
def test_malformed_receipts_never_register_target_assets(fixture, monkeypatch, change):
    task = create(fixture).json()
    def invalid(*args, **kwargs):
        value = result_for(args, kwargs)
        if change in ('target_path', 'asset_path'):
            value[change] = '/machine/unrelated'
        elif change == 'digest':
            value['manifest_sha256'] = 'f' * 64
        elif change == 'total':
            value['bytes_total'] += 1
        else:
            value['manifest']['files'][0]['path'] = '../escape'
            value['manifest_sha256'] = canonical_digest(value['manifest'])
        return value
    final = run_copy(fixture, task['id'], monkeypatch, invalid)
    assert final['state'] == 'FAILED' and final['target_asset_id'] is None
    assert len(fixture[0].client.get('/api/v1/assets', headers=ADMIN).json()) == 1


@pytest.mark.parametrize('drift', ['path', 'metadata', 'roots', 'fingerprint'])
def test_frozen_binding_drift_blocks_worker_and_retry(fixture, monkeypatch, drift):
    platform, manager, _, target, asset, connections = fixture
    task = create(fixture).json()
    with platform.app.state.db.write() as session:
        if drift == 'path':
            session.get(Asset, asset['id']).path = '/machine/other'
        elif drift == 'metadata':
            session.get(Asset, asset['id']).metadata_json = {'description': 'changed'}
        elif drift == 'roots':
            session.get(Node, target['node_id']).roots = ['/machine', '/extra']
        else:
            session.get(RemoteConnection, connections[0].id).fingerprint = 'SHA256:changed'
    called = []
    manager.transfer_function = lambda *_a, **_k: called.append(True)
    manager._run(task['id'])
    final = receipt(fixture, task['id'])
    assert final['state'] == 'FAILED' and called == []
    assert platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 409


@pytest.mark.parametrize('drift', [None, 'custom-name', 'metadata', 'path', 'fingerprint'])
def test_generated_model_label_upgrade_preserves_copy_snapshot_and_rejects_other_drift(fixture, monkeypatch, drift):
    platform, _, source, _, asset, connections = fixture
    recipe = 'auto_pi05_full_h16_d32_sft'
    with platform.app.state.db.write() as session:
        run = add_run(session, source, name='折叠衣服训练')
        attempt = Attempt(id=uid(), run_id=run.id, node_id=source['node_id'], number=1,
            execution_token=uid(), state='SUCCEEDED', created_at=now(), code_snapshot=run.code_expectation)
        session.add(attempt)
        session.flush()
        model = session.get(Asset, asset['id'])
        model.kind = 'MODEL'
        model.name = recipe + ' / 59999'
        model.version = '59999'
        model.metadata_json = {'format': 'openpi-orbax', 'recipe': recipe, 'step': 59999}
        model.origin_attempt_id = attempt.id
        expected = f'{run.name} · {run.id[:8]} / 59999'
    task = create(fixture).json()
    with platform.app.state.db.read() as session:
        frozen = deepcopy(session.get(AssetTransfer, task['id']).binding_snapshot)
    platform.app.state.db.initialize()
    with platform.app.state.db.write() as session:
        model = session.get(Asset, asset['id'])
        assert model.name == expected
        assert session.get(AssetTransfer, task['id']).name == expected + ' · 资产同步'
        if drift == 'custom-name': model.name = '另一个手动名称'
        elif drift == 'metadata': model.metadata_json = {**model.metadata_json, 'changed': True}
        elif drift == 'path': model.path += '-changed'
        elif drift == 'fingerprint': session.get(RemoteConnection, connections[0].id).fingerprint = 'SHA256:changed'
    final = run_copy(fixture, task['id'], monkeypatch)
    with platform.app.state.db.read() as session:
        assert session.get(AssetTransfer, task['id']).binding_snapshot == frozen
        if drift is None:
            assert final['state'] == 'SUCCEEDED'
            assert session.get(Asset, final['target_asset_id']).name == expected
        else:
            assert final['state'] == 'FAILED'
            assert final['error']['code'] == 'ASSET_TRANSFER_BINDING_CHANGED'


def test_retry_uses_same_owned_target_and_a_new_attempt(fixture, monkeypatch):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    def failed(*_args, **_kwargs):
        raise APIError(503, 'TRANSFER_NETWORK', 'closed: ' + SECRET)
    first = run_copy(fixture, task['id'], monkeypatch, failed)
    assert first['state'] == 'FAILED' and SECRET not in str(first)
    retried = platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN)
    assert retried.status_code == 202 and retried.json()['attempt'] == 2
    final = run_copy(fixture, task['id'], monkeypatch)
    assert final['state'] == 'SUCCEEDED' and final['attempt'] == 2
    assert final['target_name'] == task['target_name']


@pytest.mark.parametrize('failures,expected_state,expected_calls', [(1, 'SUCCEEDED', 2), (2, 'FAILED', 2)])
def test_control_ssh_connection_failure_retries_once_before_copy(
        fixture, monkeypatch, failures, expected_state, expected_calls):
    _, manager, *_ = fixture
    task = create(fixture).json()
    remotes, copied, waits = [], [], []

    class FlakyRemote(FakeRemote):
        calls = 0

        def connect(self):
            self.calls += 1
            if self.calls <= failures:
                raise RemoteError('SSH_CONNECTION_FAILED', 'SSH unavailable: ' + SECRET)
            super().connect()

    def remote(_connection):
        value = FlakyRemote() if not remotes else FakeRemote()
        remotes.append(value)
        return value

    monkeypatch.setattr(manager.remote_manager, '_remote', remote)
    monkeypatch.setattr(manager._stop, 'wait', lambda seconds: waits.append(seconds) or False)

    def transfer(*args, **kwargs):
        copied.append(True)
        return result_for(args, kwargs)

    manager.transfer_function = transfer
    manager._run(task['id'])
    final = receipt(fixture, task['id'])
    assert final['state'] == expected_state and final['attempt'] == 1
    assert remotes[0].calls == expected_calls and waits == [.5]
    assert copied == ([True] if expected_state == 'SUCCEEDED' else [])
    assert '0.5 秒后重试一次' in final['logs'] and SECRET not in str(final)
    assert all(r.closed for r in remotes)
    if expected_state == 'FAILED':
        assert final['error']['code'] == 'SSH_CONNECTION_FAILED'


@pytest.mark.parametrize('error', [
    RemoteError('SSH_AUTH_FAILED', 'Authentication failed'),
    RemoteError('SSH_HOST_KEY_MISMATCH', 'Host key changed'),
    RemoteError('SSH_CANCELLED', 'Cancelled'),
    APIError(503, 'SSH_CONNECTION_FAILED', 'Unrelated typed failure'),
])
def test_control_ssh_authentication_host_key_and_other_errors_are_not_retried(fixture, monkeypatch, error):
    _, manager, *_ = fixture
    task = create(fixture).json()
    calls, copied = [], []

    class FailedRemote(FakeRemote):
        def connect(self):
            calls.append(True)
            raise error

    monkeypatch.setattr(manager.remote_manager, '_remote', lambda _connection: FailedRemote())
    monkeypatch.setattr(manager._stop, 'wait', lambda _seconds: pytest.fail('Must not retry this error'))
    manager.transfer_function = lambda *_args, **_kwargs: copied.append(True)
    manager._run(task['id'])
    final = receipt(fixture, task['id'])
    assert final['state'] == 'FAILED' and final['error']['code'] == error.code
    assert calls == [True] and copied == []
    assert '0.5 秒后重试一次' not in final['logs']


def test_cancellation_during_control_ssh_retry_wait_prevents_reconnect_and_copy(fixture, monkeypatch):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    waiting, release = threading.Event(), threading.Event()
    calls, copied = [], []

    class FailedRemote(FakeRemote):
        def connect(self):
            calls.append(True)
            raise RemoteError('SSH_CONNECTION_FAILED', 'Connection temporarily unavailable')

    def retry_wait(seconds):
        assert seconds == .5
        waiting.set()
        assert release.wait(5)
        return manager._stop.is_set()

    monkeypatch.setattr(manager.remote_manager, '_remote', lambda _connection: FailedRemote())
    monkeypatch.setattr(manager._stop, 'wait', retry_wait)
    manager.transfer_function = lambda *_args, **_kwargs: copied.append(True)
    worker = threading.Thread(target=manager._run, args=(task['id'],), daemon=True)
    manager._workers[task['id']] = worker
    worker.start()
    try:
        assert waiting.wait(5)
        response = platform.client.post(BASE + '/' + task['id'] + '/cancel', headers=ADMIN)
        assert response.status_code == 202
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert receipt(fixture, task['id'])['state'] == 'CANCELLED'
    assert calls == [True] and copied == []


@pytest.mark.parametrize('failure,code,message', [
    (socket.timeout(), 'TRANSFER_CONNECTION_TIMEOUT', '暂未响应'),
    (TimeoutError(), 'TRANSFER_CONNECTION_TIMEOUT', '暂未响应'),
    (EOFError(), 'TRANSFER_CONNECTION_CLOSED', '已断开'),
    (ConnectionResetError(SECRET), 'TRANSFER_CONNECTION_CLOSED', '已断开'),
    (BrokenPipeError(SECRET), 'TRANSFER_CONNECTION_CLOSED', '已断开'),
    (Exception(), 'ASSET_TRANSFER_FAILED', '数据同步失败'),
    (ValueError(' \n '), 'ASSET_TRANSFER_FAILED', '数据同步失败'),
    (TransferError('TRANSFER_IDLE_TIMEOUT', 'No progress: ' + SECRET),
     'TRANSFER_IDLE_TIMEOUT', 'No progress:'),
    (APIError(503, 'SSH_CONNECTION_FAILED', 'Disconnected: ' + SECRET),
     'SSH_CONNECTION_FAILED', 'Disconnected:'),
    (TransferError('TRANSFER_EMPTY_MESSAGE', ''), 'TRANSFER_EMPTY_MESSAGE', '数据同步失败'),
])
def test_worker_failures_have_safe_nonempty_actionable_messages(fixture, monkeypatch, failure, code, message):
    platform, _, *_ = fixture
    task = create(fixture).json()

    def fail(*_args, **kwargs):
        kwargs['progress']({'stage': 'TRANSFERRING', 'bytes_done': 3, 'bytes_total': 7,
                            'files_total': 1, 'percent': 42.9})
        raise failure

    final = run_copy(fixture, task['id'], monkeypatch, fail)
    assert final['state'] == 'FAILED' and final['attempt'] == 1
    assert final['error']['code'] == code and message in final['error']['message']
    assert final['error']['message'].strip() and final['error']['message'] in final['logs']
    assert SECRET not in str(final)
    assert final['bytes_transferred'] == 3 and final['progress_pct'] == 42.9
    assert final['target_asset_id'] is None
    assert len(platform.client.get('/api/v1/assets', headers=ADMIN).json()) == 1
    # Recovery stays explicit: reporting the failure must not launch another attempt.
    retry = platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN)
    assert retry.status_code == 202 and retry.json()['attempt'] == 2


def test_cancellation_before_worker_first_database_read_is_finalized(fixture):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    gate = threading.Event()
    worker = threading.Thread(target=lambda: (gate.wait(5), manager._run(task['id'])), daemon=True)
    manager._workers[task['id']] = worker
    worker.start()
    try:
        cancelled = platform.client.post(BASE + '/' + task['id'] + '/cancel', headers=ADMIN)
        assert cancelled.status_code == 202 and cancelled.json()['state'] == 'CANCELLING'
        assert platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 409
    finally:
        gate.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert receipt(fixture, task['id'])['state'] == 'CANCELLED'


def test_cancelled_or_stale_progress_cannot_reactivate_a_transfer(fixture):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    assert platform.client.post(BASE + '/' + task['id'] + '/cancel', headers=ADMIN).status_code == 202
    manager._progress(task['id'], 1, {'stage': 'TRANSFERRING', 'bytes_done': 7})
    assert receipt(fixture, task['id'])['state'] == 'CANCELLED'
    assert platform.client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 202
    manager._progress(task['id'], 1, {'stage': 'VERIFYING', 'bytes_done': 7})
    current = receipt(fixture, task['id'])
    assert current['state'] == 'QUEUED' and current['attempt'] == 2 and current['bytes_transferred'] == 0


def test_stale_generation_result_cannot_register_an_asset(fixture, monkeypatch):
    platform, _, *_ = fixture
    task = create(fixture).json()
    def previous_worker(*args, **kwargs):
        with platform.app.state.db.write() as session:
            row = session.get(AssetTransfer, task['id'])
            row.attempt += 1
            row.state = 'QUEUED'
        return result_for(args, kwargs)
    final = run_copy(fixture, task['id'], monkeypatch, previous_worker)
    assert final['state'] == 'QUEUED' and final['attempt'] == 2
    assert final['target_asset_id'] is None
    assert len(platform.client.get('/api/v1/assets', headers=ADMIN).json()) == 1


def test_queued_cancel_and_restart_keep_recoverable_receipts(fixture):
    platform, manager, *_ = fixture
    first = create(fixture).json()
    assert platform.client.post(BASE + '/' + first['id'] + '/cancel', headers=ADMIN).json()['state'] == 'CANCELLED'
    retried = platform.client.post(BASE + '/' + first['id'] + '/retry', headers=ADMIN)
    assert retried.status_code == 202 and retried.json()['attempt'] == 2
    manager.start()
    assert receipt(fixture, first['id'])['state'] == 'INTERRUPTED'
    assert platform.client.post(BASE + '/' + first['id'] + '/retry', headers=ADMIN).status_code == 202
    manager.close()
    assert receipt(fixture, first['id'])['state'] == 'INTERRUPTED'


def test_active_transfers_block_asset_mutation_storage_and_agent_maintenance(fixture):
    platform, _, source, target, asset, _ = fixture
    task = create(fixture).json()
    assert platform.client.post(f"/api/v1/assets/{asset['id']}/archive", headers=ADMIN).status_code == 409
    assert platform.client.patch(f"/api/v1/assets/{asset['id']}/dataset-profile", headers=ADMIN,
                                 json={'dataset_profile': profile()}).status_code == 409
    for node in (source, target):
        with platform.app.state.db.write() as session:
            assert _in_use(session, node['node_id'])
            with pytest.raises(APIError) as exc:
                DeploymentManager._assert_idle(session, node['node_id'])
            assert exc.value.code == 'NODE_BUSY'
        response = platform.client.post(f"/api/v1/nodes/{node['node_id']}/storage-configurations", headers=ADMIN,
            json={'expected_output_root': '/machine/output', 'expected_cache_root': None,
                  'output_root': '/machine/new-output', 'cache_root': '/machine/cache'})
        assert response.status_code == 409 and response.json()['error']['code'] == 'STORAGE_CONFIGURE_BUSY'
    with platform.app.state.db.write() as session:
        session.get(AssetTransfer, task['id']).state = 'FAILED'
        for node in (source, target):
            assert not _in_use(session, node['node_id'])
            DeploymentManager._assert_idle(session, node['node_id'])
    assert platform.client.patch(f"/api/v1/assets/{asset['id']}/dataset-profile", headers=ADMIN,
                                 json={'dataset_profile': profile()}).status_code == 200


def test_copy_is_read_only_and_can_run_while_source_training_is_running(fixture):
    platform, _, source, _, *_ = fixture
    with platform.app.state.db.write() as session:
        run = add_run(session, source)
        run.state = 'RUNNING'
    assert create(fixture).status_code == 202
    with platform.app.state.db.read() as session:
        assert session.get(Run, run.id).state == 'RUNNING'


def test_listing_is_bounded_filterable_and_viewer_cannot_write(fixture):
    platform, _, _, target, asset, _ = fixture
    task = create(fixture).json()
    assert platform.client.get(BASE, headers=ADMIN, params={'source_asset_id': asset['id'], 'limit': 1}).json() == [task]
    assert platform.client.get(BASE, headers=ADMIN, params={'target_node_id': target['node_id']}).json() == [task]
    assert platform.client.get(BASE, headers=ADMIN, params={'source_asset_id': uid()}).json() == []
    assert platform.client.get(BASE, headers=ADMIN, params={'limit': 201}).status_code == 422
    assert platform.client.get(BASE).status_code == 401
    platform.app.state.auth.create_user('sync-viewer', 'sync-test-password-123', 'viewer')
    viewer = TestClient(platform.app, base_url='https://testserver')
    try:
        login = viewer.post('/api/v1/auth/login', json={'username': 'sync-viewer', 'password': 'sync-test-password-123'})
        assert login.status_code == 200
        viewer.headers['X-CSRF-Token'] = login.json()['csrf_token']
        assert viewer.get(BASE).status_code == 200
        assert viewer.post(BASE, json={}).status_code == 403
        assert viewer.post(BASE + '/' + task['id'] + '/cancel').status_code == 403
    finally:
        viewer.close()


def test_polling_queries_never_load_manifest_or_binding_snapshots(fixture):
    platform, manager, *_ = fixture
    task = create(fixture).json()
    engine = platform.app.state.db.engine
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith('SELECT') and 'FROM asset_transfers' in statement:
            statements.append(statement)

    event.listen(engine, 'before_cursor_execute', capture)
    try:
        listing = platform.client.get(BASE, headers=ADMIN)
        detail = platform.client.get(BASE + '/' + task['id'], headers=ADMIN)
        assert listing.status_code == detail.status_code == 200
        assert not manager._cancelled(task['id'], 1)
    finally:
        event.remove(engine, 'before_cursor_execute', capture)
    assert len(statements) == 3, 'Polling must not trigger lazy JSON fetches'
    for statement in statements:
        projection = statement.split('FROM', 1)[0]
        assert not re.search(r'\basset_transfers\.(manifest|binding_snapshot)\b', projection)
    assert 'asset_transfers.state' in statements[-1] and 'asset_transfers.attempt' in statements[-1]
    assert 'asset_transfers.logs' not in statements[-1]
    for item in [*listing.json(), detail.json()]:
        assert 'manifest' not in item and 'binding_snapshot' not in item
        assert item['manifest_sha256'] is None and item['state'] == 'QUEUED'


@pytest.mark.parametrize('legacy_error', [
    {'code': 'ASSET_TRANSFER_FAILED', 'message': ''},
    {'code': 'ASSET_TRANSFER_FAILED', 'message': ' \n '},
    {'code': 'ASSET_TRANSFER_FAILED', 'message': None},
    {'code': 'ASSET_TRANSFER_FAILED'},
])
def test_historical_empty_failure_messages_are_readable_without_rewriting_receipts(fixture, legacy_error):
    platform, _, *_ = fixture
    task = create(fixture).json()
    norm_error = {**legacy_error, 'code': 'HISTORICAL_NORM_ERROR'}
    legacy_norm = {'state': 'FAILED', 'error': norm_error}
    with platform.app.state.db.write() as session:
        row = session.get(AssetTransfer, task['id'])
        row.state = 'FAILED'
        row.error = legacy_error
        row.include_norm_stats = True
        row.norm_sync = legacy_norm
        row.logs = 'Historical log kept intact'
        updated_at = row.updated_at
    detail = receipt(fixture, task['id'])
    listing = platform.client.get(BASE, headers=ADMIN).json()
    for item in [detail, *listing]:
        assert item['error']['code'] == legacy_error['code']
        assert item['norm_sync']['error']['code'] == norm_error['code']
        assert item['error']['message'] == item['norm_sync']['error']['message'] == (
            '本次同步未记录具体错误原因；已传文件保留，可手动重试继续。')
        assert 'TIMEOUT' not in str(item['error'])
    with platform.app.state.db.read() as session:
        row = session.get(AssetTransfer, task['id'])
        assert row.error == legacy_error and row.norm_sync == legacy_norm
        assert row.updated_at == updated_at and row.logs == 'Historical log kept intact'
