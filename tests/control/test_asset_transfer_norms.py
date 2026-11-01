"""Optional norm copies remain one task and bind only verified target assets."""

import pytest
from sqlalchemy import event, select, text

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.asset_transfers import active_asset_transfers
from vla_control.common import APIError
from vla_control.models import Asset

from .conftest import ADMIN
from .test_asset_transfers import BASE, create, fixture, receipt, result_for, run_copy


@pytest.fixture
def paired(fixture):
    platform, _, source, _, dataset, _ = fixture
    response = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': source['node_id'], 'kind': 'NORM', 'name': 'folding norm',
        'path': '/machine/norm/norm_stats.json', 'version': 'v3', 'metadata': {'description': 'Raw 14D stats'}})
    assert response.status_code == 201, response.text
    norm = response.json()
    with platform.app.state.db.write() as session:
        session.get(Asset, dataset['id']).norm_asset_id = norm['id']
    return fixture, norm


def copied_norm(*args, **kwargs):
    return result_for(args, kwargs, file_name='norm_stats.json' if args[2].endswith('norm_stats.json') else None)


@pytest.mark.parametrize('legacy', [False, True])
def test_raw_inserts_that_omit_norm_fields_default_to_false_in_fresh_and_migrated_schema(fixture, legacy):
    platform = fixture[0]
    first = create(fixture).json()
    if legacy:
        with platform.app.state.db.engine.begin() as connection:
            connection.exec_driver_sql('ALTER TABLE asset_transfers DROP COLUMN norm_sync')
            connection.exec_driver_sql('ALTER TABLE asset_transfers DROP COLUMN include_norm_stats')
        platform.app.state.db.initialize()
    columns = [column.name for column in AssetTransfer.__table__.columns
               if column.name not in {'include_norm_stats', 'norm_sync'}]
    values = [':new_id' if name == 'id' else ':new_key' if name == 'request_key' else name for name in columns]
    with platform.app.state.db.write() as session:
        session.execute(text(f'INSERT INTO asset_transfers ({",".join(columns)}) '
                             f'SELECT {",".join(values)} FROM asset_transfers WHERE id=:source_id'),
                        {'new_id': 'raw-legacy-writer', 'new_key': 'raw-key', 'source_id': first['id']})
    with platform.app.state.db.read() as session:
        value, storage_type = session.execute(text('SELECT include_norm_stats,typeof(include_norm_stats) '
            'FROM asset_transfers WHERE id=:id'), {'id': 'raw-legacy-writer'}).one()
        assert value == 0 and storage_type == 'integer'
        inserted = session.get(AssetTransfer, 'raw-legacy-writer')
        assert inserted.include_norm_stats is False and inserted.norm_sync is None
        assert session.get(AssetTransfer, first['id']).include_norm_stats is False


def test_bound_norm_is_optional_and_plain_copy_does_not_keep_source_asset_id(paired, monkeypatch):
    fixture, norm = paired
    task = create(fixture).json()
    assert task['include_norm_stats'] is False and task['norm_sync'] is None
    final = run_copy(fixture, task['id'], monkeypatch)
    assert final['state'] == 'SUCCEEDED'
    with fixture[0].app.state.db.read() as session:
        target = session.get(Asset, final['target_asset_id'])
        assert target.norm_asset_id is None
        assert session.get(Asset, norm['id']).node_id == fixture[2]['node_id']


def test_pair_registers_both_assets_atomically_and_rebinds_dataset(paired, monkeypatch):
    fixture, norm = paired
    task = create(fixture, include_norm_stats=True).json()
    assert task['include_norm_stats'] is True
    assert task['norm_sync']['source_asset_id'] == norm['id']
    assert task['norm_sync']['state'] == 'QUEUED'
    assert task['norm_sync']['target_name'] == 'norm-stats-' + task['id']
    calls = []
    def copy(*args, **kwargs):
        calls.append((args[2], args[4], kwargs['target_name']))
        kwargs['progress']({'stage': 'TRANSFERRING', 'bytes_total': 7, 'bytes_done': 3, 'files_total': 1,
                            'files_done': None, 'percent': 42, 'speed_bps': 50})
        current = receipt(fixture, task['id'])
        assert current['state'] == 'TRANSFERRING' and current['target_asset_id'] is None
        if len(calls) == 2:
            assert current['bytes_total'] == 14 and current['bytes_transferred'] == 10
            assert current['files_transferred'] is None
            assert current['norm_sync']['files_transferred'] is None
            assert current['norm_sync']['progress_pct'] == 42
        return copied_norm(*args, **kwargs)
    final = run_copy(fixture, task['id'], monkeypatch, copy)
    assert len(calls) == 2 and calls[0][1] == task['id'] and calls[1][1] == task['id'] + '-norm'
    assert final['state'] == 'SUCCEEDED' and final['progress_pct'] == 100
    assert final['bytes_total'] == final['bytes_transferred'] == 14
    assert final['files_total'] == final['files_transferred'] == 2
    assert final['norm_sync']['state'] == 'SUCCEEDED'
    assert final['norm_sync']['asset_path'].endswith('/norm-stats-' + task['id'] + '/norm_stats.json')
    with fixture[0].app.state.db.read() as session:
        target_dataset = session.get(Asset, final['target_asset_id'])
        target_norm = session.get(Asset, final['norm_sync']['target_asset_id'])
        assert target_dataset.norm_asset_id == target_norm.id != norm['id']
        assert target_dataset.node_id == target_norm.node_id == fixture[3]['node_id']
        assert target_norm.kind == 'NORM' and target_norm.version == 'v3'
        assert target_norm.metadata_json['description'] == 'Raw 14D stats'
        assert target_norm.metadata_json['asset_sync']['transfer_id'] == task['id']
        assert target_norm.metadata_json['asset_sync']['part'] == 'norm_stats'
        assert session.get(Asset, fixture[4]['id']).norm_asset_id == norm['id']


def test_paired_creation_locks_assets_before_nodes_like_asset_archival(paired):
    fixture, norm = paired
    engine = fixture[0].app.state.db.engine
    observed = []
    def capture(_connection, _cursor, statement, parameters, _context, _executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            observed.append((statement, parameters))
    event.listen(engine, 'before_cursor_execute', capture)
    try:
        response = create(fixture, include_norm_stats=True)
        assert response.status_code == 202, response.text
    finally:
        event.remove(engine, 'before_cursor_execute', capture)
    norm_query = next(i for i, (sql, args) in enumerate(observed) if 'FROM assets' in sql and norm['id'] in args)
    node_query = next(i for i, (sql, _) in enumerate(observed) if 'FROM nodes' in sql)
    assert norm_query < node_query, 'Paired transfer and NORM archive must agree on asset-before-node lock order'


def test_second_copy_failure_registers_neither_asset_and_retry_uses_stable_identities(paired, monkeypatch):
    fixture, norm = paired
    task = create(fixture, include_norm_stats=True, idempotency_key='pair').json()
    calls = []
    fail_norm = True
    def copy(*args, **kwargs):
        calls.append((args[2], args[4], kwargs['target_name']))
        if args[4].endswith('-norm') and fail_norm:
            raise APIError(503, 'TRANSFER_NETWORK', 'norm connection interrupted')
        return copied_norm(*args, **kwargs)
    failed = run_copy(fixture, task['id'], monkeypatch, copy)
    assert failed['state'] == failed['norm_sync']['state'] == 'FAILED'
    assert failed['target_asset_id'] is None and failed['norm_sync']['target_asset_id'] is None
    with fixture[0].app.state.db.read() as session:
        assert len(list(session.scalars(select(Asset)))) == 2
        assert session.get(AssetTransfer, task['id']).manifest['bytes_total'] == 7
    retried = fixture[0].client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN)
    assert retried.status_code == 202 and retried.json()['attempt'] == 2
    fail_norm = False
    final = run_copy(fixture, task['id'], monkeypatch, copy)
    assert final['state'] == 'SUCCEEDED' and calls[:2] == calls[2:]
    with fixture[0].app.state.db.read() as session:
        assert len(list(session.scalars(select(Asset)))) == 4
    assert create(fixture, include_norm_stats=True, idempotency_key='pair').json()['id'] == task['id']
    assert fixture[0].client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 409


@pytest.mark.parametrize('stage', ['dataset', 'norm'])
def test_cancellation_during_either_copy_never_registers_half_a_pair(paired, monkeypatch, stage):
    fixture, _ = paired
    task = create(fixture, include_norm_stats=True).json()
    calls = []
    def copy(*args, **kwargs):
        part = 'norm' if args[4].endswith('-norm') else 'dataset'
        calls.append(part)
        if stage == part:
            assert fixture[0].client.post(BASE + '/' + task['id'] + '/cancel', headers=ADMIN).status_code == 202
        return copied_norm(*args, **kwargs)
    final = run_copy(fixture, task['id'], monkeypatch, copy)
    assert final['state'] == final['norm_sync']['state'] == 'CANCELLED'
    assert final['target_asset_id'] is None
    assert calls == (['dataset'] if stage == 'dataset' else ['dataset', 'norm'])
    with fixture[0].app.state.db.read() as session:
        assert len(list(session.scalars(select(Asset)))) == 2
    fixture[1]._progress(task['id'], 1, {'stage': 'TRANSFERRING', 'bytes_done': 7}, part='norm')
    assert receipt(fixture, task['id'])['state'] == 'CANCELLED'


@pytest.mark.parametrize('drift', ['path', 'version', 'metadata', 'binding', 'kind', 'state', 'node'])
def test_norm_identity_and_binding_are_frozen_for_worker_and_retry(paired, monkeypatch, drift):
    fixture, norm = paired
    task = create(fixture, include_norm_stats=True).json()
    with fixture[0].app.state.db.write() as session:
        row = session.get(Asset, norm['id'])
        if drift == 'binding': session.get(Asset, fixture[4]['id']).norm_asset_id = None
        elif drift == 'path': row.path = '/machine/other.json'
        elif drift == 'version': row.version = 'changed'
        elif drift == 'metadata': row.metadata_json = {'changed': True}
        elif drift == 'kind': row.kind = 'MODEL'
        elif drift == 'state': row.state = 'ARCHIVED'
        else: row.node_id = fixture[3]['node_id']
    def forbidden(*_args, **_kwargs): pytest.fail('drift must block before opening transfer')
    final = run_copy(fixture, task['id'], monkeypatch, forbidden)
    assert final['state'] == 'FAILED' and final['target_asset_id'] is None
    assert fixture[0].client.post(BASE + '/' + task['id'] + '/retry', headers=ADMIN).status_code == 409


def test_plain_legacy_snapshot_and_idempotency_are_unaffected_by_later_default_binding(paired, monkeypatch):
    fixture, norm = paired
    task = create(fixture, include_norm_stats=False, idempotency_key='old').json()
    with fixture[0].app.state.db.write() as session:
        row = session.get(AssetTransfer, task['id'])
        assert 'norm_asset' not in row.binding_snapshot
        from vla_protocol import canonical_digest
        assert row.request_digest == canonical_digest({'source_asset_id': fixture[4]['id'],
            'target_node_id': fixture[3]['node_id'], 'target_directory': '/machine/copied',
            'target_name': 'folding-copy', 'name': None})
        session.get(Asset, fixture[4]['id']).norm_asset_id = None
    assert create(fixture, idempotency_key='old').json()['id'] == task['id']
    final = run_copy(fixture, task['id'], monkeypatch)
    assert final['state'] == 'SUCCEEDED' and final['norm_sync'] is None


def test_companion_is_reserved_against_archival_and_competing_destinations(paired):
    fixture, norm = paired
    task = create(fixture, include_norm_stats=True).json()
    with fixture[0].app.state.db.read() as session:
        assert [row.id for row in active_asset_transfers(session, asset_id=norm['id'])] == [task['id']]
    assert fixture[0].client.post('/api/v1/assets/' + norm['id'] + '/archive', headers=ADMIN).status_code == 409
    assert create(fixture, target_name=task['norm_sync']['target_name']).status_code == 409
    cancelled = fixture[0].client.post(BASE + '/' + task['id'] + '/cancel', headers=ADMIN).json()
    assert cancelled['norm_sync']['state'] == 'CANCELLED'
    assert fixture[0].client.post('/api/v1/assets/' + norm['id'] + '/archive', headers=ADMIN).status_code == 200


@pytest.mark.parametrize('value', ['true', 1, None, [], {}])
def test_include_norm_stats_is_a_strict_boolean(fixture, value):
    assert create(fixture, include_norm_stats=value).status_code == 422


def test_include_norm_stats_requires_a_bound_available_norm(fixture):
    response = create(fixture, include_norm_stats=True)
    assert response.status_code == 409 and response.json()['error']['code'] == 'ASSET_TRANSFER_NORM_UNAVAILABLE'
    assert fixture[0].client.get(BASE, headers=ADMIN).json() == []


def test_changed_binding_from_an_open_dialog_requires_reconfirmation(paired):
    fixture, norm = paired
    response = create(fixture, include_norm_stats=True, expected_norm_asset_id='previous-norm')
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'ASSET_TRANSFER_NORM_BINDING_CHANGED'
    assert fixture[0].client.get(BASE, headers=ADMIN).json() == []
    accepted = create(fixture, include_norm_stats=True, expected_norm_asset_id=norm['id'])
    assert accepted.status_code == 202, accepted.text


def test_malformed_norm_receipt_does_not_register_either_copy(paired, monkeypatch):
    fixture, _ = paired
    task = create(fixture, include_norm_stats=True).json()
    def invalid(*args, **kwargs):
        result = copied_norm(*args, **kwargs)
        if args[4].endswith('-norm'): result['manifest_sha256'] = '0' * 64
        return result
    final = run_copy(fixture, task['id'], monkeypatch, invalid)
    assert final['state'] == final['norm_sync']['state'] == 'FAILED'
    with fixture[0].app.state.db.read() as session:
        assert len(list(session.scalars(select(Asset)))) == 2
