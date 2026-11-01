"""The additive norm binding/companion-copy upgrade preserves the old database."""

import re

from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
import pytest

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.models import Asset, Run

from .conftest import ADMIN


def _legacy_table(cursor, name, remove):
    """Rebuild only isolated fixture tables without the newly added columns.

    SQLite cannot DROP a column named by a table-level foreign key. Reusing
    its CREATE TABLE declaration and retaining every other column, constraint,
    index and row yields the immediately preceding schema without editing any
    production database or weakening foreign keys during the migration itself.
    """
    ddl = cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0]
    indexes = [row[0] for row in cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (name,))]
    columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({name})") if row[1] not in remove]
    projection = ','.join('"' + column + '"' for column in columns)
    rows = cursor.execute(f'SELECT {projection} FROM "{name}"').fetchall()
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(value) for value in remove) + r')\b')
    old_ddl = '\n'.join(line for line in ddl.splitlines() if not pattern.search(line))
    old_ddl = re.sub(r',\s*\)$', '\n)', old_ddl)
    cursor.execute(f'DROP TABLE "{name}"')
    cursor.execute(old_ddl)
    if rows:
        cursor.executemany(f'INSERT INTO "{name}" ({projection}) VALUES ({",".join("?" for _ in columns)})', rows)
    for statement in indexes:
        if not pattern.search(statement):
            cursor.execute(statement)
    return columns


def _snapshot(database, queries):
    with database.engine.connect() as connection:
        return {name: [tuple(row) for row in connection.exec_driver_sql(query)] for name, query in queries.items()}


def test_norm_upgrade_is_idempotent_and_preserves_assets_training_sessions_and_copy_progress(platform):
    source, target = platform.register(), platform.register(name='target')
    assets = []
    for kind in ('DATASET', 'NORM'):
        response = platform.client.post('/api/v1/assets', headers=ADMIN, json={
            'node_id': source['node_id'], 'kind': kind, 'name': 'historical ' + kind,
            'path': '/machine/' + kind.lower(), 'metadata': {'custom': 'kept'}})
        assert response.status_code == 201, response.text
        assets.append(response.json())
    dataset, norm = assets
    run = platform.run(source)
    transfer_id = uid()
    with platform.app.state.db.write() as session:
        row = session.get(Run, run['id'])
        row.inputs = {'dataset': dataset['id'], 'norm': norm['id']}
        session.add(AssetTransfer(id=transfer_id, request_key=uid(), request_digest='c' * 64,
            name='historical partial transfer', source_asset_id=dataset['id'],
            source_node_id=source['node_id'], source_path=dataset['path'], source_kind='DATASET',
            target_node_id=target['node_id'], target_directory='/machine/copied', target_name='historical',
            state='TRANSFERRING', attempt=3, bytes_total=1000, bytes_transferred=350,
            files_total=10, files_transferred=3, speed_bytes_per_second=25, progress_pct=35,
            binding_snapshot={'historical': True}, logs='keep this progress\n', created_at=now(), updated_at=now()))
    password = 'migration-preserves-current-session-password'
    platform.app.state.auth.create_user('norm-migration-viewer', password, 'viewer')
    login = platform.client.post('/api/v1/auth/login', json={'username': 'norm-migration-viewer', 'password': password})
    assert login.status_code == 200
    database = platform.app.state.db
    queries = {table: f'SELECT * FROM {table} ORDER BY id' for table in
               ('nodes', 'profiles', 'runs', 'commands', 'users', 'user_sessions')}
    before = _snapshot(database, queries)
    raw = database.engine.raw_connection()
    try:
        raw.commit()
        cursor = raw.cursor()
        cursor.execute('PRAGMA foreign_keys=OFF')
        asset_columns = _legacy_table(cursor, 'assets', {'norm_asset_id'})
        transfer_columns = _legacy_table(cursor, 'asset_transfers', {'include_norm_stats', 'norm_sync'})
        raw.commit()
        cursor.execute('PRAGMA foreign_keys=ON')
        assert cursor.execute('PRAGMA foreign_keys').fetchone()[0] == 1
    finally:
        raw.close()
    preserved_queries = {
        **queries,
        'assets': 'SELECT ' + ','.join('"' + c + '"' for c in asset_columns) + ' FROM assets ORDER BY id',
        'asset_transfers': 'SELECT ' + ','.join('"' + c + '"' for c in transfer_columns) + ' FROM asset_transfers ORDER BY id',
    }
    legacy = _snapshot(database, preserved_queries)
    assert {key: legacy[key] for key in queries} == before
    database.initialize()
    database.initialize()
    assert _snapshot(database, preserved_queries) == legacy
    assert 'norm_asset_id' in {c['name'] for c in inspect(database.engine).get_columns('assets')}
    assert {'include_norm_stats', 'norm_sync'} <= {c['name'] for c in inspect(database.engine).get_columns('asset_transfers')}
    with database.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT norm_asset_id FROM assets').fetchall() == [(None,), (None,)]
        assert tuple(connection.exec_driver_sql('SELECT include_norm_stats,norm_sync FROM asset_transfers').one()) == (0, None)
    # Re-initializing after a real new binding must preserve that relationship.
    with database.write() as session:
        session.get(Asset, dataset['id']).norm_asset_id = norm['id']
    database.initialize()
    with database.read() as session:
        assert session.get(Asset, dataset['id']).norm_asset_id == norm['id']
        assert session.get(AssetTransfer, transfer_id).bytes_transferred == 350
        assert session.get(AssetTransfer, transfer_id).state == 'TRANSFERRING'
    assert _snapshot(database, queries) == before
    with pytest.raises(IntegrityError):
        with database.write() as session:
            session.get(Asset, dataset['id']).norm_asset_id = 'missing-norm'
            session.flush()
