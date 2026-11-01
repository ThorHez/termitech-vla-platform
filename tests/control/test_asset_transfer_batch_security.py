"""Independent authorization and account-bound retry checks for batch copies."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.asset_transfer_batch_models import AssetTransferBatch
from vla_control.asset_transfer_models import AssetTransfer
from vla_control.auth_models import User
from vla_control.common import uid
from vla_control.models import Asset, Command

from .test_asset_transfers import fixture


BASE = '/api/v1/asset-transfer-batches'
PASSWORD = 'isolated-batch-copy-security-password'


@pytest.fixture
def secured(fixture):
    platform = fixture[0]
    for username, role in (('batch-owner', 'operator'), ('batch-admin', 'admin'), ('batch-reader', 'viewer')):
        platform.app.state.auth.create_user(username, PASSWORD, role)
    launched = []
    fixture[1]._launch = launched.append
    client = TestClient(platform.app, base_url='https://testserver')
    yield fixture, client, launched
    client.close()


def login(value, username='batch-owner'):
    client = value[1]
    client.cookies.clear()
    client.headers.pop('X-CSRF-Token', None)
    client.headers.pop('X-Asset-Transfer-Batch-User', None)
    response = client.post('/api/v1/auth/login', json={'username': username, 'password': PASSWORD})
    assert response.status_code == 200
    account = response.json()
    client.headers.update({'X-CSRF-Token': account['csrf_token'], 'X-Asset-Transfer-Batch-User': account['user']['id']})
    return account


def body(value, **changes):
    source = value[0][4]
    return {'idempotency_key': uid(), 'target_node_id': value[0][3]['node_id'],
            'target_directory': '/machine/copied',
            'entries': [{'source_asset_id': source['id'], 'target_name': 'batch-copy', 'include_norm_stats': False}],
            **changes}


def counts(value):
    with value[0][0].app.state.db.read() as session:
        return {model.__tablename__: session.scalar(select(func.count()).select_from(model))
                for model in (Asset, AssetTransfer, AssetTransferBatch, Command)}


def test_private_batch_requires_cookie_csrf_and_asset_permission(secured):
    client = secured[1]
    before = counts(secured)
    assert client.post(BASE, json=body(secured)).status_code == 401
    assert client.post(BASE, json=body(secured), headers={'Authorization': 'Bearer test-admin'}).status_code == 401
    login(secured)
    csrf = client.headers.pop('X-CSRF-Token')
    assert client.post(BASE, json=body(secured)).status_code == 403
    client.headers['X-CSRF-Token'] = csrf
    assert client.post(BASE, json=body(secured), headers={'Origin': 'https://foreign.example'}).status_code == 403
    login(secured, 'batch-reader')
    assert client.post(BASE, json=body(secured)).status_code == 403
    assert counts(secured) == before and secured[2] == []


def test_valid_bootstrap_bearer_cannot_own_a_batch(fixture):
    from .conftest import ADMIN
    launched = []
    fixture[1]._launch = launched.append
    client = fixture[0].client
    value = (fixture, client, launched)
    assert client.post(BASE, headers=ADMIN, json=body(value)).status_code == 403
    assert counts(value)['asset_transfers'] == 0 and launched == []


@pytest.mark.parametrize('username', ['batch-owner', 'batch-admin'])
def test_changed_account_header_blocks_even_administrator(secured, username):
    login(secured, username)
    before = counts(secured)
    response = secured[1].post(BASE, json=body(secured), headers={'X-Asset-Transfer-Batch-User': 'previous-account'})
    assert response.status_code == 403
    assert counts(secured) == before and secured[2] == []


def test_same_retry_key_is_scoped_to_account_and_does_not_relaunch(secured):
    first_user = login(secured)
    request = body(secured, idempotency_key='same-key-different-accounts')
    first = secured[1].post(BASE, json=request)
    assert first.status_code == 202, first.text
    first_ids = [row['id'] for row in first.json()['transfers']]
    assert secured[2] == first_ids
    replay = secured[1].post(BASE, json=request)
    assert replay.status_code == 202 and replay.json() == first.json()
    assert secured[2] == first_ids
    other = login(secured, 'batch-admin')
    assert other['user']['id'] != first_user['user']['id']
    request['target_directory'] = '/machine/other-account'
    second = secured[1].post(BASE, json=request)
    assert second.status_code == 202, second.text
    second_ids = [row['id'] for row in second.json()['transfers']]
    assert set(first_ids).isdisjoint(second_ids)
    assert secured[2] == first_ids + second_ids
    assert counts(secured)['asset_transfer_batches'] == 2


def test_disabled_owner_cannot_replay_previously_committed_request(secured):
    account = login(secured)
    request = body(secured)
    assert secured[1].post(BASE, json=request).status_code == 202
    before, launched = counts(secured), list(secured[2])
    with secured[0][0].app.state.db.write() as session:
        session.get(User, account['user']['id']).enabled = False
    assert secured[1].post(BASE, json=request).status_code == 401
    assert counts(secured) == before and secured[2] == launched


@pytest.mark.parametrize('length', [0, 21])
def test_empty_or_over_limit_batch_does_not_queue_anything(secured, length):
    login(secured)
    before = counts(secured)
    request = body(secured)
    request['entries'] = [{'source_asset_id': secured[0][4]['id'], 'target_name': f'copy-{index}'}
                          for index in range(length)]
    assert secured[1].post(BASE, json=request).status_code == 422
    assert counts(secured) == before and secured[2] == []


def test_invalid_later_entry_cannot_create_earlier_background_copy(secured):
    login(secured)
    request = body(secured)
    request['entries'].append({'source_asset_id': 'missing', 'target_name': 'second-copy'})
    before = counts(secured)
    response = secured[1].post(BASE, json=request)
    assert response.status_code == 404, response.text
    assert counts(secured) == before and secured[2] == []


def test_changed_payload_under_committed_key_is_conflict_without_new_launch(secured):
    login(secured)
    request = body(secured)
    assert secured[1].post(BASE, json=request).status_code == 202
    before, launched = counts(secured), list(secured[2])
    request['entries'][0]['target_name'] = 'another-target'
    assert secured[1].post(BASE, json=request).status_code == 409
    assert counts(secured) == before and secured[2] == launched
