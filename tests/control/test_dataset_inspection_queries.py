"""Metadata queries remain scoped, read-only and independent of saved assets."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import offset_time
from vla_control.models import Asset, Command, Node

from .conftest import ADMIN, ENROLL


def result(path="/machine/data"):
    return {"schema_version": 1, "path": path, "source": {"info_sha256": "a" * 64, "tasks_sha256": None},
            "features": [{"key": "actions", "dtype": "float32", "shape": [16, 10], "names": []}],
            "suggestions": {}, "processing": {}, "evidence": [], "warnings": []}


def create(platform, node, path="/machine/data", headers=ADMIN, **extra):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/dataset-inspections", headers=headers,
                                 json={"path": path, **extra})


def get(platform, node, query, headers=ADMIN):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/dataset-inspections/{query['id']}", headers=headers)


def ack(platform, node, query, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{query['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": status, "result": result(query["path"]) if value is None else value})


def test_query_cache_safe_ack_and_asset_data_are_independent(platform):
    node = platform.register()
    registered = platform.client.post('/api/v1/assets', headers=ADMIN, json={
        'node_id': node['node_id'], 'name': 'data', 'kind': 'DATASET', 'path': '/machine/data', 'metadata': {'keep': True}}).json()
    response = create(platform, node, "/machine/data/")
    assert response.status_code == 202, response.text
    assert response.headers["Cache-Control"] == "no-store"
    query = response.json()
    assert query["path"] == "/machine/data" and query["status"] == "PENDING" and query["inspection"] is None
    assert create(platform, node).json()["id"] == query["id"]
    command = platform.commands(node)[0]
    assert command["type"] == "DATASET_INSPECT" and set(command["payload"]) == {"path", "expires_at"}
    assert ack(platform, node, query).status_code == 200
    ready = get(platform, node, query).json()
    assert ready["status"] == "READY" and ready["inspection"] == result()
    assert create(platform, node).json() == ready
    assert ack(platform, node, query, {"error": {"code": "DATASET_INSPECT_FAILED"}}, status="REJECTED").status_code == 200
    assert get(platform, node, query).json() == ready
    with platform.app.state.db.read() as session:
        asset = session.get(Asset, registered['id'])
        assert asset.metadata_json == {'keep': True} and asset.state == registered['state']
        assert session.get(Command, query['id']).result == result()


def test_metadata_query_auth_node_ownership_and_offline_checks(platform):
    node, other = platform.register(), platform.register(name="other")
    query = create(platform, node).json()
    for headers in ({}, ENROLL, node["headers"]):
        assert create(platform, node, headers=headers).status_code == 401
        assert get(platform, node, query, headers=headers).status_code == 401
    assert get(platform, other, query).status_code == 404
    assert ack(platform, other, query).status_code == 403
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-1000)
    response = create(platform, node)
    assert response.status_code == 409 and response.json()['error']['code'] == 'NODE_OFFLINE'


@pytest.mark.parametrize("path", ["/outside/data", "/machine/../other", "/machine/data\n", "/machine/data\x85", "relative"])
def test_invalid_paths_never_reach_agent_queue(platform, path):
    node = platform.register()
    assert create(platform, node, path).status_code == 422
    assert platform.commands(node) == []


def test_expiry_and_root_changes_reject_late_or_historical_results(platform):
    node = platform.register()
    query = create(platform, node).json()
    with platform.app.state.db.write() as session:
        command = session.get(Command, query["id"])
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert get(platform, node, query).json()['error']['code'] == 'DATASET_INSPECT_TIMEOUT'
    ack(platform, node, query)
    assert get(platform, node, query).json()['error']['code'] == 'DATASET_INSPECT_TIMEOUT'
    fresh = create(platform, node).json()
    assert fresh['id'] != query['id']
    ack(platform, node, fresh)
    with platform.app.state.db.write() as session:
        session.get(Node, node['node_id']).roots = ['/different']
    checked = get(platform, node, fresh).json()
    assert checked['status'] == 'FAILED' and checked['inspection'] is None
    assert checked['error']['code'] == 'PATH_OUTSIDE_ROOTS'


@pytest.mark.parametrize('change', ['path', 'source', 'extra', 'feature-shape', 'unsafe-code', 'oversized'])
def test_invalid_agent_metadata_cannot_enter_form_results(platform, change):
    node = platform.register()
    query = create(platform, node).json()
    value = result()
    if change == 'path': value['path'] = '/machine/other'
    if change == 'source': value['source']['info_sha256'] = 'bad'
    if change == 'extra': value['raw_private_file'] = 'must-not-leak'
    if change == 'feature-shape': value['features'][0]['shape'] = [True, 10]
    if change == 'unsafe-code': value['suggestions'] = {'python': "__import__('os')"}
    if change == 'oversized': value['warnings'] = ['x' * 1001]
    assert ack(platform, node, query, value).status_code == 200
    checked = get(platform, node, query).json()
    assert checked['status'] == 'FAILED' and checked['inspection'] is None
    assert checked['error']['code'] == 'DATASET_INSPECT_INVALID_RESULT'
    assert 'must-not-leak' not in str(checked)


@pytest.mark.parametrize('code,expected', [('COMMAND_UNSUPPORTED', 'DATASET_INSPECT_AGENT_UNSUPPORTED'),
    ('DATASET_INSPECT_PERMISSION_DENIED', 'DATASET_INSPECT_PERMISSION_DENIED'), ('UNEXPECTED_PRIVATE_TRACE', 'DATASET_INSPECT_FAILED')])
def test_old_agents_and_read_failures_return_safe_messages(platform, code, expected):
    node = platform.register();query = create(platform, node).json()
    ack(platform, node, query, {'error': {'code': code, 'message': 'must-not-leak private traceback'}}, 'REJECTED')
    checked = get(platform, node, query).json()
    assert checked['error']['code'] == expected and 'must-not-leak' not in str(checked)


def test_pending_limit_and_unknown_request_fields(platform):
    node = platform.register()
    assert create(platform, node, extra='unsafe').status_code == 422
    for index in range(10): assert create(platform, node, f'/machine/data-{index}').status_code == 202
    assert create(platform, node, '/machine/data-too-many').status_code == 429
    assert len(platform.commands(node)) == 10


@pytest.mark.parametrize('role,status', [('admin', 202), ('operator', 202), ('viewer', 403)])
def test_metadata_access_requires_asset_write_permission_and_csrf(platform, role, status):
    node = platform.register()
    password = 'inspection-test-local-password'
    platform.app.state.auth.create_user('inspect-' + role, password, role)
    client = TestClient(platform.app, base_url='https://testserver')
    try:
        logged = client.post('/api/v1/auth/login', json={'username': 'inspect-' + role, 'password': password})
        assert logged.status_code == 200
        url = f"/api/v1/nodes/{node['node_id']}/dataset-inspections"
        csrf = {'X-CSRF-Token': logged.json()['csrf_token']}
        response = client.post(url, json={'path': '/machine/data'}, headers=csrf)
        assert response.status_code == status, response.text
        if status == 202:
            assert client.get(url + '/' + response.json()['id']).status_code == 200
            assert client.post(url, json={'path': '/machine/data'}).json()['error']['code'] == 'CSRF_REJECTED'
            assert client.post(url, json={'path': '/machine/data'}, headers={**csrf, 'Origin': 'https://foreign.invalid'}).json()['error']['code'] == 'ORIGIN_REJECTED'
    finally: client.close()
