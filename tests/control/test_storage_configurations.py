"""Storage changes persist only with matching Agent proof on an idle node."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import APIError, now, offset_time, uid
from vla_control.models import Command, Node, Preview, Run
from vla_control.profile_rebindings import require_not_pending

from .conftest import ADMIN


def body(**changes):
    return {"expected_output_root": "/machine/output", "expected_cache_root": None,
            "output_root": "/machine/output-2", "cache_root": "/machine/cache-2", **changes}


def path(node):
    return f"/api/v1/nodes/{node['node_id']}/storage-configurations"


def create(platform, node, value=None, headers=ADMIN):
    return platform.client.post(path(node), headers=headers, json=body() if value is None else value)


def get(platform, node, operation):
    return platform.client.get(path(node) + '/' + operation['id'], headers=ADMIN)


def receipt(**changes):
    return {"persisted": True, "output_root": body()["output_root"], "cache_root": body()["cache_root"],
            "directories": {"output": {"path": body()["output_root"], "free_bytes": 1024, "total_bytes": 2048},
                            "cache": {"path": body()["cache_root"], "free_bytes": 4096, "total_bytes": 8192}}, **changes}


def ack(platform, node, operation, result=None, status='APPLIED'):
    return platform.client.post(f"/agent/v1/commands/{operation['id']}/ack", headers=node['headers'],
        json={'node_id': node['node_id'], 'status': status, 'result': receipt() if result is None else result})


def current(platform, node):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()


def add_run(session, node, state):
    run = Run(id=uid(), node_id=node['node_id'], profile_id='default', name='old run', template_id='smoke',
              state=state, created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1,
              code_expectation={}, queue_reason=None, error=None)
    session.add(run)
    return run


def test_persisted_storage_ack_updates_node_invalidates_old_preview_and_reuses_id(platform):
    node = platform.register()
    preview = platform.preview(node)
    before = current(platform, node)
    queued = create(platform, node)
    assert queued.status_code == 202
    operation = queued.json()
    assert operation['status'] == 'PENDING' and operation['storage'] is None
    assert create(platform, node).json() == operation
    assert current(platform, node)['output_root'] == before['output_root']
    value = receipt(private_output='must not be persisted')
    assert ack(platform, node, operation, value).status_code == 200
    ready = get(platform, node, operation).json()
    assert ready['status'] == 'READY'
    assert ready['storage'] == {key: receipt()[key] for key in ('output_root', 'cache_root', 'directories')}
    updated = current(platform, node)
    assert updated['output_root'] == body()['output_root'] and updated['cache_root'] == body()['cache_root']
    assert create(platform, node).json() == ready
    assert ack(platform, node, operation, value).status_code == 200
    assert ack(platform, node, operation, receipt(output_root='/machine/other')).status_code == 409
    with platform.app.state.db.read() as session:
        old = session.get(Preview, preview['id'])
        assert old.status == 'FAILED' and old.error['code'] == 'STORAGE_CHANGED_SINCE_PREFLIGHT'
        assert 'private_output' not in session.get(Command, operation['id']).result['agent_result']


def test_pending_change_blocks_new_training_writeback_and_runtime_changes(platform):
    node = platform.register()
    create(platform, node)
    response = platform.client.post('/api/v1/run-previews', headers=ADMIN, json={
        'name':'blocked', 'node_id':node['node_id'], 'profile_id':'default', 'template_id':'smoke',
        'inputs':{}, 'parameters':{}, 'gpu_count':1})
    assert response.status_code == 409 and response.json()['error']['code'] == 'STORAGE_CONFIGURE_PENDING'
    with platform.app.state.db.read() as session:
        with pytest.raises(APIError) as error:
            require_not_pending(session, node['node_id'], 'default', '/machine/code')
        assert error.value.code == 'STORAGE_CONFIGURE_PENDING'
    management = platform.client.get(f"/api/v1/nodes/{node['node_id']}/management", headers=ADMIN).json()
    assert management['can_disconnect'] is False


@pytest.mark.parametrize('state', ['QUEUED','STARTING','PREPARING','RUNNING','CANCELLING','STOPPING','UNKNOWN'])
def test_any_nonterminal_run_blocks_storage_changes(platform, state):
    node = platform.register()
    with platform.app.state.db.write() as session:
        add_run(session, node, state)
    response = create(platform, node)
    assert response.status_code == 409 and response.json()['error']['code'] == 'STORAGE_CONFIGURE_BUSY'


@pytest.mark.parametrize('kind', ['START','PREFLIGHT','RECIPE_WRITE','PROFILE_REBIND','POLICY_EXPORT'])
def test_pending_agent_operations_block_storage_changes(platform, kind):
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.add(Command(id=uid(), node_id=node['node_id'], type=kind, status='PENDING', payload={}, created_at=now()))
    response = create(platform, node)
    assert response.status_code == 409 and response.json()['error']['code'] == 'STORAGE_CONFIGURE_BUSY'


@pytest.mark.parametrize('change', [
    {'output_root':'relative'}, {'cache_root':'/outside/cache'}, {'output_root':'/machine/../other'},
    {'cache_root':'/machine/cache\n'}, {'cache_root':'/'}, {'expected_output_root':'bad'},
    {'output_root':'/machine/out\x00put'}, {'extra':True}, {'cache_root':''},
])
def test_invalid_or_unmanaged_paths_never_dispatch(platform, change):
    node = platform.register()
    response = create(platform, node, body(**change))
    assert response.status_code == 422
    assert platform.commands(node) == []


def test_cas_conflicts_and_distinct_pending_changes_are_rejected(platform):
    node = platform.register()
    stale = create(platform, node, body(expected_output_root='/machine/old'))
    assert stale.status_code == 409 and stale.json()['error']['code'] == 'STORAGE_CONFIGURE_CONFLICT'
    create(platform, node)
    different = create(platform, node, body(output_root='/machine/third'))
    assert different.status_code == 409 and different.json()['error']['code'] == 'STORAGE_CONFIGURE_PENDING'


@pytest.mark.parametrize('mutate', [
    lambda value:value.update(persisted=False), lambda value:value.update(output_root='/machine/wrong'),
    lambda value:value['directories']['output'].update(path='/machine/wrong'),
    lambda value:value['directories']['cache'].update(free_bytes=True),
    lambda value:value['directories']['output'].update(free_bytes=4096),
    lambda value:value['directories']['output'].update(total_bytes=-1),
    lambda value:value.update(directories={}),
])
def test_invalid_or_uncertain_receipts_keep_change_pending(platform, mutate):
    node = platform.register(); operation = create(platform, node).json()
    value = deepcopy(receipt()); mutate(value)
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, node, operation).json()['status'] == 'PENDING'
    assert current(platform, node)['output_root'] == '/machine/output'


def test_expiry_and_uncertain_failure_do_not_unlock_potentially_persisted_change(platform):
    node = platform.register(); operation = create(platform, node).json()
    with platform.app.state.db.write() as session:
        command = session.get(Command, operation['id'])
        command.payload = {**command.payload, 'expires_at':offset_time(-1)}
    assert get(platform, node, operation).json()['status'] == 'PENDING'
    assert ack(platform, node, operation, {'error':{'code':'STORAGE_CONFIGURE_FAILED'}}, status='REJECTED').status_code == 200
    assert get(platform, node, operation).json()['status'] == 'PENDING'
    assert ack(platform, node, operation).status_code == 200
    assert get(platform, node, operation).json()['status'] == 'READY'


def test_proven_unmodified_failure_unlocks_without_accepting_agent_message(platform):
    node = platform.register(); operation = create(platform, node).json()
    value = {'error':{'code':'STORAGE_CONFIGURE_UNWRITABLE','persisted':False,'message':'arbitrary text'}}
    assert ack(platform, node, operation, value, status='REJECTED').status_code == 200
    failed = get(platform, node, operation).json()
    assert failed['status'] == 'FAILED' and failed['error']['code'] == 'STORAGE_CONFIGURE_UNWRITABLE'
    assert 'arbitrary text' not in str(failed)
    assert create(platform, node).json()['id'] != operation['id']


def test_old_agents_explicitly_report_unsupported_and_cannot_change_paths(platform):
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.get(Node, node['node_id']).metrics = {'capabilities':['PREFLIGHT']}
    response = create(platform, node)
    assert response.status_code == 409 and response.json()['error']['code'] == 'STORAGE_CONFIGURE_UNSUPPORTED'
    with platform.app.state.db.write() as session:
        session.get(Node, node['node_id']).metrics = {}
    operation = create(platform, node).json()
    assert ack(platform, node, operation, {'error':{'code':'COMMAND_UNSUPPORTED'}}, status='REJECTED').status_code == 200
    assert get(platform, node, operation).json()['error']['code'] == 'STORAGE_CONFIGURE_UNSUPPORTED'


def test_cache_root_can_return_to_project_default(platform):
    node = platform.register()
    operation = create(platform, node, body(cache_root=None)).json()
    value = receipt(cache_root=None, directories={'output':receipt()['directories']['output'], 'cache':None})
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, node, operation).json()['storage']['cache_root'] is None


def test_operation_requires_machine_management_permission_and_csrf_reauthentication(platform):
    node = platform.register(); password = 'storage-directory-admin-password'
    for role in ['admin','operator','viewer']:
        platform.app.state.auth.create_user('storage-' + role, password, role)
        with TestClient(platform.app, base_url='https://testserver') as client:
            session = client.post('/api/v1/auth/login', json={'username':'storage-' + role, 'password':password}).json()
            client.headers['X-CSRF-Token'] = session['csrf_token']
            response = client.post(path(node), json=body())
            assert response.status_code == 403
            assert response.json()['error']['code'] == ('REAUTH_REQUIRED' if role == 'admin' else 'FORBIDDEN')
            if role == 'admin':
                assert client.post('/api/v1/auth/reauth', json={'password':password}).status_code == 200
                assert client.post(path(node), json=body()).status_code == 202
            else:
                assert client.get(path(node)).status_code == 403


def test_wrong_node_cannot_read_or_ack_another_nodes_operation(platform):
    node = platform.register(); other = platform.register(name='other')
    operation = create(platform, node).json()
    assert get(platform, other, operation).status_code == 404
    assert ack(platform, other, operation).status_code == 403
    assert create(platform, node, headers={}).status_code == 401
    assert create(platform, node, headers=node['headers']).status_code == 401


def test_concurrent_distinct_changes_enqueue_one_command(platform):
    node = platform.register(); barrier = Barrier(2)
    def submit(index):
        barrier.wait(timeout=5)
        return create(platform, node, body(output_root=f'/machine/target-{index}'))
    with ThreadPoolExecutor(max_workers=2) as workers:
        responses = list(workers.map(submit, range(2)))
    assert sorted(response.status_code for response in responses) == [202,409]
    assert len([row for row in platform.commands(node) if row['type'] == 'STORAGE_CONFIGURE']) == 1


def test_concurrent_identical_changes_reuse_one_durable_command(platform):
    node = platform.register(); barrier = Barrier(2)
    def submit(_):
        barrier.wait(timeout=5)
        return create(platform, node)
    with ThreadPoolExecutor(max_workers=2) as workers:
        responses = list(workers.map(submit, range(2)))
    assert all(response.status_code == 202 for response in responses)
    assert len({response.json()['id'] for response in responses}) == 1


@pytest.mark.parametrize('change', ['node', 'run'])
def test_late_persisted_receipt_stays_locked_when_state_changed_or_work_became_active(platform, change):
    node = platform.register(); operation = create(platform, node).json()
    with platform.app.state.db.write() as session:
        if change == 'node':
            session.get(Node, node['node_id']).output_root = '/machine/manual'
        else:
            add_run(session, node, 'UNKNOWN')
    assert ack(platform, node, operation).status_code == 200
    checked = get(platform, node, operation).json()
    assert checked['status'] == 'PENDING' and checked['error']['code'] == 'STORAGE_CONFIGURE_CONFIRMATION_PENDING'


def test_unstructured_agent_error_cannot_break_ack_parser_or_inject_message(platform):
    node = platform.register(); operation = create(platform, node).json()
    assert ack(platform, node, operation, {'error':{'code':{}, 'persisted':False}}, status='REJECTED').status_code == 200
    assert get(platform, node, operation).json()['error']['code'] == 'STORAGE_CONFIGURE_FAILED'


@pytest.mark.parametrize('mode', ['missing', 'wrong_output', 'wrong_cache', 'old_cache_paths'])
def test_openpi_preflight_cannot_authorize_stale_or_ignored_storage_configuration(platform, mode):
    from .test_automatic_preflight import acknowledge, pending
    node, identifier, command, code = pending(platform)
    with platform.app.state.db.write() as session:
        machine = session.get(Node, node['node_id'])
        machine.cache_root = '/machine/current-cache'
    if mode != 'missing':
        code['automatic_preflight']['storage_configuration'] = {
            'output_root':'/machine/output', 'cache_root':'/machine/current-cache'}
        if mode == 'wrong_output':
            code['automatic_preflight']['storage_configuration']['output_root'] = '/machine/old-output'
        if mode == 'wrong_cache':
            code['automatic_preflight']['storage_configuration']['cache_root'] = '/machine/old-cache'
    checked = acknowledge(platform, node, identifier, command, code)
    expected = 'AGENT_STORAGE_CONFIGURATION_UNSUPPORTED' if mode == 'missing' else 'STORAGE_CHANGED_SINCE_PREFLIGHT'
    assert checked['status'] == 'FAILED' and checked['error']['code'] == expected


def test_openpi_storage_configuration_is_preserved_and_matches_effective_cache(platform):
    from vla_protocol import canonical_digest
    from .test_automatic_preflight import acknowledge, pending
    node, identifier, command, code = pending(platform)
    cache_root = '/machine/current-cache'
    with platform.app.state.db.write() as session:
        session.get(Node, node['node_id']).cache_root = cache_root
    evidence = code['automatic_preflight']
    evidence['storage_configuration'] = {'output_root':'/machine/output', 'cache_root':cache_root}
    cache = evidence['plan']['training_cache']
    cache['datasets_cache'] = cache_root + '/huggingface/datasets'
    cache['paths'] = {'HF_DATASETS_CACHE':cache['datasets_cache']}
    for builder in cache['builders']:
        builder['cache_dir'] = cache['datasets_cache'] + '/parquet/default'
    evidence['plan_digest'] = canonical_digest(evidence['plan'])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked['status'] == 'READY'
    assert checked['code_snapshot']['automatic_preflight']['storage_configuration'] == evidence['storage_configuration']


def test_additive_database_upgrade_preserves_old_machine_and_business_data(platform):
    from sqlalchemy import inspect
    node = platform.register()
    before = current(platform, node)
    with platform.app.state.db.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE nodes DROP COLUMN cache_root')
    platform.app.state.db.initialize()
    assert 'cache_root' in {item['name'] for item in inspect(platform.app.state.db.engine).get_columns('nodes')}
    assert current(platform, node) == before
