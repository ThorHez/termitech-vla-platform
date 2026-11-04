"""Training sharing requires an explicit, frozen budget and live admission checks."""

from copy import deepcopy

import pytest
from sqlalchemy import select
from vla_protocol import canonical_digest
from vla_protocol.training_gpu_policy import TRAINING_SHARED_GPU_CAPABILITY

from vla_control.models import Allocation, GPUReservation, Node, Preview
from vla_control.common import now, offset_time
from vla_control.policy_deployment_models import PolicyDeployment

from .conftest import ADMIN, automatic_checks, automatic_preflight, confirmed_run_request, gpu, snapshot
from .test_task_recipe import inputs


POLICY = {"mode": "shared", "memory_mb": 51200}


def occupied(uuid="GPU-7", **changes):
    return gpu(uuid, memory_total_mb=81920, memory_used_mb=20480, memory_free_mb=61440,
               processes=[{"pid": 546536, "kind": "compute", "ownership": "external"}], **changes)


def node(platform, rows=None, *, capability=True):
    row = platform.register(adapter="openpi-sft", gpus=[occupied()] if rows is None else rows)
    if capability:
        with platform.app.state.db.write() as session:
            session.get(Node, row["node_id"]).metrics = {"capabilities": [TRAINING_SHARED_GPU_CAPABILITY]}
    return row


def request(platform, worker, **changes):
    return platform.client.post('/api/v1/run-previews', headers=ADMIN, json={
        "name": "shared GPU training", "node_id": worker["node_id"], "profile_id": "default",
        "template_id": "openpi-sft", "recipe": "shared_recipe", "inputs": inputs(platform, worker),
        "parameters": {"steps": 3, "batch_size": 2}, "gpu_count": 1,
        "requested_gpu_uuids": ["GPU-7"], "training_gpu_policy": POLICY, **changes})


def pending(platform, worker, **changes):
    response = request(platform, worker, **changes)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(c for c in platform.commands(worker) if c["type"] == "PREFLIGHT" and c["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    code = snapshot(recipe=payload["recipe"], automatic_preflight=automatic_preflight(payload))
    if payload.get("training_gpu_policy") is not None:
        policy = payload["training_gpu_policy"]
        code["training_gpu_policy"] = deepcopy(policy)
        selected = payload["requested_gpu_uuids"]
        with platform.app.state.db.read() as session:
            inventory = session.get(Node, worker["node_id"]).gpus
        code["training_gpu_check"] = {"policy": deepcopy(policy), "gpu_uuids": list(selected),
                                      "devices": [deepcopy(next(g for g in inventory if g["uuid"] == uuid)) for uuid in selected]}
        plan = code["automatic_preflight"]["plan"]
        plan["training_gpu_policy"] = deepcopy(policy)
        code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    return identifier, command, code


def acknowledge(platform, worker, identifier, command, code):
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=worker["headers"],
        json={"node_id": worker["node_id"], "status": "APPLIED", "result": {"checks": automatic_checks(), "code_snapshot": code}})
    assert response.status_code == 200, response.text
    return platform.client.get(f'/api/v1/run-previews/{identifier}', headers=ADMIN).json()


def ready(platform, worker, **changes):
    parts = pending(platform, worker, **changes)
    preview = acknowledge(platform, worker, *parts)
    assert preview["status"] == "READY", preview
    return preview


def submit(platform, preview):
    return platform.client.post('/api/v1/runs', headers={**ADMIN, 'Idempotency-Key': preview['id']},
                                json=confirmed_run_request(preview))


def test_external_compute_can_schedule_only_with_explicit_frozen_shared_policy(platform):
    worker = node(platform)
    preview = ready(platform, worker)
    assert preview['request']['training_gpu_policy'] == POLICY
    response = submit(platform, preview)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run['training_gpu_policy'] == run['code_expectation']['training_gpu_policy'] == POLICY
    assert platform.schedule() == 1
    start = next(c for c in platform.commands(worker) if c['type'] == 'START')
    assert start['payload']['training_gpu_policy'] == POLICY
    assert start['payload']['gpu_uuids'] == ['GPU-7']
    with platform.app.state.db.read() as session:
        assert session.scalar(select(Allocation).where(Allocation.node_id == worker['node_id'])).gpu_uuid == 'GPU-7'


def test_legacy_exclusive_shape_still_waits_for_an_occupied_card(platform):
    worker = node(platform)
    identifier, command, code = pending(platform, worker, training_gpu_policy={'mode': 'exclusive'})
    assert 'training_gpu_policy' not in command['payload']
    preview = acknowledge(platform, worker, identifier, command, code)
    assert preview['status'] == 'READY'
    assert 'training_gpu_policy' not in preview['request']
    response = submit(platform, preview)
    assert response.status_code == 201, response.text
    assert 'training_gpu_policy' not in response.json()
    assert platform.schedule() == 0
    assert platform.get_run(response.json())['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'


@pytest.mark.parametrize('changes', [
    {'training_gpu_policy': {'mode': 'shared'}},
    {'training_gpu_policy': {'mode': 'shared', 'memory_mb': True}},
    {'training_gpu_policy': {'mode': 'shared', 'memory_mb': 255}},
    {'training_gpu_policy': {**POLICY, 'allow_existing': True}},
    {'requested_gpu_uuids': None}, {'template_id': 'smoke'},
    {'template_id': 'openpi-norm', 'gpu_count': 0, 'requested_gpu_uuids': None},
])
def test_invalid_or_implicit_sharing_cannot_reach_agent(platform, changes):
    worker = node(platform)
    response = request(platform, worker, **changes)
    assert response.status_code == 422, response.text
    assert platform.commands(worker) == []


def test_older_agent_cannot_silently_ignore_shared_training_policy(platform):
    worker = node(platform, capability=False)
    response = request(platform, worker)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'AGENT_TRAINING_SHARED_GPU_UNSUPPORTED'
    assert platform.commands(worker) == []


@pytest.mark.parametrize('damage', ['missing_policy', 'missing_evidence', 'wrong_gpu', 'different_budget', 'insufficient_memory', 'unknown_process', 'plan_budget'])
def test_incomplete_or_mismatched_shared_preflight_cannot_authorize_start(platform, damage):
    worker = node(platform)
    identifier, command, code = pending(platform, worker)
    if damage == 'missing_policy':code.pop('training_gpu_policy')
    elif damage == 'missing_evidence':code.pop('training_gpu_check')
    elif damage == 'wrong_gpu':code['training_gpu_check']['gpu_uuids'] = ['GPU-other']
    elif damage == 'different_budget':code['training_gpu_check']['policy']['memory_mb'] = 1024
    elif damage == 'insufficient_memory':code['training_gpu_check']['devices'][0]['memory_free_mb'] = 52000
    elif damage == 'unknown_process':code['training_gpu_check']['devices'][0]['processes'][0]['kind'] = 'unknown'
    elif damage == 'plan_budget':
        code['automatic_preflight']['plan']['training_gpu_policy']['memory_mb'] = 1024
        code['automatic_preflight']['plan_digest'] = canonical_digest(code['automatic_preflight']['plan'])
    preview = acknowledge(platform, worker, identifier, command, code)
    assert preview['status'] == 'FAILED', preview
    assert submit(platform, preview).status_code == 409


def test_submission_rechecks_budget_evidence_in_previously_ready_preview(platform):
    worker = node(platform)
    preview = ready(platform, worker)
    with platform.app.state.db.write() as session:
        row = session.get(Preview, preview['id'])
        code = deepcopy(row.code_snapshot);code['training_gpu_check']['policy']['memory_mb'] = 256;row.code_snapshot = code
    response = submit(platform, preview)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'AGENT_TRAINING_GPU_CHECK_INCOMPLETE'


@pytest.mark.parametrize('changed', [
    {'memory_free_mb': 52000}, {'processes': None},
    {'processes': [{'pid': 1, 'kind': 'compute', 'ownership': 'platform'}]},
])
def test_scheduler_rechecks_actual_remaining_memory_and_processes(platform, changed):
    worker = node(platform)
    preview = ready(platform, worker);response = submit(platform, preview);assert response.status_code == 201
    with platform.app.state.db.write() as session:
        row = session.get(Node, worker['node_id']);row.gpus = [{**row.gpus[0], **changed}]
    assert platform.schedule() == 0
    assert platform.get_run(response.json())['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'


def test_platform_allocation_cannot_be_shared_even_when_memory_is_available(platform):
    worker = node(platform)
    first = submit(platform, ready(platform, worker));assert first.status_code == 201
    assert platform.schedule() == 1
    second = submit(platform, ready(platform, worker));assert second.status_code == 201
    assert platform.schedule() == 0
    assert platform.get_run(second.json())['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'


def test_shared_run_does_not_bypass_active_inference_reservation(platform):
    worker = node(platform)
    preview = ready(platform, worker);response = submit(platform, preview);assert response.status_code == 201
    # Existing tests use bare ORM fixtures to exercise admission against an active resource state.
    with platform.app.state.db.write() as session:
        session.add(PolicyDeployment(id='occupied-policy', name='active inference', node_id=worker['node_id'], profile_id='default',
            checkpoint_path='/machine/model', config_name='shared_recipe', port=8765, state='RUNNING',
            created_at='2026-01-01', updated_at='2026-01-01', gpu_uuid='GPU-7'))
    assert platform.schedule() == 0
    assert platform.get_run(response.json())['queue_reason'] == 'REQUESTED_GPU_UNAVAILABLE'


def test_another_users_reservation_cannot_be_borrowed_by_shared_training(platform):
    worker = node(platform)
    platform.client.base_url = 'https://testserver'
    platform.app.state.auth.create_user('training-owner', 'training-owner-password-123', 'admin')
    user = platform.app.state.auth.create_user('reserved-owner', 'reserved-owner-password-123', 'operator')
    login = platform.client.post('/api/v1/auth/login', json={'username': 'training-owner', 'password': 'training-owner-password-123'})
    assert login.status_code == 200
    platform.client.headers['X-CSRF-Token'] = login.json()['csrf_token']
    preview = ready(platform, worker);response = submit(platform, preview);assert response.status_code == 201
    with platform.app.state.db.write() as session:
        session.add(GPUReservation(node_id=worker['node_id'], gpu_uuid='GPU-7', user_id=user['id'],
                                   created_at=now(), expires_at=offset_time(3600)))
    assert platform.schedule() == 0
    assert platform.get_run(response.json())['queue_reason'] == 'REQUESTED_GPU_RESERVED'
    assert request(platform, worker).json()['error']['code'] == 'GPU_RESERVED_BY_OTHER_USER'


def test_shared_submission_cannot_override_frozen_budget(platform):
    worker = node(platform)
    preview = ready(platform, worker)
    response = platform.client.post('/api/v1/runs', headers={**ADMIN, 'Idempotency-Key': preview['id']},
        json={**confirmed_run_request(preview), 'training_gpu_policy': {'mode': 'shared', 'memory_mb': 1024}})
    assert response.status_code == 422
    result = submit(platform, preview)
    assert result.status_code == 201 and result.json()['training_gpu_policy'] == POLICY


def test_scheduler_cannot_dispatch_shared_run_after_agent_capability_is_lost(platform):
    worker = node(platform)
    result = submit(platform, ready(platform, worker));assert result.status_code == 201
    with platform.app.state.db.write() as session:
        session.get(Node, worker['node_id']).metrics = {}
    assert platform.schedule() == 0
    assert platform.get_run(result.json())['queue_reason'] == 'TRAINING_SHARED_GPU_UNSUPPORTED'


def test_explicit_card_order_is_bound_to_gpu_preflight_evidence(platform):
    worker = node(platform, [occupied('GPU-7'), occupied('GPU-4')])
    identifier, command, code = pending(platform, worker, requested_gpu_uuids=['GPU-7', 'GPU-4'], gpu_count=2)
    assert command['payload']['requested_gpu_uuids'] == ['GPU-7', 'GPU-4']
    code['training_gpu_check']['devices'].reverse()
    preview = acknowledge(platform, worker, identifier, command, code)
    assert preview['status'] == 'FAILED'
    assert preview['error']['code'] == 'AGENT_TRAINING_GPU_CHECK_INCOMPLETE'
