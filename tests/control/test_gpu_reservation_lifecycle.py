"""Account revocation and expiring reservations cannot grant new execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.account_lifecycle import task_execution_authorized, training_execution_authorized
from vla_control.auth_models import User
from vla_control.common import now, offset_time, uid
from vla_control.gpu_reservations import reservation_owners
from vla_control.models import Allocation, Attempt, GPUReservation, Run
from vla_control.policy_deployment_models import PolicyDeployment
from vla_control.settings import Settings

from .conftest import gpu
from .test_gpu_reservations import accounts, login, path, reserve
from .test_dataset_norm_migration import _legacy_table


def queued(platform, node, owner, *, count=0, state="QUEUED"):
    run_id = uid()
    with platform.app.state.db.write() as session:
        session.add(Run(id=run_id, name="account lifecycle", node_id=node["node_id"],
            profile_id="default", template_id="smoke", state=state, owner_user_id=owner,
            created_at=now(), updated_at=now(), inputs={}, parameters={"steps": 3},
            gpu_count=count, code_expectation={}))
    return run_id


def test_limit_counts_all_nodes_and_failed_replace_preserves_previous_reservations(platform):
    first = platform.register(name="first", gpus=[gpu("GPU-1"), gpu("GPU-2"), gpu("GPU-3")])
    second = platform.register(name="second", gpus=[gpu("GPU-4"), gpu("GPU-5")])
    accounts(platform)
    login(platform.client, "bob")
    assert reserve(platform, first, ["GPU-1"]).status_code == 200
    assert reserve(platform, second, ["GPU-4"]).status_code == 200
    assert platform.client.get("/api/v1/gpu-reservation-policy").json() == {
        "max_per_user": 2, "ttl_seconds": 28800, "reserved_count": 2, "remaining_count": 0}
    response = reserve(platform, first, ["GPU-1", "GPU-2"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "GPU_RESERVATION_LIMIT"
    assert [r["gpu_uuid"] for r in platform.client.get(path(first)).json()] == ["GPU-1"]
    assert reserve(platform, first, []).status_code == 200
    assert reserve(platform, second, ["GPU-4", "GPU-5"]).status_code == 200


def test_same_account_concurrent_reservations_on_different_nodes_cannot_exceed_limit(platform):
    nodes = [platform.register(name=f"node-{i}", gpus=[gpu(f"GPU-{i}-1"), gpu(f"GPU-{i}-2")]) for i in range(2)]
    accounts(platform)
    clients = [TestClient(platform.app, base_url="https://testserver") for _ in nodes]
    try:
        for client in clients:
            login(client, "bob")
        barrier = Barrier(2)
        def request(index):
            barrier.wait(timeout=5)
            return clients[index].put(path(nodes[index]), json={"gpu_uuids": [f"GPU-{index}-1", f"GPU-{index}-2"]})
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(request, range(2)))
        assert sorted(r.status_code for r in responses) == [200, 409]
        assert next(r for r in responses if r.status_code == 409).json()["error"]["code"] == "GPU_RESERVATION_LIMIT"
        with platform.app.state.db.read() as session:
            assert len(list(session.scalars(select(GPUReservation)))) == 2
    finally:
        for client in clients:
            client.close()


def test_expired_and_legacy_reservations_do_not_block_or_count_against_limit(platform):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2"), gpu("GPU-3")])
    accounts(platform)
    alice = login(platform.client, "alice")
    with platform.app.state.db.write() as session:
        session.add(GPUReservation(node_id=node["node_id"], gpu_uuid="GPU-1", user_id=alice,
            created_at=offset_time(-90000), expires_at=None))
        session.add(GPUReservation(node_id=node["node_id"], gpu_uuid="GPU-2", user_id=alice,
            created_at=now(), expires_at=offset_time(-1)))
    assert platform.client.get(path(node)).json() == []
    with platform.app.state.db.read() as session:
        assert reservation_owners(session, node["node_id"]) == {}
    response = reserve(platform, node, ["GPU-1", "GPU-2"])
    assert response.status_code == 200
    for row in response.json():
        remaining = datetime.fromisoformat(row["expires_at"]) - datetime.fromisoformat(now())
        assert 28790 < remaining.total_seconds() <= 28800
    login(platform.client, "bob")
    # Explicit expiry makes the physical card available to another account.
    with platform.app.state.db.write() as session:
        for row in session.scalars(select(GPUReservation)):
            row.expires_at = offset_time(-1)
    assert reserve(platform, node, ["GPU-1"]).status_code == 200


def test_legacy_fresh_reservation_expires_eight_hours_after_creation(platform):
    node = platform.register()
    accounts(platform)
    alice = login(platform.client, "alice")
    created = offset_time(-3600)
    with platform.app.state.db.write() as session:
        session.add(GPUReservation(node_id=node["node_id"], gpu_uuid="GPU-1", user_id=alice,
                                   created_at=created, expires_at=None))
    row = platform.client.get(path(node)).json()[0]
    assert (datetime.fromisoformat(row["expires_at"]) - datetime.fromisoformat(created)).total_seconds() == 28800


def test_reservation_expiry_does_not_release_or_preempt_active_training(platform):
    node = platform.register()
    accounts(platform)
    bob = login(platform.client, "bob")
    run_id = queued(platform, node, bob, count=1)
    assert platform.schedule() == 1
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    with platform.app.state.db.write() as session:
        session.get(GPUReservation, (node["node_id"], "GPU-1")).expires_at = offset_time(-1)
    login(platform.client, "alice")
    assert platform.client.get(path(node)).json() == []
    assert reserve(platform, node, ["GPU-1"]).json()["error"]["code"] == "GPU_IN_USE"
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).state == "STARTING"
        assert session.get(Allocation, (node["node_id"], "GPU-1")) is not None
    assert not any(command["type"] == "CANCEL" for command in platform.commands(node))


def test_existing_reservations_migrate_without_gaining_unlimited_lifetime(platform):
    node = platform.register()
    accounts(platform)
    bob = login(platform.client, "bob")
    with platform.app.state.db.write() as session:
        session.add(GPUReservation(node_id=node["node_id"], gpu_uuid="GPU-1", user_id=bob,
                                   created_at=offset_time(-90000), expires_at=None))
    database = platform.app.state.db
    raw = database.engine.raw_connection()
    try:
        raw.commit()
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        _legacy_table(cursor, "gpu_reservations", {"expires_at"})
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    database.initialize()
    with database.read() as session:
        row = session.get(GPUReservation, (node["node_id"], "GPU-1"))
        assert row.user_id == bob and row.expires_at is None
    assert platform.client.get(path(node)).json() == []


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}])
def test_account_revocation_freezes_queue_releases_idle_and_preserves_running_allocation(platform, change):
    node = platform.register(gpus=[gpu("GPU-1"), gpu("GPU-2")])
    accounts(platform)
    bob = login(platform.client, "bob")
    run_id = queued(platform, node, bob)
    active_id = queued(platform, node, bob, count=1, state="RUNNING")
    with platform.app.state.db.write() as session:
        session.add(Attempt(id="active-attempt", run_id=active_id, node_id=node["node_id"], number=1,
            execution_token="existing-execution-token", state="RUNNING", created_at=now(), gpu_uuids=["GPU-1"], last_event_seq=0))
        session.flush()
        session.add(Allocation(node_id=node["node_id"], gpu_uuid="GPU-1", attempt_id="active-attempt", created_at=now()))
    assert reserve(platform, node, ["GPU-1", "GPU-2"]).status_code == 200
    platform.app.state.auth.update_user(None, bob, change)
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).queue_reason in {"ACCOUNT_DISABLED", "ACCOUNT_PERMISSION_REVOKED"}
        assert session.get(GPUReservation, (node["node_id"], "GPU-2")) is None
        assert session.get(GPUReservation, (node["node_id"], "GPU-1")) is not None
    assert platform.schedule() == 0
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).state == "QUEUED"
        assert session.get(Run, run_id).queue_reason == "ACCOUNT_EXECUTION_REVOKED"
        assert session.get(Run, active_id).state == "RUNNING"
        assert session.get(Allocation, (node["node_id"], "GPU-1")) is not None
        assert reservation_owners(session, node["node_id"]) == {}
    assert not any(command["type"] in {"START", "CANCEL"} for command in platform.commands(node))
    platform.app.state.auth.update_user(None, bob, {"enabled": True, "role": "operator"})
    assert platform.schedule() == 1


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}, {"must_change_password": True}])
def test_scheduler_rechecks_account_even_when_revocation_hook_was_not_called(platform, change):
    node = platform.register()
    accounts(platform)
    bob = login(platform.client, "bob")
    run_id = queued(platform, node, bob)
    with platform.app.state.db.write() as session:
        user = session.get(User, bob)
        for field, value in change.items():
            setattr(user, field, value)
    assert platform.schedule() == 0
    with platform.app.state.db.read() as session:
        assert session.get(Run, run_id).queue_reason == "ACCOUNT_EXECUTION_REVOKED"
        assert list(session.scalars(select(Attempt))) == []


def test_legacy_execution_requires_empty_accounts_and_explicit_compatibility(platform):
    node = platform.register()
    queued(platform, node, None)
    with platform.app.state.db.read() as session:
        assert not training_execution_authorized(session, None)
        assert training_execution_authorized(session, None, allow_legacy=True)
    accounts(platform)
    assert platform.schedule() == 0
    with platform.app.state.db.read() as session:
        assert not training_execution_authorized(session, None, allow_legacy=True)


def test_reservation_policy_settings_are_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("VLA_GPU_RESERVATION_MAX_PER_USER", "4")
    monkeypatch.setenv("VLA_GPU_RESERVATION_TTL_SECONDS", "7200")
    settings = Settings.from_env()
    assert settings.gpu_reservation_max_per_user == 4 and settings.gpu_reservation_ttl_seconds == 7200
    for changes in ({"gpu_reservation_max_per_user": 0}, {"gpu_reservation_max_per_user": 257},
                    {"gpu_reservation_ttl_seconds": 59}, {"gpu_reservation_ttl_seconds": 7 * 86400 + 1}):
        with pytest.raises(ValueError):
            replace(Settings(enrollment_token="independent-enrollment"), **changes).validate()


@pytest.mark.parametrize("case,expected", [("admin-disabled-owner", True), ("demoted-admin", False),
                                         ("owner-operator", True), ("admin-legacy-owner", True)])
def test_explicit_execution_actor_requires_current_admin_for_another_owner(platform, case, expected):
    accounts(platform)
    with platform.app.state.db.write() as session:
        alice = session.scalar(select(User).where(User.username == "alice"))
        bob = session.scalar(select(User).where(User.username == "bob"))
        actor, owner = alice.id, bob.id
        if case == "admin-disabled-owner":
            bob.enabled = False
        elif case == "demoted-admin":
            alice.role = "operator"
        elif case == "owner-operator":
            actor = bob.id
        else:
            owner = None
        assert task_execution_authorized(session, owner, actor) is expected


def test_revoking_intervention_actor_preserves_its_active_policy_reservation(platform):
    node = platform.register()
    accounts(platform)
    alice = login(platform.client, "alice")
    platform.app.state.auth.create_user("backup-admin", "backup-admin-test-password", "admin")
    with platform.app.state.db.write() as session:
        bob = session.scalar(select(User).where(User.username == "bob")).id
        session.add(PolicyDeployment(id=uid(), name="explicit administrator intervention", node_id=node["node_id"],
            profile_id="default", checkpoint_path="/machine/checkpoint", config_name="pi0_base", port=8000,
            owner_user_id=bob, execution_user_id=alice, state="RUNNING", gpu_uuid="GPU-1",
            created_at=now(), updated_at=now()))
    login(platform.client, "bob")
    assert reserve(platform, node, ["GPU-1"]).json()["error"]["code"] == "GPU_IN_USE"
    login(platform.client, "alice")
    assert reserve(platform, node, ["GPU-1"]).status_code == 200
    platform.app.state.auth.update_user(None, alice, {"enabled": False})
    with platform.app.state.db.read() as session:
        assert session.get(GPUReservation, (node["node_id"], "GPU-1")) is not None
        assert reservation_owners(session, node["node_id"]) == {}
