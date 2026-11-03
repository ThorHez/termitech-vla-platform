import json
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

from vla_agent.resource_guard import filter_inventory, protected_gpus, require_gpu_access
from vla_agent.safety import AgentError


NODE = "0fcb9ef5-8fc7-4feb-9603-44fc05dae5cc"
OWNER = "2deed412-593b-4478-ab53-6397539ded62"
OTHER_OWNER = "33de1368-8972-4896-8cae-e7c0a859f82f"


@pytest.fixture
def guarded(tmp_path):
    path = tmp_path / "production.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE nodes(id TEXT PRIMARY KEY,gpus TEXT);
            CREATE TABLE gpu_reservations(
                node_id TEXT NOT NULL,gpu_uuid TEXT NOT NULL,user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,PRIMARY KEY(node_id,gpu_uuid));
            CREATE TABLE gpu_allocations(node_id TEXT,gpu_uuid TEXT);
            CREATE TABLE policy_deployments(node_id TEXT,gpu_uuid TEXT,state TEXT);
        """)
        db.execute("INSERT INTO nodes VALUES (?,?)", (NODE, json.dumps([
            {"uuid": "GPU-free"}, {"uuid": "GPU-reserved"}, {"uuid": "GPU-assigned"}, {"uuid": "GPU-policy"}])))
        db.execute("INSERT INTO gpu_reservations VALUES (?,?,?,?)", (NODE, "GPU-reserved", OWNER, "2026-11-03"))
        db.execute("INSERT INTO gpu_allocations VALUES (?,?)", (NODE, "GPU-assigned"))
        db.execute("INSERT INTO policy_deployments VALUES (?,?,?)", (NODE, "GPU-policy", "STARTING"))
    return {"resource_guard": {"database": str(path), "node_id": NODE}}


def test_no_guard_keeps_existing_agent_behavior():
    inventory = [{"uuid": "GPU-a"}]
    assert filter_inventory({}, inventory) == (inventory, None)
    assert protected_gpus({}) is None
    require_gpu_access({}, ["GPU-a"])


@pytest.mark.parametrize("gpu", ["GPU-reserved", "GPU-assigned", "GPU-policy"])
def test_production_reservations_allocations_and_inference_are_rejected(guarded, gpu):
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, [gpu])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"
    require_gpu_access(guarded, ["GPU-free"])


def test_inventory_retains_truthful_telemetry_only_for_eligible_test_devices(guarded):
    inventory = [{"uuid": gpu, "memory_used_mb": 12, "processes": []}
                 for gpu in ("GPU-free", "GPU-reserved", "GPU-assigned", "GPU-policy")]
    actual, status = filter_inventory(guarded, inventory)
    assert actual == inventory[:1]
    assert status["protected_gpus"] == 3 and status["total_gpus"] == 4
    assert len(inventory) == 4  # Never mutate the physical telemetry object.


def test_reads_live_wal_without_modifying_production_rows(guarded):
    path = Path(guarded["resource_guard"]["database"])
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO gpu_reservations VALUES (?,?,?,?)", (NODE, "GPU-free", OWNER, "2026-11-03"))
        writer.commit()
        before = list(writer.iterdump())
        with pytest.raises(AgentError) as error:
            require_gpu_access(guarded, ["GPU-free"])
        assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"
        assert list(writer.iterdump()) == before


@pytest.mark.parametrize("failure", ["missing", "corrupt", "schema", "wrong_node", "symlink"])
def test_unverifiable_authority_fails_closed(guarded, failure, tmp_path):
    path = Path(guarded["resource_guard"]["database"])
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"not sqlite")
    elif failure == "schema":
        with sqlite3.connect(path) as db:
            db.execute("DROP TABLE gpu_reservations")
    elif failure == "wrong_node":
        guarded["resource_guard"]["node_id"] = "29da3dfc-ff41-4b65-a46f-5794e522cb30"
    else:
        alias = tmp_path / "alias.db"
        alias.symlink_to(path)
        guarded["resource_guard"]["database"] = str(alias)
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-free"])
    assert error.value.code == "RESOURCE_GUARD_UNAVAILABLE"
    inventory, status = filter_inventory(guarded, [{"uuid": "GPU-free"}])
    assert inventory == [] and status["status"] == "UNAVAILABLE"


def test_unknown_physical_device_is_not_accepted(guarded):
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-not-on-this-host"])
    assert error.value.code == "RESOURCE_GUARD_SCOPE_MISMATCH"


def test_configured_cross_instance_gpu_locks_require_same_inode(guarded, tmp_path):
    source = tmp_path / "production-locks"
    target = tmp_path / "test-state" / "locks"
    source.mkdir()
    target.mkdir(parents=True)
    guarded["state_dir"] = str(target.parent)
    guarded["resource_guard"]["lock_directory"] = str(source)
    for gpu in ("GPU-free", "GPU-reserved", "GPU-assigned", "GPU-policy"):
        name = "gpu-" + hashlib.sha256(gpu.encode()).hexdigest() + ".lock"
        (source / name).touch()
        os.link(source / name, target / name)
    require_gpu_access(guarded, ["GPU-free"])
    name = "gpu-" + hashlib.sha256(b"GPU-free").hexdigest() + ".lock"
    (target / name).unlink()
    (target / name).touch()
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-free"])
    assert error.value.code == "RESOURCE_GUARD_UNAVAILABLE"


def test_agent_rejects_start_before_registering_or_spawning_attempt(guarded, agent_fixture):
    from vla_agent.main import Agent
    config = {**agent_fixture["config"], **guarded}
    agent = Agent(config)
    payload = {**agent_fixture["payload"], "gpu_uuids": ["GPU-reserved"]}
    response = agent.handle_command({"id": "guarded-start", "type": "START", "payload": payload})
    assert response["status"] == "REJECTED"
    assert response["result"]["error"]["code"] == "GPU_RESERVED_BY_PRODUCTION"
    assert agent.state.attempt(payload["attempt_id"]) is None
    assert agent.runners == {}


def test_policy_worker_rechecks_guard_before_importing_model(guarded):
    from vla_agent.policy_deployment_worker import check_gpu_before_load
    import vla_agent
    import vla_protocol
    manifest = {"gpu_uuid": "GPU-reserved", "gpu_policy": {"mode": "exclusive", "memory_mb": None},
                "agent_python": sys.executable,
                "agent_root": str(Path(vla_agent.__file__).resolve().parent.parent),
                "protocol_root": str(Path(vla_protocol.__file__).resolve().parent.parent), **guarded}
    with pytest.raises(ValueError, match="GPU_RESERVED_BY_PRODUCTION"):
        check_gpu_before_load(manifest)


def share_reserved_card(guarded):
    guarded["resource_guard"]["shared_reservations"] = {
        "owner_id": OWNER, "gpu_uuids": ["GPU-reserved"]}


def test_only_explicit_same_owner_reservation_can_be_shared_without_db_writes(guarded):
    share_reserved_card(guarded)
    path = guarded["resource_guard"]["database"]
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("INSERT INTO gpu_reservations VALUES (?,?,?,?)", (NODE, "GPU-free", OWNER, "2026-11-03"))
        db.commit()
        before = list(db.iterdump())
        require_gpu_access(guarded, ["GPU-reserved"])
        inventory, status = filter_inventory(guarded, [{"uuid": "GPU-reserved"}, {"uuid": "GPU-free"}])
        assert inventory == [{"uuid": "GPU-reserved"}]
        assert status["status"] == "AVAILABLE"
        with pytest.raises(AgentError, match="正式环境预约或分配") as error:
            require_gpu_access(guarded, ["GPU-free"])
        assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"
        assert list(db.iterdump()) == before


def test_explicit_card_owned_by_other_user_stays_blocked(guarded):
    share_reserved_card(guarded)
    require_gpu_access(guarded, ["GPU-reserved"])
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("UPDATE gpu_reservations SET user_id=? WHERE node_id=? AND gpu_uuid=?",
                   (OTHER_OWNER, NODE, "GPU-reserved"))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"


def test_shared_reservation_never_overrides_training_allocation(guarded):
    share_reserved_card(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("INSERT INTO gpu_allocations VALUES (?,?)", (NODE, "GPU-reserved"))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"


@pytest.mark.parametrize("state", ["PENDING", "STARTING", "RUNNING", "STOPPING", "UNKNOWN"])
def test_shared_reservation_never_overrides_active_inference(guarded, state):
    share_reserved_card(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("INSERT INTO policy_deployments VALUES (?,?,?)", (NODE, "GPU-reserved", state))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"


@pytest.mark.parametrize("shared", [
    None, False, {}, {"owner_id": OWNER}, {"owner_id": OWNER, "gpu_uuids": [], "extra": True},
    {"owner_id": None, "gpu_uuids": ["GPU-reserved"]},
    {"owner_id": "", "gpu_uuids": ["GPU-reserved"]},
    {"owner_id": OWNER.upper(), "gpu_uuids": ["GPU-reserved"]},
    {"owner_id": OWNER, "gpu_uuids": []},
    {"owner_id": OWNER, "gpu_uuids": "GPU-reserved"},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-reserved", "GPU-reserved"]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-reserved", []]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-unknown"]},
    {"owner_id": OWNER, "gpu_uuids": [""]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-reserved\n"]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-x;exec"]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-" + "a" * 197]},
    {"owner_id": OWNER, "gpu_uuids": ["GPU-" + str(i) for i in range(257)]},
])
def test_bad_shared_reservation_policy_fails_closed(guarded, shared):
    guarded["resource_guard"]["shared_reservations"] = shared
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-free"])
    assert error.value.code == "RESOURCE_GUARD_UNAVAILABLE"
    inventory, status = filter_inventory(guarded, [{"uuid": "GPU-free"}])
    assert inventory == [] and status["status"] == "UNAVAILABLE"


def upgrade_authority(guarded):
    stamp = datetime.now(timezone.utc)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("ALTER TABLE gpu_reservations ADD COLUMN expires_at TEXT")
        db.execute("CREATE TABLE users(id TEXT PRIMARY KEY,enabled BOOLEAN,role TEXT,must_change_password BOOLEAN)")
        db.execute("INSERT INTO users VALUES (?,1,'operator',0)", (OWNER,))
        db.execute("UPDATE gpu_reservations SET created_at=?,expires_at=?",
                   (stamp.isoformat(), (stamp + timedelta(hours=8)).isoformat()))
    return stamp


@pytest.mark.parametrize("revocation", ["expired", "legacy_expired", "disabled", "viewer", "password_change"])
def test_upgraded_authority_releases_only_inactive_human_reservations(guarded, revocation):
    stamp = upgrade_authority(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        if revocation == "expired":
            db.execute("UPDATE gpu_reservations SET expires_at=?", ((stamp - timedelta(seconds=1)).isoformat(),))
        elif revocation == "legacy_expired":
            db.execute("UPDATE gpu_reservations SET created_at=?,expires_at=NULL", ((stamp - timedelta(hours=9)).isoformat(),))
        elif revocation == "disabled":
            db.execute("UPDATE users SET enabled=0")
        elif revocation == "viewer":
            db.execute("UPDATE users SET role='viewer'")
        else:
            db.execute("UPDATE users SET must_change_password=1")
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        before = list(db.iterdump())
        require_gpu_access(guarded, ["GPU-reserved"])
        for device in ("GPU-assigned", "GPU-policy"):
            with pytest.raises(AgentError) as error:
                require_gpu_access(guarded, [device])
            assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"
        assert list(db.iterdump()) == before


def test_upgraded_unexpired_legacy_row_keeps_default_eight_hour_protection(guarded):
    stamp = upgrade_authority(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("UPDATE gpu_reservations SET created_at=?,expires_at=NULL", ((stamp - timedelta(hours=1)).isoformat(),))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"


@pytest.mark.parametrize("value", ["", "invalid", "2026-11-03", "2026-11-03T12:00:00", "99999-01-01T00:00:00+00:00"])
def test_upgraded_malformed_reservation_expiry_fails_closed(guarded, value):
    upgrade_authority(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("UPDATE gpu_reservations SET expires_at=?", (value,))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-free"])
    assert error.value.code == "RESOURCE_GUARD_UNAVAILABLE"


def test_expired_reservation_cannot_unblock_active_allocation_even_when_shared(guarded):
    stamp = upgrade_authority(guarded)
    share_reserved_card(guarded)
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("UPDATE gpu_reservations SET expires_at=?", ((stamp - timedelta(seconds=1)).isoformat(),))
        db.execute("INSERT INTO gpu_allocations VALUES (?,?)", (NODE, "GPU-reserved"))
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"


def test_partial_authority_upgrade_preserves_conservative_legacy_protection(guarded):
    with sqlite3.connect(guarded["resource_guard"]["database"]) as db:
        db.execute("ALTER TABLE gpu_reservations ADD COLUMN expires_at TEXT")
        db.execute("UPDATE gpu_reservations SET expires_at='2000-01-01T00:00:00+00:00'")
    with pytest.raises(AgentError) as error:
        require_gpu_access(guarded, ["GPU-reserved"])
    assert error.value.code == "GPU_RESERVED_BY_PRODUCTION"
