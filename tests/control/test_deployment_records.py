"""Workflow ownership stays separate from transfer IO and Agent executions."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select
from vla_protocol import canonical_digest

from vla_control.auth_models import AuditEvent
from vla_control.checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer
from vla_control.common import APIError, now, offset_time, uid
from vla_control.deployment_record_models import (
    DeploymentRecord, DeploymentRecordRuntime, DeploymentRecordTransfer,
)
from vla_control.deployment_records import _assign_historical_transfer
from vla_control.models import Allocation, Asset, Command, Node, Profile
from vla_control.policy_deployment_models import PolicyDeployment

from .conftest import ADMIN
from .test_checkpoint_transfers import (
    BASE as TRANSFERS, PORTABLE, SECRET, ack_export, complete,
    create as create_transfer, get as get_transfer, transfer,
)
from .test_policy_deployment_restart import enable_restart, terminal
from .test_policy_deployments import (
    BASE as RUNTIMES, ack, action, add_run, command, create as create_runtime, running,
)
from .test_recipe_details import register


RECORDS = "/api/v1/deployment-records"


def records(platform):
    response = platform.client.get(RECORDS, headers=ADMIN)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def detail(platform, identifier):
    response = platform.client.get(RECORDS + "/" + identifier, headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()


def snapshot(platform, *, include_records=False):
    models = [CheckpointTransfer, PolicyDeployment, PolicyDeploymentTransfer, Command, Allocation, Asset]
    if include_records:
        models += [DeploymentRecord, DeploymentRecordRuntime, DeploymentRecordTransfer]
    with platform.app.state.db.read() as session:
        return {model.__tablename__: [dict(row) for row in session.execute(
            select(model.__table__).order_by(*model.__table__.primary_key.columns)).mappings()]
            for model in models}


def cancel(fixture, child):
    platform, _, source, *_ = fixture
    response = platform.client.post(TRANSFERS + "/" + child["id"] + "/cancel", headers=ADMIN)
    assert response.status_code in (200, 202), response.text
    for pending in platform.commands(source):
        if pending["type"] == "POLICY_EXPORT_CANCEL" and pending["payload"].get("transfer_id") == child["id"]:
            result = {"state": "FAILED", "error": {"code": "POLICY_EXPORT_CANCELLED", "message": "fixture stopped export"}}
            assert ack_export(fixture, pending, result=result).status_code == 200
    assert get_transfer(fixture, child["id"])["state"] == "CANCELLED"


def succeed(fixture, child):
    platform, _, source, *_ = fixture
    pending = next(item for item in platform.commands(source)
                   if item["type"] == "POLICY_EXPORT" and item["payload"].get("transfer_id") == child["id"])
    assert ack_export(fixture, pending).status_code == 200
    complete(fixture, child)
    result = get_transfer(fixture, child["id"])
    assert result["state"] == "SUCCEEDED"
    return result


def deploy(fixture, child, **changes):
    platform, _, _, target, run, *_ = fixture
    return platform.client.post(RUNTIMES, headers=ADMIN, json={
        "name": "workflow runtime", "node_id": target["node_id"], "profile_id": "default",
        "checkpoint_path": child["target_path"], "source_run_id": run.id,
        "transfer_id": child["id"], "port": 8000, **changes})


@pytest.fixture
def history(transfer):
    """A cancelled isolated workflow and a distinct exact successful runtime."""
    platform, _, source, *_ = transfer
    original = create_transfer(transfer, record_name="historical cancelled workflow").json()
    pending = next(item for item in platform.commands(source) if item["type"] == "POLICY_EXPORT")
    assert ack_export(transfer, pending).status_code == 200
    cancel(transfer, original)
    successful = create_transfer(transfer, record_name="successful workflow").json()
    successful = succeed(transfer, successful)
    response = deploy(transfer, successful, record_id=successful["record_id"])
    assert response.status_code == 202, response.text
    return transfer, get_transfer(transfer, original["id"]), successful, response.json()


def test_preparation_creates_a_parent_atomically_without_gpu_or_model_start(transfer, monkeypatch):
    platform, _, source, target, *_ = transfer
    monkeypatch.setattr("vla_control.scheduler.gpu_available", lambda *_: pytest.fail("Preparation cannot require idle GPU"))
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).gpus = []
    response = create_transfer(transfer, name="sync child", record_name="parent record")
    assert response.status_code == 202, response.text
    child = response.json()
    parent = detail(platform, child["record_id"])
    assert parent["name"] == "parent record" and child["name"] == "sync child"
    assert parent["state"] == "PREPARING" and parent["deployment_id"] is None and parent["deployment"] is None
    assert parent["node_id"] == target["node_id"] and parent["profile_id"] == "default"
    assert parent["transfer_ids"] == [child["id"]] and parent["selected_transfer_id"] == child["id"]
    assert parent["transfers"][0]["record_id"] == parent["id"]
    assert len(records(platform)) == 1
    assert not any(item["type"] == "POLICY_DEPLOY_START" for item in platform.commands(target))
    assert [item["type"] for item in platform.commands(source)] == ["POLICY_EXPORT"]


def test_cancel_and_retry_remain_children_of_the_same_durable_parent(transfer):
    platform, *_ = transfer
    child = create_transfer(transfer, record_name="one intent").json()
    parent_id = child["record_id"]
    cancel(transfer, child)
    cancelled = detail(platform, parent_id)
    assert cancelled["state"] == "CANCELLED" and cancelled["deployment_id"] is None
    response = platform.client.post(TRANSFERS + "/" + child["id"] + "/retry", headers=ADMIN)
    assert response.status_code == 202, response.text
    assert response.json()["record_id"] == parent_id and response.json()["attempt"] == 2
    renewed = detail(platform, parent_id)
    assert renewed["state"] == "PREPARING" and renewed["transfer_ids"] == [child["id"]]
    assert [item["id"] for item in records(platform)] == [parent_id]


def test_a_new_sync_inside_the_existing_parent_keeps_cancelled_history_newest_first(transfer):
    platform, *_ = transfer
    first = create_transfer(transfer, record_name="one parent").json()
    cancel(transfer, first)
    second = create_transfer(transfer, record_id=first["record_id"], source_checkpoint_path="/machine/output/owned/20000")
    assert second.status_code == 202, second.text
    second = second.json()
    assert second["record_id"] == first["record_id"] and second["id"] != first["id"]
    parent = detail(platform, first["record_id"])
    assert parent["transfer_ids"] == [second["id"], first["id"]]
    assert parent["selected_transfer_id"] == second["id"] and parent["state"] == "PREPARING"
    assert parent["transfers"][1]["state"] == "CANCELLED" and len(records(platform)) == 1


def test_new_named_intents_do_not_reuse_other_parents_active_transfers(transfer):
    platform, *_ = transfer
    first = create_transfer(transfer, record_name="intent A").json()
    duplicate = create_transfer(transfer, record_id=first["record_id"])
    assert duplicate.status_code == 202 and duplicate.json()["id"] == first["id"]
    second = create_transfer(transfer, record_name="intent B")
    assert second.status_code == 202, second.text
    second = second.json()
    assert second["id"] != first["id"] and second["record_id"] != first["record_id"]
    assert detail(platform, first["record_id"])["transfer_ids"] == [first["id"]]
    assert detail(platform, second["record_id"])["transfer_ids"] == [second["id"]]
    assert create_transfer(transfer, record_id=second["record_id"]).json()["id"] == second["id"]


def test_legacy_anonymous_transfer_retry_keeps_its_exact_active_parent(transfer):
    first = create_transfer(transfer)
    second = create_transfer(transfer)
    assert first.status_code == second.status_code == 202
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["record_id"] == first.json()["record_id"]
    assert len(records(transfer[0])) == 1


@pytest.mark.parametrize("state,expected", [
    ("QUEUED", "PREPARING"), ("EXPORTING", "PREPARING"), ("TRANSFERRING", "PREPARING"),
    ("VERIFYING", "PREPARING"), ("SUCCEEDED", "READY_TO_DEPLOY"),
    ("CANCELLED", "CANCELLED"), ("FAILED", "FAILED"), ("INTERRUPTED", "INTERRUPTED"),
])
def test_parent_without_runtime_projects_preparation_state_without_creating_a_service(transfer, state, expected):
    platform, *_ = transfer
    child = create_transfer(transfer, record_name="state projection").json()
    with platform.app.state.db.write() as session:
        session.get(CheckpointTransfer, child["id"]).state = state
    record = detail(platform, child["record_id"])
    assert record["state"] == expected and record["deployment"] is None
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(PolicyDeployment))) == []


def test_empty_record_is_a_draft_and_gets_are_read_only(platform):
    identifier = uid()
    with platform.app.state.db.write() as session:
        session.add(DeploymentRecord(id=identifier, name="not started", created_at=now()))
    before = snapshot(platform, include_records=True)
    for _ in range(2):
        parent = detail(platform, identifier)
        assert parent["state"] == "DRAFT" and parent["transfer_ids"] == []
        assert parent["deployment_id"] is parent["selected_transfer_id"] is parent["deployment"] is None
        assert parent["transfers"] == []
        assert len(records(platform)) == 1
    assert snapshot(platform, include_records=True) == before


def test_runtime_binding_is_exact_and_parent_rejects_a_second_runtime(history):
    fixture, historical, successful, runtime = history
    platform, _, _, target, *_ = fixture
    parent = detail(platform, runtime["record_id"])
    assert parent["deployment_id"] == runtime["id"] and parent["deployment"]["id"] == runtime["id"]
    assert parent["selected_transfer_id"] == successful["id"]
    assert parent["transfer_ids"] == [successful["id"]]
    assert historical["id"] not in parent["transfer_ids"]
    before = snapshot(platform, include_records=True)
    response = deploy(fixture, successful, record_id=runtime["record_id"], port=8001)
    assert response.status_code == 409, response.text
    response = create_transfer(fixture, record_id=runtime["record_id"])
    assert response.status_code == 409, response.text
    assert snapshot(platform, include_records=True) == before
    start = command(platform, target, "POLICY_DEPLOY_START")
    assert start["payload"]["deployment_id"] == runtime["id"]
    assert "record_id" not in start["payload"] and "record_name" not in start["payload"]


def test_cross_parent_or_unknown_binding_cannot_create_a_runtime_or_claim_another_sync(transfer):
    platform, *_ = transfer
    first = create_transfer(transfer, record_name="owner").json()
    first = succeed(transfer, first)
    other = create_transfer(transfer, record_name="different intent").json()
    before = snapshot(platform, include_records=True)
    for parent_id in (other["record_id"], uid()):
        response = deploy(transfer, first, record_id=parent_id)
        assert response.status_code in (404, 409, 422), response.text
        assert snapshot(platform, include_records=True) == before
    response = create_transfer(transfer, record_id=uid())
    assert response.status_code == 404, response.text
    assert snapshot(platform, include_records=True) == before


def test_summary_and_detail_preserve_effective_unknown_runtime_state(history):
    fixture, _, _, runtime = history
    platform, _, _, target, *_ = fixture
    with platform.app.state.db.write() as session:
        session.get(PolicyDeployment, runtime["id"]).state = "RUNNING"
        session.get(Node, target["node_id"]).last_seen = offset_time(-3600)
    parent = detail(platform, runtime["record_id"])
    assert parent["state"] == parent["deployment"]["state"] == "UNKNOWN"
    assert next(item for item in records(platform) if item["id"] == parent["id"])["state"] == "UNKNOWN"


def test_aliases_and_gets_never_merge_unrelated_same_run_preparations_or_leak_secrets(history):
    fixture, historical, successful, runtime = history
    platform, *_ = fixture
    before = snapshot(platform, include_records=True)
    for identifier in (runtime["record_id"], runtime["id"], successful["id"]):
        assert detail(platform, identifier)["id"] == runtime["record_id"]
    assert detail(platform, historical["id"])["id"] == historical["record_id"] != runtime["record_id"]
    for parent in records(platform):
        response = platform.client.get(RECORDS + "/" + parent["id"], headers=ADMIN)
        for private in (SECRET, "encrypted_password", "binding_snapshot", "portable_config", "frozen_request"):
            assert private not in response.text
    assert snapshot(platform, include_records=True) == before


def test_record_ids_require_human_read_permission_and_unknown_ids_fail_closed(transfer):
    platform, _, source, *_ = transfer
    child = create_transfer(transfer, record_name="private management record").json()
    for path in (RECORDS, RECORDS + "/" + child["record_id"]):
        assert platform.client.get(path).status_code == 401
        assert platform.client.get(path, headers=source["headers"]).status_code == 401
    assert platform.client.get(RECORDS + "/" + uid(), headers=ADMIN).status_code == 404


def test_viewer_can_read_record_metadata_but_cannot_mutate_its_children(transfer):
    platform, *_ = transfer
    child = create_transfer(transfer, record_name="viewer record").json()
    password = "fixture-record-viewer-password"
    platform.app.state.auth.create_user("record-admin", password, "admin")
    platform.app.state.auth.create_user("record-viewer", password, "viewer")
    client = TestClient(platform.app, base_url="https://testserver")
    try:
        login = client.post("/api/v1/auth/login", json={"username": "record-viewer", "password": password})
        assert login.status_code == 200, login.text
        headers = {"X-CSRF-Token": login.json()["csrf_token"]}
        before = snapshot(platform, include_records=True)
        assert client.get(RECORDS).status_code == 200
        assert client.get(RECORDS + "/" + child["record_id"]).status_code == 200
        for operation in ("cancel", "retry"):
            response = client.post(TRANSFERS + "/" + child["id"] + "/" + operation, headers=headers)
            assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN"
        assert snapshot(platform, include_records=True) == before
    finally:
        client.close()


def test_backfill_is_idempotent_and_uses_only_existing_exact_runtime_transfer_links(history):
    fixture, historical, successful, runtime = history
    platform, *_ = fixture
    before = snapshot(platform)
    with platform.app.state.db.write() as session:
        session.execute(delete(DeploymentRecordTransfer))
        session.execute(delete(DeploymentRecordRuntime))
        session.execute(delete(DeploymentRecord))
    platform.app.state.db.initialize()
    initial = records(platform)
    assert len(initial) == 2
    runtime_parent = detail(platform, runtime["id"])
    orphan_parent = detail(platform, historical["id"])
    assert runtime_parent["id"] != orphan_parent["id"]
    assert runtime_parent["transfer_ids"] == [successful["id"]]
    assert orphan_parent["transfer_ids"] == [historical["id"]] and orphan_parent["deployment"] is None
    assert orphan_parent["state"] == "CANCELLED"
    populated = snapshot(platform, include_records=True)
    platform.app.state.db.initialize()
    assert snapshot(platform, include_records=True) == populated
    assert snapshot(platform) == before


def test_legacy_shared_transfer_keeps_canonical_owner_and_exact_selected_reference_during_backfill(transfer):
    platform, _, _, target, *_ = transfer
    child = succeed(transfer, create_transfer(transfer, record_name="original owner").json())
    first = deploy(transfer, child, record_id=child["record_id"]).json()
    witnesses = {"portable_digest": child["portable_digest"], "checkpoint_manifest_digest": child["manifest_sha256"]}
    assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_START"), running(**witnesses)).status_code == 200
    assert action(platform, first, "stop").status_code == 202
    assert ack(platform, target, command(platform, target, "POLICY_DEPLOY_STOP"), {"state": "STOPPED", **witnesses}).status_code == 200
    response = deploy(transfer, child, port=8001)
    assert response.status_code == 202, response.text
    second = response.json()
    assert second["record_id"] != first["record_id"]
    assert get_transfer(transfer, child["id"])["record_id"] == first["record_id"]
    selected = detail(platform, second["record_id"])
    assert selected["deployment_id"] == second["id"] and selected["selected_transfer_id"] == child["id"]
    assert selected["transfer_ids"] == [child["id"]]
    assert selected["transfers"][0]["record_id"] == first["record_id"]
    before = snapshot(platform)
    with platform.app.state.db.write() as session:
        session.execute(delete(DeploymentRecordTransfer))
        session.execute(delete(DeploymentRecordRuntime))
        session.execute(delete(DeploymentRecord))
    platform.app.state.db.initialize()
    assert len(records(platform)) == 2
    earliest_parent = detail(platform, first["id"])
    later_parent = detail(platform, second["id"])
    assert get_transfer(transfer, child["id"])["record_id"] == earliest_parent["id"]
    assert later_parent["id"] != earliest_parent["id"]
    assert later_parent["selected_transfer_id"] == child["id"] and later_parent["transfer_ids"] == [child["id"]]
    assert detail(platform, child["id"])["id"] == earliest_parent["id"]
    migrated = snapshot(platform, include_records=True)
    platform.app.state.db.initialize()
    assert snapshot(platform, include_records=True) == migrated
    assert snapshot(platform) == before


def test_explicit_historical_assignment_retains_audit_and_merged_alias_without_business_mutation(history):
    fixture, historical, successful, runtime = history
    platform, *_ = fixture
    before = snapshot(platform)
    with platform.app.state.db.write() as session:
        assigned = _assign_historical_transfer(session, historical["id"], runtime["record_id"], actor_username="fixture-explicit-repair")
        assert assigned.id == runtime["record_id"]
    parent = detail(platform, runtime["record_id"])
    assert parent["transfer_ids"] == [successful["id"], historical["id"]]
    assert parent["selected_transfer_id"] == successful["id"]
    assert parent["deployment_id"] == runtime["id"] and parent["state"] == "PENDING"
    assert len(records(platform)) == 1
    assert detail(platform, historical["id"])["id"] == parent["id"]
    assert detail(platform, historical["record_id"])["id"] == parent["id"]
    with platform.app.state.db.read() as session:
        old = session.get(DeploymentRecord, historical["record_id"])
        assert old is not None and old.merged_into_id == parent["id"]
        link = session.get(DeploymentRecordTransfer, historical["id"])
        assert link.provenance["origin"] == "explicit_historical_assignment"
        assert link.provenance["previous_record_id"] == old.id
        assert link.provenance["reference_transfer_id"] == successful["id"]
        event = session.scalar(select(AuditEvent).where(AuditEvent.action == "deployment_record.attach_historical_transfer"))
        assert event.actor_username == "fixture-explicit-repair" and event.details["transfer_id"] == historical["id"]
    assert snapshot(platform) == before
    with platform.app.state.db.write() as session:
        _assign_historical_transfer(session, historical["id"], parent["id"])
        assert len(list(session.scalars(select(AuditEvent).where(AuditEvent.action == "deployment_record.attach_historical_transfer")))) == 1


@pytest.mark.parametrize("field", ["source_node_id", "source_run_id", "source_checkpoint_path", "target_node_id", "target_profile_id", "target_directory", "binding_snapshot", "portable_digest", "portable_config"])
def test_historical_assignment_rejects_any_contract_difference_atomically(history, field):
    fixture, historical, _, runtime = history
    platform, _, source, *_ = fixture
    with platform.app.state.db.write() as session:
        row = session.get(CheckpointTransfer, historical["id"])
        if field == "source_run_id":
            row.source_run_id = add_run(session, source).id
        elif field == "source_node_id":
            row.source_node_id = fixture[3]["node_id"]
        elif field == "target_node_id":
            row.target_node_id = source["node_id"]
        elif field == "target_profile_id":
            original = session.get(Profile, (row.target_node_id, row.target_profile_id))
            alternate = Profile(node_id=row.target_node_id, id="another-profile", name="alternate",
                code_dir=original.code_dir, python=original.python, adapter=original.adapter,
                recipe=original.recipe, execution_mode=original.execution_mode, metadata_json={})
            session.add(alternate)
            session.flush()
            row.target_profile_id = alternate.id
        elif field == "binding_snapshot":
            row.binding_snapshot = {**row.binding_snapshot, "changed_source": True}
        elif field == "portable_digest":
            row.portable_digest = "f" * 64
        elif field == "portable_config":
            row.portable_config = {"tampered": True}
        else:
            setattr(row, field, getattr(row, field) + "-other")
        session.flush()
        with pytest.raises(APIError):
            _assign_historical_transfer(session, historical["id"], runtime["record_id"])
        assert session.get(DeploymentRecordTransfer, historical["id"]).record_id == historical["record_id"]
        assert session.get(DeploymentRecord, historical["record_id"]).merged_into_id is None


@pytest.mark.parametrize("state", ["QUEUED", "EXPORTING", "TRANSFERRING", "VERIFYING", "CANCELLING"])
def test_historical_assignment_requires_a_terminal_transfer(history, state):
    fixture, historical, _, runtime = history
    platform, *_ = fixture
    with platform.app.state.db.write() as session:
        session.get(CheckpointTransfer, historical["id"]).state = state
        with pytest.raises(APIError) as error:
            _assign_historical_transfer(session, historical["id"], runtime["record_id"])
        assert error.value.status == 409
        assert session.get(DeploymentRecordTransfer, historical["id"]).record_id == historical["record_id"]


def test_historical_assignment_cannot_take_a_transfer_from_a_parent_with_a_runtime(history):
    fixture, historical, _, runtime = history
    platform, _, _, target, *_ = fixture
    with platform.app.state.db.write() as session:
        legacy = PolicyDeployment(id=uid(), name="existing historical runtime", node_id=target["node_id"],
            profile_id="default", checkpoint_path="/machine/existing-model", config_name="pi0_base", port=8100,
            state="STOPPED", logs="", checks=[], created_at=now(), updated_at=now())
        session.add(legacy)
        session.flush()
        session.add(DeploymentRecordRuntime(deployment_id=legacy.id, record_id=historical["record_id"]))
        session.flush()
        with pytest.raises(APIError) as error:
            _assign_historical_transfer(session, historical["id"], runtime["record_id"])
        assert error.value.code == "DEPLOYMENT_RECORD_CONFLICT"
        assert session.get(DeploymentRecordTransfer, historical["id"]).record_id == historical["record_id"]


@pytest.mark.parametrize("target_mode", ("no-runtime", "failed-reference"))
def test_historical_assignment_requires_an_existing_runtime_and_its_successful_exact_reference(history, target_mode):
    fixture, historical, successful, runtime = history
    platform, *_ = fixture
    with platform.app.state.db.write() as session:
        target_id = runtime["record_id"]
        if target_mode == "no-runtime":
            target_id = uid()
            session.add(DeploymentRecord(id=target_id, name="unverified draft", created_at=now()))
        else:
            session.get(CheckpointTransfer, successful["id"]).state = "FAILED"
        session.flush()
        with pytest.raises(APIError) as error:
            _assign_historical_transfer(session, historical["id"], target_id)
        assert error.value.code == "DEPLOYMENT_RECORD_TRANSFER_MISMATCH"
        assert session.get(DeploymentRecordTransfer, historical["id"]).record_id == historical["record_id"]


def durable_export_witness(history, corruption=None):
    """Model cancellation after a later retry cleared the cached portable JSON."""
    fixture, historical, _, _ = history
    platform, _, source, target, *_ = fixture
    with platform.app.state.db.write() as session:
        row = session.get(CheckpointTransfer, historical["id"])
        row.attempt = 3
        row.portable_config = None
        # The normal fixture initially ACKs EXPORT directly as READY. Retire
        # that shortcut so only the explicit historical STATUS witness below
        # can satisfy this test's positive or negative evidence path.
        for old in session.scalars(select(Command).where(
                Command.payload["transfer_id"].as_string() == row.id)):
            if (old.result or {}).get("state") == "READY":
                old.status = "REJECTED"
        payload = {"transfer_id": row.id, "transfer_attempt": 2, "export_id": row.id + "-2"}
        result = {"state": "READY", "portable_config": deepcopy(PORTABLE), "portable_digest": canonical_digest(PORTABLE)}
        witness = Command(id=uid(), type="POLICY_EXPORT_STATUS", node_id=source["node_id"],
            status="APPLIED", payload=payload, result=result, created_at=now())
        if corruption == "node":
            witness.node_id = target["node_id"]
        elif corruption == "transfer":
            witness.payload = {**payload, "transfer_id": uid()}
        elif corruption == "export-id":
            witness.payload = {**payload, "export_id": "unrelated-export-2"}
        elif corruption in ("future-attempt", "zero-attempt", "bool-attempt", "string-attempt"):
            attempt = {"future-attempt": 4, "zero-attempt": 0, "bool-attempt": True, "string-attempt": "2"}[corruption]
            witness.payload = {**payload, "transfer_attempt": attempt, "export_id": row.id + "-" + str(attempt)}
        elif corruption == "digest":
            witness.result = {**result, "portable_digest": "f" * 64}
        elif corruption == "config":
            witness.result = {**result, "portable_config": {"tampered": True}}
        elif corruption == "unapplied":
            witness.status = "PENDING"
        elif corruption == "not-ready":
            witness.result = {**result, "state": "RUNNING"}
        elif corruption == "command-type":
            witness.type = "POLICY_DEPLOY_STATUS"
        elif corruption == "wrong-export-source-run":
            witness.type = "POLICY_EXPORT"
            witness.payload = {**payload, "source_training": {"run_id": uid()}}
        session.add(witness)
        session.flush()
        return witness.id


def test_cancelled_history_uses_exact_durable_export_witness_without_repopulating_source_config(history):
    fixture, historical, successful, runtime = history
    platform, *_ = fixture
    witness_id = durable_export_witness(history)
    before = snapshot(platform)
    with platform.app.state.db.write() as session:
        _assign_historical_transfer(session, historical["id"], runtime["record_id"], actor_username="explicit-verified-history")
        original = session.get(CheckpointTransfer, historical["id"])
        assert original.state == "CANCELLED" and original.attempt == 3 and original.portable_config is None
        assert original.portable_digest == successful["portable_digest"]
        link = session.get(DeploymentRecordTransfer, historical["id"])
        assert link.provenance["portable_witness_command_id"] == witness_id
        event = session.scalar(select(AuditEvent).where(AuditEvent.action == "deployment_record.attach_historical_transfer"))
        assert event.details["portable_witness_command_id"] == witness_id
    assert snapshot(platform) == before
    assert detail(platform, runtime["record_id"])["transfer_ids"] == [successful["id"], historical["id"]]


@pytest.mark.parametrize("corruption", [
    "node", "transfer", "export-id", "future-attempt", "zero-attempt", "bool-attempt", "string-attempt",
    "digest", "config", "unapplied", "not-ready", "command-type", "wrong-export-source-run",
])
def test_cancelled_history_rejects_inexact_or_unverified_durable_export_witness(history, corruption):
    fixture, historical, _, runtime = history
    platform, *_ = fixture
    durable_export_witness(history, corruption)
    before = snapshot(platform, include_records=True)
    with platform.app.state.db.write() as session:
        with pytest.raises(APIError) as error:
            _assign_historical_transfer(session, historical["id"], runtime["record_id"])
        assert error.value.code == "DEPLOYMENT_RECORD_TRANSFER_MISMATCH"
        assert session.get(DeploymentRecordTransfer, historical["id"]).record_id == historical["record_id"]
        assert session.get(DeploymentRecord, historical["record_id"]).merged_into_id is None
        assert session.scalar(select(AuditEvent.id).where(AuditEvent.action == "deployment_record.attach_historical_transfer")) is None
    assert snapshot(platform, include_records=True) == before


def test_restart_keeps_the_same_parent_and_existing_runtime_execution_fencing(platform):
    node = register(platform)
    enable_restart(platform, node)
    runtime, old_start = terminal(platform, node)
    parent_id = runtime["record_id"]
    response = action(platform, runtime, "start")
    assert response.status_code == 202, response.text
    token = response.json()["execution_id"]
    current_start = command(platform, node, "POLICY_DEPLOY_START")
    assert current_start["payload"]["deployment_id"] == runtime["id"]
    assert current_start["payload"]["execution_id"] == token
    assert ack(platform, node, current_start, running(execution_id=token)).status_code == 200
    assert ack(platform, node, old_start, running()).status_code == 200
    parent = detail(platform, parent_id)
    assert parent["deployment_id"] == runtime["id"] and parent["deployment"]["execution_id"] == token
    assert parent["state"] == "RUNNING" and len(records(platform)) == 1
