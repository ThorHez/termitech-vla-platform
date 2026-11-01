"""Optional model filtering preserves old receipts, snapshots and active rows."""

from copy import deepcopy

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from vla_protocol import canonical_digest

from vla_control.asset_transfer_batch_models import AssetTransferBatch
from vla_control.asset_transfer_batches import AssetTransferBatchInput
from vla_control.asset_transfer_models import AssetTransfer
from vla_control.asset_transfers import AssetTransferInput
from vla_control.common import uid
from vla_control.models import Asset

from .test_asset_transfers import BASE, create, fixture, receipt, result_for, run_copy
from .test_asset_transfer_batches import BATCH, body as batch_body, counts, group


@pytest.fixture
def model(fixture):
    with fixture[0].app.state.db.write() as session:
        source = session.get(Asset, fixture[4]["id"])
        source.kind, source.path = "MODEL", "/machine/models/checkpoint"
        source.metadata_json = {"description": "model weights"}
    return fixture


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_single_exclusion_is_strict_boolean_before_any_queue(model, value):
    response = create(model, exclude_train_state=value)
    assert response.status_code == 422
    with model[0].app.state.db.read() as session:
        assert list(session.scalars(select(AssetTransfer))) == []


@pytest.mark.parametrize("kind", ["DATASET", "NORM"])
def test_only_model_assets_allow_exclusion_true_and_false_stays_supported(fixture, kind):
    with fixture[0].app.state.db.write() as session:
        session.get(Asset, fixture[4]["id"]).kind = kind
    response = create(fixture, exclude_train_state=True)
    assert response.status_code == 422 and response.json()["error"]["code"] == "ASSET_TRANSFER_EXCLUDE_MODEL_ONLY"
    with fixture[0].app.state.db.read() as session:
        assert list(session.scalars(select(AssetTransfer))) == []
    accepted = create(fixture, exclude_train_state=False)
    assert accepted.status_code == 202 and accepted.json()["exclude_train_state"] is False


@pytest.mark.parametrize("single_file", [False, True])
def test_model_filter_is_frozen_passed_to_worker_and_recorded_on_target(model, monkeypatch, single_file):
    if single_file:
        with model[0].app.state.db.write() as session:
            session.get(Asset, model[4]["id"]).path = "/machine/models/weights.safetensors"
    task = create(model, exclude_train_state=True, idempotency_key="model-without-state").json()
    assert task["exclude_train_state"] is True
    with model[0].app.state.db.read() as session:
        stored = session.get(AssetTransfer, task["id"])
        assert stored.exclude_train_state is True and stored.binding_snapshot["exclude_train_state"] is True
    observed = []
    def transfer(*args, **kwargs):
        observed.append(kwargs["exclude_train_state"])
        return result_for(args, kwargs, file_name="weights.safetensors" if single_file else None)
    final = run_copy(model, task["id"], monkeypatch, transfer)
    assert final["state"] == "SUCCEEDED" and observed == [True]
    assert final["training_preflight_required"] is True
    with model[0].app.state.db.read() as session:
        target = session.get(Asset, final["target_asset_id"])
        source = session.get(Asset, model[4]["id"])
        metadata = target.metadata_json["asset_sync"]
        assert metadata["exclude_train_state"] is True
        assert metadata["excluded_root_directories"] == ["train_state"]
        assert metadata["training_preflight_required"] is True
        assert "resume_supported" not in metadata and "training_state_complete" not in metadata
        assert source.metadata_json == {"description": "model weights"}


def test_false_omission_keeps_historical_digest_snapshot_worker_call_and_replay(model, monkeypatch):
    options = {"source_asset_id": model[4]["id"], "target_node_id": model[3]["node_id"],
               "target_directory": "/machine/copied", "target_name": "folding-copy", "idempotency_key": "old-model-copy"}
    old_fields = AssetTransferInput(**options).model_dump(exclude={"idempotency_key", "exclude_train_state", "include_norm_stats", "expected_norm_asset_id"})
    task = create(model, idempotency_key=options["idempotency_key"]).json()
    with model[0].app.state.db.read() as session:
        stored = session.get(AssetTransfer, task["id"])
        assert stored.request_digest == canonical_digest(old_fields)
        assert stored.exclude_train_state is False and "exclude_train_state" not in stored.binding_snapshot
    replay = create(model, idempotency_key=options["idempotency_key"], exclude_train_state=False)
    assert replay.status_code == 202 and replay.json() == task
    assert create(model, idempotency_key=options["idempotency_key"], exclude_train_state=True).status_code == 409
    def old_transfer(*args, **kwargs):
        assert "exclude_train_state" not in kwargs
        return result_for(args, kwargs)
    final = run_copy(model, task["id"], monkeypatch, old_transfer)
    assert final["state"] == "SUCCEEDED"
    with model[0].app.state.db.read() as session:
        metadata = session.get(Asset, final["target_asset_id"]).metadata_json["asset_sync"]
        assert "exclude_train_state" not in metadata and "excluded_root_directories" not in metadata


def test_retry_keeps_filter_and_rejects_new_filter_under_same_idempotency_key(model, monkeypatch):
    task = create(model, exclude_train_state=True, idempotency_key="filtered-retry").json()
    with model[0].app.state.db.write() as session:
        session.get(AssetTransfer, task["id"]).state = "FAILED"
    response = model[0].client.post(BASE + "/" + task["id"] + "/retry",
                                    headers={"Authorization": "Bearer test-admin"}, json={"exclude_train_state": False})
    assert response.status_code == 202 and response.json()["exclude_train_state"] is True
    assert response.json()["attempt"] == 2
    def transfer(*args, **kwargs):
        assert kwargs["exclude_train_state"] is True
        return result_for(args, kwargs)
    assert run_copy(model, task["id"], monkeypatch, transfer)["state"] == "SUCCEEDED"
    assert create(model, exclude_train_state=False, idempotency_key="filtered-retry").status_code == 409


def test_retry_detects_corrupted_filter_snapshot_instead_of_silently_changing_scope(model):
    task = create(model, exclude_train_state=True).json()
    with model[0].app.state.db.write() as session:
        stored = session.get(AssetTransfer, task["id"])
        stored.state, stored.exclude_train_state = "FAILED", False
    response = model[0].client.post(BASE + "/" + task["id"] + "/retry", headers={"Authorization": "Bearer test-admin"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_TRANSFER_BINDING_CHANGED"


def test_batch_each_entry_has_independent_filter_and_frozen_idempotency_scope(group):
    request = batch_body(group)
    request["entries"][1]["exclude_train_state"] = True
    response = group[1].post(BATCH, json=request)
    assert response.status_code == 202, response.text
    result = response.json()
    assert [row["exclude_train_state"] for row in result["transfers"]] == [False, True, False]
    with group[0][0].app.state.db.read() as session:
        for item in result["transfers"]:
            row = session.get(AssetTransfer, item["id"])
            assert row.exclude_train_state is (row.source_kind == "MODEL")
            assert row.binding_snapshot.get("exclude_train_state", False) == row.exclude_train_state
    assert group[1].post(BATCH, json=request).json() == result
    request["entries"][1]["exclude_train_state"] = False
    assert group[1].post(BATCH, json=request).status_code == 409
    assert counts(group)[:2] == (3, 1)


@pytest.mark.parametrize("entry_index,value", [(0, True), (2, True), (1, None), (1, 1), (1, "true")])
def test_invalid_batch_filter_never_partially_queues_other_selected_assets(group, entry_index, value):
    request = batch_body(group)
    request["entries"][entry_index]["exclude_train_state"] = value
    before = counts(group)
    response = group[1].post(BATCH, json=request)
    assert response.status_code == 422, response.text
    assert counts(group) == before and group[3] == []


def test_batch_old_digest_and_receipt_replay_with_explicit_false_after_upgrade(group):
    request = batch_body(group)
    response = group[1].post(BATCH, json=request)
    assert response.status_code == 202, response.text
    legacy = response.json()
    for item in legacy["transfers"]:
        item.pop("exclude_train_state")
    # Existing receipts remain frozen; pre-feature request digests have no
    # false field and remain valid when the new browser explicitly sends it.
    old_request = AssetTransferBatchInput.model_validate(request).model_dump()
    for entry in old_request["entries"]:
        entry.pop("exclude_train_state")
    with group[0][0].app.state.db.write() as session:
        row = session.scalar(select(AssetTransferBatch).where(AssetTransferBatch.request_key == request["idempotency_key"]))
        assert row.request_digest == canonical_digest(old_request)
        row.result = deepcopy(legacy)
    for entry in request["entries"]:
        entry["exclude_train_state"] = False
    replay = group[1].post(BATCH, json=request)
    assert replay.status_code == 202 and replay.json() == legacy
    assert len(group[3]) == 3 and counts(group)[:2] == (3, 1)


def test_additive_filter_migration_preserves_historical_active_progress_and_defaults(model):
    task = create(model, idempotency_key="before-filter-upgrade").json()
    database = model[0].app.state.db
    with database.write() as session:
        row = session.get(AssetTransfer, task["id"])
        row.state, row.attempt, row.bytes_total, row.bytes_transferred = "TRANSFERRING", 3, 1000, 375
        row.progress_pct, row.logs = 37.5, "preserve active progress\n"
    columns = [column.name for column in AssetTransfer.__table__.columns if column.name != "exclude_train_state"]
    projection = ",".join('"' + name + '"' for name in columns)
    with database.engine.begin() as connection:
        before = connection.exec_driver_sql("SELECT " + projection + " FROM asset_transfers").fetchall()
        connection.exec_driver_sql("ALTER TABLE asset_transfers DROP COLUMN exclude_train_state")
        assert "exclude_train_state" not in {column["name"] for column in inspect(connection).get_columns("asset_transfers")}
    database.initialize()
    database.initialize()
    with database.engine.begin() as connection:
        assert connection.exec_driver_sql("SELECT " + projection + " FROM asset_transfers").fetchall() == before
        assert connection.exec_driver_sql("SELECT exclude_train_state,typeof(exclude_train_state) FROM asset_transfers").one() == (0, "integer")
        # Raw historical writers that omit the new column also receive false.
        values = [":id" if name == "id" else ":key" if name == "request_key" else '"' + name + '"' for name in columns]
        connection.execute(text("INSERT INTO asset_transfers (" + projection + ") SELECT " + ",".join(values) + " FROM asset_transfers WHERE id=:source"),
                           {"id": uid(), "key": uid(), "source": task["id"]})
    with database.read() as session:
        assert all(row.exclude_train_state is False for row in session.scalars(select(AssetTransfer)))
        existing = session.get(AssetTransfer, task["id"])
        assert existing.state == "TRANSFERRING" and existing.bytes_transferred == 375
        assert "exclude_train_state" not in existing.binding_snapshot
    with pytest.raises(IntegrityError):
        with database.write() as session:
            session.get(AssetTransfer, task["id"]).exclude_train_state = None
            session.flush()
    assert receipt(model, task["id"])["exclude_train_state"] is False
