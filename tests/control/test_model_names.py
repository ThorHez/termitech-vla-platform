"""Generated labels may improve; checkpoint and copy identities must not move."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now
from vla_control.db import Database
from vla_control.model_names import (backfill_model_names, equivalent_model_names,
                                     model_asset_name, model_output_name)
from vla_control.models import Asset, Attempt, Node, Profile, Run


RECIPE = "auto_pi05_full_h16_d32_sft"
RUN = "abcdefgh-1234-5678-9012-123456789012"
LEGACY = RECIPE + " / 59999"
CANONICAL = "叠衣服 · abcdefgh / 59999"


def output(**changes):
    return {"kind": "MODEL", "name": LEGACY, "version": "59999",
            "metadata": {"format": "openpi-orbax", "recipe": RECIPE, "step": 59999}, **changes}


def test_output_name_uses_run_identity_and_keeps_recipe_unchanged():
    value = output()
    before = deepcopy(value)
    assert model_output_name(SimpleNamespace(id=RUN, name="叠衣服"), SimpleNamespace(number=1), value) == CANONICAL
    assert value == before
    assert model_output_name(SimpleNamespace(id=RUN, name="叠衣服"), SimpleNamespace(number=2), value) == "叠衣服 · abcdefgh · 执行 2 / 59999"


@pytest.mark.parametrize("changes", [
    {"kind": "NORM"}, {"name": "我选的最好模型"}, {"version": "60000"}, {"version": "５９９９９"},
    {"metadata": {"format": "other", "recipe": RECIPE, "step": 59999}},
    {"metadata": {"format": "openpi-orbax", "recipe": " ", "step": 59999}},
    {"metadata": {"format": "openpi-orbax", "recipe": RECIPE, "step": True}},
    {"metadata": {"format": "openpi-orbax", "recipe": RECIPE, "step": -1}},
    {"metadata": {"format": "openpi-orbax", "recipe": RECIPE, "step": "59999"}},
])
def test_non_generated_output_names_are_preserved(changes):
    value = output(**changes)
    assert model_output_name(SimpleNamespace(id=RUN, name="叠衣服"), SimpleNamespace(number=1), value) == value["name"]


def test_truncation_preserves_id_attempt_and_step_and_zero_padded_versions():
    run, attempt = SimpleNamespace(id=RUN, name="训" * 200), SimpleNamespace(number=12)
    result = model_output_name(run, attempt, output())
    assert len(result) == 180 and result.endswith(" · abcdefgh · 执行 12 / 59999")
    value = output(name=RECIPE + " / 00059999", version="00059999")
    assert model_output_name(SimpleNamespace(id=RUN, name="叠衣服"), SimpleNamespace(number=1), value) == CANONICAL


@pytest.fixture
def db():
    database = Database("sqlite://")
    database.initialize()
    timestamp = now()
    with database.write() as session:
        for number in range(1, 4):
            node_id = f"node{number}"
            session.add(Node(id=node_id, token_hash=str(number) * 64, name=node_id, hostname=node_id,
                last_seen=timestamp, created_at=timestamp, roots=["/data"], output_root="/data/output"))
            session.flush()
            session.add(Profile(node_id=node_id, id="default", name="runtime", code_dir="/data/code",
                                python="/usr/bin/python3", adapter="openpi-sft"))
        session.flush()
        session.add(Run(id=RUN, name="叠衣服", node_id="node1", profile_id="default", template_id="openpi-sft",
            state="SUCCEEDED", created_at=timestamp, updated_at=timestamp, deleted_at=timestamp,
            inputs={}, parameters={}, gpu_count=1, code_expectation={"recipe": RECIPE}))
        session.flush()
        session.add(Attempt(id="attempt", run_id=RUN, node_id="node1", number=1, execution_token="x"*64,
            state="SUCCEEDED", created_at=timestamp, output_dir="/data/output/attempt"))
        session.flush()
        session.add(Asset(id="original", node_id="node1", name=LEGACY, kind="MODEL",
            path="/data/output/attempt/checkpoints/59999", version="59999", state="UNAVAILABLE",
            origin_attempt_id="attempt", metadata_json=output()["metadata"], created_at=timestamp))
    yield database
    database.close()


def add_copy(session, identifier="copy", source_id="original", *, name=LEGACY, **sync_changes):
    source = session.get(Asset, source_id)
    sync = {"source_asset_id": source_id, "source_node_id": source.node_id,
            "source_path": source.path, "byte_verified": True, **sync_changes}
    asset = Asset(id=identifier, node_id="node2" if source.node_id != "node2" else "node3", name=name,
        kind="MODEL", path=f"/data/models/{identifier}", version="59999", state="REGISTERED",
        metadata_json={"format": "openpi-orbax", "asset_sync": sync}, created_at=now())
    session.add(asset);session.flush()
    return asset


def transfer(session, identifier="sync", source_id="original", *, name=LEGACY + " · 资产同步"):
    source = session.get(Asset, source_id)
    row = AssetTransfer(id=identifier, request_key=identifier, request_digest="f"*64, name=name,
        source_asset_id=source.id, source_node_id=source.node_id, source_path=source.path, source_kind="MODEL",
        target_node_id="node3", target_directory="/data/models", target_name=identifier,
        state="FAILED", binding_snapshot={"source_asset":{"id":source.id,"name":source.name,
          "path":source.path,"metadata":deepcopy(source.metadata_json)}}, created_at=now(), updated_at=now())
    session.add(row);session.flush()
    return row


def test_existing_historical_original_and_verified_copy_chain_names(db):
    with db.write() as session:
        first = add_copy(session)
        second = add_copy(session, "copy2", first.id)
        assert model_asset_name(session, session.get(Asset, "original")) == CANONICAL
        assert model_asset_name(session, second) == CANONICAL
        assert equivalent_model_names(session, second.id, LEGACY, CANONICAL)
        assert not equivalent_model_names(session, second.id, LEGACY, "my preferred name")
        assert not equivalent_model_names(session, second.id, "anything", "anything")


@pytest.mark.parametrize("changes", [
    {"byte_verified": False}, {"byte_verified": 1}, {"source_node_id": "node3"},
    {"source_path": "/data/someone-else"}, {"source_asset_id": "missing"},
])
def test_forged_copy_lineage_is_not_renamed(db, changes):
    with db.write() as session:
        copy = add_copy(session, **changes)
        assert model_asset_name(session, copy) == LEGACY
        assert not equivalent_model_names(session, copy.id, LEGACY, CANONICAL)


def test_cycle_and_custom_source_or_copy_names_are_preserved(db):
    with db.write() as session:
        one = add_copy(session, "cycle1")
        two = add_copy(session, "cycle2", one.id)
        one.metadata_json = {"asset_sync":{"byte_verified":True,"source_asset_id":two.id,
            "source_node_id":two.node_id,"source_path":two.path}}
        assert model_asset_name(session, one) == LEGACY
        custom = add_copy(session, "custom-copy", name="my preferred copy")
        assert model_asset_name(session, custom) == "my preferred copy"
        session.get(Asset,"original").name = "my best model"
        inherited = add_copy(session, "inherited")
        assert model_asset_name(session,inherited) == LEGACY
        assert not equivalent_model_names(session, "original", LEGACY, CANONICAL)


def test_wrong_origin_node_is_rejected_without_falling_back_to_copy_metadata(db):
    with db.write() as session:
        copy = add_copy(session)
        # Deliberately corrupt the detached in-memory identity. The database FK
        # already forbids persisting this cross-node attempt; the helper also
        # refuses it before it can influence any display or snapshot check.
        with session.no_autoflush:
            copy.origin_attempt_id = "attempt"
            assert model_asset_name(session,copy) == LEGACY
        session.expunge(copy)


def test_bulk_migration_updates_only_generated_names_and_is_idempotent(db):
    with db.write() as session:
        first = add_copy(session)
        add_copy(session,"copy2",first.id)
        add_copy(session,"custom",name="user chosen")
        add_copy(session,"forged",source_path="/data/not-source")
        transfer(session)
        transfer(session,"copy-sync",first.id)
        transfer(session,"custom-sync",name="please keep this transfer title")
    with db.engine.begin() as connection:
        assets_before = {r["id"]:dict(r) for r in connection.execute(select(Asset.__table__)).mappings()}
        transfers_before = {r["id"]:dict(r) for r in connection.execute(select(AssetTransfer.__table__)).mappings()}
        assert backfill_model_names(connection) == {"assets":3,"asset_transfers":2}
        assets_after = {r["id"]:dict(r) for r in connection.execute(select(Asset.__table__)).mappings()}
        transfers_after = {r["id"]:dict(r) for r in connection.execute(select(AssetTransfer.__table__)).mappings()}
        for identifier,row in assets_before.items():
            assert {k:v for k,v in row.items() if k!="name"} == {k:v for k,v in assets_after[identifier].items() if k!="name"}
        for identifier,row in transfers_before.items():
            assert {k:v for k,v in row.items() if k!="name"} == {k:v for k,v in transfers_after[identifier].items() if k!="name"}
        assert assets_after["original"]["name"] == assets_after["copy2"]["name"] == CANONICAL
        assert assets_after["custom"]["name"] == "user chosen"
        assert assets_after["forged"]["name"] == LEGACY
        assert transfers_after["sync"]["name"] == transfers_after["copy-sync"]["name"] == CANONICAL+" · 资产同步"
        assert transfers_after["custom-sync"]["name"] == "please keep this transfer title"
        assert backfill_model_names(connection) == {"assets":0,"asset_transfers":0}


def test_bulk_migration_query_count_does_not_scale_with_chain_length(db):
    with db.write() as session:
        source = "original"
        for index in range(120):
            source = add_copy(session,f"chain{index}",source).id
    statements=[]
    def record(_connection,_cursor,statement,*_):statements.append(statement)
    event.listen(db.engine,"before_cursor_execute",record)
    try:
        with db.engine.begin() as connection:
            assert backfill_model_names(connection)["assets"] == 121
    finally:event.remove(db.engine,"before_cursor_execute",record)
    assert len(statements) <= 6
