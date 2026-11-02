"""Unified task pages retain older training and sync history at bounded sizes."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.models import Asset, Attempt, Command, Event, Node, Run

from .conftest import ADMIN, ENROLL
from .test_asset_transfers import fixture


BASE = "/api/v1/background-tasks"
TRAINING_ACTIVE = {"QUEUED", "STARTING", "RUNNING", "CANCELLING", "UNKNOWN"}
SYNC_ACTIVE = {"QUEUED", "PREPARING", "SCANNING", "TRANSFERRING", "VERIFYING", "CANCELLING"}


def timestamp(index):
    return (datetime(2026, 11, 1, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat()


def training(fixture, index, **changes):
    _, _, source, _, _, _ = fixture
    values = {"id": f"run-{index:04}", "name": f"training task {index}", "node_id": source["node_id"],
        "profile_id": "default", "template_id": "smoke", "state": "SUCCEEDED", "created_at": timestamp(index),
        "updated_at": timestamp(index), "inputs": {}, "parameters": {"steps": 100}, "gpu_count": 0,
        "code_expectation": {"recipe": "smoke"}}
    return Run(**{**values, **changes})


def sync(fixture, index, **changes):
    _, _, source, target, asset, _ = fixture
    values = {"id": f"sync-{index:04}", "request_key": f"sync-request-{index:04}", "request_digest": "c" * 64,
        "name": f"sync task {index}", "source_asset_id": asset["id"], "source_node_id": source["node_id"],
        "source_path": asset["path"], "source_kind": "DATASET", "target_node_id": target["node_id"],
        "target_directory": "/machine/imports", "target_name": f"copy-{index:04}", "state": "SUCCEEDED",
        "created_at": timestamp(index), "updated_at": timestamp(index),
        "binding_snapshot": {"private_receipt": "not-public"}}
    return AssetTransfer(**{**values, **changes})


def save(fixture, *records):
    with fixture[0].app.state.db.write() as session:
        session.add_all(records)


def get(platform, **params):
    response = platform.client.get(BASE, headers=ADMIN, params=params)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def keys(page):
    return [entry["key"] for entry in page["items"]]


def snapshots(platform):
    with platform.app.state.db.read() as session:
        return {model.__tablename__: [dict(row) for row in session.execute(
            select(model.__table__).order_by(*model.__table__.primary_key.columns)).mappings()]
            for model in (Run, AssetTransfer, Attempt, Event, Asset, Command)}


@pytest.fixture
def mixed(fixture):
    run_states = ["QUEUED", "RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELLED"]
    sync_states = ["PREPARING", "SCANNING", "CANCELLING", "SUCCEEDED", "INTERRUPTED", "CANCELLED"]
    records = [training(fixture, index * 2, state=run_states[index % len(run_states)]) for index in range(26)]
    records += [sync(fixture, index * 2 + 1, state=sync_states[index % len(sync_states)]) for index in range(26)]
    save(fixture, *records)
    expected = sorted([(row.created_at, row.id, "training" if isinstance(row, Run) else "data-sync", row.state)
                       for row in records], reverse=True)
    return fixture, expected


def test_default_pages_merge_both_kinds_before_slicing_and_remain_stable(mixed):
    fixture, expected = mixed
    platform = fixture[0]
    pages = [get(platform, page=number) for number in (1, 2, 3)]
    assert [len(page["items"]) for page in pages] == [20, 20, 12]
    for number, page in enumerate(pages, 1):
        assert {field: page[field] for field in ("total", "total_all", "page", "page_size", "total_pages")} == {
            "total": 52, "total_all": 52, "page": number, "page_size": 20, "total_pages": 3}
    all_keys = [key for page in pages for key in keys(page)]
    assert len(set(all_keys)) == 52
    assert all_keys == [f"{kind}:{identifier}" for _, identifier, kind, _ in expected]
    assert get(platform, page=2) == pages[1]
    assert {entry["kind"] for entry in pages[0]["items"]} == {"training", "data-sync"}


@pytest.mark.parametrize("kind", ["ALL", "training", "data-sync"])
@pytest.mark.parametrize("status", ["ALL", "ACTIVE", "SUCCEEDED", "FAILED", "CANCELLED"])
def test_type_and_status_filters_apply_before_page_counts_and_slices(mixed, kind, status):
    fixture, expected = mixed
    def selected(row):
        _, _, item_kind, state = row
        if kind != "ALL" and kind != item_kind:
            return False
        if status == "ACTIVE":
            return state in (TRAINING_ACTIVE if item_kind == "training" else SYNC_ACTIVE)
        if status == "FAILED":
            return state in {"FAILED", "INTERRUPTED"}
        return status == "ALL" or state == status
    wanted = [row for row in expected if selected(row)]
    page = get(fixture[0], type=kind, status=status, page_size=5, page=2)
    total_pages = max(1, (len(wanted) + 4) // 5)
    current = min(2, total_pages)
    assert page["total"] == len(wanted) and page["total_all"] == 52
    assert page["page"] == current and page["total_pages"] == total_pages and page["page_size"] == 5
    assert keys(page) == [f"{item_kind}:{identifier}" for _, identifier, item_kind, _ in wanted[(current - 1) * 5:current * 5]]


def test_search_finds_machine_labels_hostnames_task_names_and_copy_paths(fixture):
    platform, _, source, target, *_ = fixture
    with platform.app.state.db.write() as session:
        session.get(Node, source["node_id"]).display_name = "Source Workbench"
        session.get(Node, source["node_id"]).hostname = "source-linux-host"
        session.get(Node, target["node_id"]).display_name = "Target Thor"
        session.get(Node, target["node_id"]).hostname = "target-jetson-host"
    save(fixture, training(fixture, 0, name="叠衣 NeedLe task"),
         sync(fixture, 1, source_path="/machine/source-only-dataset",
              target_directory="/machine/destination-only", target_path="/machine/destination-only/exact-copy"))
    both = {"training:run-0000", "data-sync:sync-0001"}
    for query in ("  source WORKBENCH  ", "source-linux-host", source["node_id"]):
        assert set(keys(get(platform, q=query))) == both
    for query in ("target thor", "target-jetson-host", "source-only-dataset", "destination-only", "exact-copy"):
        assert keys(get(platform, q=query)) == ["data-sync:sync-0001"]
    for query in ("needle", "叠衣", "模型训练"):
        assert keys(get(platform, q=query)) == ["training:run-0000"]
    assert keys(get(platform, q="数据同步")) == ["data-sync:sync-0001"]
    assert keys(get(platform, q="   ")) == ["data-sync:sync-0001", "training:run-0000"]


@pytest.mark.parametrize("query,expected", [("%", ["training:run-0000"]), ("_", ["training:run-0000"]),
                                            ("100%_", ["training:run-0000"]), ("100AB", ["training:run-0001"])])
def test_search_treats_sql_wildcards_as_literal_text(fixture, query, expected):
    save(fixture, training(fixture, 0, name="literal 100%_exact"), training(fixture, 1, name="literal 100ABexact"))
    assert keys(get(fixture[0], q=query)) == expected


def test_soft_deleted_records_are_excluded_from_global_and_filtered_totals(fixture):
    save(fixture, training(fixture, 0), sync(fixture, 1),
         training(fixture, 2, name="deleted-only-needle", deleted_at=now()),
         sync(fixture, 3, name="deleted-only-needle", deleted_at=now()))
    page = get(fixture[0])
    assert page["total"] == page["total_all"] == 2
    assert keys(page) == ["data-sync:sync-0001", "training:run-0000"]
    filtered = get(fixture[0], q="deleted-only-needle", page=99)
    assert filtered["items"] == [] and filtered["total"] == 0 and filtered["total_all"] == 2
    assert filtered["page"] == filtered["total_pages"] == 1


def test_stable_ties_keep_same_id_tasks_distinct_across_page_boundaries(fixture):
    save(fixture, training(fixture, 0, id="shared-id", created_at=timestamp(2)),
         sync(fixture, 1, id="shared-id", created_at=timestamp(2)),
         training(fixture, 2, id="z-last-tie", created_at=timestamp(2)))
    first, second, third = [get(fixture[0], page=index, page_size=1) for index in (1, 2, 3)]
    assert keys(first) + keys(second) + keys(third) == ["training:z-last-tie", "training:shared-id", "data-sync:shared-id"]
    assert get(fixture[0], page=2, page_size=1) == second


def test_out_of_range_pages_clamp_after_removing_the_last_page_record(fixture):
    platform = fixture[0]
    save(fixture, *(training(fixture, index) for index in range(21)))
    before = get(platform, page=99)
    assert before["page"] == before["total_pages"] == 2 and keys(before) == ["training:run-0000"]
    response = platform.client.delete("/api/v1/runs/run-0000", headers=ADMIN)
    assert response.status_code == 204, response.text
    after = get(platform, page=2)
    assert after["page"] == after["total_pages"] == 1 and after["total"] == 20 and len(after["items"]) == 20
    empty = get(platform, q="no-such-task", page=1000)
    assert empty["items"] == [] and empty["total"] == 0 and empty["page"] == empty["total_pages"] == 1


def test_older_tasks_remain_accessible_beyond_each_legacy_list_limit(fixture):
    platform = fixture[0]
    save(fixture, *(training(fixture, index) for index in range(501)), *(sync(fixture, index) for index in range(501)))
    page = get(platform, page_size=100, page=11)
    assert page["total"] == page["total_all"] == 1002 and page["total_pages"] == page["page"] == 11
    assert keys(page) == ["data-sync:sync-0000", "training:run-0000"]
    assert keys(get(platform, q="run-0000")) == ["training:run-0000"]
    assert keys(get(platform, q="sync-0000")) == ["data-sync:sync-0000"]
    assert get(platform, type="training", page_size=100, page=6)["total"] == 501
    assert keys(get(platform, type="training", page_size=100, page=6)) == ["training:run-0000"]
    assert keys(get(platform, type="data-sync", page_size=100, page=6)) == ["data-sync:sync-0000"]


def test_items_keep_rich_progress_and_output_payloads_without_mutating_history(fixture):
    platform, _, source, _, source_asset, _ = fixture
    run = training(fixture, 0, inputs={"dataset": source_asset["id"]})
    copy = sync(fixture, 1, state="TRANSFERRING", bytes_total=1000, bytes_transferred=250,
                progress_pct=25, files_total=10, files_transferred=-1, logs="copy progress\n")
    save(fixture, run, copy)
    attempt_id = uid()
    with platform.app.state.db.write() as session:
        session.add(Attempt(id=attempt_id, run_id=run.id, node_id=source["node_id"], number=1, execution_token=uid(),
            state="SUCCEEDED", created_at=timestamp(0), latest_metrics={"step": 100, "loss": 0.125},
            resources={"gpu": 1}, output_dir="/machine/output/finished"))
        session.flush()
        session.add(Asset(id=uid(), node_id=source["node_id"], name="trained checkpoint", kind="MODEL",
            path="/machine/output/finished/100", version="100", state="READY", metadata_json={"step": 100},
            created_at=timestamp(0), origin_attempt_id=attempt_id))
    before = snapshots(platform)
    page = get(platform)
    rich_run = next(item["item"] for item in page["items"] if item["kind"] == "training")
    rich_copy = next(item["item"] for item in page["items"] if item["kind"] == "data-sync")
    assert rich_run == platform.client.get(f"/api/v1/runs/{run.id}", headers=ADMIN).json()
    assert rich_copy == platform.client.get(f"/api/v1/asset-transfers/{copy.id}", headers=ADMIN).json()
    assert rich_run["attempts"][0]["latest_metrics"]["loss"] == 0.125
    assert rich_run["outputs"][0]["kind"] == "MODEL" and rich_run["input_assets"][0]["id"] == source_asset["id"]
    assert rich_copy["bytes_transferred"] == 250 and rich_copy["files_transferred"] is None
    assert "binding_snapshot" not in rich_copy and "manifest" not in rich_copy
    assert snapshots(platform) == before


@pytest.mark.parametrize("params", [{"page": 0}, {"page": -1}, {"page": "1.5"}, {"page_size": 0},
    {"page_size": 101}, {"page_size": "many"}, {"type": "deployment"}, {"type": "TRAINING"},
    {"status": "UNKNOWN"}, {"status": "done"}, {"q": "x" * 201}])
def test_invalid_pagination_and_filter_parameters_are_rejected(fixture, params):
    assert fixture[0].client.get(BASE, headers=ADMIN, params=params).status_code == 422


def test_viewers_can_read_pages_but_agent_and_enrollment_credentials_cannot(fixture):
    platform, _, source, *_ = fixture
    save(fixture, training(fixture, 0), sync(fixture, 1))
    for headers in ({}, ENROLL, source["headers"]):
        assert platform.client.get(BASE, headers=headers).status_code == 401
    username, password = "background-viewer", "background-task-read-password"
    platform.app.state.auth.create_user(username, password, "viewer")
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert login.status_code == 200, login.text
        response = client.get(BASE, params={"page_size": 1, "page": 2})
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 2 and keys(response.json()) == ["training:run-0000"]
