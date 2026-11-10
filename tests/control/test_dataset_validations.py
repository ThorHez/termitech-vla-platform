"""Dataset registration queues bounded CPU checks without launching training."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import now, offset_time, uid
from vla_control.dataset_validations import INTERNAL_KEY, queue_validation
from vla_control.models import Allocation, Asset, Attempt, Command, Node, Profile, Run

from .conftest import ADMIN, ENROLL, profile_inspection
from .test_asset_transfers import create as create_transfer, fixture, result_for, run_copy
from .test_dataset_batches import batch, scan, submit
from .test_dataset_profiles import profile as dataset_profile


def worker(platform, name="dataset-worker", *, supported=True):
    node = platform.register(name=name, adapter="openpi-sft")
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).metrics = {"capabilities": ["DATASET_VALIDATE"] if supported else []}
    return node


def register(platform, node, **changes):
    body = {"kind": "DATASET", "name": "dataset", "node_id": node["node_id"], "path": "/machine/data",
            "metadata": {"dataset_profile": dataset_profile()}, **changes}
    if body["kind"] == "DATASET" and body["metadata"].get("dataset_profile"):
        body["dataset_inspection_id"] = profile_inspection(platform, node["node_id"], body["path"], body["metadata"]["dataset_profile"])
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    return response.json()


def get(platform, asset):
    response = platform.client.get(f"/api/v1/assets/{asset['id']}/validation", headers=ADMIN)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def create(platform, asset, **body):
    return platform.client.post(f"/api/v1/assets/{asset['id']}/validations", headers=ADMIN, json=body)


def result(asset, profile_id="default"):
    return {"schema_version": 1, "scope": "dataset_source", "path": asset["path"], "profile_id": profile_id,
        "source_fingerprint": "f" * 64,
        "checks": [{"name": name, "ok": True, "message": "verified"} for name in (
            "source_metadata", "dataset_timestamps", "raw_samples", "dataset_cache")],
        "timestamp_contract": {"schema_version": 1, "status": "verified", "scope": "all parquet timestamp/episode_index rows",
            "effective_fps": 20, "expected_interval_seconds": .05, "tolerance_s": 1e-4, "checked_rows": 6,
            "checked_pairs": 4, "checked_episodes": 2, "parquet_files": 2, "metadata_sha256": "a" * 64,
            "episodes_sha256": "b" * 64, "data_manifest_digest": "c" * 64,
            "runtime": {"module": "lerobot.common.datasets.lerobot_dataset", "module_path": "/machine/code/lerobot_dataset.py",
                "module_sha256": "d" * 64, "constructor_signature": "(self, tolerance_s=0.0001)",
                "validator": "check_timestamps_sync", "validator_path": "/machine/code/utils.py", "validator_sha256": "e" * 64}},
        "raw_samples": {"selection": "first_middle_last", "checked_samples": 3, "sample_indices": [0, 3, 5],
            "samples": [{"index": index, "fields": {"actions": {"shape": [7], "dtype": "float32"}}} for index in (0, 3, 5)]},
        "runtime": {"python_executable": "/usr/bin/python3", "python_version": "3.11.15",
            "loader_path": "/machine/code/lerobot_dataset.py", "loader_sha256": "d" * 64,
            "validator_path": "/machine/code/utils.py", "validator_sha256": "e" * 64}}


def ack(platform, node, report, value, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{report['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": status, "result": value})


def test_registration_queues_one_dataset_scoped_command_and_safe_ack_updates_only_validation(platform):
    node = worker(platform)
    asset = register(platform, node)
    report = asset["validation"]
    assert report["status"] == "PENDING" and report["scope"] == "dataset_source" and report["profile_id"] == "default"
    assert INTERNAL_KEY not in asset["metadata"]
    command = platform.commands(node)[0]
    assert command["id"] == report["id"] and command["type"] == "DATASET_VALIDATE"
    assert set(command["payload"]) == {"asset_id", "path", "profile_id", "registered_profile", "input_digest", "expires_at",
                                      "source_registration"}
    assert command["payload"]["source_registration"] is True
    assert command["payload"]["asset_id"] == asset["id"] and command["payload"]["input_digest"] == report["input_digest"]
    assert command["payload"]["registered_profile"] == asset["metadata"]["dataset_profile"]
    assert create(platform, asset).json()["id"] == report["id"]
    assert ack(platform, node, report, result(asset)).status_code == 200
    success = get(platform, asset)
    assert success["status"] == "SUCCEEDED" and success["completed_at"]
    assert success["source_fingerprint"] == "f" * 64 and success["raw_samples"]["checked_samples"] == 3
    listing = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    assert listing[0]["validation"] == success and listing[0]["metadata"] == asset["metadata"]
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).state == "REGISTERED"
        assert list(session.scalars(select(Run))) == list(session.scalars(select(Attempt))) == list(session.scalars(select(Allocation))) == []
        assert session.get(Command, report["id"]).result == result(asset)
    # Completion is immutable if a persisted Agent acknowledgement is retried.
    assert ack(platform, node, report, {"error": {"code": "LATE_ERROR", "message": "ignore"}}, "REJECTED").status_code == 200
    assert get(platform, asset) == success
    assert create(platform, asset).json()["id"] != report["id"]


@pytest.mark.parametrize("registered_path,returned_path", [
    ("/machine/data/", "/machine/data"),
    ("/machine//data/./", "/machine/data"),
    ("///machine/data///", "/machine/data"),
    ("/machine/data", "/machine//data/./"),
    ("/machine/data set/", "/machine/data set"),
    ("/machine/data /", "/machine/data "),
])
def test_equivalent_remote_path_spellings_accept_agent_evidence_without_changing_registration(
        platform, registered_path, returned_path):
    node = worker(platform)
    # This regression covers path identity, independent of optional image/profile registration.
    asset = register(platform, node, path=registered_path, metadata={})
    value = {**result(asset), "path": returned_path}
    assert ack(platform, node, asset["validation"], value).status_code == 200
    report = get(platform, asset)
    assert report["status"] == "SUCCEEDED" and report["error"] is None
    assert report["input_digest"] == asset["validation"]["input_digest"]
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).path == registered_path
        command = session.get(Command, report["id"])
        assert command.payload["path"] == registered_path
        assert command.result == value


@pytest.mark.parametrize("returned_path", [
    "/machine/data-other", "/another/data", "/machine/data ",
    "/machine/link/../data", "machine/data", "/machine/data\x00", "/machine/data\n",
    "/machine/data\x7f", "/machine/data\x85", "", None, [], {}, True,
])
def test_different_or_unsafe_remote_paths_cannot_validate_a_dataset(platform, returned_path):
    node = worker(platform)
    asset = register(platform, node, path="/machine/data/")
    value = {**result(asset), "path": returned_path}
    assert ack(platform, node, asset["validation"], value).status_code == 200
    report = get(platform, asset)
    assert report["status"] == "FAILED" and report["error"]["code"] == "DATASET_VALIDATION_INVALID_RESULT"
    assert report["source_fingerprint"] is None and report["raw_samples"] is None


@pytest.mark.parametrize("corruption", ["profile", "fps", "sample"])
def test_equivalent_paths_do_not_bypass_other_evidence_checks(platform, corruption):
    node = worker(platform)
    asset = register(platform, node, path="/machine/data/", metadata={"dataset_profile": dataset_profile(fps=20)})
    value = {**result(asset), "path": "/machine/data"}
    if corruption == "profile":
        value["profile_id"] = "foreign"
    elif corruption == "fps":
        value["timestamp_contract"].update(effective_fps=30, expected_interval_seconds=1 / 30)
    else:
        value["raw_samples"]["sample_indices"] = [0, 2, 5]
    assert ack(platform, node, asset["validation"], value).status_code == 200
    report = get(platform, asset)
    assert report["status"] == "FAILED" and report["error"]["code"] == "DATASET_VALIDATION_INVALID_RESULT"


@pytest.mark.parametrize("condition,code", [("offline", "DATASET_VALIDATION_NODE_OFFLINE"),
    ("unsupported", "DATASET_VALIDATION_AGENT_UNSUPPORTED"), ("no-profile", "DATASET_VALIDATION_PROFILE_UNAVAILABLE"),
    ("many-profiles", "DATASET_VALIDATION_PROFILE_REQUIRED")])
def test_unavailable_validation_preserves_successful_registration_with_blocked_status(platform, condition, code):
    node = worker(platform, supported=condition != "unsupported")
    with platform.app.state.db.write() as session:
        if condition == "offline":
            session.get(Node, node["node_id"]).last_seen = offset_time(-1000)
        elif condition == "no-profile":
            session.get(Profile, (node["node_id"], "default")).active = False
        elif condition == "many-profiles":
            session.add(Profile(node_id=node["node_id"], id="another", name="second", code_dir="/machine/other",
                python="/usr/bin/python3", adapter="openpi-sft", recipe="pi0_base", metadata_json={}, active=True))
    asset = register(platform, node)
    assert asset["validation"]["status"] == "BLOCKED" and asset["validation"]["error"]["code"] == code
    assert platform.commands(node) == []
    if condition == "many-profiles":
        chosen = create(platform, asset, profile_id="another")
        assert chosen.status_code == 202 and chosen.json()["status"] == "PENDING"
        assert chosen.json()["profile_id"] == "another"


@pytest.mark.parametrize("change", ["metadata", "path", "profile", "roots"])
def test_changed_source_or_profile_is_stale_and_old_ack_cannot_restore_success(platform, change):
    node = worker(platform)
    asset = register(platform, node)
    old = asset["validation"]
    with platform.app.state.db.write() as session:
        if change == "metadata":
            row = session.get(Asset, asset["id"])
            row.metadata_json = {**row.metadata_json, "notes": "new source declaration"}
        elif change == "path":
            session.get(Asset, asset["id"]).path = "/machine/new-data"
        elif change == "profile":
            session.get(Profile, (node["node_id"], "default")).python = "/usr/local/bin/python3"
        else:
            session.get(Node, node["node_id"]).roots = ["/machine", "/another"]
    assert get(platform, asset)["status"] == "STALE"
    assert ack(platform, node, old, result(asset)).status_code == 200
    assert get(platform, asset)["status"] == "STALE"
    assert create(platform, asset).json()["id"] != old["id"]


def test_profile_edit_invalidates_previous_request_and_late_ack_never_overwrites_newer_request(platform):
    node = worker(platform)
    asset = register(platform, node)
    old = asset["validation"]
    updated = {**asset["metadata"]["dataset_profile"], "task_description": "new declaration"}
    response = platform.client.patch(f"/api/v1/assets/{asset['id']}/dataset-profile", headers=ADMIN,
                                     json={"dataset_profile": updated})
    assert response.status_code == 200 and response.json()["validation"]["status"] == "STALE"
    fresh = create(platform, asset).json()
    assert fresh["id"] != old["id"] and fresh["status"] == "PENDING"
    assert ack(platform, node, old, result(asset)).status_code == 200
    assert get(platform, asset)["id"] == fresh["id"] and get(platform, asset)["status"] == "PENDING"


@pytest.mark.parametrize("corruption", ["path", "profile", "extra", "raw-array", "partial-timestamps", "sample-index", "huge", "nonfinite",
                                       "missing-check", "failed-check", "validator-path", "validator-sha"])
def test_malformed_agent_results_never_become_reusable_validation(platform, corruption):
    node = worker(platform)
    asset = register(platform, node)
    value = result(asset)
    if corruption in {"path", "profile"}:
        value["path" if corruption == "path" else "profile_id"] = "/machine/foreign" if corruption == "path" else "foreign"
    elif corruption == "extra":
        value["arbitrary"] = "not accepted"
    elif corruption == "raw-array":
        value["raw_samples"]["samples"][0]["fields"]["actions"]["values"] = [1, 2, 3]
    elif corruption == "partial-timestamps":
        value["timestamp_contract"]["checked_pairs"] = 1
    elif corruption == "sample-index":
        value["raw_samples"]["sample_indices"] = [0, 2, 5]
    elif corruption == "huge":
        value["checks"][0]["message"] = "x" * 300000
    elif corruption == "missing-check":
        value["checks"] = value["checks"][:-1]
    elif corruption == "failed-check":
        value["checks"][0]["ok"] = False
    elif corruption in {"validator-path", "validator-sha"}:
        value["runtime"]["validator_path" if corruption == "validator-path" else "validator_sha256"] = (
            "/machine/code/other.py" if corruption == "validator-path" else "0" * 64)
    else:
        # JSON finite validation runs before a successful receipt can be saved.
        value["timestamp_contract"]["effective_fps"] = "NaN"
    assert ack(platform, node, asset["validation"], value).status_code == 200
    report = get(platform, asset)
    assert report["status"] == "FAILED" and report["error"]["code"] == "DATASET_VALIDATION_INVALID_RESULT"
    assert report["raw_samples"] is None


def test_timestamp_receipt_must_use_the_registered_fps(platform):
    node = worker(platform)
    asset = register(platform, node, metadata={"dataset_profile": dataset_profile(fps=30)})
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    report = get(platform, asset)
    assert report["status"] == "FAILED" and report["error"]["code"] == "DATASET_VALIDATION_INVALID_RESULT"


def test_expired_request_and_agent_failure_have_bounded_persisted_status(platform):
    node = worker(platform)
    asset = register(platform, node)
    old = asset["validation"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, old["id"])
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert ack(platform, node, old, result(asset)).status_code == 200
    assert get(platform, asset)["error"]["code"] == "DATASET_VALIDATION_EXPIRED"
    fresh = create(platform, asset).json()
    assert ack(platform, node, fresh, {"error": {"code": "DATASET_TIMESTAMPS_INVALID", "message": "bad interval"}}, "REJECTED").status_code == 200
    assert get(platform, asset)["error"] == {"code": "DATASET_TIMESTAMPS_INVALID", "message": "bad interval"}


def test_client_cannot_forge_internal_receipt_and_non_dataset_assets_do_not_get_validation(platform):
    node = worker(platform)
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={"kind": "DATASET", "name": "forged",
        "node_id": node["node_id"], "path": "/machine/data", "metadata": {INTERNAL_KEY: {"status": "SUCCEEDED"}}})
    assert response.status_code == 422
    asset = register(platform, node, kind="MODEL", metadata={})
    assert "validation" not in asset
    assert create(platform, asset).status_code == 404
    assert platform.client.get(f"/api/v1/assets/{asset['id']}/validation", headers=ADMIN).status_code == 404


@pytest.mark.parametrize("role,expected", [("admin", 202), ("operator", 403), ("viewer", 403)])
def test_write_permission_csrf_read_permissions_and_machine_ownership(platform, role, expected):
    node = worker(platform)
    asset = register(platform, node)
    path = f"/api/v1/assets/{asset['id']}"
    for headers in ({}, ENROLL, node["headers"]):
        assert platform.client.get(path + "/validation", headers=headers).status_code == 401
        assert platform.client.post(path + "/validations", headers=headers, json={}).status_code == 401
    other = worker(platform, name="other")
    assert ack(platform, other, asset["validation"], result(asset)).status_code == 403
    password = "dataset-background-validation-password"
    platform.app.state.auth.create_user("dataset-validation-" + role, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "dataset-validation-" + role, "password": password})
        assert login.status_code == 200
        assert client.get(path + "/validation").status_code == 200
        assert client.post(path + "/validations", json={}).status_code == 403
        response = client.post(path + "/validations", json={}, headers={"X-CSRF-Token": login.json()["csrf_token"]})
        assert response.status_code == expected, response.text


def test_unknown_profile_is_rejected_and_registered_assets_without_history_return_null(platform):
    node = worker(platform)
    asset = register(platform, node)
    assert create(platform, asset, profile_id="foreign").status_code == 422
    with platform.app.state.db.write() as session:
        row = session.get(Asset, asset["id"])
        row.metadata_json = {key: value for key, value in row.metadata_json.items() if key != INTERNAL_KEY}
    assert get(platform, asset) is None
    assert platform.client.get("/api/v1/assets/missing/validation", headers=ADMIN).status_code == 404


def test_validation_delivery_waits_until_all_training_executions_have_ended(platform):
    node = worker(platform)
    asset = register(platform, node)
    from .test_policy_deployments import add_run
    with platform.app.state.db.write() as session:
        run = add_run(session, node, state="RUNNING")
        command = Command(id=uid(), node_id=node["node_id"], type="CANCEL", status="PENDING", payload={}, created_at=now())
        session.add(command)
    assert [item["type"] for item in platform.commands(node)] == ["CANCEL"]
    with platform.app.state.db.write() as session:
        session.get(Run, run.id).state = "SUCCEEDED"
    commands = platform.commands(node)
    assert commands[0]["type"] == "CANCEL"
    assert commands[-1]["id"] == asset["validation"]["id"]


def test_transfer_registration_uses_fresh_target_validation_and_never_copies_source_receipt(fixture, monkeypatch):
    platform, _, source, target, asset, _ = fixture
    with platform.app.state.db.write() as session:
        for node in (source, target):
            session.get(Node, node["node_id"]).metrics = {"capabilities": ["STORAGE_CONFIGURE", "DATASET_VALIDATE"]}
        report = queue_validation(session, session.get(Asset, asset["id"]))
    transfer = create_transfer(fixture).json()
    def copy(*args, **kwargs):
        # A new source-only validation receipt must not invalidate an existing
        # transport snapshot, which freezes user metadata rather than UI state.
        assert ack(platform, source, report, result(asset)).status_code == 200
        return result_for(args, kwargs)
    final = run_copy(fixture, transfer["id"], monkeypatch, copy)
    assert final["state"] == "SUCCEEDED", final
    target_asset = next(item for item in platform.client.get("/api/v1/assets", headers=ADMIN).json()
                        if item["id"] == final["target_asset_id"])
    assert target_asset["validation"]["status"] == "PENDING"
    assert target_asset["validation"]["id"] != report["id"]
    assert INTERNAL_KEY not in target_asset["metadata"]
    command = next(item for item in platform.commands(target) if item["type"] == "DATASET_VALIDATE")
    assert command["payload"]["asset_id"] == target_asset["id"] and command["payload"]["path"] == target_asset["path"]


def test_batch_registration_automatically_queues_each_new_dataset_and_replay_does_not_duplicate_jobs(batch):
    app, client, _, node_id, *_ = batch
    value = scan(batch)
    with app.state.db.write() as session:
        node = session.get(Node, node_id)
        node.status, node.last_seen, node.metrics = "ONLINE", now(), {"capabilities": ["DATASET_VALIDATE"]}
        session.add(Profile(node_id=node_id, id="default", name="runtime", code_dir="/machine/code", python="/usr/bin/python3",
                            adapter="openpi-sft", recipe="pi0_base", metadata_json={}, active=True))
    key = uid()
    response = submit(batch, value, key=key)
    assert response.status_code == 200, response.text
    created = response.json()["created"]
    assert len(created) == 2 and all(item["validation"]["status"] == "PENDING" for item in created)
    with app.state.db.read() as session:
        commands = list(session.scalars(select(Command).where(Command.type == "DATASET_VALIDATE")))
        assert {item.payload["asset_id"] for item in commands} == {item["id"] for item in created}
    assert submit(batch, value, key=key).json() == response.json()
    with app.state.db.read() as session:
        assert len(list(session.scalars(select(Command).where(Command.type == "DATASET_VALIDATE")))) == 2
