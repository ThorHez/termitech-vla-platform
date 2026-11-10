"""Routed metadata discovery, ownership and atomic registration on synthetic SSH."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.auth_models import User
from vla_control.common import now, offset_time, uid
from vla_control.dataset_batch_models import DatasetBatchRegistration, DatasetScan
from vla_control.dataset_batches import BatchInput
from vla_control.main import create_app
from vla_control.models import Asset, Command, Node, Run
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.settings import Settings


PASSWORD = "isolated-dataset-batch-password"
PROFILE = {"schema_version": 1, "format": "lerobot", "robot_type": "ARX5", "arm_mode": "single",
           "action_representation": "unknown", "task_description": "Fold cloth.",
           "cameras": [{"name": "top", "key": "observation.images.top", "kind": "rgb"}],
           "field_annotations": [{"key": "actions", "role": "action", "representation": "composite",
               "value_mode": "delta", "component": "other", "arm": "left", "shape": [16, 10], "horizon": 16}]}


@pytest.fixture
def batch(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "batch.db"),
        enrollment_token="isolated-batch-agent", scheduler_enabled=False, session_cookie_secure=False))
    app.state.auth.create_user("batch-admin", PASSWORD, "admin")
    manager = app.state.dataset_batch_manager
    manager._launch = lambda _: None
    connection_id, node_id, stamp = uid(), uid(), now()
    with app.state.db.write() as session:
        session.add(Node(id=node_id, name="fixture", hostname="fixture", status="OFFLINE", token_hash="b" * 64,
            roots=["/machine"], output_root="/machine/out", created_at=stamp, last_seen=stamp, metrics={}, gpus=[]))
        session.add(RemoteConnection(id=connection_id, host="192.0.2.10", port=22, username="fixture-user",
            encrypted_password=app.state.remote_manager.vault.encrypt("fixture-ssh-password"),
            fingerprint="SHA256:fixture-pinned-host", probe_result={}, install_dir="/machine/agent",
            created_at=stamp, expires_at=offset_time(-60)))
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection_id, node_id=node_id,
            idempotency_key=uid(), request_digest="c" * 64, target_key=uid(), onboarding_id=uid(), state="SUCCEEDED",
            stage="SUCCEEDED", logs=[], history=[], agent_config={"_operation": "install"}, install_dir="/machine/agent",
            control_plane_url="https://fixture", connection_mode="direct", tunnel_state="NOT_REQUIRED",
            created_at=stamp, updated_at=stamp, finished_at=stamp))
    remotes = []
    class Remote:
        def __init__(self, **values):
            self.values, self.closed = values, False
            remotes.append(self)
        def connect(self):
            assert self.values["expected_fingerprint"] == "SHA256:fixture-pinned-host"
            assert self.values["password"] == "fixture-ssh-password"
        def close(self):
            self.closed = True
    app.state.remote_manager._remote_factory = Remote
    def scanner(remote, path, roots, **options):
        assert roots == ["/machine"] and path == "/machine/datasets"
        assert options["recursive"] is True and options["max_datasets"] == 200
        return {"path": path, "entries": [
            {"path": path + "/" + name, "name": name, "relative_path": name,
             "info_sha256": hashlib.sha256(name.encode()).hexdigest(), "feature_signature": "f" * 64,
             "features": [{"key": "observation.images.top", "dtype": "video", "shape": [3, 224, 224], "names": []}],
             "summary": {"fps": 20, "robot_type": "ARX5", "total_frames": 100, "task_description": "Fold cloth.", "private_token": "never-return"},
             "error": None if name != "bad" else {"code": "INVALID_METADATA", "message": "Metadata unreadable"}}
            for name in ("first", "nested/second", "bad")], "warnings": [], "truncated": False,
            "scanned_directories": 6, "skipped_directories": 0}
    manager.scanner = scanner
    with TestClient(app, base_url="http://testserver") as client:
        auth = client.post("/api/v1/auth/login", json={"username": "batch-admin", "password": PASSWORD}).json()
        client.headers.update({"X-CSRF-Token": auth["csrf_token"], "X-Dataset-Scan-User": auth["user"]["id"]})
        yield app, client, manager, node_id, connection_id, auth, remotes


@pytest.fixture
def operator_batch(batch):
    app, client, manager, node_id, connection_id, _, remotes = batch
    app.state.auth.create_user("batch-operator", PASSWORD, "operator")
    client.cookies.clear()
    response = client.post("/api/v1/auth/login", json={"username": "batch-operator", "password": PASSWORD})
    assert response.status_code == 200, response.text
    authenticated = response.json()
    client.headers.update({"X-CSRF-Token": authenticated["csrf_token"],
                           "X-Dataset-Scan-User": authenticated["user"]["id"]})
    return app, client, manager, node_id, connection_id, authenticated, remotes


def base(fixture):
    return f"/api/v1/nodes/{fixture[3]}/dataset-scans"


def scan(fixture, ready=True):
    response = fixture[1].post(base(fixture), json={"path": "/machine/datasets", "recursive": True, "max_depth": 4})
    assert response.status_code == 202, response.text
    value = response.json()
    if ready:
        fixture[2]._run(value["id"])
        response = fixture[1].get(base(fixture) + "/" + value["id"])
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["state"] == "READY", value
    return value


def submit(fixture, value, entries=None, key=None):
    return fixture[1].post(base(fixture) + "/" + value["id"] + "/register", json={
        "idempotency_key": key or uid(), "entries": entries or [
            {"path": "/machine/datasets/first", "name": "first cloth", "version": "v1", "profile": deepcopy(PROFILE)},
            {"path": "/machine/datasets/nested/second", "name": "second cloth", "version": "v2", "profile": deepcopy(PROFILE)}]})


def assets(fixture):
    return fixture[1].get("/api/v1/assets").json()


def test_offline_agent_scan_uses_pinned_ssh_and_only_returns_whitelisted_metadata(batch):
    value = scan(batch)
    assert len(value["entries"]) == 3 and value["entries"][2]["error"]
    assert value["entries"][0]["summary"] == {"fps": 20, "robot_type": "ARX5", "total_frames": 100, "task_description": "Fold cloth."}
    assert "never-return" not in json.dumps(value) and "password" not in json.dumps(value)
    assert len(batch[6]) == 1 and batch[6][0].closed
    assert assets(batch) == []
    with batch[0].state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(Command)) == 0
        assert session.scalar(select(func.count()).select_from(Run)) == 0


def test_batch_registers_selected_datasets_and_preserves_source_horizon_and_profile(batch):
    value = scan(batch)
    response = submit(batch, value)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["created_count"] == 2 and result["skipped_count"] == 0
    assert {a["path"] for a in result["created"]} == {"/machine/datasets/first", "/machine/datasets/nested/second"}
    for asset in result["created"]:
        assert asset["kind"] == "DATASET" and asset["state"] == "REGISTERED" and asset["node_id"] == batch[3]
        assert asset["registered_by_user_id"] == batch[5]["user"]["id"]
        label = asset["metadata"]["dataset_profile"]["field_annotations"][0]
        assert label["representation"] == "composite" and label["horizon"] == 16
        assert asset["metadata"]["batch_registration"]["metadata_only"] is True


def test_existing_dataset_paths_are_skipped_including_archived_and_preview_is_updated(batch):
    value = scan(batch)
    first = submit(batch, value).json()
    with batch[0].state.db.write() as session:
        session.get(Asset, first["created"][0]["id"]).state = "ARCHIVED"
    again = submit(batch, value)
    assert again.status_code == 200 and again.json()["created_count"] == 0 and again.json()["skipped_count"] == 2
    assert len(assets(batch)) == 2
    refreshed = batch[1].get(base(batch) + "/" + value["id"]).json()
    assert refreshed["entries"][0]["existing_asset_id"]
    assert "ARCHIVED" in {e.get("existing_state") for e in refreshed["entries"]}


def test_committed_idempotent_replay_is_frozen_even_after_scan_expiry_and_binding_change(batch):
    value = scan(batch)
    key = uid()
    first = submit(batch, value, key=key)
    with batch[0].state.db.write() as session:
        session.get(DatasetScan, value["id"]).expires_at = offset_time(-1)
        session.get(RemoteConnection, batch[4]).fingerprint = "SHA256:changed"
    replay = submit(batch, value, key=key)
    assert replay.status_code == 200 and replay.json() == first.json()
    entries = [{"path": "/machine/datasets/first", "name": "different"}]
    conflict = submit(batch, value, entries=entries, key=key)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "DATASET_BATCH_IDEMPOTENCY_CONFLICT"
    assert len(assets(batch)) == 2


@pytest.mark.parametrize("case", ["missing", "bad_metadata", "outside", "duplicate", "invalid_profile", "unknown", "ambiguous_norm"])
def test_invalid_registration_never_leaves_partial_assets(batch, case):
    value = scan(batch)
    entry = {"path": "/machine/datasets/first", "name": "good", "norm_path": "/machine/norm"}
    bad = {"path": "/machine/datasets/nested/second", "name": "bad"}
    if case == "missing": bad["path"] = "/machine/datasets/not-found"
    elif case == "bad_metadata": bad["path"] = "/machine/datasets/bad"
    elif case == "outside": bad["path"] = "/outside/data"
    elif case == "duplicate": bad["path"] = "/machine/datasets//first/./"
    elif case == "invalid_profile": bad["profile"] = {**deepcopy(PROFILE), "processing": {}}
    elif case == "unknown": bad["user_id"] = "foreign"
    elif case == "ambiguous_norm": bad.update(norm_path="/machine/norm", norm_asset_id=None)
    response = submit(batch, value, entries=[entry, bad])
    assert response.status_code == 422, response.text
    assert assets(batch) == []
    with batch[0].state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(DatasetBatchRegistration)) == 0


def test_norm_creation_is_shared_and_atomic_with_datasets(batch):
    value = scan(batch)
    entries = [{"path": path, "name": "cloth", "norm_path": "/machine/stats/norm_stats.json"}
               for path in ("/machine/datasets/first", "/machine/datasets/nested/second")]
    response = submit(batch, value, entries=entries)
    assert response.status_code == 200, response.text
    assert len({a["norm_asset_id"] for a in response.json()["created"]}) == 1
    all_assets = assets(batch)
    assert len(all_assets) == 3 and len([a for a in all_assets if a["kind"] == "NORM"]) == 1


def test_invalid_late_norm_rolls_back_earlier_norm_and_all_datasets(batch):
    value = scan(batch)
    entries = [{"path": "/machine/datasets/first", "name": "first", "norm_path": "/machine/new-norm"},
               {"path": "/machine/datasets/nested/second", "name": "second", "norm_asset_id": "missing"}]
    assert submit(batch, value, entries=entries).status_code == 404
    assert assets(batch) == []


def mixed_norm_scan(batch):
    original = batch[2].scanner
    def four_datasets(*args, **kwargs):
        value = original(*args, **kwargs)
        for name in ("unbound", "reuse"):
            entry = deepcopy(value["entries"][0])
            entry.update(path=value["path"] + "/" + name, name=name, relative_path=name,
                         info_sha256=hashlib.sha256(name.encode()).hexdigest())
            value["entries"].append(entry)
        return value
    batch[2].scanner = four_datasets
    return scan(batch)


def norm_asset(batch, path, state="REGISTERED"):
    asset_id = uid()
    with batch[0].state.db.write() as session:
        session.add(Asset(id=asset_id, node_id=batch[3], kind="NORM", path=path,
                          name="fixture stats", version="v1", state=state,
                          metadata_json={"fixture": "preserve"}, created_at=now()))
    return asset_id


@pytest.mark.parametrize("unbound", [{}, {"norm_asset_id": None}])
def test_each_dataset_keeps_independent_norm_binding_and_path_reuse_on_replay(batch, unbound):
    value = mixed_norm_scan(batch)
    selected_norm = norm_asset(batch, "/machine/stats/existing")
    entries = [
        {"path": "/machine/datasets/first", "name": "existing id", "norm_asset_id": selected_norm},
        {"path": "/machine/datasets/nested/second", "name": "new path", "norm_path": "/machine/stats/new"},
        {"path": "/machine/datasets/unbound", "name": "no norm", **unbound},
        {"path": "/machine/datasets/reuse", "name": "existing path", "norm_path": "/machine//stats/./existing/"},
    ]
    key = uid()
    response = submit(batch, value, entries=entries, key=key)
    assert response.status_code == 200, response.text
    result = response.json()
    created = {asset["path"]: asset for asset in result["created"]}
    assert result["created_count"] == 4 and result["skipped_count"] == 0
    assert created[entries[0]["path"]]["norm_asset_id"] == selected_norm
    assert created[entries[3]["path"]]["norm_asset_id"] == selected_norm
    assert created[entries[2]["path"]]["norm_asset_id"] is None
    new_norm_id = created[entries[1]["path"]]["norm_asset_id"]
    assert new_norm_id and new_norm_id != selected_norm
    norms = {asset["id"]: asset for asset in assets(batch) if asset["kind"] == "NORM"}
    assert len(norms) == 2 and norms[new_norm_id]["path"] == "/machine/stats/new"
    assert norms[selected_norm]["metadata"] == {"fixture": "preserve"}
    # A retry must return exactly the original independent bindings, rather
    # than re-evaluating unavailable defaults or creating another path asset.
    with batch[0].state.db.write() as session:
        session.get(Asset, selected_norm).state = "UNAVAILABLE"
        session.get(DatasetScan, value["id"]).expires_at = offset_time(-1)
    replay = submit(batch, value, entries=entries, key=key)
    assert replay.status_code == 200 and replay.json() == result
    assert len(assets(batch)) == 6
    with batch[0].state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(DatasetBatchRegistration)) == 1


def test_invalid_per_dataset_norm_path_rolls_back_mixed_batch_and_same_key_can_retry(batch, monkeypatch):
    from vla_control import dataset_batches

    value = mixed_norm_scan(batch)
    selected_norm = norm_asset(batch, "/machine/stats/existing")
    unavailable_norm = norm_asset(batch, "/machine/stats/z-unavailable", state="ARCHIVED")
    entries = [
        {"path": "/machine/datasets/first", "name": "existing id", "norm_asset_id": selected_norm},
        {"path": "/machine/datasets/nested/second", "name": "new path", "norm_path": "/machine/stats/a-new"},
        {"path": "/machine/datasets/unbound", "name": "no norm", "norm_asset_id": None},
        {"path": "/machine/datasets/reuse", "name": "invalid path", "norm_path": "/machine/stats/z-unavailable"},
    ]
    original = dataset_batches.resolve_norm_path
    resolved = []
    def observe_resolution(session, **kwargs):
        result = original(session, **kwargs)
        resolved.append((kwargs["path"], result.id))
        return result
    monkeypatch.setattr(dataset_batches, "resolve_norm_path", observe_resolution)
    before = assets(batch)
    key = uid()
    response = submit(batch, value, entries=entries, key=key)
    assert response.status_code == 409 and response.json()["error"]["code"] == "NORM_ASSET_UNAVAILABLE"
    # The first path was actually flushed before the later path failed, so
    # these assertions cover transaction rollback, not only early validation.
    assert resolved and resolved[0][0] == "/machine/stats/a-new"
    assert assets(batch) == before
    with batch[0].state.db.read() as session:
        assert session.get(Asset, resolved[0][1]) is None
        assert session.scalar(select(func.count()).select_from(DatasetBatchRegistration)) == 0
    repeat = submit(batch, value, entries=entries, key=key)
    assert repeat.status_code == 409 and repeat.json() == response.json()
    assert assets(batch) == before
    # Failed transactions do not consume the idempotency key. After fixing
    # just that dependency, the identical request can commit and replay.
    with batch[0].state.db.write() as session:
        session.get(Asset, unavailable_norm).state = "REGISTERED"
    fixed = submit(batch, value, entries=entries, key=key)
    assert fixed.status_code == 200, fixed.text
    saved = fixed.json()
    bindings = {asset["path"]: asset["norm_asset_id"] for asset in saved["created"]}
    assert bindings[entries[0]["path"]] == selected_norm
    assert bindings[entries[2]["path"]] is None
    assert bindings[entries[3]["path"]] == unavailable_norm
    assert bindings[entries[1]["path"]] not in {None, selected_norm, unavailable_norm}
    assert submit(batch, value, entries=entries, key=key).json() == saved
    assert len(assets(batch)) == 7
    with batch[0].state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(DatasetBatchRegistration)) == 1


@pytest.mark.parametrize("change,code", [("expiry", "DATASET_SCAN_EXPIRED"), ("roots", "DATASET_SCAN_BINDING_CHANGED"),
    ("fingerprint", "DATASET_SCAN_BINDING_CHANGED"), ("credentials", "DATASET_SCAN_BINDING_CHANGED"),
    ("retired", "NODE_CONNECTION_REMOVED")])
def test_uncommitted_scan_binding_must_still_be_current(batch, change, code):
    value = scan(batch)
    with batch[0].state.db.write() as session:
        if change == "expiry": session.get(DatasetScan, value["id"]).expires_at = offset_time(-1)
        elif change == "roots": session.get(Node, batch[3]).roots = ["/machine", "/extra"]
        elif change == "retired": session.get(Node, batch[3]).status = "DISCONNECTED"
        else: setattr(session.get(RemoteConnection, batch[4]), change if change == "fingerprint" else "encrypted_password", "changed")
    response = submit(batch, value)
    assert response.status_code in {409, 410} and response.json()["error"]["code"] == code
    assert assets(batch) == []


def test_foreign_owner_and_scan_user_switch_fail_closed(batch):
    value = scan(batch)
    batch[0].state.auth.create_user("other-admin", PASSWORD, "admin")
    with TestClient(batch[0], base_url="http://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "other-admin", "password": PASSWORD}).json()
        client.headers["X-CSRF-Token"] = login["csrf_token"]
        assert client.get(base(batch) + "/" + value["id"]).status_code == 404
        response = client.post(base(batch) + "/" + value["id"] + "/register", json={"idempotency_key": uid(), "entries": [{"path": "/machine/datasets/first", "name": "bad"}]})
        assert response.status_code == 404
    changed = batch[1].get(base(batch) + "/" + value["id"], headers={"X-Dataset-Scan-User": "other"})
    assert changed.status_code == 403


@pytest.mark.parametrize("role,status", [("viewer", 403), ("operator", 202), ("admin", 202)])
def test_scan_permission_and_csrf_are_enforced(batch, role, status):
    batch[0].state.auth.create_user("role-" + role, PASSWORD, role)
    with TestClient(batch[0], base_url="http://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "role-" + role, "password": PASSWORD}).json()
        payload = {"path": "/machine/datasets"}
        assert client.post(base(batch), json=payload).status_code == 403
        response = client.post(base(batch), json=payload, headers={"X-CSRF-Token": login["csrf_token"]})
        assert response.status_code == status, response.text
        assert client.post(base(batch), json=payload, headers={"X-CSRF-Token": login["csrf_token"], "Origin": "https://foreign.invalid"}).status_code == 403


def test_operator_can_scan_and_register_new_datasets_with_initial_norm_but_cannot_modify_assets(operator_batch):
    app, client, _, _, _, authenticated, _ = operator_batch
    assert "datasets.register" in authenticated["permissions"] and "assets.write" not in authenticated["permissions"]
    value = scan(operator_batch)
    entries = [{"path": "/machine/datasets/first", "name": "operator dataset", "version": "v1",
                "profile": deepcopy(PROFILE), "norm_path": "/machine/norms/first.json"}]
    csrf = client.headers.pop("X-CSRF-Token")
    denied = submit(operator_batch, value, entries=entries)
    assert denied.status_code == 403 and assets(operator_batch) == []
    client.headers["X-CSRF-Token"] = csrf
    response = submit(operator_batch, value, entries=entries)
    assert response.status_code == 200, response.text
    created = response.json()["created"][0]
    before = assets(operator_batch)
    assert len(before) == 2 and {row["kind"] for row in before} == {"DATASET", "NORM"}
    assert created["norm_asset_id"] == next(row["id"] for row in before if row["kind"] == "NORM")
    assert all(created["metadata"]["dataset_profile"][key] == value for key, value in PROFILE.items())
    for method, path, body in (
        ("PATCH", "/dataset-profile", {"profile": deepcopy(PROFILE)}),
        ("PUT", "/norm-binding", {"norm_asset_id": None}),
        ("POST", "/validation/cancel", {}),
    ):
        denied = client.request(method, "/api/v1/assets/" + created["id"] + path, json=body)
        assert denied.status_code == 403, (method, path, denied.text)
    assert assets(operator_batch) == before
    with app.state.db.read() as session:
        assert session.get(DatasetScan, value["id"]).user_id == authenticated["user"]["id"]
    archived = client.post("/api/v1/assets/" + created["id"] + "/archive")
    assert archived.status_code == 200, archived.text
    assert archived.json()["registered_by_user_id"] == authenticated["user"]["id"]
    assert archived.json()["state"] == "ARCHIVED"


def test_operator_repeat_registration_preserves_existing_asset_and_does_not_create_unused_norm(operator_batch):
    value = scan(operator_batch)
    first = submit(operator_batch, value)
    assert first.status_code == 200, first.text
    before = assets(operator_batch)
    response = submit(operator_batch, value, entries=[
        {"path": "/machine/datasets/first", "name": "attempted replacement", "version": "other",
         "profile": {**deepcopy(PROFILE), "task_description": "Changed"},
         "norm_path": "/machine/norms/unused.json"}])
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 0 and response.json()["skipped_count"] == 1
    assert assets(operator_batch) == before


@pytest.mark.parametrize("phase", ["queued", "result"])
def test_operator_scan_worker_rechecks_registration_rights_before_ssh_and_before_saving_results(operator_batch, monkeypatch, phase):
    app, _, manager, _, _, authenticated, remotes = operator_batch
    value = scan(operator_batch, ready=False)
    user_id = authenticated["user"]["id"]
    if phase == "queued":
        app.state.auth.update_user(None, user_id, {"role": "viewer"})
        monkeypatch.setattr(manager.remote_manager, "_remote", lambda *_: pytest.fail("Revoked scan must not open SSH"))
    else:
        original = manager.scanner

        def revoke_before_results(*args, **kwargs):
            result = original(*args, **kwargs)
            app.state.auth.update_user(None, user_id, {"role": "viewer"})
            return result

        monkeypatch.setattr(manager, "scanner", revoke_before_results)
    manager._run(value["id"])
    with app.state.db.read() as session:
        row = session.get(DatasetScan, value["id"])
        assert row.state == "FAILED" and row.error["code"] == "FORBIDDEN" and row.result is None
        assert session.scalar(select(func.count()).select_from(Asset)) == 0
    assert len(remotes) == (0 if phase == "queued" else 1)
    assert all(remote.closed for remote in remotes)


def test_accepted_scan_remains_authorized_when_admin_becomes_operator(batch):
    app, client, manager, _, _, authenticated, _ = batch
    app.state.auth.create_user("remaining-admin", PASSWORD, "admin")
    value = scan(batch, ready=False)
    app.state.auth.update_user(None, authenticated["user"]["id"], {"role": "operator"})
    manager._run(value["id"])
    with app.state.db.read() as session:
        assert session.get(DatasetScan, value["id"]).state == "READY"
    response = client.post("/api/v1/auth/login", json={"username": "batch-admin", "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    assert client.get(base(batch) + "/" + value["id"]).status_code == 200
    assert submit(batch, value).status_code == 200


@pytest.mark.parametrize("payload", [{"path": "/"}, {"path": "/machine/../outside"}, {"path": "relative"},
    {"path": "/machine/line\nname"}, {"path": "/machine/datasets", "max_depth": 9},
    {"path": "/machine/datasets", "recursive": 1}, {"path": "/machine/datasets", "host": "evil"}])
def test_scan_request_path_and_limits_cannot_escape(batch, payload):
    response = batch[1].post(base(batch), json=payload)
    assert response.status_code == 422
    assert batch[6] == []


def test_malformed_remote_result_is_not_saved_or_registered(batch):
    value = scan(batch, ready=False)
    batch[2].scanner = lambda *args, **kwargs: {"path": "/machine/datasets", "entries": [{"path": "/machine/secret", "name": "bad"}], "warnings": [], "truncated": False}
    batch[2]._run(value["id"])
    result = batch[1].get(base(batch) + "/" + value["id"]).json()
    assert result["state"] == "FAILED" and result["entries"] == []
    assert submit(batch, value).status_code == 409
    assert assets(batch) == []


def test_restart_marks_pending_scan_failed_without_changing_assets(batch):
    value = scan(batch, ready=False)
    batch[2].close()
    batch[2]._started = False
    batch[2].start()
    result = batch[1].get(base(batch) + "/" + value["id"]).json()
    assert result["state"] == "FAILED" and result["error"]["code"] == "DATASET_SCAN_INTERRUPTED"
    assert assets(batch) == []


def test_concurrent_same_key_returns_one_frozen_receipt_and_one_set_of_assets(batch):
    value = scan(batch)
    body = BatchInput.model_validate({"idempotency_key": uid(), "entries": [
        {"path": "/machine/datasets/first", "name": "first", "norm_path": "/machine/stats"},
        {"path": "/machine/datasets/nested/second", "name": "second", "norm_path": "/machine/stats"}]})
    barrier = threading.Barrier(2)
    def register():
        barrier.wait()
        return batch[2].register(batch[5]["user"]["id"], batch[3], value["id"], body)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: register(), range(2)))
    assert results[0] == results[1]
    assert len(assets(batch)) == 3
    with batch[0].state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(DatasetBatchRegistration)) == 1


def test_expired_preview_body_is_pruned_without_losing_committed_idempotent_receipt(batch):
    value = scan(batch)
    key = uid()
    first = submit(batch, value, key=key).json()
    with batch[0].state.db.write() as session:
        session.get(DatasetScan, value["id"]).expires_at = offset_time(-1)
    scan(batch, ready=False)  # admission also prunes expired heavy previews
    with batch[0].state.db.read() as session:
        assert session.get(DatasetScan, value["id"]).result is None
    assert submit(batch, value, key=key).json() == first
    assert submit(batch, value).status_code == 410


def test_partial_scan_warning_and_large_skip_count_are_preserved(batch):
    original = batch[2].scanner
    def truncated(*args, **kwargs):
        value = original(*args, **kwargs)
        value.update(truncated=True, warnings=["Depth limit reached; results incomplete"], skipped_directories=20000)
        return value
    batch[2].scanner = truncated
    value = scan(batch)
    assert value["truncated"] and value["skipped_directories"] == 20000
    assert "incomplete" in value["warnings"][0]
