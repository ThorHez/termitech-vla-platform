"""Independent account, scope and atomicity checks for folder registration."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.auth_models import User, UserSession
from vla_control.common import now, offset_time, uid
from vla_control.dataset_batch_models import DatasetBatchRegistration, DatasetScan
from vla_control.dataset_batches import _scan_result
from vla_control.models import Asset, Command, Node
from vla_control.remote_models import RemoteConnection

from .test_checkpoint_transfers import bind_ssh


PASSWORD = "batch-security-fixture-123"
PARENT = "/machine/datasets"


@pytest.fixture
def secured(platform):
    node = platform.register()
    connection = bind_ssh(platform, node, "192.0.2.41")
    platform.app.state.dataset_batch_manager._launch = lambda _id: None
    for name, role in (("batch-owner", "admin"), ("batch-admin", "admin"),
                       ("batch-operator", "operator"), ("batch-reader", "viewer")):
        platform.app.state.auth.create_user(name, PASSWORD, role)
    client = TestClient(platform.app, base_url="https://testserver")
    yield platform, client, node, connection
    client.close()


def login(fixture, username="batch-owner"):
    client = fixture[1]
    client.cookies.clear()
    client.headers.pop("X-Dataset-Batch-User", None)
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200
    value = response.json()
    client.headers.update({"X-CSRF-Token": value["csrf_token"], "X-Dataset-Batch-User": value["user"]["id"]})
    return value


def base(fixture):
    return f"/api/v1/nodes/{fixture[2]['node_id']}/dataset-scans"


def create(fixture):
    response = fixture[1].post(base(fixture), json={"path": PARENT})
    assert response.status_code == 202, response.text
    return response.json()


def ready(fixture):
    scan = create(fixture)
    entries = [{"path": PARENT + "/" + name, "relative_path": name, "name": name,
        "info_sha256": "a" * 64, "feature_signature": "b" * 64,
        "summary": {"fps": 20}, "error": None} for name in ("a", "b")]
    value = {"path": PARENT, "entries": entries, "warnings": [], "truncated": False,
             "scanned_directories": 3, "skipped_directories": 0}
    with fixture[0].app.state.db.write() as session:
        row = session.get(DatasetScan, scan["id"])
        row.state, row.result = "READY", _scan_result(value, PARENT, ["/machine"])
    return scan


def body(key=None, **changes):
    return {"idempotency_key": key or uid(), "entries": [
        {"path": PARENT + "/a", "name": "dataset a", "version": "v1", **changes}]}


def register(fixture, scan, value):
    return fixture[1].post(base(fixture) + "/" + scan["id"] + "/register", json=value)


def counts(fixture):
    with fixture[0].app.state.db.read() as session:
        return {model.__tablename__: session.scalar(select(func.count()).select_from(model))
                for model in (Asset, Command, DatasetScan, DatasetBatchRegistration)}


def test_scan_requires_cookie_account_csrf_and_asset_permission(secured):
    platform, client, *_ = secured
    assert client.post(base(secured), json={"path": PARENT}).status_code == 401
    assert client.post(base(secured), json={"path": PARENT}, headers={"Authorization": "Bearer test-admin"}).status_code == 401
    login(secured)
    csrf = client.headers.pop("X-CSRF-Token")
    assert client.post(base(secured), json={"path": PARENT}).status_code == 403
    client.headers["X-CSRF-Token"] = csrf
    assert client.post(base(secured), json={"path": PARENT}, headers={"Origin": "https://foreign.example"}).status_code == 403
    login(secured, "batch-reader")
    assert client.post(base(secured), json={"path": PARENT}).status_code == 403
    assert client.get(base(secured) + "/unknown-scan").status_code == 403
    assert client.post(base(secured) + "/unknown-scan/register", json=body()).status_code == 403
    assert counts(secured)["dataset_scans"] == 0


def test_even_a_valid_bootstrap_bearer_cannot_own_a_private_scan(platform):
    node = platform.register()
    headers = {"Authorization": "Bearer test-admin"}
    assert platform.client.get("/api/v1/assets", headers=headers).status_code == 200
    response = platform.client.post(f"/api/v1/nodes/{node['node_id']}/dataset-scans",
                                    headers=headers, json={"path": PARENT})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "DATASET_BATCH_ACCOUNT_REQUIRED"


@pytest.mark.parametrize("method", ["GET", "REGISTER"])
@pytest.mark.parametrize("username", ["batch-admin", "batch-operator"])
def test_another_authorized_account_cannot_read_or_register_private_scan(secured, method, username):
    login(secured)
    scan = ready(secured)
    before = counts(secured)
    login(secured, username)
    response = (secured[1].get(base(secured) + "/" + scan["id"]) if method == "GET"
                else register(secured, scan, body()))
    assert response.status_code == 404
    assert counts(secured) == before


@pytest.mark.parametrize("header", ["X-Dataset-Batch-User", "X-Dataset-Scan-User"])
def test_stale_account_header_prevents_scan_and_registration(secured, header):
    previous = login(secured)
    scan = ready(secured)
    login(secured, "batch-admin")
    secured[1].headers[header] = previous["user"]["id"]
    before = counts(secured)
    assert secured[1].post(base(secured), json={"path": PARENT}).status_code == 403
    assert secured[1].get(base(secured) + "/" + scan["id"]).status_code == 403
    assert register(secured, scan, body()).status_code == 403
    assert counts(secured) == before


@pytest.mark.parametrize("path", ["/", "/etc", "relative/path", "/machine/../etc", "/machine/\nsecret"])
def test_scan_parent_cannot_escape_or_scan_the_whole_filesystem(secured, path):
    login(secured)
    before = counts(secured)
    response = secured[1].post(base(secured), json={"path": path})
    assert response.status_code == 422
    assert counts(secured) == before


@pytest.mark.parametrize("path", ["/machine/not-discovered", "/etc/dataset"])
def test_unscanned_paths_cannot_be_registered_even_with_a_valid_scan(secured, path):
    login(secured)
    scan = ready(secured)
    value = body(norm_path="/machine/new-norm")
    value["entries"][0]["path"] = path
    before = counts(secured)
    assert register(secured, scan, value).status_code == 422
    assert counts(secured) == before


def test_duplicate_aliases_in_one_batch_create_no_assets_or_norms(secured):
    login(secured)
    scan = ready(secured)
    value = body(norm_path="/machine/new-norm")
    value["entries"].append({**value["entries"][0], "path": PARENT + "//a/", "name": "alias"})
    before = counts(secured)
    assert register(secured, scan, value).status_code == 422
    assert counts(secured) == before


def test_batch_validation_rolls_back_automatic_norms_and_earlier_entries(secured):
    login(secured)
    scan = ready(secured)
    with secured[0].app.state.db.write() as session:
        wrong = Asset(id=uid(), node_id=secured[2]["node_id"], name="not norm", kind="MODEL",
                      path="/machine/model", version="v1", state="REGISTERED", metadata_json={}, created_at=now())
        session.add(wrong)
    value = body(norm_path="/machine/new-norm")
    value["entries"].append({"path": PARENT + "/b", "name": "b", "norm_asset_id": wrong.id})
    before = counts(secured)
    assert register(secured, scan, value).status_code == 409
    assert counts(secured) == before


@pytest.mark.parametrize("mutation", ["roots", "ssh", "retired", "expired"])
def test_changed_machine_authority_or_expired_discovery_cannot_create_assets(secured, mutation):
    login(secured)
    scan = ready(secured)
    with secured[0].app.state.db.write() as session:
        if mutation == "roots": session.get(Node, secured[2]["node_id"]).roots = ["/machine", "/another"]
        elif mutation == "ssh": session.get(RemoteConnection, secured[3].id).fingerprint = "SHA256:changed"
        elif mutation == "retired": session.get(Node, secured[2]["node_id"]).status = "DISCONNECTED"
        else: session.get(DatasetScan, scan["id"]).expires_at = offset_time(-1)
    before = counts(secured)
    assert register(secured, scan, body(norm_path="/machine/new-norm")).status_code in {409, 410}
    assert counts(secured) == before


def test_committed_idempotent_replay_survives_scan_expiry_and_returns_original_receipt(secured):
    login(secured)
    scan = ready(secured)
    value = body(norm_path="/machine/shared-norm")
    first = register(secured, scan, value)
    assert first.status_code == 200, first.text
    before = counts(secured)
    with secured[0].app.state.db.write() as session:
        session.get(DatasetScan, scan["id"]).expires_at = offset_time(-1)
        session.get(RemoteConnection, secured[3].id).fingerprint = "SHA256:changed"
    replay = register(secured, scan, value)
    assert replay.status_code == 200 and replay.json() == first.json()
    assert counts(secured) == before
    changed = deepcopy(value)
    changed["entries"][0]["name"] = "different request"
    assert register(secured, scan, changed).status_code == 409
    assert counts(secured) == before


def test_privilege_revocation_prevents_registration_and_job_visibility(secured):
    user = login(secured)
    scan = ready(secured)
    with secured[0].app.state.db.write() as session:
        session.get(User, user["user"]["id"]).enabled = False
    before = counts(secured)
    assert register(secured, scan, body()).status_code == 401
    assert secured[1].get(base(secured) + "/" + scan["id"]).status_code == 401
    assert counts(secured) == before
