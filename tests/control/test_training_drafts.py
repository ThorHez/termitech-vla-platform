"""Private, bounded unfinished configurations cannot cross account boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.auth_models import User, UserSession
from vla_control.common import offset_time
from vla_control.main import create_app
from vla_control.models import TrainingDraft
from vla_control.settings import Settings
from vla_control.training_drafts import MAX_CONTENT_BYTES, MAX_DEPTH, MAX_NODES


PASSWORD = "draft-test-password-123"
PATH = "/api/v1/training-draft"


@pytest.fixture
def drafts(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "drafts.sqlite3"),
                        enrollment_token="separate-agent-token", scheduler_enabled=False,
                        session_cookie_secure=False)
    app = create_app(settings)
    for username, role in (("alice", "admin"), ("bob", "operator"), ("reader", "viewer")):
        app.state.auth.create_user(username, PASSWORD, role)
    with TestClient(app) as client:
        yield app, client


def login(client, username):
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()


def save(client, content, revision=0):
    return client.put(PATH, json={"expected_revision": revision, "content": content})


def clear(client, revision):
    return client.request("DELETE", PATH, json={"expected_revision": revision})


def test_partial_drafts_are_private_durable_data_not_training_requests(drafts):
    app, client = drafts
    alice = login(client, "alice")
    empty = client.get(PATH)
    assert empty.status_code == 200
    assert empty.json() == {"content": None, "revision": 0, "updated_at": None}
    content = {"schema_version": 1, "request": {"name": "half finished", "inputs": {},
               "dataset_bindings": [{"dataset_asset_id": "", "weight": None}],
               "parameters": {"batch_size": "unfinished"}},
               "incomplete_numbers": [{"path": ["request", "gpu_count"], "value": "NaN"}]}
    response = save(client, content)
    assert response.status_code == 200, response.text
    assert response.json()["content"] == content and response.json()["revision"] == 1
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["updated_at"]
    assert client.get(PATH).json() == response.json()
    with app.state.db.read() as session:
        assert session.get(TrainingDraft, alice["user"]["id"]).content == content
    # A new authenticated session restores the same account's draft.
    login(client, "alice")
    assert client.get(PATH).json()["content"] == content


def test_two_users_cannot_read_write_or_clear_each_others_draft(drafts):
    app, client = drafts
    alice = login(client, "alice")
    assert save(client, {"request": {"name": "private alice"}}).status_code == 200
    bob = login(client, "bob")
    assert client.get(PATH).json()["content"] is None
    assert save(client, {"request": {"name": "private bob"}}).status_code == 200
    # Even an administrator cannot target another account through the body or URL.
    assert client.put(PATH, json={"expected_revision": 1, "content": {},
                                 "user_id": alice["user"]["id"]}).status_code == 422
    assert client.get(PATH + "/" + alice["user"]["id"]).status_code == 404
    assert client.get(PATH, params={"user_id": alice["user"]["id"]}).json()["content"]["request"]["name"] == "private bob"
    assert clear(client, 1).status_code == 200
    login(client, "alice")
    assert client.get(PATH).json()["content"]["request"]["name"] == "private alice"
    with app.state.db.read() as session:
        assert session.get(TrainingDraft, bob["user"]["id"]).content is None


def test_revision_conflicts_and_clear_tombstones_prevent_lost_updates(drafts):
    _, client = drafts
    login(client, "alice")
    first = save(client, {"request": {"name": "first"}}).json()
    second = save(client, {"request": {"name": "second"}}, first["revision"])
    assert second.status_code == 200 and second.json()["revision"] == 2
    stale = save(client, {"request": {"name": "stale"}}, first["revision"])
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "TRAINING_DRAFT_CONFLICT"
    assert clear(client, first["revision"]).status_code == 409
    cleared = clear(client, 2)
    assert cleared.status_code == 200 and cleared.json()["content"] is None and cleared.json()["revision"] == 3
    assert save(client, {"request": {"name": "delayed"}}, 2).status_code == 409
    assert client.get(PATH).json() == cleared.json()
    assert save(client, {"request": {"name": "new draft"}}, 3).json()["revision"] == 4


def test_captured_owner_assertion_blocks_delayed_requests_after_account_switch(drafts):
    _, client = drafts
    alice = login(client, "alice")
    alice_id = alice["user"]["id"]
    assert client.get(PATH, headers={"X-Training-Draft-User": alice_id}).status_code == 200
    bob = login(client, "bob")
    assert save(client, {"request": {"name": "bob's current work"}}).status_code == 200
    stale_headers = {"X-Training-Draft-User": alice_id}
    requests = [
        client.get(PATH, headers=stale_headers),
        client.put(PATH, headers=stale_headers, json={"expected_revision": 1, "content": {"request": {"name": "delayed Alice"}}}),
        client.request("DELETE", PATH, headers=stale_headers, json={"expected_revision": 1}),
    ]
    for response in requests:
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "TRAINING_DRAFT_USER_CHANGED"
    current = client.get(PATH, headers={"X-Training-Draft-User": bob["user"]["id"]})
    assert current.json()["content"]["request"]["name"] == "bob's current work"
    assert current.json()["revision"] == 1


def test_clearing_absent_draft_also_blocks_delayed_initial_save(drafts):
    _, client = drafts
    login(client, "alice")
    assert clear(client, 0).json()["revision"] == 1
    assert save(client, {"request": {"name": "old initial save"}}, 0).status_code == 409
    assert client.get(PATH).json()["content"] is None


def test_concurrent_saves_have_one_winner_and_one_conflict(drafts):
    app, client = drafts
    authentication = login(client, "alice")
    cookies = dict(client.cookies)
    csrf = authentication["csrf_token"]
    barrier = Barrier(2)
    def worker(label):
        with TestClient(app) as other:
            other.cookies.update(cookies)
            other.headers["X-CSRF-Token"] = csrf
            barrier.wait(timeout=5)
            return save(other, {"request": {"name": label}})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(worker, ["one", "two"]))
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert next(response for response in responses if response.status_code == 409).json()["error"]["code"] == "TRAINING_DRAFT_CONFLICT"
    assert client.get(PATH).json()["content"]["request"]["name"] in {"one", "two"}


def test_viewer_agent_and_unauthenticated_access_are_rejected(drafts):
    _, client = drafts
    assert client.get(PATH).status_code == 401
    assert client.get(PATH, headers={"Authorization": "Bearer separate-agent-token"}).status_code == 401
    login(client, "reader")
    assert client.get(PATH).status_code == 403
    assert save(client, {}).status_code == 403
    assert clear(client, 0).status_code == 403


def test_legacy_administrator_without_personal_identity_cannot_use_drafts(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "legacy.sqlite3"),
                             enrollment_token="separate-enrollment", admin_token="legacy-only",
                             legacy_admin_enabled=True, scheduler_enabled=False,
                             session_cookie_secure=False))
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer legacy-only"
        response = client.get(PATH)
        assert response.status_code == 403 and response.json()["error"]["code"] == "TRAINING_DRAFT_ACCOUNT_REQUIRED"
        assert save(client, {}).status_code == 403
        assert clear(client, 0).status_code == 403


def test_csrf_origin_and_changed_account_state_are_enforced(drafts):
    app, client = drafts
    authentication = login(client, "alice")
    client.headers.pop("X-CSRF-Token")
    assert save(client, {}).json()["error"]["code"] == "CSRF_REJECTED"
    assert clear(client, 0).json()["error"]["code"] == "CSRF_REJECTED"
    client.headers["X-CSRF-Token"] = authentication["csrf_token"]
    assert client.put(PATH, json={"expected_revision": 0, "content": {}},
                      headers={"Origin": "https://untrusted.example"}).json()["error"]["code"] == "ORIGIN_REJECTED"
    with app.state.db.write() as session:
        session.get(User, authentication["user"]["id"]).enabled = False
    assert client.get(PATH).status_code == 401
    assert save(client, {}).status_code == 401


def test_expired_session_cannot_read_or_write_retained_draft(drafts):
    app, client = drafts
    authentication = login(client, "alice")
    assert save(client, {"request": {"name": "keep"}}).status_code == 200
    with app.state.db.write() as session:
        row = session.scalar(select(UserSession).where(UserSession.user_id == authentication["user"]["id"]))
        row.expires_at = offset_time(-1)
    assert client.get(PATH).status_code == 401
    assert save(client, {}).status_code == 401
    login(client, "alice")
    assert client.get(PATH).json()["content"]["request"]["name"] == "keep"


@pytest.mark.parametrize("value", [True, -1, 1.5, "0", None, 2**53])
def test_revision_must_be_strict_nonnegative_safe_integer(drafts, value):
    _, client = drafts
    login(client, "alice")
    assert save(client, {}, value).status_code == 422


@pytest.mark.parametrize("key", ["preview_id", "confirmed_config_sha256", "code_snapshot", "idempotency_key",
                                 "execution_token", "csrf_token", "password", "admin_token", "enrollment_token",
                                 "access_token", "refresh_token", "csrfToken", "execution-token", "PreviewID",
                                 "sessionToken", "Authorization", "cookie", "password_hash"])
def test_credentials_and_previous_approval_are_never_cached(drafts, key):
    _, client = drafts
    login(client, "alice")
    response = save(client, {"request": {"parameters": {key: "do not store"}}})
    assert response.status_code == 422 and response.json()["error"]["code"] == "TRAINING_DRAFT_INVALID"
    assert client.get(PATH).json()["revision"] == 0


def test_legitimate_tokenizer_and_data_keys_are_not_blanket_rejected(drafts):
    _, client = drafts
    login(client, "alice")
    content = {"request": {"inputs": {"key": "asset"}, "parameters": {"tokenizer": "gemma", "key_masks": [True]}}}
    assert save(client, content).json()["content"] == content


def test_json_size_depth_and_node_limits_apply_before_storage(drafts):
    _, client = drafts
    login(client, "alice")
    assert save(client, {"value": "x" * MAX_CONTENT_BYTES}).status_code == 413
    nested = {}
    for _ in range(MAX_DEPTH + 1):
        nested = {"child": nested}
    assert save(client, nested).status_code == 422
    assert save(client, {"values": [None] * MAX_NODES}).status_code == 422
    assert client.get(PATH).json()["content"] is None


@pytest.mark.parametrize("raw", ['{"expected_revision":0,"content":{"value":NaN}}',
                                 '{"expected_revision":0,"content":{"value":Infinity}}',
                                 '{"expected_revision":0,"content":{"value":-Infinity}}',
                                 'not json', '[]', '{"expected_revision":0,"content":null}'])
def test_malformed_and_nonfinite_json_is_rejected(drafts, raw):
    _, client = drafts
    login(client, "alice")
    response = client.put(PATH, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "TRAINING_DRAFT_INVALID"
