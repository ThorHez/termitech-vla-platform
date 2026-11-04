"""Private, bounded unfinished configurations cannot cross account boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.auth_models import User, UserSession
from vla_control.common import offset_time
from vla_control.main import create_app
from vla_control.models import TrainingDraft, TrainingTemplateDraft
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


def save(client, content, revision=0, template_id=None):
    params = {} if template_id is None else {"template_id": template_id}
    return client.put(PATH, params=params, json={"expected_revision": revision, "content": content})


def clear(client, revision, template_id=None):
    params = {} if template_id is None else {"template_id": template_id}
    return client.request("DELETE", PATH, params=params, json={"expected_revision": revision})


def test_templates_have_independent_drafts_revisions_and_clear_tombstones(drafts):
    app, client = drafts
    user_id = login(client, "alice")["user"]["id"]
    contents = {
        "openpi-sft": {"request": {"template_id": "openpi-sft", "name": "my training",
            "parameters": {"batch_size": 8}, "dataset_bindings": [{"dataset_asset_id": "training-data"}]}},
        "openpi-norm": {"request": {"template_id": "openpi-norm", "name": "my normalization",
            "parameters": {"num_batches": None}, "norm_datasets": [{"dataset_asset_id": "norm-data"}]}},
    }
    for template_id, content in contents.items():
        empty = client.get(PATH, params={"template_id": template_id})
        assert empty.json() == {"content": None, "revision": 0, "updated_at": None, "template_id": template_id}
        saved = save(client, content, template_id=template_id)
        assert saved.status_code == 200, saved.text
        assert saved.json()["revision"] == 1
        assert saved.json()["template_id"] == template_id
        assert client.get(PATH, params={"template_id": template_id}).json() == saved.json()
    changed = {"request": {"template_id": "openpi-sft", "name": "edited training"}}
    assert save(client, changed, 1, "openpi-sft").json()["revision"] == 2
    assert save(client, contents["openpi-sft"], 1, "openpi-sft").status_code == 409
    assert clear(client, 1, "openpi-sft").status_code == 409
    cleared = clear(client, 2, "openpi-sft")
    assert cleared.status_code == 200
    assert cleared.json()["template_id"] == "openpi-sft"
    assert cleared.json()["content"] is None and cleared.json()["revision"] == 3
    assert save(client, changed, 2, "openpi-sft").status_code == 409
    other = client.get(PATH, params={"template_id": "openpi-norm"}).json()
    assert other["content"] == contents["openpi-norm"] and other["revision"] == 1
    assert client.get(PATH).json()["revision"] == 0
    with app.state.db.read() as session:
        assert session.get(TrainingTemplateDraft, (user_id, "openpi-sft")).content is None
        assert session.get(TrainingTemplateDraft, (user_id, "openpi-norm")).content == contents["openpi-norm"]
        assert session.get(TrainingDraft, user_id) is None
    login(client, "alice")
    assert client.get(PATH, params={"template_id": "openpi-norm"}).json() == other
    assert clear(client, 0, "smoke").json()["revision"] == 1
    assert save(client, {"request": {"template_id": "smoke"}}, 0, "smoke").status_code == 409


def test_scoped_drafts_remain_private_and_reject_mismatched_template_content(drafts):
    _, client = drafts
    alice_id = login(client, "alice")["user"]["id"]
    content = {"request": {"template_id": "openpi-sft", "name": "Alice private training"}}
    assert save(client, content, template_id="openpi-sft").status_code == 200
    assert save(client, content, template_id="openpi-norm").status_code == 422
    assert save(client, {}, template_id="openpi-norm").status_code == 422
    assert client.get(PATH, params={"template_id": "openpi-norm"}).json()["revision"] == 0
    login(client, "bob")
    assert client.get(PATH, params={"template_id": "openpi-sft", "user_id": alice_id}).json()["revision"] == 0
    content["request"]["name"] = "Bob private training"
    assert save(client, content, template_id="openpi-sft").status_code == 200
    assert clear(client, 1, "openpi-sft").status_code == 200
    assert client.get(PATH, params={"template_id": "openpi-sft"},
                      headers={"X-Training-Draft-User": alice_id}).status_code == 403
    login(client, "alice")
    restored = client.get(PATH, params={"template_id": "openpi-sft"}).json()
    assert restored["content"]["request"]["name"] == "Alice private training" and restored["revision"] == 1


@pytest.mark.parametrize("same_template", [True, False])
def test_concurrent_scoped_saves_conflict_only_for_the_same_template(drafts, same_template):
    app, client = drafts
    authentication = login(client, "alice")
    cookies = dict(client.cookies)
    barrier = Barrier(2)
    template_ids = ["openpi-sft", "openpi-sft" if same_template else "openpi-norm"]
    def worker(entry):
        index, template_id = entry
        with TestClient(app) as other:
            other.cookies.update(cookies)
            other.headers["X-CSRF-Token"] = authentication["csrf_token"]
            barrier.wait(timeout=5)
            return save(other, {"request": {"template_id": template_id, "name": str(index)}},
                        template_id=template_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(worker, enumerate(template_ids)))
    assert sorted(response.status_code for response in responses) == ([200, 409] if same_template else [200, 200])
    for index, template_id in enumerate(template_ids):
        current = client.get(PATH, params={"template_id": template_id}).json()
        assert current["revision"] == 1
        assert current["content"]["request"]["name"] in ({"0", "1"} if same_template else {str(index)})


@pytest.mark.parametrize("template_id", ["", " ", "x" * 101, "invalid\nname", " leading", "trailing "])
def test_invalid_template_scope_is_rejected_for_every_operation(drafts, template_id):
    _, client = drafts
    login(client, "alice")
    assert client.get(PATH, params={"template_id": template_id}).status_code == 422
    assert save(client, {"request": {"template_id": template_id}}, template_id=template_id).status_code == 422
    assert clear(client, 0, template_id).status_code == 422


def test_duplicate_template_scope_is_rejected(drafts):
    _, client = drafts
    login(client, "alice")
    assert client.get(PATH, params=[("template_id", "openpi-sft"), ("template_id", "openpi-norm")]).status_code == 422


def test_upgrade_preserves_legacy_draft_and_sessions_and_never_repopulates_cleared_scope(drafts):
    app, client = drafts
    alice_id = login(client, "alice")["user"]["id"]
    content = {"schema_version": 1, "request": {"template_id": "openpi-sft", "name": "unfinished old training"},
               "special_values": [{"path": ["request", "parameters", "batch_size"], "value": "nan"}]}
    assert save(client, content).status_code == 200
    with app.state.db.write() as session:
        row = session.get(TrainingDraft, alice_id)
        row.revision = 7
    before = client.get(PATH).json()
    # An existing deployment has only the old table. The additive upgrade must
    # create its replacement without dropping users, sessions or legacy data.
    TrainingTemplateDraft.__table__.drop(app.state.db.engine)
    app.state.db.initialize()
    assert client.get("/api/v1/auth/me").status_code == 200
    assert client.get(PATH).json() == before
    assert client.get(PATH, params={"template_id": "openpi-sft"}).json() == {**before, "template_id": "openpi-sft"}
    assert client.get(PATH, params={"template_id": "openpi-norm"}).json()["revision"] == 0
    replacement = {"request": {"template_id": "openpi-sft", "name": "current training"}}
    assert save(client, replacement, 7, "openpi-sft").json()["revision"] == 8
    assert client.get(PATH).json() == before
    assert clear(client, 8, "openpi-sft").json()["revision"] == 9
    assert save(client, replacement, 7).json()["revision"] == 8
    app.state.db.initialize()
    app.state.db.initialize()
    scoped = client.get(PATH, params={"template_id": "openpi-sft"}).json()
    assert scoped["content"] is None and scoped["revision"] == 9
    legacy = client.get(PATH).json()
    assert legacy["content"] == replacement and legacy["revision"] == 8


def test_upgrade_does_not_guess_template_for_unidentified_legacy_draft(drafts):
    app, client = drafts
    login(client, "alice")
    content = {"request": {"name": "no template identity"}}
    assert save(client, content).status_code == 200
    app.state.db.initialize()
    assert client.get(PATH).json()["content"] == content
    for template_id in ("openpi-sft", "openpi-norm", "smoke"):
        assert client.get(PATH, params={"template_id": template_id}).json()["revision"] == 0


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
