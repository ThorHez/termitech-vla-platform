"""Reusable inputs remain private, revisioned inert data across human sessions."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from vla_control.auth_models import User, UserSession
from vla_control.common import offset_time
from vla_control.db import Database
from vla_control.main import create_app
from vla_control.models import Attempt, Preview, Run, TrainingDraft, TrainingTemplate
from vla_control.settings import Settings
from vla_control.training_drafts import MAX_CONTENT_BYTES, MAX_DEPTH, MAX_NODES
from vla_control.training_templates import MAX_REQUEST_BYTES


PASSWORD = "template-test-password-123"
PATH = "/api/v1/training-templates"


@pytest.fixture
def templates(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "templates.sqlite3"),
                             enrollment_token="separate-agent-token", scheduler_enabled=False,
                             session_cookie_secure=False))
    for username, role in (("alice", "admin"), ("bob", "operator"), ("reader", "viewer")):
        app.state.auth.create_user(username, PASSWORD, role)
    with TestClient(app) as client:
        yield app, client


def login(client, username):
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    result = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
    return result.json()


def create(client, name="双臂联合训练", content=None, description="保留机器、模型和处理组合"):
    return client.post(PATH, json={"name": name, "description": description,
                                  "content": {} if content is None else content})


def update(client, template, **changes):
    return client.put(PATH + "/" + template["id"], json={
        "expected_revision": template["revision"], "name": template["name"],
        "description": template["description"], "content": template["content"], **changes})


def remove(client, template, **changes):
    return client.request("DELETE", PATH + "/" + template["id"],
                          json={"expected_revision": template["revision"], **changes})


def test_multiple_templates_are_durable_private_inputs_without_preflight_or_launch(templates):
    app, client = templates
    authentication = login(client, "alice")
    assert client.get(PATH).json() == []
    content = {"schema_version": 1, "request": {"name": "", "node_id": "saved-machine",
               "profile_id": "saved-environment", "template_id": "openpi-sft", "inputs": {"model": "model-a"},
               "parameters": {"batch_size": 32}, "dataset_bindings": [{"dataset_asset_id": "", "norm_asset_id": "",
               "weight": 1, "processor": {"name": "ComposableDataConfig", "parameters": {"composition": {"action_key": "actions"}}}}]},
               "step": 0, "advanced_recipe": False, "edited_parameters": ["auto_config.action_horizon:automatic"]}
    first = create(client, content=content)
    assert first.status_code == 201, first.text
    saved = first.json()
    assert saved["content"] == content and saved["revision"] == 1
    assert saved["created_at"] == saved["updated_at"] and saved["created_at"]
    assert "user_id" not in saved
    second = create(client, name="另一种训练", content={"request": {"parameters": {"batch_size": "未填完"}}}).json()
    listed = client.get(PATH)
    assert listed.status_code == 200 and listed.headers["Cache-Control"] == "no-store"
    assert [row["id"] for row in listed.json()] == [second["id"], saved["id"]]
    assert all("content" not in row and "user_id" not in row for row in listed.json())
    assert client.get(PATH + "/" + saved["id"]).json() == saved
    # A new browser/session restores independently saved templates, not approvals.
    login(client, "alice")
    assert client.get(PATH + "/" + saved["id"]).json() == saved
    with app.state.db.read() as session:
        assert session.get(TrainingTemplate, saved["id"]).user_id == authentication["user"]["id"]
        assert session.scalar(select(func.count()).select_from(TrainingTemplate)) == 2
        for model in (Run, Preview, Attempt):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_edit_name_description_content_with_revision_then_delete(templates):
    app, client = templates
    login(client, "alice")
    original = create(client).json()
    result = update(client, original, name=" 改名训练 ", description="详细说明\n第二行", content={"request": {"gpu_count": 2}})
    assert result.status_code == 200, result.text
    edited = result.json()
    assert edited["name"] == "改名训练" and edited["description"] == "详细说明\n第二行"
    assert edited["content"] == {"request": {"gpu_count": 2}} and edited["revision"] == 2
    assert edited["created_at"] == original["created_at"] and edited["updated_at"] >= original["updated_at"]
    assert client.get(PATH + "/" + edited["id"]).json() == edited
    assert remove(client, edited).json() == {"id": edited["id"], "deleted": True}
    assert client.get(PATH).json() == []
    assert client.get(PATH + "/" + edited["id"]).status_code == 404
    assert update(client, edited).status_code == 404
    assert remove(client, edited).status_code == 404
    with app.state.db.read() as session:
        assert session.get(TrainingTemplate, edited["id"]) is None


def test_even_administrators_cannot_read_overwrite_delete_or_select_other_owners(templates):
    _, client = templates
    alice = login(client, "alice")["user"]["id"]
    owned_alice = create(client, name="Alice only").json()
    bob = login(client, "bob")["user"]["id"]
    assert client.get(PATH).json() == []
    owned_bob = create(client, name="Bob only").json()
    for foreign, username in [(owned_alice, "bob"), (owned_bob, "alice")]:
        login(client, username)
        assert client.get(PATH + "/" + foreign["id"]).status_code == 404
        assert update(client, foreign, name="Cannot overwrite").status_code == 404
        assert remove(client, foreign).status_code == 404
    own_rows = client.get(PATH, params={"user_id": bob}).json()
    assert [row["id"] for row in own_rows] == [owned_alice["id"]]
    assert create(client, name="still Alice").status_code == 201
    for method, path, body in [
        ("POST", PATH, {"name": "wrong owner", "content": {}, "user_id": bob}),
        ("PUT", PATH + "/" + owned_alice["id"], {"expected_revision": 1, "name": "wrong", "description": "", "content": {}, "user_id": bob}),
        ("DELETE", PATH + "/" + owned_alice["id"], {"expected_revision": 1, "user_id": bob}),
    ]:
        assert client.request(method, path, json=body).status_code == 422
    assert alice != bob
    assert client.get(PATH + "/" + owned_alice["id"]).json()["name"] == "Alice only"


def test_captured_user_assertion_blocks_delayed_actions_after_account_switch(templates):
    _, client = templates
    alice = login(client, "alice")["user"]["id"]
    alice_template = create(client).json()
    bob = login(client, "bob")["user"]["id"]
    bob_template = create(client, name="Bob current").json()
    stale = {"X-Training-Template-User": alice}
    requests = [
        client.get(PATH, headers=stale), client.get(PATH + "/" + bob_template["id"], headers=stale),
        client.post(PATH, headers=stale, json={"name": "delayed", "content": {}}),
        client.put(PATH + "/" + bob_template["id"], headers=stale,
                   json={"expected_revision": 1, "name": "delayed", "description": "", "content": {}}),
        client.request("DELETE", PATH + "/" + bob_template["id"], headers=stale, json={"expected_revision": 1}),
    ]
    for response in requests:
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "TRAINING_TEMPLATE_USER_CHANGED"
    own = client.get(PATH + "/" + bob_template["id"], headers={"X-Training-Template-User": bob})
    assert own.json() == bob_template
    login(client, "alice")
    assert client.get(PATH + "/" + alice_template["id"]).json() == alice_template


def test_stale_update_or_delete_never_overwrites_a_newer_revision(templates):
    _, client = templates
    login(client, "alice")
    first = create(client).json()
    latest = update(client, first, content={"request": {"name": "new work"}}).json()
    for response in [update(client, first, content={"request": {"name": "stale"}}), remove(client, first)]:
        assert response.status_code == 409 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_CONFLICT"
    assert client.get(PATH + "/" + first["id"]).json() == latest


def test_saving_a_trimmed_existing_name_overwrites_the_same_record_and_invalidates_old_cas(templates):
    _, client = templates
    login(client, "alice")
    first = create(client, name=" 可复用训练 ", content={"request": {"name": "old"}}).json()
    response = create(client, name="可复用训练", description="新说明", content={"request": {"name": "new"}})
    assert response.status_code == 200, response.text
    latest = response.json()
    assert latest["id"] == first["id"] and latest["name"] == "可复用训练"
    assert latest["revision"] == 2 and latest["created_at"] == first["created_at"]
    assert latest["description"] == "新说明" and latest["content"] == {"request": {"name": "new"}}
    assert len(client.get(PATH).json()) == 1
    for stale in [update(client, first), remove(client, first)]:
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "TRAINING_TEMPLATE_CONFLICT"
    assert update(client, latest, description="CAS 仍可更新").json()["revision"] == 3
    assert create(client, name="另一个名称").status_code == 201


def test_same_name_in_another_account_is_independent_even_for_administrators(templates):
    _, client = templates
    login(client, "alice")
    alice = create(client, name="shared name", content={"request": {"name": "alice"}}).json()
    login(client, "bob")
    bob_response = create(client, name=" shared name ", content={"request": {"name": "bob"}})
    assert bob_response.status_code == 201
    bob = bob_response.json()
    assert bob["id"] != alice["id"] and bob["revision"] == 1
    assert create(client, name="shared name", description="bob update").json()["revision"] == 2
    login(client, "alice")
    assert client.get(PATH + "/" + alice["id"]).json() == alice
    assert [row["id"] for row in client.get(PATH).json()] == [alice["id"]]


def test_put_rename_collision_is_explicit_and_never_deletes_or_overwrites_the_other_template(templates):
    _, client = templates
    login(client, "alice")
    first = create(client, name="first", content={"request": {"name": "one"}}).json()
    second = create(client, name="second", content={"request": {"name": "two"}}).json()
    response = update(client, first, name=" second ", content={"request": {"name": "unwanted"}})
    assert response.status_code == 409 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_NAME_CONFLICT"
    assert client.get(PATH + "/" + first["id"]).json() == first
    assert client.get(PATH + "/" + second["id"]).json() == second
    assert update(client, first, name="third").status_code == 200
    # Another account's identically named record does not block a rename.
    login(client, "bob")
    own = create(client, name="bob").json()
    assert update(client, own, name="second").status_code == 200


def test_concurrent_same_name_posts_create_one_record_then_overwrite_in_revision_order(templates):
    app, client = templates
    authentication = login(client, "alice")
    cookies = dict(client.cookies)
    barrier = Barrier(2)

    def worker(label):
        with TestClient(app) as other:
            other.cookies.update(cookies)
            other.headers["X-CSRF-Token"] = authentication["csrf_token"]
            barrier.wait(timeout=5)
            return create(other, name="same name", content={"request": {"name": label}})

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(worker, ["one", "two"]))
    assert sorted(response.status_code for response in responses) == [200, 201]
    assert len({response.json()["id"] for response in responses}) == 1
    assert sorted(response.json()["revision"] for response in responses) == [1, 2]
    listed = client.get(PATH).json()
    assert len(listed) == 1 and listed[0]["revision"] == 2
    current = client.get(PATH + "/" + listed[0]["id"]).json()
    assert current == next(response.json() for response in responses if response.status_code == 200)
    with app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(TrainingTemplate)) == 1


def test_concurrent_post_and_put_rename_share_one_owner_lock_and_never_create_duplicates(templates):
    app, client = templates
    authentication = login(client, "alice")
    original = create(client, name="source", content={"request": {"name": "original"}}).json()
    cookies, barrier = dict(client.cookies), Barrier(2)

    def worker(operation):
        with TestClient(app) as other:
            other.cookies.update(cookies)
            other.headers["X-CSRF-Token"] = authentication["csrf_token"]
            barrier.wait(timeout=5)
            if operation == "rename":
                return update(other, original, name="target")
            return create(other, name="target", content={"request": {"name": "posted"}})

    with ThreadPoolExecutor(max_workers=2) as pool:
        renamed, posted = list(pool.map(worker, ["rename", "post"]))
    assert (renamed.status_code, posted.status_code) in {(200, 200), (409, 201)}
    if renamed.status_code == 409:
        assert renamed.json()["error"]["code"] == "TRAINING_TEMPLATE_NAME_CONFLICT"
    rows = client.get(PATH).json()
    targets = [row for row in rows if row["name"] == "target"]
    assert len(targets) == 1
    current = client.get(PATH + "/" + targets[0]["id"]).json()
    assert current["content"] == {"request": {"name": "posted"}}


def test_concurrent_initialization_serializes_legacy_duplicate_merge_and_index_creation(templates):
    app, client = templates
    authentication = login(client, "alice")
    original = create(client, name="legacy race").json()
    with app.state.db.engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX uq_training_templates_user_name")
    newer = offset_time(60)
    with app.state.db.write() as session:
        session.add(TrainingTemplate(id="latest-race", user_id=authentication["user"]["id"],
            name="legacy race", description="latest", content={"request": {"name": "latest"}},
            revision=5, created_at=original["created_at"], updated_at=newer))
    barrier = Barrier(2)

    def initialize(_):
        database = Database(app.state.settings.database_url)
        try:
            barrier.wait(timeout=5)
            database.initialize()
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(initialize, range(2)))
    assert client.get(PATH).json() == [{"id": "latest-race", "name": "legacy race", "description": "latest",
        "revision": 6, "created_at": original["created_at"], "updated_at": newer}]
    assert client.get(PATH + "/latest-race").json()["content"] == {"request": {"name": "latest"}}


def test_initialization_merges_legacy_duplicate_names_deterministically_and_is_idempotent(templates):
    app, client = templates
    alice = login(client, "alice")["user"]["id"]
    with app.state.db.read() as session:
        bob = session.scalar(select(User.id).where(User.username == "bob"))
    # Model the pre-index schema in this isolated database only. Existing rows
    # may have newer contents with lower revisions, or equal update timestamps.
    with app.state.db.engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX uq_training_templates_user_name")
    old = "2026-10-25T00:00:00.000000+00:00"
    newest = "2026-10-27T00:00:00.000000+00:00"
    with app.state.db.write() as session:
        for identifier, owner, name, revision, updated, content in [
            ("older-high-revision", alice, "legacy", 10, old, "older"),
            ("a-newest", alice, "legacy", 2, newest, "newest deterministic winner"),
            ("z-newest", alice, "legacy", 4, newest, "timestamp tie loser"),
            ("other-owner", bob, "legacy", 3, old, "other owner unchanged"),
            ("unique-record", alice, "unique", 9, old, "unrelated template unchanged"),
        ]:
            session.add(TrainingTemplate(id=identifier, user_id=owner, name=name, description=content,
                        content={"request": {"name": content}}, revision=revision, created_at=old, updated_at=updated))
        session.add(TrainingDraft(user_id=alice, content={"request": {"name": "unrelated active draft"}}, revision=8, updated_at=old))
    with app.state.db.read() as session:
        sessions_before = [(row.id, row.token_hash, row.expires_at) for row in session.scalars(select(UserSession))]
    app.state.db.initialize()
    with app.state.db.read() as session:
        rows = list(session.scalars(select(TrainingTemplate).order_by(TrainingTemplate.id)))
        assert [row.id for row in rows] == ["a-newest", "other-owner", "unique-record"]
        winner = session.get(TrainingTemplate, "a-newest")
        assert winner.revision == 11 and winner.updated_at == newest and winner.created_at == old
        assert winner.content == {"request": {"name": "newest deterministic winner"}}
        assert winner.description == "newest deterministic winner"
        assert session.get(TrainingTemplate, "other-owner").revision == 3
        assert session.get(TrainingTemplate, "unique-record").revision == 9
        draft = session.get(TrainingDraft, alice)
        assert draft.revision == 8 and draft.content["request"]["name"] == "unrelated active draft"
        assert [(row.id, row.token_hash, row.expires_at) for row in session.scalars(select(UserSession))] == sessions_before
        before = [(row.id, row.revision, row.content, row.description, row.updated_at) for row in rows]
    app.state.db.initialize()
    with app.state.db.read() as session:
        after = [(row.id, row.revision, row.content, row.description, row.updated_at)
                 for row in session.scalars(select(TrainingTemplate).order_by(TrainingTemplate.id))]
        assert after == before
    indices = inspect(app.state.db.engine).get_indexes("training_templates")
    assert any(index["name"] == "uq_training_templates_user_name" and index["unique"]
               and index["column_names"] == ["user_id", "name"] for index in indices)
    # The database guard still rejects a writer that bypasses the API lock.
    with pytest.raises(IntegrityError), app.state.db.write() as session:
        session.add(TrainingTemplate(id="bypass", user_id=alice, name="legacy", description="",
                    content={}, revision=1, created_at=old, updated_at=old))
    winner = client.get(PATH + "/a-newest").json()
    assert update(client, winner, expected_revision=2).status_code == 409
    assert client.get(PATH + "/older-high-revision").status_code == 404
    response = create(client, name=" legacy ", description="after migration", content={"request": {"name": "new"}})
    assert response.status_code == 200 and response.json()["id"] == "a-newest" and response.json()["revision"] == 12


def test_concurrent_updates_have_one_winner_and_one_revision_conflict(templates):
    app, client = templates
    auth = login(client, "alice")
    original = create(client).json()
    cookies = dict(client.cookies)
    barrier = Barrier(2)

    def worker(name):
        with TestClient(app) as other:
            other.cookies.update(cookies)
            other.headers["X-CSRF-Token"] = auth["csrf_token"]
            barrier.wait(timeout=5)
            return update(other, original, name=name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["first", "second"]))
    assert sorted(result.status_code for result in results) == [200, 409]
    current = client.get(PATH + "/" + original["id"]).json()
    assert current["name"] in {"first", "second"} and current["revision"] == 2


def test_viewer_agent_unauthenticated_and_legacy_callers_cannot_use_templates(templates, tmp_path):
    _, client = templates
    for headers in [{}, {"Authorization": "Bearer separate-agent-token"}]:
        assert client.get(PATH, headers=headers).status_code == 401
        assert client.post(PATH, headers=headers, json={"name": "no", "content": {}}).status_code == 401
    login(client, "reader")
    for method, path in [("GET", PATH), ("POST", PATH), ("GET", PATH + "/missing"),
                         ("PUT", PATH + "/missing"), ("DELETE", PATH + "/missing")]:
        assert client.request(method, path, json={}).status_code == 403
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "legacy.sqlite3"),
                             enrollment_token="enrollment", admin_token="legacy-only", legacy_admin_enabled=True,
                             scheduler_enabled=False, session_cookie_secure=False))
    with TestClient(app) as legacy:
        legacy.headers["Authorization"] = "Bearer legacy-only"
        for method, path in [("GET", PATH), ("POST", PATH), ("GET", PATH + "/missing"),
                             ("PUT", PATH + "/missing"), ("DELETE", PATH + "/missing")]:
            response = legacy.request(method, path, json={})
            assert response.status_code == 403 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_ACCOUNT_REQUIRED"


def test_csrf_origin_disabled_and_expired_accounts_still_gate_templates(templates):
    app, client = templates
    authentication = login(client, "alice")
    current = create(client).json()
    client.headers.pop("X-CSRF-Token")
    for response in [create(client), update(client, current), remove(client, current)]:
        assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"
    client.headers["X-CSRF-Token"] = authentication["csrf_token"]
    for method, path, body in [("POST", PATH, {"name": "bad origin", "content": {}}),
            ("PUT", PATH + "/" + current["id"], {"expected_revision": 1, "name": "bad", "description": "", "content": {}}),
            ("DELETE", PATH + "/" + current["id"], {"expected_revision": 1})]:
        response = client.request(method, path, json=body, headers={"Origin": "https://untrusted.example"})
        assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_REJECTED"
    with app.state.db.write() as session:
        row = session.scalar(select(UserSession).where(UserSession.user_id == authentication["user"]["id"]))
        row.expires_at = offset_time(-1)
    assert client.get(PATH).status_code == 401 and update(client, current).status_code == 401
    login(client, "alice")
    with app.state.db.write() as session:
        session.get(User, authentication["user"]["id"]).enabled = False
    assert client.get(PATH).status_code == 401 and create(client).status_code == 401


@pytest.mark.parametrize("key", ["preview_id", "confirmed_config_sha256", "code_snapshot", "execution_token",
    "idempotency_key", "csrfToken", "password", "enrollment-token", "admin_token", "Authorization", "cookie",
    "sessionToken", "access_token", "refresh_token", "password_hash"])
def test_template_content_cannot_store_credentials_or_previous_approval(templates, key):
    _, client = templates
    login(client, "alice")
    response = create(client, content={"nested": [{key: "never persist"}]})
    assert response.status_code == 422 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_INVALID"
    assert client.get(PATH).json() == []


@pytest.mark.parametrize("revision", [0, True, -1, 1.5, "1", None, 2**53])
def test_update_and_delete_require_a_strict_positive_safe_revision(templates, revision):
    _, client = templates
    login(client, "alice")
    template = create(client).json()
    assert update(client, template, expected_revision=revision).status_code == 422
    assert remove(client, template, expected_revision=revision).status_code == 422
    assert client.get(PATH + "/" + template["id"]).json() == template


@pytest.mark.parametrize("name,description", [("", ""), ("   ", ""), ("n" * 101, ""), (True, ""),
    ("a\x00b", ""), ("a\nb", ""), ("normal", "d" * 2001), ("normal", []), ("normal", "a\x7fb")])
def test_names_and_descriptions_are_bounded_text(templates, name, description):
    _, client = templates
    login(client, "alice")
    response = create(client, name=name, description=description)
    assert response.status_code == 422 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_INVALID"


def test_optional_create_description_and_bounded_incomplete_inert_content(templates):
    _, client = templates
    login(client, "alice")
    saved = client.post(PATH, json={"name": "no description", "content": {"request": {"parameters": {"tokenizer_name": "valid"}}}})
    assert saved.status_code == 201 and saved.json()["description"] == ""
    assert create(client, content=[]).status_code == 422
    assert create(client, content={"value": "x" * MAX_CONTENT_BYTES}).status_code == 413
    nested = {}
    for _ in range(MAX_DEPTH + 1):
        nested = {"nested": nested}
    assert create(client, content=nested).status_code == 422
    assert create(client, content={"many": [None] * MAX_NODES}).status_code == 422
    response = client.post(PATH, content=b" " * (MAX_REQUEST_BYTES + 1))
    assert response.status_code == 413 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_TOO_LARGE"
    assert len(client.get(PATH).json()) == 1


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b'{"name":"n","content":{"number":NaN}}',
    b'{"name":"n","content":{"number":Infinity}}', b'{"name":"\\ud800","content":{}}',
    b'{"name":"n","content":{"invalid":"\\ud800"}}', b"\xff"])
def test_malformed_nonfinite_and_invalid_unicode_input_is_rejected(templates, raw):
    _, client = templates
    login(client, "alice")
    response = client.post(PATH, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "TRAINING_TEMPLATE_INVALID"
    assert client.get(PATH).json() == []
