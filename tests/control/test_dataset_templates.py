"""Dataset formats are durable private data, not asset identities or launches."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from vla_control.common import uid
from vla_control.dataset_template_models import DatasetFormatTemplate
from vla_control import dataset_templates as module
from vla_control.main import create_app
from vla_control.models import Asset, Attempt, Command, Run
from vla_control.settings import Settings


PASSWORD = "dataset-format-template-123"
PATH = "/api/v1/dataset-format-templates"
PROFILE = {
    "schema_version": 1, "format": "lerobot", "robot_type": "ARX5", "arm_mode": "bimanual",
    "action_representation": "joint", "action_mode": "absolute", "rotation_representation": "none",
    "cameras": [{"name": "head", "key": "observation.images.head", "kind": "rgb"},
                {"name": "wrist", "key": "observation.images.wrist", "kind": "depth"}],
    "task_description": "Flatten and fold cloth.", "fps": 30, "state_dim": 14, "action_dim": 14,
    "feature_keys": ["observation.state", "actions"],
    "field_annotations": [
        {"key": "observation.state", "role": "state", "representation": "joint", "value_mode": "absolute",
         "component": "joint", "arm": "shared", "shape": [14]},
        {"key": "actions", "role": "action", "representation": "joint", "value_mode": "absolute",
         "component": "joint", "arm": "shared", "shape": [50, 14], "horizon": 50},
    ],
}


@pytest.fixture
def templates(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "templates.db"),
                        enrollment_token="dataset-template-agent", scheduler_enabled=False,
                        session_cookie_secure=False)
    app = create_app(settings)
    for username, role in (("alice", "admin"), ("bob", "operator"), ("viewer", "viewer")):
        app.state.auth.create_user(username, PASSWORD, role)
    with TestClient(app) as client:
        yield app, client, settings


def login(client, username="alice"):
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    body = response.json()
    client.headers["X-CSRF-Token"] = body["csrf_token"]
    return body


def create(client, name="ARX5 双臂关节", profile=None, **extra):
    return client.post(PATH, json={"name": name, "profile": deepcopy(PROFILE if profile is None else profile), **extra})


def update(client, template, **changes):
    return client.put(PATH + "/" + template["id"], json={"name": template["name"],
        "profile": template["profile"], "expected_revision": template["revision"], **changes})


def remove(client, template, **changes):
    return client.request("DELETE", PATH + "/" + template["id"],
                          json={"expected_revision": template["revision"], **changes})


def test_create_restore_across_session_and_process_without_mutating_assets(templates):
    app, client, settings = templates
    owner = login(client)["user"]["id"]
    result = create(client)
    assert result.status_code == 201, result.text
    saved = result.json()
    assert saved["revision"] == 1 and saved["created_at"] == saved["updated_at"]
    assert saved["profile"]["field_annotations"][1]["shape"] == [50, 14]
    assert saved["profile"]["field_annotations"][1]["horizon"] == 50
    assert saved["profile"]["cameras"] == PROFILE["cameras"]
    assert "processing" not in saved["profile"] and "user_id" not in saved
    listing = client.get(PATH)
    assert listing.status_code == 200 and listing.headers["Cache-Control"] == "no-store"
    assert [row["id"] for row in listing.json()] == [saved["id"]]
    assert "profile" not in listing.json()[0]
    login(client)
    assert client.get(PATH + "/" + saved["id"]).json() == saved
    second_app = create_app(settings)
    with TestClient(second_app) as second_client:
        login(second_client)
        assert second_client.get(PATH + "/" + saved["id"]).json() == saved
    with app.state.db.read() as session:
        assert session.get(DatasetFormatTemplate, saved["id"]).user_id == owner
        for model in (Asset, Attempt, Command, Run):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_same_trimmed_name_updates_existing_template_instead_of_duplicating(templates):
    _, client, _ = templates
    login(client)
    original = create(client, name=" 相同格式 ").json()
    altered = deepcopy(PROFILE)
    altered["task_description"] = "A new editable task default"
    result = create(client, name="相同格式", profile=altered)
    assert result.status_code == 200, result.text
    saved = result.json()
    assert saved["id"] == original["id"] and saved["revision"] == 2
    assert saved["created_at"] == original["created_at"]
    assert saved["profile"]["task_description"] == altered["task_description"]
    assert len(client.get(PATH).json()) == 1


def test_revision_updates_and_deletes_fail_on_stale_content(templates):
    _, client, _ = templates
    login(client)
    original = create(client).json()
    changed = update(client, original, name="renamed")
    assert changed.status_code == 200, changed.text
    saved = changed.json()
    assert saved["revision"] == 2 and saved["created_at"] == original["created_at"]
    for response in (update(client, original), remove(client, original)):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "DATASET_TEMPLATE_CONFLICT"
    assert remove(client, saved).json() == {"id": saved["id"], "deleted": True}
    assert client.get(PATH + "/" + saved["id"]).status_code == 404
    assert client.get(PATH).json() == []


def test_account_isolation_even_administrator_and_same_name_per_owner(templates):
    _, client, _ = templates
    login(client)
    alice = create(client, name="private format").json()
    login(client, "bob")
    assert client.get(PATH).json() == []
    bob = create(client, name="private format").json()
    assert alice["id"] != bob["id"]
    for template, user in ((alice, "bob"), (bob, "alice")):
        login(client, user)
        for result in (client.get(PATH + "/" + template["id"]), update(client, template), remove(client, template)):
            assert result.status_code == 404
            assert result.json()["error"]["code"] == "NOT_FOUND"
        assert len(client.get(PATH).json()) == 1


def test_rename_cannot_overwrite_a_different_template(templates):
    _, client, _ = templates
    login(client)
    first, second = create(client, "first").json(), create(client, "second").json()
    result = update(client, first, name=second["name"])
    assert result.status_code == 409
    assert result.json()["error"]["code"] == "DATASET_TEMPLATE_NAME_CONFLICT"
    assert len(client.get(PATH).json()) == 2


@pytest.mark.parametrize("field,value", [
    ("processing", None), ("processing", {"action_key": "actions"}),
    ("node_id", "other-machine"), ("norm_asset_id", "foreign-norm"), ("path", "/another/dataset"),
    ("status", "VALIDATED"), ("preflight", {"ok": True}), ("robot_type", "bad\x00name"),
    ("schema_version", True), ("fps", "30"), ("cameras", []), ("task_description", ""),
])
def test_only_strict_registration_profile_semantics_can_be_saved(templates, field, value):
    _, client, _ = templates
    login(client)
    profile = {**deepcopy(PROFILE), field: value}
    result = create(client, profile=profile)
    assert result.status_code == 422, result.text
    assert result.json()["error"]["code"] == "DATASET_TEMPLATE_INVALID"
    assert client.get(PATH).json() == []


def test_original_action_horizon_contract_is_preserved_not_model_prediction_length(templates):
    _, client, _ = templates
    login(client)
    invalid = deepcopy(PROFILE)
    invalid["field_annotations"][1]["horizon"] = 16
    assert create(client, profile=invalid).status_code == 422
    single = deepcopy(PROFILE)
    single["field_annotations"][1].update(shape=[14], horizon=1)
    result = create(client, profile=single)
    assert result.status_code == 201, result.text
    assert result.json()["profile"]["field_annotations"][1]["horizon"] == 1


@pytest.mark.parametrize("payload", [
    {"name": " ", "profile": PROFILE}, {"name": "x" * 101, "profile": PROFILE},
    {"name": "line\nbreak", "profile": PROFILE}, {"name": True, "profile": PROFILE},
    {"name": "format", "profile": PROFILE, "user_id": "different"},
    {"name": "format", "profile": PROFILE, "asset_id": "copied-identity"},
])
def test_template_names_and_request_keys_are_strict(templates, payload):
    _, client, _ = templates
    login(client)
    assert client.post(PATH, json=payload).status_code == 422


def test_body_is_bounded_and_rejects_non_json_constants_duplicate_keys_and_invalid_utf8(templates):
    _, client, _ = templates
    login(client)
    result = client.post(PATH, content=b" " * (module.MAX_REQUEST_BYTES + 1))
    assert result.status_code == 413
    assert result.json()["error"]["code"] == "DATASET_TEMPLATE_TOO_LARGE"
    for body in (b'{"name":"one","name":"two","profile":{}}', b'{"name":"one","profile":{"fps":NaN}}',
                 b"\xff", b"[" * 1000, b"null"):
        assert client.post(PATH, content=body).status_code == 422
    assert client.get(PATH).json() == []


@pytest.mark.parametrize("revision", [True, 0, -1, "1", module.MAX_EXPECTED_REVISION + 1])
def test_update_delete_revision_is_a_bounded_integer(templates, revision):
    _, client, _ = templates
    login(client)
    saved = create(client).json()
    assert update(client, saved, expected_revision=revision).status_code == 422
    assert remove(client, saved, expected_revision=revision).status_code == 422


def test_limit_is_per_owner_and_same_name_updates_still_work_at_limit(templates, monkeypatch):
    _, client, _ = templates
    monkeypatch.setattr(module, "MAX_TEMPLATES_PER_USER", 2)
    login(client)
    first = create(client, "one").json()
    create(client, "two")
    result = create(client, "three")
    assert result.status_code == 409 and result.json()["error"]["code"] == "DATASET_TEMPLATE_LIMIT"
    result = create(client, "one")
    assert result.status_code == 200 and result.json()["id"] == first["id"]
    login(client, "bob")
    assert create(client, "three").status_code == 201


def test_same_name_concurrent_upserts_are_one_revisioned_row(templates):
    app, client, _ = templates
    owner = login(client)["user"]["id"]
    barrier = Barrier(2)

    def save(task):
        barrier.wait()
        return module.create_template(app.state.db, owner, {"name": "one format", "profile": deepcopy(PROFILE)})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))
    assert sorted(status for _, status in results) == [200, 201]
    assert len({row["id"] for row, _ in results}) == 1
    listing = client.get(PATH).json()
    assert len(listing) == 1 and listing[0]["revision"] == 2


def test_saturated_revision_cannot_overflow_but_remains_deletable(templates):
    app, client, _ = templates
    login(client)
    saved = create(client).json()
    with app.state.db.write() as session:
        session.get(DatasetFormatTemplate, saved["id"]).revision = module.MAX_EXPECTED_REVISION
    saved["revision"] = module.MAX_EXPECTED_REVISION
    assert create(client).status_code == 409
    assert update(client, saved).status_code == 409
    assert remove(client, saved).status_code == 200


def test_owner_name_unique_index_and_additive_initialize_are_idempotent(templates):
    app, client, _ = templates
    owner = login(client)["user"]["id"]
    saved = create(client).json()
    app.state.db.initialize()
    app.state.db.initialize()
    assert client.get(PATH + "/" + saved["id"]).status_code == 200
    indexes = inspect(app.state.db.engine).get_indexes("dataset_format_templates")
    assert any(index["unique"] and index["column_names"] == ["user_id", "name"] for index in indexes)
    with pytest.raises(IntegrityError), app.state.db.write() as session:
        session.add(DatasetFormatTemplate(id=uid(), user_id=owner, name=saved["name"], profile=PROFILE,
                    revision=1, created_at=saved["created_at"], updated_at=saved["updated_at"]))
