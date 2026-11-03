"""Private configuration writes recheck revocation after asynchronous bodies."""

import json
import importlib

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select

from vla_control.auth_models import User, UserSession
from vla_control.common import offset_time
from vla_control.dataset_template_models import DatasetFormatTemplate
from vla_control.main import create_app
from vla_control.models import TrainingDraft, TrainingTemplate
from vla_control.settings import Settings

from .test_dataset_templates import PROFILE


PASSWORD = "private-configuration-revocation-password"


@pytest.fixture
def private_configuration(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "private-revocation.db"),
        enrollment_token="separate-private-revocation-enrollment", scheduler_enabled=False,
        session_cookie_secure=False))
    user = app.state.auth.create_user("writer", PASSWORD, "operator")
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/login", json={"username": "writer", "password": PASSWORD})
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        yield app, client, user["id"]


def snapshot(app, model):
    with app.state.db.read() as session:
        return [dict(row) for row in session.execute(select(model.__table__)).mappings()]


@pytest.mark.parametrize("surface", ["draft", "training-template", "dataset-template"])
@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize("revocation,boundary", [
    (name, "body") for name in ("disabled", "downgraded", "logout", "idle", "expired")
] + [("logout", "worker")])
def test_configuration_cannot_commit_after_authorization_changes_while_body_streams(
        private_configuration, monkeypatch, surface, operation, revocation, boundary):
    app, client, user_id = private_configuration
    if surface == "draft":
        base, model = "/api/v1/training-draft", TrainingDraft
        identifier = ""
        method = "DELETE" if operation == "delete" else "PUT"
        if operation != "create":
            assert client.put(base, json={"expected_revision": 0, "content": {"before": True}}).status_code == 200
        body = {"expected_revision": 0 if operation == "create" else 1}
        if operation != "delete":
            body["content"] = {"after": True}
    else:
        dataset = surface == "dataset-template"
        base = "/api/v1/dataset-format-templates" if dataset else "/api/v1/training-templates"
        model = DatasetFormatTemplate if dataset else TrainingTemplate
        create_body = {"name": "before", **({"profile": PROFILE} if dataset else {"content": {"before": True}})}
        identifier = ""
        if operation != "create":
            created = client.post(base, json=create_body)
            assert created.status_code == 201, created.text
            identifier = "/" + created.json()["id"]
        method = {"create": "POST", "update": "PUT", "delete": "DELETE"}[operation]
        body = {"expected_revision": 1} if operation == "delete" else {
            "name": "after", **({"profile": PROFILE} if dataset else {"description": "changed", "content": {"after": True}})}
        if operation == "update":
            body["expected_revision"] = 1
    before = snapshot(app, model)
    encoded = json.dumps(body).encode()

    def revoke():
        with app.state.db.write() as session:
            user = session.get(User, user_id)
            if revocation == "disabled":
                user.enabled = False
            elif revocation == "downgraded":
                user.role = "viewer"
            elif revocation == "logout":
                session.execute(delete(UserSession).where(UserSession.user_id == user_id))
            else:
                row = session.scalar(select(UserSession).where(UserSession.user_id == user_id))
                if revocation == "idle":
                    row.last_activity_at = offset_time(-3601)
                else:
                    row.expires_at = offset_time(-1)
    def stream():
        yield encoded[:1]
        if boundary == "body":
            revoke()
        yield encoded[1:]

    if boundary == "worker":
        module = importlib.import_module("vla_control." + {
            "draft": "training_drafts", "training-template": "training_templates",
            "dataset-template": "dataset_templates"}[surface])
        run_in_threadpool = module.run_in_threadpool
        async def after_queueing(function, *args, **kwargs):
            # The HTTP handler finished its last authorization check, then
            # waited for a worker while another request logged the account out.
            revoke()
            return await run_in_threadpool(function, *args, **kwargs)
        monkeypatch.setattr(module, "run_in_threadpool", after_queueing)

    response = client.request(method, base + identifier, content=stream(), headers={"Content-Type": "application/json"})
    assert response.status_code in {401, 403}, response.text
    assert snapshot(app, model) == before
