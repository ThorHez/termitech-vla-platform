"""Dataset creators may archive their own datasets, but cannot edit shared assets."""

import pytest
from sqlalchemy import delete, func, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control.models import Asset, Command

from .test_dataset_profiles import profile
from .test_multiuser_task_permissions import team  # noqa: F401


def body(node, **changes):
    return {"node_id": node["node_id"], "name": "new dataset", "kind": "DATASET",
            "path": "/machine/datasets/new", "metadata": {}, **changes}


def counts(platform):
    with platform.app.state.db.read() as session:
        return tuple(session.scalar(select(func.count()).select_from(model)) for model in (Asset, Command))


@pytest.mark.parametrize("name", ["team-admin", "alice", "reader"])
@pytest.mark.parametrize("kind", ["DATASET", "MODEL", "NORM"])
def test_registration_is_granted_only_for_the_allowed_role_and_asset_kind(team, name, kind):
    platform, node, _, clients = team
    before = counts(platform)
    response = clients[name].post("/api/v1/assets", json=body(node, kind=kind))
    allowed = name == "team-admin" or name == "alice" and kind == "DATASET"
    assert response.status_code == (201 if allowed else 403), response.text
    if allowed:
        assert response.json()["kind"] == kind
        assert counts(platform)[0] == before[0] + 1
    else:
        assert counts(platform) == before


@pytest.mark.parametrize("creator", ["alice", "team-admin"])
def test_operator_can_archive_only_own_registered_datasets_and_cannot_edit_existing_assets(team, creator):
    _, node, _, clients = team
    created = clients[creator].post("/api/v1/assets", json=body(node, metadata={"dataset_profile": profile()}))
    assert created.status_code == 201, created.text
    asset = created.json()
    root = f"/api/v1/assets/{asset['id']}"
    for method, suffix, payload in (
        ("PATCH", "/dataset-profile", {"dataset_profile": {**profile(), "fps": 10}}),
        ("PUT", "/norm-binding", {"norm_path": "/machine/norms/replacement"}),
        ("POST", "/validations", {}),
        ("POST", "/validation/cancel", {}),
    ):
        response = clients["alice"].request(method, root + suffix, json=payload)
        assert response.status_code == 403, (method, suffix, response.text)
    current = next(row for row in clients["alice"].get("/api/v1/assets").json() if row["id"] == asset["id"])
    assert current == asset
    response = clients["alice"].post(root + "/archive")
    assert response.status_code == (200 if creator == "alice" else 403), response.text
    assert clients["team-admin"].post(root + "/archive").status_code == 200


@pytest.mark.parametrize("binding", ["existing", "path"])
def test_operator_can_set_initial_profile_and_norm_without_modifying_existing_assets(team, binding):
    _, node, _, clients = team
    admin = clients["team-admin"]
    norm = admin.post("/api/v1/assets", json=body(node, name="shared norm", kind="NORM", path="/machine/norms/shared"))
    assert norm.status_code == 201, norm.text
    original = norm.json()
    declaration = {"norm_asset_id": original["id"]} if binding == "existing" else {"norm_path": original["path"]}
    source_profile = {**profile(), "fps": 20}
    created = clients["alice"].post("/api/v1/assets", json=body(node,
        metadata={"dataset_profile": source_profile}, **declaration))
    assert created.status_code == 201, created.text
    asset = created.json()
    assert asset["norm_asset_id"] == original["id"]
    assert asset["metadata"]["dataset_profile"]["fps"] == source_profile["fps"]
    current = next(row for row in admin.get("/api/v1/assets").json() if row["id"] == original["id"])
    assert current == original


def test_operator_initial_norm_path_may_register_only_the_new_datasets_binding(team):
    platform, node, _, clients = team
    before = counts(platform)[0]
    created = clients["alice"].post("/api/v1/assets", json=body(node, norm_path="/machine/norms/new"))
    assert created.status_code == 201, created.text
    assets = clients["alice"].get("/api/v1/assets").json()
    assert len(assets) == before + 2
    norm = next(row for row in assets if row["id"] == created.json()["norm_asset_id"])
    assert norm["kind"] == "NORM" and norm["path"] == "/machine/norms/new"
    assert clients["alice"].post(f"/api/v1/assets/{norm['id']}/archive").status_code == 403


@pytest.mark.parametrize("change", ["disabled", "viewer", "logout", "idle", "expired", "password-change"])
def test_registration_rechecks_original_session_before_any_asset_or_norm_write(team, monkeypatch, change):
    platform, node, users, clients = team
    identifier = users["alice"]["id"]
    original = platform.app.state.auth.require
    changed = []

    def revoke_after_admission(request, permission=None):
        principal = original(request, permission)
        if request.url.path == "/api/v1/assets" and request.method == "POST" and not changed:
            changed.append(True)
            if change in {"disabled", "viewer"}:
                platform.app.state.auth.update_user(None, identifier,
                    {"enabled": False} if change == "disabled" else {"role": "viewer"})
            elif change == "password-change":
                from vla_control.auth_models import User
                with platform.app.state.db.write() as session:
                    session.get(User, identifier).must_change_password = True
            else:
                with platform.app.state.db.write() as session:
                    if change == "logout":
                        session.execute(delete(UserSession).where(UserSession.user_id == identifier))
                    else:
                        active = session.scalar(select(UserSession).where(UserSession.user_id == identifier))
                        if change == "idle":
                            active.last_activity_at = offset_time(-3601)
                        else:
                            active.expires_at = offset_time(-1)
        return principal

    before = counts(platform)
    monkeypatch.setattr(platform.app.state.auth, "require", revoke_after_admission)
    response = clients["alice"].post("/api/v1/assets", json=body(node, norm_path="/machine/norms/new"))
    assert changed and response.status_code in {401, 403}, response.text
    assert counts(platform) == before


@pytest.mark.parametrize("failure", ["csrf", "other-account", "outside-roots", "norm-outside-roots"])
def test_registration_keeps_request_identity_csrf_and_path_boundaries(team, failure):
    platform, node, users, clients = team
    client = clients["alice"]
    request_body = body(node)
    headers = {}
    if failure == "csrf":
        client.headers.pop("X-CSRF-Token")
    elif failure == "other-account":
        headers["X-Operation-User"] = users["bob"]["id"]
    elif failure == "outside-roots":
        request_body["path"] = "/outside/dataset"
    else:
        request_body["norm_path"] = "/outside/norm"
    before = counts(platform)
    response = client.post("/api/v1/assets", headers=headers, json=request_body)
    assert response.status_code == (422 if "outside-roots" in failure else 403), response.text
    assert counts(platform) == before
