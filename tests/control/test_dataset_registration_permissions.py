"""Dataset edits and asset archiving use the recorded registrant."""

import pytest
from sqlalchemy import delete, func, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control.models import Asset, Command
from .conftest import profile_inspection

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
def test_registration_is_granted_for_all_supported_kinds_to_admin_and_operator(team, name, kind):
    platform, node, users, clients = team
    before = counts(platform)
    response = clients[name].post("/api/v1/assets", json=body(node, kind=kind))
    allowed = name in {"team-admin", "alice"}
    assert response.status_code == (201 if allowed else 403), response.text
    if allowed:
        assert response.json()["kind"] == kind
        assert response.json()["registered_by_user_id"] == users[name]["id"]
        assert counts(platform)[0] == before[0] + 1
    else:
        assert counts(platform) == before


def test_registrant_can_edit_then_archive_own_dataset(team):
    platform, node, _, clients = team
    alice, admin = clients["alice"], clients["team-admin"]
    created = alice.post("/api/v1/assets", json=body(node, metadata={"custom": "preserve"}))
    assert created.status_code == 201, created.text
    asset = created.json()
    root = f"/api/v1/assets/{asset['id']}"
    declared = profile(fps=10)
    patch = {"dataset_profile": declared,
             "dataset_inspection_id": profile_inspection(platform, node["node_id"], asset["path"], declared)}
    updated = alice.patch(root + "/dataset-profile", json=patch)
    assert updated.status_code == 200, updated.text
    changed = updated.json()
    assert changed["metadata"]["dataset_profile"]["fps"] == 10
    assert changed["metadata"]["custom"] == "preserve"
    for key in ("id", "path", "node_id", "name", "version", "registered_by_user_id"):
        assert changed[key] == asset[key]

    shared = admin.post("/api/v1/assets", json=body(node, kind="NORM", path="/machine/norms/shared"))
    assert shared.status_code == 201, shared.text
    original_norm = shared.json()
    for binding in ({"norm_asset_id": original_norm["id"]}, {"norm_path": original_norm["path"]}):
        bound = alice.put(root + "/norm-binding", json=binding)
        assert bound.status_code == 200, bound.text
        assert bound.json()["norm_asset_id"] == original_norm["id"]
    assets = {row["id"]: row for row in admin.get("/api/v1/assets").json()}
    assert assets[original_norm["id"]] == original_norm

    fresh_binding = alice.put(root + "/norm-binding", json={"norm_path": "/machine/norms/new"})
    assert fresh_binding.status_code == 200, fresh_binding.text
    fresh_id = fresh_binding.json()["norm_asset_id"]
    assets = {row["id"]: row for row in alice.get("/api/v1/assets").json()}
    assert assets[fresh_id]["kind"] == "NORM"
    assert assets[fresh_id]["registered_by_user_id"] is None
    assert alice.post(f"/api/v1/assets/{fresh_id}/archive").status_code == 403
    unbound = alice.put(root + "/norm-binding", json={"norm_asset_id": None})
    assert unbound.status_code == 200 and unbound.json()["norm_asset_id"] is None

    assert alice.post(root + "/validation/cancel").status_code == 403
    assert alice.post(root + "/archive").status_code == 200
    assert alice.patch(root + "/dataset-profile", json=patch).status_code == 409
    assert alice.put(root + "/norm-binding", json={"norm_asset_id": None}).status_code == 409


@pytest.mark.parametrize("target", ["other-dataset", "historical-dataset", "viewer-dataset", "own-model", "own-norm"])
def test_dataset_write_is_limited_to_a_live_operator_owning_the_target_dataset(team, target):
    platform, node, users, clients = team
    kind = {"own-model": "MODEL", "own-norm": "NORM"}.get(target, "DATASET")
    creator = "bob" if target == "other-dataset" else "alice"
    created = clients[creator].post("/api/v1/assets", json=body(
        node, kind=kind, metadata={"registered_by_user_id": users["alice"]["id"]}))
    assert created.status_code == 201, created.text
    asset = created.json()
    if target in {"historical-dataset", "viewer-dataset"}:
        with platform.app.state.db.write() as session:
            session.get(Asset, asset["id"]).registered_by_user_id = (
                users["reader"]["id"] if target == "viewer-dataset" else None)
    client = clients["reader" if target == "viewer-dataset" else "alice"]
    root = f"/api/v1/assets/{asset['id']}"
    before = client.get("/api/v1/assets").json()
    before_counts = counts(platform)
    patch = client.patch(root + "/dataset-profile", json={"dataset_profile": profile(fps=10)})
    bind = client.put(root + "/norm-binding", json={"norm_path": "/machine/norms/new"})
    if kind != "DATASET":
        status, code = 409, "ASSET_KIND_MISMATCH"
    elif target == "viewer-dataset":
        status, code = 403, "FORBIDDEN"
    else:
        status, code = 403, "RESOURCE_ACCESS_DENIED"
    for response in (patch, bind):
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code
    assert client.get("/api/v1/assets").json() == before
    assert counts(platform) == before_counts


@pytest.mark.parametrize("method,suffix", [("PATCH", "/dataset-profile"), ("PUT", "/norm-binding")])
def test_dataset_edits_recheck_the_current_account_before_writing(team, monkeypatch, method, suffix):
    platform, node, users, clients = team
    client = clients["alice"]
    created = client.post("/api/v1/assets", json=body(node))
    assert created.status_code == 201, created.text
    asset = created.json()
    path = f"/api/v1/assets/{asset['id']}{suffix}"
    declared = profile(fps=10)
    payload = ({"dataset_profile": declared,
                "dataset_inspection_id": profile_inspection(platform, node["node_id"], asset["path"], declared)}
               if method == "PATCH" else {"norm_path": "/machine/norms/new"})
    original = platform.app.state.auth.require
    revoked = False

    def revoke_after_admission(request, permission=None):
        nonlocal revoked
        principal = original(request, permission)
        if request.method == method and request.url.path == path and not revoked:
            revoked = True
            platform.app.state.auth.update_user(None, users["alice"]["id"], {"role": "viewer"})
        return principal

    admin = clients["team-admin"]
    before = admin.get("/api/v1/assets").json()
    before_counts = counts(platform)
    monkeypatch.setattr(platform.app.state.auth, "require", revoke_after_admission)
    response = client.request(method, path, json=payload)
    assert revoked and response.status_code in {401, 403}, response.text
    assert admin.get("/api/v1/assets").json() == before
    assert counts(platform) == before_counts


@pytest.mark.parametrize("binding", ["existing", "path"])
def test_operator_can_set_initial_profile_and_norm_without_modifying_existing_assets(team, binding):
    platform, node, _, clients = team
    admin = clients["team-admin"]
    norm = admin.post("/api/v1/assets", json=body(node, name="shared norm", kind="NORM", path="/machine/norms/shared"))
    assert norm.status_code == 201, norm.text
    original = norm.json()
    declaration = {"norm_asset_id": original["id"]} if binding == "existing" else {"norm_path": original["path"]}
    source_profile = {**profile(), "fps": 20}
    created = clients["alice"].post("/api/v1/assets", json=body(node,
        metadata={"dataset_profile": source_profile},
        dataset_inspection_id=profile_inspection(platform, node["node_id"], "/machine/datasets/new", source_profile), **declaration))
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
    assert norm["registered_by_user_id"] is None
    assert clients["alice"].post(f"/api/v1/assets/{norm['id']}/archive").status_code == 403


@pytest.mark.parametrize("change", ["disabled", "viewer", "logout", "idle", "expired", "password-change"])
@pytest.mark.parametrize("kind", ["DATASET", "NORM", "MODEL"])
def test_registration_rechecks_original_session_before_any_asset_or_norm_write(team, monkeypatch, change, kind):
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
    declaration = {"norm_path": "/machine/norms/new"} if kind == "DATASET" else {}
    response = clients["alice"].post("/api/v1/assets", json=body(node, kind=kind, **declaration))
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
