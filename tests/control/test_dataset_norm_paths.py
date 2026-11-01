"""Path binding registers metadata atomically and never claims file validation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.dataset_norm_bindings import canonical_norm_path, resolve_norm_path
from vla_control.models import Asset, Command, Run

from .conftest import ADMIN, ENROLL
from .test_dataset_norm_bindings import register_asset, listed


def bind_path(platform, dataset, path, **changes):
    return platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=ADMIN,
                               json={"norm_path": path, **changes})


def norms(platform, node_id=None):
    params = {"kind": "NORM"}
    if node_id:
        params["node_id"] = node_id
    return platform.client.get("/api/v1/assets", headers=ADMIN, params=params).json()


def test_register_dataset_and_norm_path_in_one_request(platform):
    node = platform.register()
    response = register_asset(platform, node, name="Cloth", norm_path="/machine/norms/raw14/")
    assert response.status_code == 201, response.text
    dataset = response.json()
    norm = listed(platform, dataset["norm_asset_id"])
    assert norm["node_id"] == dataset["node_id"] and norm["kind"] == "NORM"
    assert norm["path"] == "/machine/norms/raw14" and norm["state"] == "REGISTERED"
    assert norm["name"] == "Cloth · norm_stats" and norm["version"] == "v1"
    assert norm["metadata"] == {"registration_source": "dataset_norm_binding"}
    assert dataset["metadata"] == {"custom": {"keep": True}}
    assert dataset["path"] == "/machine/dataset"
    assert platform.commands(node) == []


def test_existing_dataset_auto_registers_norm_then_replays_reuse_it(platform):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    original = deepcopy(dataset)
    for path in ("/machine/norm//raw14/", " /machine/norm/./raw14 ", "/machine/norm/raw14"):
        response = bind_path(platform, dataset, path)
        assert response.status_code == 200, response.text
        dataset = response.json()
    assert len(norms(platform)) == 1
    assert dataset == {**original, "norm_asset_id": norms(platform)[0]["id"]}


def test_existing_registered_norm_is_reused_without_rewriting_its_metadata_or_version(platform):
    node = platform.register()
    norm = register_asset(platform, node, "NORM", name="Manual", path="/machine/stats//raw/",
                          version="original-v7", metadata={"custom": {"manual": True}}).json()
    first = register_asset(platform, node, norm_path="/machine/stats/raw").json()
    second = register_asset(platform, node).json()
    assert bind_path(platform, second, "/machine/stats/./raw/").json()["norm_asset_id"] == norm["id"]
    assert first["norm_asset_id"] == norm["id"] and len(norms(platform)) == 1
    assert listed(platform, norm["id"]) == norm


def test_same_path_on_another_machine_or_another_kind_is_not_binding_authority(platform):
    first, second = platform.register(), platform.register(name="other")
    foreign = register_asset(platform, second, "NORM", path="/machine/shared").json()
    wrong_kind = register_asset(platform, first, "MODEL", path="/machine/shared").json()
    dataset = register_asset(platform, first, norm_path="/machine/shared").json()
    assert dataset["norm_asset_id"] not in {foreign["id"], wrong_kind["id"]}
    assert len(norms(platform, first["node_id"])) == 1
    assert len(norms(platform)) == 2


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_unavailable_same_path_registration_is_not_reactivated_or_duplicated(platform, state):
    node = platform.register()
    norm = register_asset(platform, node, "NORM", path="/machine/stats/").json()
    dataset = register_asset(platform, node).json()
    with platform.app.state.db.write() as session:
        session.get(Asset, norm["id"]).state = state
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    for response in (bind_path(platform, dataset, "/machine/stats"),
                     register_asset(platform, node, norm_path="/machine/stats")):
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "NORM_ASSET_UNAVAILABLE"
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


def test_usable_path_match_is_preferred_to_old_archived_duplicate(platform):
    node = platform.register()
    old = register_asset(platform, node, "NORM", path="/machine/stats").json()
    ready = register_asset(platform, node, "NORM", path="/machine/stats/").json()
    with platform.app.state.db.write() as session:
        session.get(Asset, old["id"]).state = "ARCHIVED"
    dataset = register_asset(platform, node, norm_path="/machine/stats").json()
    assert dataset["norm_asset_id"] == ready["id"] and len(norms(platform)) == 2


def test_explicit_stats_file_is_preserved_and_not_equated_to_a_directory_without_remote_stat(platform):
    node = platform.register()
    first = register_asset(platform, node, norm_path="/machine/norms/norm_stats.json").json()
    second = register_asset(platform, node, norm_path="/machine/norms").json()
    assert first["norm_asset_id"] != second["norm_asset_id"]
    assert {row["path"] for row in norms(platform)} == {"/machine/norms", "/machine/norms/norm_stats.json"}


@pytest.mark.parametrize("path", ["", " ", "relative/stats", "/machine/../secret", "/other/stats", "/",
    "/machine/bad\x00name", "/machine/bad\nname", "/machine/bad\x85name", "/machine/back\\slash", "/machine/" + "x" * 4096])
def test_invalid_paths_leave_dataset_binding_and_catalog_unchanged(platform, path):
    node = platform.register()
    valid = register_asset(platform, node, "NORM").json()
    dataset = register_asset(platform, node, norm_asset_id=valid["id"]).json()
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    for response in (bind_path(platform, dataset, path), register_asset(platform, node, norm_path=path)):
        assert response.status_code == 422, response.text
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


@pytest.mark.parametrize("extra", [{"norm_asset_id": None}, {"norm_asset_id": "both"}])
def test_id_and_path_are_mutually_exclusive_including_explicit_null(platform, extra):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    for response in (bind_path(platform, dataset, "/machine/stats", **extra),
                     register_asset(platform, node, norm_path="/machine/stats", **extra)):
        assert response.status_code == 422, response.text
    assert norms(platform) == [] and listed(platform, dataset["id"]) == dataset


@pytest.mark.parametrize("kind", ["MODEL", "NORM"])
def test_other_asset_kinds_cannot_auto_register_a_bound_norm(platform, kind):
    node = platform.register()
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    response = register_asset(platform, node, kind, norm_path="/machine/stats")
    assert response.status_code in {409, 422}, response.text
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


@pytest.mark.parametrize("case", ["missing", "MODEL", "ARCHIVED", "UNAVAILABLE"])
def test_invalid_dataset_is_checked_before_registering_any_norm(platform, case):
    node = platform.register()
    dataset = register_asset(platform, node, "MODEL" if case == "MODEL" else "DATASET").json()
    if case in {"ARCHIVED", "UNAVAILABLE"}:
        with platform.app.state.db.write() as session:
            session.get(Asset, dataset["id"]).state = case
    if case == "missing":
        dataset["id"] = "missing"
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    response = bind_path(platform, dataset, "/machine/stats")
    assert response.status_code in {404, 409}, response.text
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


def test_dataset_create_failure_rolls_back_auto_registration(platform):
    node = platform.register()
    response = register_asset(platform, node, path="/outside/data", norm_path="/machine/stats")
    assert response.status_code == 422, response.text
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == []


def _pending_transfer(platform, source, dataset):
    target = platform.register(name="target")
    with platform.app.state.db.write() as session:
        session.add(AssetTransfer(id=uid(), request_key=uid(), request_digest="a" * 64, name="fixture transfer",
            source_asset_id=dataset["id"], source_node_id=source["node_id"], source_path=dataset["path"],
            source_kind="DATASET", target_node_id=target["node_id"], target_directory="/machine/copy",
            target_name="dataset", state="QUEUED", binding_snapshot={}, logs="", created_at=now(), updated_at=now()))


def test_changed_binding_during_sync_rolls_back_the_new_norm_but_current_path_can_replay(platform):
    node = platform.register()
    dataset = register_asset(platform, node, norm_path="/machine/original").json()
    _pending_transfer(platform, node, dataset)
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    assert bind_path(platform, dataset, "/machine/original/").status_code == 200
    response = bind_path(platform, dataset, "/machine/new-stats")
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_IN_USE"
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


def test_path_replay_preserves_current_legacy_duplicate_identity_during_sync(platform):
    node = platform.register()
    first, second = [register_asset(platform, node, "NORM", name=name, path="/machine/stats").json()
                     for name in ("old", "current")]
    dataset = register_asset(platform, node, norm_asset_id=second["id"]).json()
    _pending_transfer(platform, node, dataset)
    result = bind_path(platform, dataset, "/machine/stats/")
    assert result.status_code == 200 and result.json()["norm_asset_id"] == second["id"]
    assert result.json()["norm_asset_id"] != first["id"]


def test_concurrent_dataset_registrations_share_one_norm_metadata_identity(platform):
    node = platform.register()
    barrier = Barrier(2)

    def create(number):
        barrier.wait(timeout=10)
        return register_asset(platform, node, name="dataset " + str(number), norm_path="/machine/shared-stats")

    with ThreadPoolExecutor(max_workers=2) as workers:
        rows = list(workers.map(create, (1, 2)))
    assert all(row.status_code == 201 for row in rows), [row.text for row in rows]
    assert rows[0].json()["norm_asset_id"] == rows[1].json()["norm_asset_id"]
    assert len(norms(platform)) == 1


def test_norm_binding_by_path_does_not_rewrite_frozen_jobs_or_enqueue_file_operations(platform):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    run = platform.run(node)
    with platform.app.state.db.read() as session:
        before = dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one())
        commands = deepcopy([(row.id, row.payload) for row in session.scalars(select(Command))])
    result = bind_path(platform, dataset, "/machine/stats")
    assert result.status_code == 200, result.text
    with platform.app.state.db.read() as session:
        assert dict(session.execute(select(Run.__table__).where(Run.id == run["id"])).mappings().one()) == before
        assert [(row.id, row.payload) for row in session.scalars(select(Command))] == commands


@pytest.mark.parametrize("role,status", [("admin", 200), ("operator", 200), ("viewer", 403)])
def test_auto_registration_uses_existing_role_csrf_and_origin_protection(platform, role, status):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    password = "path-binding-isolated-password"
    platform.app.state.auth.create_user("path-" + role, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        login = client.post("/api/v1/auth/login", json={"username": "path-" + role, "password": password})
        assert login.status_code == 200
        path = f"/api/v1/assets/{dataset['id']}/norm-binding"
        csrf = {"X-CSRF-Token": login.json()["csrf_token"]}
        response = client.put(path, json={"norm_path": "/machine/stats"}, headers=csrf)
        assert response.status_code == status, response.text
        if status == 200:
            assert client.put(path, json={"norm_path": "/machine/new"}).status_code == 403
            assert client.put(path, json={"norm_path": "/machine/new"},
                              headers={**csrf, "Origin": "https://foreign.example"}).status_code == 403


def test_nonhuman_requests_cannot_create_norm_metadata(platform):
    node = platform.register()
    dataset = register_asset(platform, node).json()
    before = platform.client.get("/api/v1/assets", headers=ADMIN).json()
    for headers in ({}, ENROLL, node["headers"]):
        response = platform.client.put(f"/api/v1/assets/{dataset['id']}/norm-binding", headers=headers,
                                       json={"norm_path": "/machine/stats"})
        assert response.status_code == 401
    assert platform.client.get("/api/v1/assets", headers=ADMIN).json() == before


def test_direct_helper_uses_registered_roots_and_never_claims_files_were_read(platform):
    node = platform.register()
    with platform.app.state.db.write() as session:
        norm = resolve_norm_path(session, node_id=node["node_id"], path="/machine/stats", dataset_name="Fixture")
        assert norm.state == "REGISTERED" and norm.metadata_json == {"registration_source": "dataset_norm_binding"}
    assert canonical_norm_path("//machine//stats/./") == "/machine/stats"
