"""Registered statistics reads use stored asset bindings and existing pinned SSH."""

from types import SimpleNamespace
import threading

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.common import now, offset_time, token_hash, uid
from vla_control import dataset_statistics as module
from vla_control.main import create_app
from vla_control.models import Asset, Command, Node, Run
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.settings import Settings


PASSWORD = "isolated-statistics-viewer-password"
SSH_PASSWORD = "isolated-statistics-ssh-password"
AGENT_TOKEN = "isolated-statistics-agent-token"


def statistics_result(dataset_path, norm_path):
    result = {
        "dataset": {"status": "READY", "path": dataset_path, "total_frames": 1200,
                    "total_episodes": 12, "fps": 20,
                    "features": [{"key": "observation.state", "dtype": "float32", "shape": [2, 7]},
                                 {"key": "observation.images.wrist", "dtype": "video", "shape": [224, 224, 3]}],
                    "camera_count": 1},
        "normalization": {"status": "UNBOUND", "features": []},
    }
    if norm_path:
        result["normalization"] = {"status": "READY", "path": norm_path, "features": [
            {"key": key, "dimension": 2, "mean": [1, 2], "std": [0.1, 0.2],
             "q01": [-1, -2], "q99": [3, 4]}
            for key in ("state", "actions")]}
    return result


@pytest.fixture
def statistics(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_REMOTE_STATE_DIR", str(tmp_path / "remote"))
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "statistics.db"),
                        enrollment_token="isolated-statistics-enrollment", scheduler_enabled=False,
                        session_cookie_secure=False)
    app = create_app(settings)
    app.state.auth.create_user("statistics-viewer", PASSWORD, "viewer")
    node_id, norm_id, data_id, connection_id, stamp = uid(), uid(), uid(), uid(), now()
    with app.state.db.write() as session:
        session.add(Node(id=node_id, name="fixture", hostname="fixture", status="ONLINE",
                         token_hash=token_hash(AGENT_TOKEN), roots=["/machine"],
                         output_root="/machine/output", created_at=stamp, last_seen=stamp,
                         metrics={}, gpus=[]))
        session.add(RemoteConnection(id=connection_id, host="192.0.2.20", port=2222,
                                    username="fixture-user", encrypted_password=app.state.remote_manager.vault.encrypt(SSH_PASSWORD),
                                    fingerprint="SHA256:statistics-pinned-host", probe_result={},
                                    install_dir="/machine/agent", created_at=stamp, expires_at=offset_time(-60)))
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection_id, node_id=node_id,
                                     idempotency_key=uid(), request_digest="c" * 64, target_key=uid(),
                                     onboarding_id=uid(), state="SUCCEEDED", stage="SUCCEEDED",
                                     logs=[], history=[], agent_config={"_operation": "install"},
                                     install_dir="/machine/agent", control_plane_url="https://fixture.invalid",
                                     connection_mode="direct", tunnel_state="NOT_REQUIRED", created_at=stamp,
                                     updated_at=stamp, finished_at=stamp))
        session.add(Asset(id=norm_id, node_id=node_id, name="fixture norm", kind="NORM",
                          path="/machine/norms/norm_stats.json", version="v1", state="REGISTERED",
                          metadata_json={"keep": "norm metadata"}, created_at=stamp))
        session.flush()
        session.add(Asset(id=data_id, node_id=node_id, name="fixture data", kind="DATASET",
                          path="/machine/dataset", version="v1", state="REGISTERED", norm_asset_id=norm_id,
                          metadata_json={"keep": "dataset metadata"}, created_at=stamp))
    remotes, reads = [], []

    class Remote:
        def __init__(self, **values):
            self.values, self.closed, self.connected = values, False, False
            remotes.append(self)

        def connect(self):
            assert self.values["host"] == "192.0.2.20"
            assert self.values["port"] == 2222
            assert self.values["username"] == "fixture-user"
            assert self.values["password"] == SSH_PASSWORD
            assert self.values["expected_fingerprint"] == "SHA256:statistics-pinned-host"
            self.connected = True
            return self

        def close(self):
            self.closed = True

    app.state.remote_manager._remote_factory = Remote
    manager = app.state.dataset_statistics_manager

    def reader(remote, dataset_path, norm_path, roots, **options):
        assert remote.connected
        reads.append({"dataset_path": dataset_path, "norm_path": norm_path, "roots": roots,
                      "options": options})
        return statistics_result(dataset_path, norm_path)

    manager.reader = reader
    with TestClient(app) as client:
        login = client.post("/api/v1/auth/login", json={"username": "statistics-viewer", "password": PASSWORD})
        assert login.status_code == 200, login.text
        yield SimpleNamespace(app=app, client=client, manager=manager, settings=settings,
                              node_id=node_id, norm_id=norm_id, data_id=data_id,
                              connection_id=connection_id, remotes=remotes, reads=reads)


def get(fixture, asset_id=None, **kwargs):
    return fixture.client.get(f"/api/v1/assets/{asset_id or fixture.data_id}/statistics", **kwargs)


def assert_read_only(fixture, expected_asset_count=2):
    with fixture.app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(Command)) == 0
        assert session.scalar(select(func.count()).select_from(Run)) == 0
        assert session.scalar(select(func.count()).select_from(Asset)) == expected_asset_count
        assert session.get(Asset, fixture.data_id).metadata_json == {"keep": "dataset metadata"}
        assert session.get(Asset, fixture.norm_id).metadata_json == {"keep": "norm metadata"}


def test_viewer_reads_bound_statistics_using_only_registered_paths_and_pinned_ssh(statistics):
    response = get(statistics)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    value = response.json()
    assert value["status"] == "READY" and value["asset_id"] == statistics.data_id
    assert value["node_id"] == statistics.node_id and value["checked_at"]
    assert value["dataset"]["total_frames"] == 1200
    assert value["dataset"]["total_episodes"] == 12
    assert value["dataset"]["fps"] == 20 and value["dataset"]["camera_count"] == 1
    assert value["normalization"]["asset_id"] == statistics.norm_id
    assert value["normalization"]["features"][0]["mean"] == [1, 2]
    assert statistics.reads[0]["dataset_path"] == "/machine/dataset"
    assert statistics.reads[0]["norm_path"] == "/machine/norms/norm_stats.json"
    assert statistics.reads[0]["roots"] == ["/machine"]
    assert len(statistics.remotes) == 1 and statistics.remotes[0].closed
    assert SSH_PASSWORD not in response.text and "encrypted_password" not in response.text
    assert_read_only(statistics)


def test_anonymous_enrollment_and_agent_credentials_do_not_grant_human_read_access(statistics):
    statistics.client.cookies.clear()
    for headers in ({}, {"Authorization": "Bearer isolated-statistics-enrollment"},
                    {"Authorization": f"Bearer {AGENT_TOKEN}"}):
        response = get(statistics, headers=headers)
        assert response.status_code == 401, response.text
    assert statistics.reads == [] and statistics.remotes == []


def test_unknown_and_non_dataset_assets_cannot_select_a_remote_file(statistics):
    assert get(statistics, uid()).status_code == 404
    assert get(statistics, statistics.norm_id).status_code in (404, 422)
    assert statistics.reads == []
    assert statistics.remotes == []


def test_query_paths_and_connection_ids_cannot_override_registered_bindings(statistics):
    response = get(statistics, params={"path": "/etc/passwd", "norm_path": "/etc/shadow",
                                     "node_id": "other", "connection_id": "untrusted"})
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        assert response.json()["status"] == "READY"
        assert statistics.reads[0]["dataset_path"] == "/machine/dataset"
        assert statistics.reads[0]["norm_path"] == "/machine/norms/norm_stats.json"
    else:
        assert statistics.reads == []


@pytest.mark.parametrize("path", ["/outside/private", "/machine/../private", "/machine/data\n"])
def test_invalid_registered_dataset_paths_never_reach_the_remote_reader(statistics, path):
    with statistics.app.state.db.write() as session:
        session.get(Asset, statistics.data_id).path = path
    response = get(statistics)
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        value = response.json()
        assert value["status"] == "FAILED"
        assert value["normalization"]["status"] != "UNBOUND"
        assert value["normalization"]["asset_id"] == statistics.norm_id
        assert not value["normalization"].get("features")
    assert statistics.reads == []


def test_bound_norm_path_outside_current_roots_is_not_read(statistics):
    with statistics.app.state.db.write() as session:
        session.get(Asset, statistics.norm_id).path = "/outside/private.json"
    response = get(statistics)
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        assert response.json()["normalization"]["status"] in {"FAILED", "UNAVAILABLE"}
    assert all(read["norm_path"] is None for read in statistics.reads)


def test_unbound_dataset_retains_metadata_without_reading_an_arbitrary_default_norm(statistics):
    with statistics.app.state.db.write() as session:
        session.get(Asset, statistics.data_id).norm_asset_id = None
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["dataset"]["status"] == "READY"
    assert value["dataset"]["total_frames"] == 1200
    assert value["normalization"]["status"] == "UNBOUND"
    assert not value["normalization"].get("features")
    assert statistics.reads[0]["norm_path"] is None
    assert_read_only(statistics)


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_unavailable_bound_norm_does_not_prevent_dataset_metadata_or_read_norm_file(statistics, state):
    with statistics.app.state.db.write() as session:
        session.get(Asset, statistics.norm_id).state = state
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["dataset"]["status"] == "READY"
    assert value["normalization"]["status"] == "UNAVAILABLE"
    assert value["status"] in {"PARTIAL", "FAILED"}
    assert statistics.reads[0]["norm_path"] is None


@pytest.mark.parametrize("invalid", ["wrong_kind", "different_node"])
def test_invalid_norm_binding_is_not_used_for_ssh_file_reads(statistics, invalid):
    with statistics.app.state.db.write() as session:
        norm = session.get(Asset, statistics.norm_id)
        if invalid == "wrong_kind":
            norm.kind = "MODEL"
        else:
            stamp, other = now(), uid()
            session.add(Node(id=other, name="other", hostname="other", status="ONLINE", token_hash="f" * 64,
                             roots=["/machine"], output_root="/machine/output", created_at=stamp,
                             last_seen=stamp, metrics={}, gpus=[]))
            session.flush()
            norm.node_id = other
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["normalization"]["status"] in {"FAILED", "UNAVAILABLE"}
    assert statistics.reads[0]["norm_path"] is None


@pytest.mark.parametrize("change", ["state", "heartbeat"])
def test_offline_or_stale_machine_is_reported_without_opening_ssh(statistics, change):
    with statistics.app.state.db.write() as session:
        node = session.get(Node, statistics.node_id)
        if change == "state":
            node.status = "OFFLINE"
        else:
            node.last_seen = offset_time(-1000)
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "OFFLINE"
    assert value["dataset"]["status"] != "READY"
    assert value["normalization"]["status"] != "READY"
    assert statistics.remotes == [] and statistics.reads == []
    assert_read_only(statistics)


@pytest.mark.parametrize("failed_part", ["dataset", "normalization"])
def test_partial_reads_keep_successful_data_and_do_not_invent_missing_statistics(statistics, failed_part):
    def reader(remote, dataset_path, norm_path, roots, **options):
        value = statistics_result(dataset_path, norm_path)
        value[failed_part] = {"status": "FAILED", "path": dataset_path if failed_part == "dataset" else norm_path,
                              "features": [], "error": {"code": "STATISTICS_NOT_FOUND", "message": "文件不存在"}}
        return value
    statistics.manager.reader = reader
    value = get(statistics).json()
    assert value["status"] == "PARTIAL"
    assert value[failed_part]["status"] == "FAILED"
    assert value[failed_part].get("features", []) == []
    other = "normalization" if failed_part == "dataset" else "dataset"
    assert value[other]["status"] == "READY"


def test_missing_quantiles_remain_absent_instead_of_becoming_zero_values(statistics):
    def reader(remote, dataset_path, norm_path, roots, **options):
        result = statistics_result(dataset_path, norm_path)
        for feature in result["normalization"]["features"]:
            feature.pop("q01")
            feature.pop("q99")
        return result
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200, response.text
    for feature in response.json()["normalization"]["features"]:
        assert "q01" not in feature and "q99" not in feature


def test_raw_feature_statistics_keep_their_real_keys_and_optional_ranges(statistics):
    def reader(remote, dataset_path, norm_path, roots, **options):
        result = statistics_result(dataset_path, norm_path)
        result["normalization"]["features"] = [{
            "key": "observation.robot0_eef_pos_wrt_start", "dimension": 3,
            "mean": [-0.3, 0, 0.4], "std": [0.2, 0, 0.1],
            "min": [-0.8, 0, -0.1], "max": [0.4, 0, 0.7],
        }]
        return result
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200 and response.json()["status"] == "READY", response.text
    features = response.json()["normalization"]["features"]
    assert len(features) == 1
    assert features[0]["key"] == "observation.robot0_eef_pos_wrt_start"
    assert features[0]["dimension"] == 3
    assert features[0]["std"] == [0.2, 0, 0.1]
    assert features[0]["min"] == [-0.8, 0, -0.1]
    assert "q01" not in features[0] and "q99" not in features[0]


def test_only_supported_metadata_and_statistics_fields_leave_the_api(statistics):
    def reader(remote, dataset_path, norm_path, roots, **options):
        value = statistics_result(dataset_path, norm_path)
        value["private_output"] = "must-not-be-returned"
        for section in value.values():
            if isinstance(section, dict):
                section["password"] = SSH_PASSWORD
                for feature in section.get("features", []):
                    feature["private_token"] = "must-not-be-returned"
        return value
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200 and response.json()["status"] == "READY", response.text
    assert SSH_PASSWORD not in response.text and "must-not-be-returned" not in response.text
    assert "private_output" not in response.text and "private_token" not in response.text


@pytest.mark.parametrize("invalid", ["nan", "infinite", "negative_std", "wrong_width", "duplicate",
                                     "wrong_path", "path_type", "error_code_type", "surrogate"])
def test_invalid_reader_statistics_cannot_be_published_as_successful_values(statistics, invalid):
    def reader(remote, dataset_path, norm_path, roots, **options):
        value = statistics_result(dataset_path, norm_path)
        feature = value["normalization"]["features"][0]
        if invalid == "nan":
            feature["mean"][0] = float("nan")
        elif invalid == "infinite":
            feature["q99"][1] = float("inf")
        elif invalid == "negative_std":
            feature["std"][0] = -1
        elif invalid == "wrong_width":
            feature["mean"] = [1]
        elif invalid == "duplicate":
            value["normalization"]["features"].append(dict(feature))
        elif invalid == "path_type":
            value["normalization"]["path"] = {"untrusted": "path"}
        elif invalid == "error_code_type":
            value["normalization"] = {"status": "FAILED", "path": norm_path, "features": [],
                                      "error": {"code": {"untrusted": "error"}, "message": "private raw output"}}
        elif invalid == "surrogate":
            feature["key"] = "invalid-\ud800-feature"
        else:
            value["normalization"]["path"] = "/outside/untrusted.json"
        return value
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "PARTIAL" and value["dataset"]["status"] == "READY"
    assert value["normalization"]["status"] == "FAILED"
    assert not value["normalization"].get("features")
    assert "/outside/untrusted.json" not in response.text
    assert "private raw output" not in response.text


def test_busy_reads_keep_existing_norm_binding_without_connecting_or_faking_unbound(statistics):
    acquired = 0
    try:
        for _ in range(4):
            assert statistics.manager._slots.acquire(blocking=False)
            acquired += 1
        response = get(statistics)
    finally:
        for _ in range(acquired):
            statistics.manager._slots.release()
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "FAILED"
    assert value["error"]["code"] == "DATASET_STATISTICS_BUSY"
    assert value["dataset"]["status"] == "FAILED"
    assert value["normalization"]["status"] == "FAILED"
    assert value["normalization"]["asset_id"] == statistics.norm_id
    assert not value["normalization"].get("features")
    assert statistics.remotes == [] and statistics.reads == []


def test_read_deadline_closes_a_blocked_connection_and_returns_a_safe_timeout(statistics, monkeypatch):
    factory = statistics.app.state.remote_manager._remote_factory
    original_connect, original_close = factory.connect, factory.close
    released = threading.Event()

    def connect(remote):
        original_connect(remote)
        assert released.wait(1), "the read deadline must interrupt a blocked SSH connection"

    def close(remote):
        original_close(remote)
        released.set()

    real_timer = module.threading.Timer
    monkeypatch.setattr(factory, "connect", connect)
    monkeypatch.setattr(factory, "close", close)
    monkeypatch.setattr(module.threading, "Timer", lambda _seconds, callback: real_timer(0.02, callback))
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "FAILED"
    assert value["error"]["code"] == "DATASET_STATISTICS_TIMEOUT"
    assert statistics.remotes[-1].closed
    assert not value["normalization"].get("features")


def test_reader_failures_are_bounded_public_errors_without_credentials_or_raw_output(statistics):
    def reader(*args, **kwargs):
        raise RuntimeError(f"private command output; password={SSH_PASSWORD}; token=do-not-return")
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "FAILED"
    assert SSH_PASSWORD not in response.text and "do-not-return" not in response.text
    assert "private command output" not in response.text
    assert statistics.remotes[-1].closed
    assert_read_only(statistics)


@pytest.mark.parametrize("change", ["binding", "roots", "dataset_path", "norm_path"])
def test_changes_during_read_cannot_publish_statistics_for_a_stale_binding(statistics, change):
    def reader(remote, dataset_path, norm_path, roots, **options):
        with statistics.app.state.db.write() as session:
            if change == "binding":
                session.get(Asset, statistics.data_id).norm_asset_id = None
            elif change == "roots":
                session.get(Node, statistics.node_id).roots = ["/another"]
            elif change == "dataset_path":
                session.get(Asset, statistics.data_id).path = "/machine/replaced-dataset"
            else:
                session.get(Asset, statistics.norm_id).path = "/machine/replaced-norm.json"
        return statistics_result(dataset_path, norm_path)
    statistics.manager.reader = reader
    response = get(statistics)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "FAILED"
    assert value["dataset"]["status"] != "READY"
    assert value["normalization"]["status"] != "READY"
    assert not value["normalization"].get("features")
    assert statistics.remotes[-1].closed


def test_no_pinned_installation_does_not_fall_back_to_another_node_connection(statistics):
    with statistics.app.state.db.write() as session:
        stamp, other = now(), uid()
        session.add(Node(id=other, name="other", hostname="other", status="ONLINE", token_hash="e" * 64,
                         roots=["/machine"], output_root="/machine/output", created_at=stamp,
                         last_seen=stamp, metrics={}, gpus=[]))
        session.flush()
        job = session.scalar(select(RemoteDeployment).where(RemoteDeployment.node_id == statistics.node_id))
        job.node_id = other
    response = get(statistics)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "FAILED"
    assert statistics.remotes == [] and statistics.reads == []
