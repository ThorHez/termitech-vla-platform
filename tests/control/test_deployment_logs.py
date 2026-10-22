"""Streaming installation diagnostics stay durable, private and exportable."""

import json
import stat

from .conftest import ADMIN
from .test_remote_deployment import PASSWORD, connection, create, payload, remote  # noqa: F401


def make_job(platform):
    tested = connection(platform).json()
    response = create(platform, payload(tested["connection_id"]))
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_detailed_logs_survive_tail_limit_and_export_without_credentials(remote):
    platform, manager = remote
    job_id = make_job(platform)
    for index in range(305):
        manager._log(job_id, "INSTALL_DEPENDENCIES",
                     f"Downloading package-{index}: {PASSWORD} test-enrollment test-admin "
                     "https://operator:source-secret@packages.example/a.whl?token=query-secret")
    current = manager.get(job_id)
    assert current["stage"] == "INSTALL_DEPENDENCIES"
    assert len(current["logs"]) == 300
    assert "package-304" in current["logs"][-1]["message"]
    exported = platform.client.get(f"/api/v1/remote-deployments/{job_id}/logs/download", headers=ADMIN)
    assert exported.status_code == 200
    assert exported.headers["cache-control"] == "no-store"
    assert "attachment" in exported.headers["content-disposition"]
    records = [json.loads(line) for line in exported.text.splitlines()]
    assert len(records) > 305
    assert any("package-0:" in entry["message"] for entry in records)
    assert any("package-304:" in entry["message"] for entry in records)
    for secret in (PASSWORD, "test-enrollment", "test-admin", "source-secret", "query-secret"):
        assert secret not in exported.text
    path = manager._log_path(job_id)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert PASSWORD not in path.read_text()


def test_logs_download_requires_admin_and_valid_job(remote):
    platform, _ = remote
    job_id = make_job(platform)
    url = f"/api/v1/remote-deployments/{job_id}/logs/download"
    assert platform.client.get(url).status_code == 401
    assert platform.client.get(url, headers={"Authorization": "Bearer test-enrollment"}).status_code == 401
    assert platform.client.get("/api/v1/remote-deployments/missing/logs/download", headers=ADMIN).status_code == 404


def test_old_deployment_falls_back_to_retained_stage_logs(remote):
    platform, manager = remote
    job_id = make_job(platform)
    manager._log_path(job_id).unlink()
    expected = manager.get(job_id)["logs"]
    response = platform.client.get(f"/api/v1/remote-deployments/{job_id}/logs/download", headers=ADMIN)
    assert response.status_code == 200
    assert [json.loads(line) for line in response.text.splitlines()] == expected


def test_log_storage_failure_does_not_abort_installation_updates(remote, monkeypatch):
    platform, manager = remote
    job_id = make_job(platform)
    path = manager._log_path(job_id)
    path.unlink()
    # Existing symlinks are never followed or overwritten by diagnostics.
    target = path.parent / "untouched.txt"
    target.write_text("untouched")
    path.symlink_to(target)
    manager._log(job_id, "INSTALL_ENVIRONMENT", "创建独立 Python 环境")
    assert manager.get(job_id)["logs"][-1]["message"] == "创建独立 Python 环境"
    assert target.read_text() == "untouched"
    exported = platform.client.get(f"/api/v1/remote-deployments/{job_id}/logs/download", headers=ADMIN)
    assert exported.status_code == 200
    assert "untouched" not in exported.text
    assert "创建独立 Python 环境" in exported.text


def test_download_repairs_a_temporary_archive_write_gap_from_database_tail(remote, monkeypatch):
    platform, manager = remote
    job_id = make_job(platform)
    manager._log(job_id, "INSTALL_ENVIRONMENT", "before gap")
    save_log = manager._save_log
    monkeypatch.setattr(manager, "_save_log", lambda *_: None)
    manager._log(job_id, "INSTALL_DEPENDENCIES", "during gap")
    monkeypatch.setattr(manager, "_save_log", save_log)
    manager._log(job_id, "INSTALL_VERIFY", "after gap")
    assert "during gap" not in manager._log_path(job_id).read_text()
    response = platform.client.get(f"/api/v1/remote-deployments/{job_id}/logs/download", headers=ADMIN)
    messages = [json.loads(line)["message"] for line in response.text.splitlines()]
    assert messages[-3:] == ["before gap", "during gap", "after gap"]
