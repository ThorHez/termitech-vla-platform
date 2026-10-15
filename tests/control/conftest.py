from dataclasses import dataclass

from fastapi.testclient import TestClient
import pytest

from vla_control.common import now
from vla_control.main import create_app
from vla_control.scheduler import tick
from vla_control.settings import Settings


ADMIN = {"Authorization": "Bearer test-admin"}
ENROLL = {"Authorization": "Bearer test-enrollment"}


def snapshot(**changes):
    return {"repo_url": "https://example.com/team/training.git", "code_dir": "/machine/code", "branch": "main",
            "commit_sha": "a" * 40, "commit_message": "training recipe", "committed_at": now(), "dirty": False,
            "changes_digest": "b" * 64, "untracked_digest": "c" * 64, "module_origin": "/machine/code/smoke_training.py",
            "captured_at": now(), "complete": True, **changes}


def gpu(uuid="GPU-1", **changes):
    return {"uuid": uuid, "name": "test gpu", "memory_total_mb": 24000, "memory_used_mb": 0,
            "utilization_pct": 0, "processes": [], **changes}


@dataclass
class Platform:
    client: TestClient
    app: object
    settings: Settings

    def register(self, name="worker", adapter="smoke", gpus=None, key=None):
        body = {"name": name, "hostname": name, "roots": ["/machine"], "output_root": "/machine/output",
                "profiles": [{"id": "default", "name": "runtime", "code_dir": "/machine/code", "python": "/usr/bin/python3",
                              "adapter": adapter, "recipe": "validated_recipe", "metadata": {"validated": True}}]}
        response = self.client.post("/agent/v1/register", headers={**ENROLL, **({"Idempotency-Key": key} if key else {})}, json=body)
        assert response.status_code == 201, response.text
        node = response.json()
        node["headers"] = {"Authorization": "Bearer " + node["token"]}
        self.heartbeat(node, gpus=[gpu()] if gpus is None else gpus)
        return node

    def heartbeat(self, node, gpus=None, attempts=None):
        response = self.client.post("/agent/v1/heartbeat", headers=node["headers"], json={"node_id": node["node_id"],
            "gpus": [gpu()] if gpus is None else gpus, "metrics": {}, "active_attempts": attempts or []})
        assert response.status_code == 200, response.text

    def commands(self, node):
        response = self.client.get("/agent/v1/commands", headers=node["headers"], params={"node_id": node["node_id"]})
        assert response.status_code == 200, response.text
        return response.json()

    def preview(self, node, inputs=None, template="smoke", count=1, code=None):
        response = self.client.post("/api/v1/run-previews", headers=ADMIN, json={"name": "training", "node_id": node["node_id"],
            "profile_id": "default", "template_id": template, "inputs": inputs or {}, "parameters": {"steps": 3}, "gpu_count": count})
        assert response.status_code == 202, response.text
        preview = response.json()
        command = next(c for c in self.commands(node) if c["type"] == "PREFLIGHT" and c["payload"]["preview_id"] == preview["id"])
        result = {"checks": [{"name": "local inputs and runtime", "ok": True, "message": "checked"}], "code_snapshot": code or snapshot()}
        response = self.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"],
                                    json={"node_id": node["node_id"], "status": "APPLIED", "result": result})
        assert response.status_code == 200, response.text
        return self.client.get(f"/api/v1/run-previews/{preview['id']}", headers=ADMIN).json()

    def run(self, node, key=None, **kwargs):
        preview = self.preview(node, **kwargs)
        response = self.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": key or preview["id"]}, json={"preflight_id": preview["id"]})
        assert response.status_code == 201, response.text
        return response.json()

    def schedule(self):
        return tick(self.app.state.db, self.settings)

    def get_run(self, run):
        return self.client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN).json()

    def emit(self, node, attempt_id, seq, kind, data, timestamp=None):
        return self.client.post("/agent/v1/events/batch", headers=node["headers"], json={"node_id": node["node_id"], "events": [
            {"attempt_id": attempt_id, "seq": seq, "type": kind, "timestamp": timestamp or now(), "data": data}]})

    def start(self, node, run, seq=1):
        current = self.get_run(run)
        attempt = current["attempts"][0]
        response = self.emit(node, attempt["id"], seq, "STARTED", {"pid": 1234, "code_snapshot": current["code_expectation"],
            "output_dir": "/machine/output/" + attempt["id"]})
        assert response.status_code == 200, response.text
        return attempt


@pytest.fixture
def platform(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "control.sqlite"), admin_token="test-admin",
                        enrollment_token="test-enrollment", scheduler_enabled=False)
    app = create_app(settings)
    with TestClient(app) as client:
        yield Platform(client, app, settings)
