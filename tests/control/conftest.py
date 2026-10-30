from dataclasses import dataclass
import hashlib

from fastapi.testclient import TestClient
import pytest

from vla_control.common import now
from vla_control.main import create_app
from vla_control.scheduler import tick
from vla_control.settings import Settings
from vla_protocol import canonical_digest


ADMIN = {"Authorization": "Bearer test-admin"}
ENROLL = {"Authorization": "Bearer test-enrollment"}

GENERATED_CONFIG_SOURCE = "# CPU fixture: resolved configuration used for confirmation tests.\nCONFIG = {\"fixture\": True}\n"
GENERATED_CONFIG_SHA256 = hashlib.sha256(GENERATED_CONFIG_SOURCE.encode("utf-8")).hexdigest()


def generated_config():
    return {"schema_version": 1, "language": "python", "filename": "training_config.py",
            "source": GENERATED_CONFIG_SOURCE, "sha256": GENERATED_CONFIG_SHA256}


def confirmed_run_request(preview):
    body = {"preflight_id": preview["id"]}
    code = (preview.get("code_snapshot") or {}).get("automatic_preflight", {}).get("plan", {}).get("generated_config")
    if code is not None:
        body["confirmed_config_sha256"] = code["sha256"]
    return body


def snapshot(**changes):
    return {"repo_url": "https://example.com/team/training.git", "code_dir": "/machine/code", "branch": "main",
            "commit_sha": "a" * 40, "commit_message": "training recipe", "committed_at": now(), "dirty": False,
            "changes_digest": "b" * 64, "untracked_digest": "c" * 64, "module_origin": "/machine/code/smoke_training.py",
            "captured_at": now(), "complete": True, "recipe": "smoke", **changes}


def sample_probe(path, *, frames=32, horizon=16, sequence=()):
    indices = sorted({0, frames // 2, frames - 1})
    return {"schema_version": 1, "status": "verified", "path": path,
            "checked_samples": len(indices), "sample_indices": indices,
            "samples": [{"index": index} for index in indices],
            "action_horizon": horizon, "action_sequence_keys": list(sequence)}


def automatic_preflight(payload):
    """Simulate the evidence a current Agent returns after its local checks."""
    paths = {role: value["path"] for role, value in payload["inputs"].items()}
    definition = payload.get("recipe_definition")
    bindings = payload.get("dataset_bindings")
    pairs = [(f"dataset_{i}", f"norm_{i}") for i in range(len(bindings))] if bindings is not None else [("dataset", "norm")]
    plan = {"recipe": payload["recipe"], "input_authority": "registered_assets", "recipe_definition_digest": definition["digest"] if definition else None,
            "requested_inputs": paths, "inputs": paths, "parameters": payload["parameters"],
            "gpu_count": payload["gpu_count"], "entrypoint": "/machine/code/scripts/train.py",
            "generated_config": generated_config(),
            "datasets": [{"path": paths[dataset], "norm_path": paths[norm], "norm_sha256": "f" * 64,
                          "dataset_asset_id": payload["inputs"][dataset]["id"], "norm_asset_id": payload["inputs"][norm]["id"],
                          "metadata_sha256": "e" * 64, "frames": 32, "episodes": 2, "factory": "LeRobotDataConfig"}
                         for dataset, norm in pairs],
            "model": {"path": paths["model"] + "/params", "checkpoint_path": paths["model"], "asset_id": payload["inputs"]["model"]["id"],
                      "metadata": [], "parameter_bytes": 1024},
            "storage": {"required_bytes": 8192, "reference_checkpoint_bytes": 4096, "model_params_bytes": 1024}}
    for dataset, (role, _) in zip(plan["datasets"], pairs):
        fps = payload["inputs"][role].get("dataset_profile", {}).get("fps")
        if type(fps) in (int, float) and 0 < fps <= 1000:
            dataset.update(effective_fps=fps, fps_source="registered")
    plan["training_cache"] = {"datasets_cache": "/machine/cache/datasets",
                              "paths": {"HF_DATASETS_CACHE": "/machine/cache/datasets"},
                              "builders": [{"path": item["path"], "cache_dir": "/machine/cache/datasets/parquet/default"}
                                           for item in plan["datasets"]]}
    if payload.get("auto_config") is not None:
        plan["auto_config"] = payload["auto_config"]
    if bindings is not None:
        plan.update(dataset_bindings=bindings, weights=[row["weight"] for row in bindings])
        for dataset, binding in zip(plan["datasets"], bindings):
            if binding.get("processor") is not None:
                dataset.update(processor=binding["processor"], factory=binding["processor"]["name"],
                               processor_identity={"name": binding["processor"]["name"], "module": "openpi.training.config",
                                                   "source": {"path": "src/openpi/training/config.py", "sha256": "d" * 64}})
                if binding["processor"]["name"] == "ComposableDataConfig":
                    dataset["sample_probe"] = sample_probe(dataset["path"], frames=dataset["frames"],
                        horizon=(payload.get("auto_config") or {}).get("action_horizon", 16))
    return {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
            "plan_digest": canonical_digest(plan)}


def automatic_checks():
    return [{"name": name, "ok": True, "message": "checked"} for name in (
        "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache",
        "output_isolation", "output_storage", "gpu_capacity", "runtime", "local_paths", "code_version", "adapter")]


def gpu(uuid="GPU-1", **changes):
    return {"uuid": uuid, "name": "test gpu", "memory_total_mb": 24000, "memory_used_mb": 0,
            "utilization_pct": 0, "processes": [], **changes}


@dataclass
class Platform:
    client: TestClient
    app: object
    settings: Settings

    def register(self, name="worker", adapter="smoke", gpus=None, key=None, recipe="validated_recipe", metadata=None):
        body = {"name": name, "hostname": name, "roots": ["/machine"], "output_root": "/machine/output",
                "profiles": [{"id": "default", "name": "runtime", "code_dir": "/machine/code", "python": "/usr/bin/python3",
                              "adapter": adapter, "recipe": "smoke" if adapter == "smoke" and recipe == "validated_recipe" else recipe,
                              "metadata": {"validated": True} if metadata is None else metadata}]}
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

    def preview(self, node, inputs=None, template="smoke", count=1, code=None, recipe=None):
        response = self.client.post("/api/v1/run-previews", headers=ADMIN, json={"name": "training", "node_id": node["node_id"],
            "profile_id": "default", "template_id": template, "recipe": recipe, "inputs": inputs or {}, "parameters": {"steps": 3}, "gpu_count": count})
        assert response.status_code == 202, response.text
        preview = response.json()
        command = next(c for c in self.commands(node) if c["type"] == "PREFLIGHT" and c["payload"]["preview_id"] == preview["id"])
        resolved_code = code or snapshot(recipe=command["payload"]["recipe"])
        checks = [{"name": "local inputs and runtime", "ok": True, "message": "checked"}]
        if template == "openpi-sft" and code is None:
            resolved_code["automatic_preflight"] = automatic_preflight(command["payload"])
            checks = automatic_checks()
        result = {"checks": checks, "code_snapshot": resolved_code}
        response = self.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"],
                                    json={"node_id": node["node_id"], "status": "APPLIED", "result": result})
        assert response.status_code == 200, response.text
        return self.client.get(f"/api/v1/run-previews/{preview['id']}", headers=ADMIN).json()

    def run(self, node, key=None, **kwargs):
        preview = self.preview(node, **kwargs)
        response = self.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": key or preview["id"]}, json=confirmed_run_request(preview))
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
                        enrollment_token="test-enrollment", scheduler_enabled=False, legacy_admin_enabled=True)
    app = create_app(settings)
    with TestClient(app) as client:
        yield Platform(client, app, settings)
