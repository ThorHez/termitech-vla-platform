"""Run-owned checkpoint/loader settings need no different dataset recipe."""

from copy import deepcopy

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.main import create_app
from vla_control.models import Run
from vla_control.settings import Settings

from .conftest import ADMIN, automatic_preflight, confirmed_run_request, gpu, snapshot
from .test_automatic_configuration import automatic_request
from .test_automatic_preflight import acknowledge
from .test_task_recipe import request_preview


OPTIONS = {"save_interval": 2000, "keep_period": 20000, "num_workers": 8,
           "warmup_steps": 2000, "decay_steps": 50000, "decay_lr": 0.00001,
           "clip_gradient_norm": 1.0, "ema_decay": 0.999, "fsdp_devices": 8, "log_interval": 10}
USER_PARAMETERS = {"steps": 60000, "batch_size": 72, "learning_rate": 0.00003, **OPTIONS}


def _report_gpus(platform, node, count):
    platform.heartbeat(node, gpus=[gpu(f"GPU-runtime-{index}") for index in range(count)])


def _preview_with_options(platform, parameters, *, gpu_count=8):
    node, request = automatic_request(platform, parameters=parameters, gpu_count=gpu_count)
    _report_gpus(platform, node, gpu_count)
    response = request_preview(platform, node, **request)
    return node, request, response


@pytest.mark.parametrize("workers", [0, 8])
def test_automatic_run_freezes_runtime_options_without_selecting_an_old_recipe(platform, workers):
    options = {**OPTIONS, "num_workers": workers}
    node, request = automatic_request(platform, parameters={**USER_PARAMETERS, **options}, gpu_count=8)
    _report_gpus(platform, node, 8)
    original = deepcopy(request)
    response = request_preview(platform, node, **request)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    assert payload["recipe"].startswith("auto_")
    assert payload.get("recipe_definition") is None
    assert payload["gpu_count"] == 8
    assert payload["dataset_bindings"] == original["dataset_bindings"]
    assert all(payload["parameters"][key] == value for key, value in options.items())
    code = snapshot(recipe=payload["recipe"], auto_config=payload["auto_config"],
                    dataset_bindings=payload["dataset_bindings"], automatic_preflight=automatic_preflight(payload))
    preview = acknowledge(platform, node, identifier, command, code)
    assert preview["status"] == "READY", preview
    run_response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier},
                                        json=confirmed_run_request(preview))
    assert run_response.status_code == 201, run_response.text
    run = run_response.json()
    assert all(run["parameters"][key] == value for key, value in options.items())
    assert run["parameters"] == {**USER_PARAMETERS, **options}
    assert run["dataset_bindings"] == original["dataset_bindings"]
    assert run["auto_config"] == original["auto_config"]
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert all(start["payload"]["parameters"][key] == value for key, value in options.items())
    assert start["payload"]["parameters"] == {**USER_PARAMETERS, **options}


def test_old_automatic_request_does_not_gain_unrequested_runtime_overrides(platform):
    node, request = automatic_request(platform)
    response = request_preview(platform, node, **request)
    assert response.status_code == 202, response.text
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    assert not set(OPTIONS).intersection(command["payload"]["parameters"])


@pytest.mark.parametrize("field,value", [
    ("save_interval", 0), ("save_interval", True), ("save_interval", 1.5),
    ("keep_period", 0), ("keep_period", None), ("keep_period", "20000"),
    ("num_workers", -1), ("num_workers", True), ("num_workers", 1.5),
    ("num_workers", 129), ("warmup_steps", -1), ("warmup_steps", True),
    ("warmup_steps", 1.5), ("decay_steps", 0), ("decay_steps", True),
    ("decay_lr", -0.1), ("decay_lr", True), ("decay_lr", "1e-5"),
    ("clip_gradient_norm", 0), ("clip_gradient_norm", -1), ("clip_gradient_norm", True),
    ("ema_decay", -0.1), ("ema_decay", 1), ("ema_decay", True),
    ("fsdp_devices", 0), ("fsdp_devices", 1.5), ("fsdp_devices", True),
    ("log_interval", 0), ("log_interval", 1.5), ("arbitrary_runtime_command", 1),
])
def test_invalid_or_unknown_runtime_option_never_reaches_agent(platform, field, value):
    node, request = automatic_request(platform, parameters={field: value})
    response = request_preview(platform, node, **request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PARAMETERS"
    assert platform.commands(node) == []


@pytest.mark.parametrize("changes", [
    {"warmup_steps": 2000, "decay_steps": 2000},
    {"warmup_steps": 2001, "decay_steps": 2000},
    {"decay_lr": 0.000031, "learning_rate": 0.00003},
    {"fsdp_devices": 3},
    {"fsdp_devices": 16},
])
def test_inconsistent_training_schedule_or_sharding_is_rejected_before_preflight(platform, changes):
    node, _, response = _preview_with_options(platform, {**USER_PARAMETERS, **changes})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "INVALID_PARAMETERS"
    assert platform.commands(node) == []


@pytest.mark.parametrize("changes", [
    {"warmup_steps": 0, "decay_steps": 1, "decay_lr": 0, "ema_decay": 0},
    {"decay_lr": 0.00003, "fsdp_devices": 4},
    {"fsdp_devices": 1, "clip_gradient_norm": 0.00001, "log_interval": 1},
])
def test_valid_schedule_boundaries_and_partial_fsdp_do_not_replace_dataset_bindings(platform, changes):
    parameters = {**USER_PARAMETERS, **changes}
    node, request, response = _preview_with_options(platform, parameters)
    assert response.status_code == 202, response.text
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    assert command["payload"]["parameters"] == parameters
    assert command["payload"]["dataset_bindings"] == request["dataset_bindings"]
    assert command["payload"]["auto_config"] == request["auto_config"]


@pytest.mark.parametrize("field", list(OPTIONS))
def test_openpi_runtime_options_are_not_added_to_smoke_requests(platform, field):
    node = platform.register()
    response = request_preview(platform, node, template_id="smoke", gpu_count=0, parameters={field: OPTIONS[field]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PARAMETERS"
    assert platform.commands(node) == []


def _login(client, username):
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": "runtime-options-test-123"})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]


@pytest.mark.parametrize("parameters", [
    USER_PARAMETERS,
    {**USER_PARAMETERS, "num_workers": 0},
    {"steps": 80000, "batch_size": 128},
])
def test_user_drafts_and_templates_preserve_new_and_legacy_parameters_without_launch(tmp_path, parameters):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "runtime-options.sqlite3"),
                             enrollment_token="unused-options-agent", scheduler_enabled=False,
                             session_cookie_secure=False))
    for username in ("alice", "bob"):
        app.state.auth.create_user(username, "runtime-options-test-123", "operator")
    content = {"schema_version": 1, "step": 1, "advanced_recipe": False,
               "edited_parameters": list(parameters), "request": {
                   "name": "new datasets, same training settings", "template_id": "openpi-sft",
                   "parameters": parameters, "inputs": {"model": "current-model"},
                   "dataset_bindings": [{"dataset_asset_id": "new-dataset", "norm_asset_id": "new-norm", "weight": 1}]}}
    with TestClient(app) as client:
        _login(client, "alice")
        saved = client.put("/api/v1/training-draft", json={"expected_revision": 0, "content": content})
        assert saved.status_code == 200, saved.text
        template = client.post("/api/v1/training-templates", json={"name": "my runtime settings", "content": content})
        assert template.status_code == 201, template.text
        template_id = template.json()["id"]
        _login(client, "bob")
        assert client.get("/api/v1/training-draft").json()["content"] is None
        assert client.get("/api/v1/training-templates").json() == []
        assert client.get("/api/v1/training-templates/" + template_id).status_code == 404
        _login(client, "alice")
        assert client.get("/api/v1/training-draft").json()["content"] == content
        assert client.get("/api/v1/training-templates/" + template_id).json()["content"] == content
        with app.state.db.read() as session:
            assert session.scalar(select(func.count()).select_from(Run)) == 0
