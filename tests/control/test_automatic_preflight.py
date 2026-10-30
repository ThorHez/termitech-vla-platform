"""Only a complete automatic Agent check for the current task authorizes a run."""

import copy

import pytest
from vla_protocol import canonical_digest

from vla_control.models import Asset, Preview, Profile

from .conftest import ADMIN, automatic_checks, automatic_preflight, confirmed_run_request, snapshot
from .test_task_recipe import inputs, request_preview


def pending(platform):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={})
    response = request_preview(platform, node, inputs=inputs(platform, node), recipe="pi05_unvalidated")
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    code = snapshot(recipe=command["payload"]["recipe"], automatic_preflight=automatic_preflight(command["payload"]))
    return node, identifier, command, code


def acknowledge(platform, node, identifier, command, code, checks=None):
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": "APPLIED", "result": {
            "checks": automatic_checks() if checks is None else checks, "code_snapshot": code}})
    assert response.status_code == 200, response.text
    return platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json()


def submit(platform, identifier):
    checked = platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json()
    return platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier},
                                json=confirmed_run_request(checked))


def test_automatic_preflight_can_submit_without_claiming_manual_or_training_acceptance(platform):
    node, identifier, command, code = pending(platform)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY"
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["code_expectation"]["automatic_preflight"] == code["automatic_preflight"]
    with platform.app.state.db.read() as session:
        assert session.get(Profile, (node["node_id"], "default")).metadata_json == {}


def fps_warning():
    return {"name": "dataset_fps", "ok": True, "severity": "warning", "code": "DATASET_FPS_MISMATCH",
            "message": "Dataset registration FPS 30 differs from metadata FPS 60"}


@pytest.mark.parametrize("required_check", ["dataset_timestamps", "dataset_samples"])
def test_legacy_preflight_without_real_dataset_check_cannot_authorize_training(platform, required_check):
    node, identifier, command, code = pending(platform)
    checks = [item for item in automatic_checks() if item["name"] != required_check]
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INCOMPLETE"
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"


@pytest.mark.parametrize("required_check,message", [
    ("dataset_timestamps", "DATASET_TIMESTAMPS_INVALID: 20 Hz 与真实 60 Hz 时间戳不符"),
    ("dataset_samples", "COMPOSITION_DATA_INVALID: 解码 RGB 图像与配置的 HWC 布局不符"),
])
def test_real_dataset_failure_stays_blocking_alongside_metadata_fps_warning(platform, required_check, message):
    node, identifier, command, code = pending(platform)
    checks = [{**item, "ok": False, "message": message}
              if item["name"] == required_check else item for item in automatic_checks()]
    checked = acknowledge(platform, node, identifier, command, code, checks=[*checks, fps_warning()])
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "PREFLIGHT_FAILED"
    assert any(check["name"] == required_check and check["ok"] is False
               and check["message"] == message for check in checked["checks"])
    assert any(check["name"] == "dataset_fps" and check["severity"] == "warning" for check in checked["checks"])
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"


@pytest.mark.parametrize("required_check", ["dataset_timestamps", "dataset_samples"])
def test_submission_rechecks_real_dataset_evidence_in_a_previously_ready_preview(platform, required_check):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        preview.checks = [check for check in preview.checks if check["name"] != required_check]
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_PREFLIGHT_INCOMPLETE"


def test_preflight_without_output_isolation_check_cannot_become_ready(platform):
    node, identifier, command, code = pending(platform)
    checks = [check for check in automatic_checks() if check["name"] != "output_isolation"]
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "AGENT_PREFLIGHT_INCOMPLETE"
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_submission_rechecks_output_isolation_in_a_previously_ready_preview(platform):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        preview.checks = [check for check in preview.checks if check["name"] != "output_isolation"]
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_PREFLIGHT_INCOMPLETE"


@pytest.mark.parametrize("field,value", [
    ("missing", None), ("schema_version", True), ("schema_version", 2), ("status", "not_applicable"),
    ("path", "/different/dataset"), ("checked_samples", 0), ("checked_samples", True), ("checked_samples", 4),
    ("sample_indices", []), ("sample_indices", [-1, 16, 31]), ("sample_indices", [0, 16, 32]),
    ("sample_indices", [False, 16, 31]), ("sample_indices", [0, 0, 31]),
    ("samples", []), ("samples", [{"index": 0}, {"index": 15}, {"index": 31}]),
    ("action_horizon", 50), ("action_horizon", True), ("action_sequence_keys", ["actions"]),
])
def test_composed_samples_require_positive_bound_evidence_despite_passing_check(platform, field, value):
    from .test_composed_asset_processing import pending as composed_pending
    node, identifier, command, code, _ = composed_pending(platform)
    evidence = code["automatic_preflight"]
    data = evidence["plan"]["datasets"][0]
    if field == "missing":
        data.pop("sample_probe")
    else:
        data["sample_probe"][field] = value
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_submission_rechecks_composed_samples_in_a_previously_ready_preview(platform):
    from .test_composed_asset_processing import pending as composed_pending
    node, identifier, command, code, _ = composed_pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = copy.deepcopy(preview.code_snapshot)
        plan = changed["automatic_preflight"]["plan"]
        plan["datasets"][0].pop("sample_probe")
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_PREFLIGHT_INVALID"


def test_fps_warning_ack_is_ready_and_still_requires_exact_config_confirmation(platform):
    node, identifier, command, code = pending(platform)
    warning = fps_warning()
    evidence = code["automatic_preflight"]
    dataset = evidence["plan"]["datasets"][0]
    dataset["warnings"] = [{"code": warning["code"], "supplied": 30, "metadata": 60,
                             "dataset_path": dataset["path"], "message": warning["message"]}]
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checks = [*automatic_checks(), warning]
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "READY"
    assert checked["checks"] == checks
    assert checked["code_snapshot"]["automatic_preflight"] == evidence
    for sha, expected in ((None, "CONFIG_CONFIRMATION_REQUIRED"), ("0" * 64, "CONFIG_CONFIRMATION_MISMATCH")):
        body = {"preflight_id": identifier}
        if sha is not None:
            body["confirmed_config_sha256"] = sha
        response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier}, json=body)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == expected
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["code_expectation"]["automatic_preflight"] == evidence


@pytest.mark.parametrize("failure", [
    {"name": "norm_contract", "ok": False, "message": "norm dimension mismatch"},
    {"name": "data_contract", "ok": False, "severity": "warning", "message": "invalid dataset shape"},
])
def test_fps_warning_ack_never_overrides_false_check(platform, failure):
    node, identifier, command, code = pending(platform)
    checks = [*automatic_checks(), fps_warning(), failure]
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "PREFLIGHT_FAILED"
    assert checked["checks"] == checks
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED"
                   for asset_id in checked["request"]["inputs"].values())


def test_old_agent_without_automatic_evidence_is_rejected_even_for_allowlisted_recipe(platform):
    node, identifier, command, code = pending(platform)
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).metadata_json = {"validated_recipes": ["pi05_unvalidated"]}
    code.pop("automatic_preflight")
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_UNSUPPORTED"
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_old_agent_without_training_cache_check_requires_new_preflight(platform):
    node, identifier, command, code = pending(platform)
    checks = [check for check in automatic_checks() if check["name"] != "training_cache"]
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INCOMPLETE"
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED"
                   for asset_id in checked["request"]["inputs"].values())


@pytest.mark.parametrize("mutation", ["missing", "empty_paths", "unknown_path_key", "outside_roots",
                                    "relative_path", "wrong_cache_root", "missing_builder", "different_dataset",
                                    "builder_outside_cache"])
def test_agent_plan_requires_complete_managed_cache_evidence(platform, mutation):
    node, identifier, command, code = pending(platform)
    plan = code["automatic_preflight"]["plan"]
    cache = plan["training_cache"]
    if mutation == "missing":
        plan.pop("training_cache")
    elif mutation == "empty_paths":
        cache["paths"] = {}
    elif mutation == "unknown_path_key":
        cache["paths"]["UNRECOGNIZED_CACHE"] = cache["datasets_cache"]
    elif mutation == "outside_roots":
        cache["paths"]["TMPDIR"] = "/outside/root"
    elif mutation == "relative_path":
        cache["paths"]["TMPDIR"] = "relative/cache"
    elif mutation == "wrong_cache_root":
        cache["paths"]["HF_DATASETS_CACHE"] = "/machine/another-cache"
    elif mutation == "missing_builder":
        cache["builders"] = []
    elif mutation == "different_dataset":
        cache["builders"][0]["path"] = "/machine/another-dataset"
    else:
        cache["builders"][0]["cache_dir"] = "/machine/another-cache"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"


@pytest.mark.parametrize("field,value", [
    ("recipe", "another_recipe"), ("recipe_definition_digest", "wrong-revision"),
    ("requested_inputs", {"dataset": "/machine/other"}), ("parameters", {"steps": 99}),
    ("gpu_count", 2), ("gpu_count", True),
])
def test_agent_plan_must_match_exact_task_even_with_valid_digest(platform, field, value):
    node, identifier, command, code = pending(platform)
    evidence = code["automatic_preflight"]
    evidence["plan"][field] = value
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_MISMATCH"


def test_agent_plan_digest_drift_is_rejected(platform):
    node, identifier, command, code = pending(platform)
    code["automatic_preflight"]["plan"]["entrypoint"] = "/machine/code/other.py"
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"


@pytest.mark.parametrize("field,value", [("entrypoint", "relative.py"), ("datasets", []), ("model", {}),
                                        ("inputs", {"dataset": "relative"}), ("storage", {})])
def test_agent_plan_needs_complete_local_asset_evidence(platform, field, value):
    node, identifier, command, code = pending(platform)
    evidence = code["automatic_preflight"]
    evidence["plan"][field] = value
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"


@pytest.mark.parametrize("checks,expected", [
    ([{"name": "runtime", "ok": True}], "AGENT_PREFLIGHT_INCOMPLETE"),
    ([], "PREFLIGHT_FAILED"),
    ([*automatic_checks(), {"name": "output_storage", "ok": False}], "PREFLIGHT_FAILED"),
])
def test_partial_or_failed_checks_never_mark_preview_or_assets_ready(platform, checks, expected):
    node, identifier, command, code = pending(platform)
    checked = acknowledge(platform, node, identifier, command, code, checks=checks)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED"
                   for asset_id in checked["request"]["inputs"].values())


@pytest.mark.parametrize("mutation,expected", [("legacy", "AGENT_PREFLIGHT_UNSUPPORTED"),
                                               ("plan", "AGENT_PREFLIGHT_INVALID"),
                                               ("checks", "AGENT_PREFLIGHT_INCOMPLETE"),
                                               ("asset_path", "AGENT_PREFLIGHT_MISMATCH")])
def test_submission_revalidates_stored_evidence_and_bound_asset_paths(platform, mutation, expected):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = copy.deepcopy(preview.code_snapshot)
        if mutation == "legacy":
            changed.pop("automatic_preflight")
        elif mutation == "plan":
            changed["automatic_preflight"]["plan"]["parameters"]["steps"] = 777
        elif mutation == "checks":
            preview.checks = [{"name": "runtime", "ok": True}]
        else:
            session.get(Asset, preview.request["inputs"]["dataset"]).path = "/machine/other_dataset"
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == expected


def test_started_evidence_drift_is_cancelled_without_releasing_gpu(platform):
    node, identifier, command, code = pending(platform)
    acknowledge(platform, node, identifier, command, code)
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.get_run(run)["attempts"][0]
    changed = copy.deepcopy(run["code_expectation"])
    changed["automatic_preflight"]["plan"]["entrypoint"] = "/machine/code/other.py"
    changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": changed,
        "output_dir": "/machine/output/" + attempt["id"]})
    assert response.status_code == 200, response.text
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == attempt["id"]
