"""Every training dataset is bound to its norm and protected throughout a run."""

import copy

import pytest
from sqlalchemy import select
from vla_protocol import canonical_digest

from vla_control.models import Asset, Preview, RunInput

from .conftest import ADMIN, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_task_recipe import request_preview


def asset(platform, node, kind, name):
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "name": name, "kind": kind, "node_id": node["node_id"], "path": "/machine/" + name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def configured(platform, node=None):
    node = node or platform.register(adapter="openpi-sft", recipe="mixed", metadata={})
    model = asset(platform, node, "MODEL", "checkpoint")
    rows = [{"dataset_asset_id": asset(platform, node, "DATASET", f"dataset-{i}"),
             "norm_asset_id": asset(platform, node, "NORM", f"norm-{i}"),
             "recipe_index": i, "weight": weight} for i, weight in enumerate((1.0, 20.0, 50.0))]
    return node, {"model": model}, rows


def pending(platform, **changes):
    node, inputs, rows = configured(platform)
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows, **changes)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    code = snapshot(recipe="mixed", dataset_bindings=rows, automatic_preflight=automatic_preflight(command["payload"]))
    return node, identifier, command, code


def test_all_pairs_frozen_serialized_scheduled_and_protected_by_run_inputs(platform):
    node, identifier, command, code = pending(platform)
    rows = code["dataset_bindings"]
    expected = {"model", "dataset_0", "norm_0", "dataset_1", "norm_1", "dataset_2", "norm_2"}
    assert command["payload"]["dataset_bindings"] == rows
    assert set(command["payload"]["inputs"]) == expected
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY" and checked["request"]["dataset_bindings"] == rows
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "READY" for asset_id in checked["request"]["inputs"].values())
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["dataset_bindings"] == rows and set(run["inputs"]) == expected
    assert len(run["input_assets"]) == 7
    with platform.app.state.db.read() as session:
        references = list(session.scalars(select(RunInput).where(RunInput.run_id == run["id"])))
        assert {reference.role for reference in references} == expected
    for key in ("dataset_asset_id", "norm_asset_id"):
        archived = platform.client.post(f"/api/v1/assets/{rows[-1][key]}/archive", headers=ADMIN)
        assert archived.status_code == 409 and archived.json()["error"]["code"] == "ASSET_IN_USE"
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["dataset_bindings"] == rows and set(start["payload"]["inputs"]) == expected
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"


@pytest.mark.parametrize("value", [[], [{}], [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": 0, "weight": 0}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": -1, "weight": 1}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": True, "weight": 1}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": 64, "weight": 1}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": 0, "weight": True}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": 0, "weight": "2"}],
    [{"dataset_asset_id": "a", "norm_asset_id": "b", "recipe_index": 0, "weight": 1}] * 65])
def test_invalid_bindings_never_dispatch(platform, value):
    node = platform.register(adapter="openpi-sft")
    response = request_preview(platform, node, inputs={}, dataset_bindings=value)
    assert response.status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("mutation,expected", [("legacy", "DATASET_BINDINGS_CONFLICT"),
    ("wrong_role_id", "DATASET_BINDINGS_MISMATCH"), ("partial_roles", "DATASET_BINDINGS_MISMATCH"),
    ("unknown_role", "DATASET_BINDINGS_MISMATCH"), ("wrong_kind", "ASSET_KIND_MISMATCH"),
    ("other_node", "ASSET_NODE_MISMATCH"), ("archived", "INPUT_UNAVAILABLE")])
def test_every_row_is_validated_against_node_asset_kind_and_flattened_inputs(platform, mutation, expected):
    node, inputs, rows = configured(platform)
    if mutation == "legacy":
        inputs["dataset"] = rows[0]["dataset_asset_id"]
    elif mutation == "wrong_role_id":
        inputs["dataset_0"] = rows[-1]["dataset_asset_id"]
    elif mutation == "partial_roles":
        inputs["dataset_0"] = rows[0]["dataset_asset_id"]
    elif mutation == "unknown_role":
        inputs["dataset_99"] = rows[-1]["dataset_asset_id"]
    elif mutation == "wrong_kind":
        rows[-1]["norm_asset_id"] = rows[-1]["dataset_asset_id"]
    elif mutation == "other_node":
        other = platform.register(adapter="openpi-sft", name="other")
        rows[-1]["norm_asset_id"] = asset(platform, other, "NORM", "remote-norm")
    else:
        platform.client.post(f"/api/v1/assets/{rows[-1]['norm_asset_id']}/archive", headers=ADMIN)
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code in {409, 422} and response.json()["error"]["code"] == expected
    assert platform.commands(node) == []


def test_indexed_roles_require_explicit_bindings(platform):
    node, inputs, rows = configured(platform)
    inputs["dataset_0"] = rows[0]["dataset_asset_id"]
    response = request_preview(platform, node, inputs=inputs)
    assert response.status_code == 422 and response.json()["error"]["code"] == "UNKNOWN_INPUT_ROLE"


def test_bindings_are_not_supported_by_smoke_template(platform):
    node = platform.register()
    _, inputs, rows = configured(platform, node)
    response = request_preview(platform, node, template_id="smoke", inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422 and response.json()["error"]["code"] == "DATASET_BINDINGS_UNSUPPORTED"


@pytest.mark.parametrize("mutation,expected", [("missing_snapshot", "AGENT_DATASET_BINDINGS_UNSUPPORTED"),
    ("missing_plan", "AGENT_DATASET_BINDINGS_UNSUPPORTED"), ("snapshot_weight", "DATASET_BINDINGS_MISMATCH"),
    ("plan_index", "DATASET_BINDINGS_MISMATCH"), ("omitted_dataset", "DATASET_BINDINGS_MISMATCH"),
    ("swapped_norm", "DATASET_BINDINGS_MISMATCH"), ("other_weight", "DATASET_BINDINGS_MISMATCH")])
def test_agent_must_prove_every_ordered_binding_and_effective_weight(platform, mutation, expected):
    node, identifier, command, code = pending(platform)
    # Keep independent structures as a JSON transport would do.
    code = copy.deepcopy(code)
    plan = code["automatic_preflight"]["plan"]
    if mutation == "missing_snapshot":
        code.pop("dataset_bindings")
    elif mutation == "missing_plan":
        plan.pop("dataset_bindings")
    elif mutation == "snapshot_weight":
        code["dataset_bindings"] = copy.deepcopy(code["dataset_bindings"])
        code["dataset_bindings"][-1]["weight"] = 2
    elif mutation == "plan_index":
        plan["dataset_bindings"] = copy.deepcopy(plan["dataset_bindings"])
        plan["dataset_bindings"][-1]["recipe_index"] = 0
    elif mutation == "omitted_dataset":
        plan["datasets"].pop()
    elif mutation == "swapped_norm":
        plan["datasets"][-1]["norm_path"] = plan["datasets"][0]["norm_path"]
    else:
        plan["weights"][-1] = 2
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED" for asset_id in checked["request"]["inputs"].values())


def test_norm_directory_asset_may_resolve_to_its_norm_stats_file(platform):
    node, identifier, command, code = pending(platform)
    plan = code["automatic_preflight"]["plan"]
    plan["datasets"][-1]["norm_path"] += "/norm_stats.json"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"


def test_norm_directory_symlink_target_is_accepted_with_matching_asset_identity(platform):
    node, identifier, command, code = pending(platform)
    plan = code["automatic_preflight"]["plan"]
    plan["datasets"][-1].update(norm_asset_path=plan["inputs"]["norm_2"],
                                norm_path="/machine/shared-normalization/norm_stats.json")
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"


@pytest.mark.parametrize("norm_asset_path", ["/machine/other-norm", None])
def test_wrong_norm_asset_identity_is_rejected_even_when_norm_file_path_matches(platform, norm_asset_path):
    node, identifier, command, code = pending(platform)
    plan = code["automatic_preflight"]["plan"]
    plan["datasets"][-1]["norm_asset_path"] = norm_asset_path
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "DATASET_BINDINGS_MISMATCH"


@pytest.mark.parametrize("mutation", ["id", "missing_roles"])
def test_submission_revalidates_binding_ids_against_frozen_flattened_roles(platform, mutation):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        request = copy.deepcopy(preview.request)
        if mutation == "id":
            request["dataset_bindings"][-1]["norm_asset_id"] = request["dataset_bindings"][0]["norm_asset_id"]
        else:
            request["inputs"] = {"model": request["inputs"]["model"]}
        preview.request = request
    response = submit(platform, identifier)
    assert response.status_code == 422 and response.json()["error"]["code"] == "DATASET_BINDINGS_MISMATCH"


def test_scheduler_checks_last_row_assets_before_allocating_gpus(platform):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    run = submit(platform, identifier).json()
    with platform.app.state.db.write() as session:
        session.get(Asset, run["dataset_bindings"][-1]["norm_asset_id"]).state = "ARCHIVED"
    assert platform.schedule() == 0
    current = platform.get_run(run)
    assert current["state"] == "FAILED" and current["error"]["code"] == "INPUT_UNAVAILABLE"


def test_started_binding_drift_is_cancelled_while_gpu_remains_reserved(platform):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    attempt = platform.get_run(run)["attempts"][0]
    changed = copy.deepcopy(run["code_expectation"])
    changed["dataset_bindings"][-1]["weight"] = 0.5
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": changed,
        "output_dir": "/machine/output/" + attempt["id"]})
    assert response.status_code == 200, response.text
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == "DATASET_BINDINGS_MISMATCH"
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == attempt["id"]
