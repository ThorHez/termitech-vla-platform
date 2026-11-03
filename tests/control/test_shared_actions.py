"""Shared action vectors survive the public request and immutable run contract."""

import copy

import pytest
from vla_protocol import canonical_digest

from .conftest import ADMIN
from .test_automatic_preflight import acknowledge, submit
from .test_composed_asset_processing import description, pending
from .test_multi_dataset_control import configured
from .test_norm_tasks import batch_outputs, start_batch_norm
from .test_task_recipe import request_preview


def action_options(arm_mode="single", representation="eef", mixed=False):
    layout = [{"kind": "shared", "arm": "shared", "width": 3, "normalize": False}]
    if mixed:
        arms = ["left"] if arm_mode == "single" else ["left", "right"]
        kinds = [("position", 3), ("rotation", 6), ("gripper", 1)] if representation == "eef" else [("joint", 6), ("gripper", 1)]
        layout.extend({"kind": kind, "arm": arm, "width": width, "normalize": True}
                      for arm in arms for kind, width in kinds)
    return {"arm_mode": arm_mode, "action_representation": representation, "action_layout": layout}


@pytest.mark.parametrize("arm_mode", ["single", "bimanual"])
@pytest.mark.parametrize("representation", ["eef", "joint"])
@pytest.mark.parametrize("mixed", [False, True])
def test_shared_actions_roundtrip_source_labels_preflight_confirmation_and_start(platform, arm_mode, representation, mixed):
    options = action_options(arm_mode, representation, mixed)
    node, identifier, command, code, profile = pending(platform, source_only=True, **options)
    expected = options["action_layout"]
    width = sum(block["width"] for block in expected)
    assert profile["field_annotations"][-1] == {
        "key": "actions", "role": "action", "representation": "composite", "value_mode": "absolute",
        "component": "other", "arm": "shared", "shape": [16, width]}
    choice = command["payload"]["dataset_bindings"][0]["processor"]
    assert choice["parameters"]["composition"]["action_layout"] == expected
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    unconfirmed = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier + "-unconfirmed"},
                                       json={"preflight_id": identifier})
    assert unconfirmed.status_code == 409
    assert unconfirmed.json()["error"]["code"] == "CONFIG_CONFIRMATION_REQUIRED"
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["dataset_bindings"][0]["processor"] == choice
    plan = run["code_expectation"]["automatic_preflight"]["plan"]
    evidence = plan["datasets"][0]["composition"]
    assert evidence["spec"]["action_layout"] == expected
    assert evidence["raw_action_dim"] == width
    assert evidence["action_loss_mask"] == [1.] * width + [0.] * (32 - width)
    assert platform.schedule() == 1
    start = next(command for command in platform.commands(node) if command["type"] == "START")
    assert start["payload"]["dataset_bindings"][0]["processor"] == choice
    assert start["payload"]["inputs"]["dataset_0"]["dataset_profile"] == profile
    assert start["payload"]["confirmed_config_sha256"] == plan["generated_config"]["sha256"]
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"


@pytest.mark.parametrize("mixed", [False, True])
def test_shared_actions_remain_separate_per_dataset_in_automatic_norm_runs(platform, mixed):
    options = action_options("bimanual", "eef" if mixed else "joint", mixed)
    node, request, run, attempt, start, output_dir = start_batch_norm(platform,
        auto_config={"action_horizon": 16, "action_dim": 32}, composed=True, composition_options=options)
    for row in start["payload"]["norm_datasets"]:
        assert row["processor"]["parameters"]["composition"]["action_layout"] == options["action_layout"]
    for dataset in run["code_expectation"]["automatic_preflight"]["plan"]["datasets"]:
        assert dataset["composition"]["spec"]["action_layout"] == options["action_layout"]
        assert dataset["composition"]["raw_action_dim"] == sum(row["width"] for row in options["action_layout"])
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0,
        "outputs": batch_outputs(request, output_dir)})
    assert response.status_code == 200, response.text
    completed = platform.get_run(run)
    assert completed["state"] == "SUCCEEDED", completed
    assert len(completed["outputs"]) == 2


@pytest.mark.parametrize("change", ["shared_kind_left_arm", "regular_kind_shared_arm", "duplicate_shared",
    "missing_right_eef", "missing_right_joint", "position_width", "rotation_width", "joint_rotation"])
def test_shared_blocks_cannot_bypass_request_validation_of_regular_per_arm_actions(platform, change):
    options = action_options("bimanual", "joint" if change in {"missing_right_joint", "joint_rotation"} else "eef", True)
    spec = description(**options)["processing"]
    layout = spec["action_layout"]
    if change == "shared_kind_left_arm":
        layout[0]["arm"] = "left"
    elif change == "regular_kind_shared_arm":
        layout[0]["kind"] = "joint"
    elif change == "duplicate_shared":
        layout.append(copy.deepcopy(layout[0]))
    elif change == "missing_right_eef":
        spec["action_layout"] = [row for row in layout if row["arm"] != "right"]
    elif change == "missing_right_joint":
        spec["action_layout"] = [row for row in layout if (row["arm"], row["kind"]) != ("right", "joint")]
    elif change == "position_width":
        next(row for row in layout if row["kind"] == "position")["width"] = 2
    elif change == "rotation_width":
        next(row for row in layout if row["kind"] == "rotation")["width"] = 3
    else:
        spec["rotation_representation"] = "rotation_6d"
    node, inputs, rows = configured(platform)
    rows = rows[:1]
    rows[0].pop("recipe_index")
    rows[0]["processor"] = {"name": "ComposableDataConfig", "parameters": {"composition": spec}}
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


def test_agent_cannot_change_shared_vector_width_after_request_freezing(platform):
    node, identifier, command, code, _ = pending(platform, **action_options("bimanual", "joint"))
    evidence = code["automatic_preflight"]
    composed = evidence["plan"]["datasets"][0]["composition"]
    composed["spec"]["action_layout"][0]["width"] = 4
    composed["digest"] = canonical_digest(composed["spec"])
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] == "AGENT_COMPOSITION_UNSUPPORTED"
