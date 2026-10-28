"""Dataset semantics are frozen per request and must be checked by the Agent."""

from copy import deepcopy

import pytest
from sqlalchemy import select
from vla_protocol import canonical_digest

from vla_control.common import snapshot_identity
from vla_control.models import Asset, Preview
from vla_control.scheduler import resolve_inputs

from .conftest import ADMIN, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_dataset_profiles import profile
from .test_multi_dataset_control import configured
from .test_task_recipe import inputs, request_preview


def describe(platform, asset_id, **changes):
    response = platform.client.patch(f"/api/v1/assets/{asset_id}/dataset-profile", headers=ADMIN,
                                     json={"dataset_profile": profile(**changes)})
    assert response.status_code == 200, response.text
    return response.json()["metadata"]["dataset_profile"]


def create_pending(platform, node, assets, **changes):
    response = request_preview(platform, node, inputs=assets, **changes)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    evidence = automatic_preflight(command["payload"])
    # Simulate the newer training-plan evidence after machine-local validation.
    for dataset in evidence["plan"]["datasets"]:
        matched = next(value for value in command["payload"]["inputs"].values()
                       if value["id"] == dataset["dataset_asset_id"])
        if "dataset_profile" in matched:
            dataset["registered_profile"] = deepcopy(matched["dataset_profile"])
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    code = snapshot(recipe=command["payload"]["recipe"], automatic_preflight=evidence)
    if command["payload"].get("dataset_bindings") is not None:
        code["dataset_bindings"] = deepcopy(command["payload"]["dataset_bindings"])
    return identifier, command, code


def single_pending(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    assets = inputs(platform, node)
    description = describe(platform, assets["dataset"])
    return node, assets, description, *create_pending(platform, node, assets)


def test_dataset_description_freezes_at_preflight_and_reaches_start_after_asset_edits(platform):
    node, assets, description, identifier, command, code = single_pending(platform)
    assert command["payload"]["inputs"]["dataset"]["dataset_profile"] == description
    assert "dataset_profile" not in command["payload"]["inputs"]["model"]
    assert "dataset_profile" not in command["payload"]["inputs"]["norm"]
    before = platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json()
    assert before["request"]["dataset_profiles"] == {"dataset": description}
    changed = describe(platform, assets["dataset"], task_description="New label for a future task")
    assert changed != description
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert checked["request"]["dataset_profiles"] == {"dataset": description}
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["dataset_profiles"] == {"dataset": description}
    assert run["code_expectation"]["dataset_profiles"] == {"dataset": description}
    assert snapshot_identity(run["code_expectation"]) == snapshot_identity(code)
    describe(platform, assets["dataset"], task_description="Another future task")
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["inputs"]["dataset"]["dataset_profile"] == description
    assert start["payload"]["code_expectation"]["dataset_profiles"] == {"dataset": description}
    assert platform.get_run(run)["dataset_profiles"] == {"dataset": description}
    with platform.app.state.db.read() as session:
        assert session.get(Asset, assets["dataset"]).metadata_json["dataset_profile"]["task_description"] == "Another future task"
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"


def test_multiple_datasets_preserve_each_registered_profile_and_legacy_row(platform):
    node, assets, bindings = configured(platform)
    first = describe(platform, bindings[0]["dataset_asset_id"], robot_type="first robot")
    third = describe(platform, bindings[2]["dataset_asset_id"], robot_type="third robot")
    identifier, command, code = create_pending(platform, node, assets, dataset_bindings=bindings)
    expected = {"dataset_0": first, "dataset_2": third}
    assert "dataset_profile" not in command["payload"]["inputs"]["dataset_1"]
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert checked["request"]["dataset_profiles"] == expected
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["dataset_profiles"] == expected
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["inputs"]["dataset_0"]["dataset_profile"] == first
    assert start["payload"]["inputs"]["dataset_2"]["dataset_profile"] == third
    assert "dataset_profile" not in start["payload"]["inputs"]["dataset_1"]


@pytest.mark.parametrize("mutation,expected", [
    ("missing", "AGENT_DATASET_PROFILE_UNSUPPORTED"), ("null", "AGENT_DATASET_PROFILE_UNSUPPORTED"),
    ("changed", "DATASET_PROFILE_MISMATCH"), ("wrong_type", "DATASET_PROFILE_MISMATCH"),
])
def test_agent_cannot_ignore_or_rewrite_the_registered_profile(platform, mutation, expected):
    node, assets, description, identifier, command, code = single_pending(platform)
    plan = code["automatic_preflight"]["plan"]
    dataset = plan["datasets"][0]
    if mutation == "missing":
        dataset.pop("registered_profile")
    elif mutation == "null":
        dataset["registered_profile"] = None
    elif mutation == "changed":
        dataset["registered_profile"]["arm_mode"] = "single"
    else:
        dataset["registered_profile"] = "unchecked"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"
    with platform.app.state.db.read() as session:
        assert session.get(Asset, assets["dataset"]).state == "REGISTERED"


def test_every_profile_in_a_multi_dataset_run_must_be_checked(platform):
    node, assets, bindings = configured(platform)
    for index in (0, 2):
        describe(platform, bindings[index]["dataset_asset_id"])
    identifier, command, code = create_pending(platform, node, assets, dataset_bindings=bindings)
    plan = code["automatic_preflight"]["plan"]
    plan["datasets"][2].pop("registered_profile")
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_DATASET_PROFILE_UNSUPPORTED"


def test_registered_fps_overrides_metadata_but_still_requires_config_confirmation(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    assets = inputs(platform, node)
    declared = describe(platform, assets["dataset"], fps=20)
    identifier, command, code = create_pending(platform, node, assets)
    evidence = code["automatic_preflight"]
    dataset = evidence["plan"]["datasets"][0]
    assert dataset["effective_fps"] == 20 and dataset["fps_source"] == "registered"
    dataset["fps"] = 60  # The unchanged info.json FPS is provenance, not authority.
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert checked["request"]["dataset_profiles"]["dataset"] == declared
    assert checked["code_snapshot"]["automatic_preflight"] == evidence
    for sha, expected in ((None, "CONFIG_CONFIRMATION_REQUIRED"), ("0" * 64, "CONFIG_CONFIRMATION_MISMATCH")):
        body = {"preflight_id": identifier}
        if sha is not None:
            body["confirmed_config_sha256"] = sha
        response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier}, json=body)
        assert response.status_code == 409 and response.json()["error"]["code"] == expected
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["code_expectation"]["automatic_preflight"] == evidence
    assert response.json()["dataset_profiles"]["dataset"]["fps"] == 20


@pytest.mark.parametrize("field,value,expected", [
    ("effective_fps", "missing", "AGENT_DATASET_FPS_UNSUPPORTED"),
    ("effective_fps", None, "DATASET_FPS_MISMATCH"),
    ("effective_fps", 60, "DATASET_FPS_MISMATCH"),
    ("effective_fps", "20", "DATASET_FPS_MISMATCH"),
    ("fps_source", "missing", "AGENT_DATASET_FPS_UNSUPPORTED"),
    ("fps_source", "metadata", "AGENT_DATASET_FPS_UNSUPPORTED"),
])
def test_agent_must_prove_registered_fps_is_the_effective_sampling_rate(platform, field, value, expected):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    assets = inputs(platform, node)
    describe(platform, assets["dataset"], fps=20)
    identifier, command, code = create_pending(platform, node, assets)
    evidence = code["automatic_preflight"]
    dataset = evidence["plan"]["datasets"][0]
    if value == "missing":
        dataset.pop(field)
    else:
        dataset[field] = value
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"
    with platform.app.state.db.read() as session:
        assert session.get(Asset, assets["dataset"]).state == "REGISTERED"


def test_each_selected_dataset_must_prove_its_own_registered_fps(platform):
    node, assets, bindings = configured(platform)
    describe(platform, bindings[0]["dataset_asset_id"], fps=20)
    describe(platform, bindings[2]["dataset_asset_id"], fps=30)
    identifier, command, code = create_pending(platform, node, assets, dataset_bindings=bindings)
    evidence = code["automatic_preflight"]
    first, legacy, third = evidence["plan"]["datasets"]
    assert first["effective_fps"] == 20 and third["effective_fps"] == 30
    assert "effective_fps" not in legacy
    third["effective_fps"] = 20
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "DATASET_FPS_MISMATCH"


def test_submission_rechecks_registered_fps_evidence(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    assets = inputs(platform, node)
    describe(platform, assets["dataset"], fps=20)
    identifier, command, code = create_pending(platform, node, assets)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = deepcopy(preview.code_snapshot)
        changed["automatic_preflight"]["plan"]["datasets"][0].pop("effective_fps")
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_DATASET_FPS_UNSUPPORTED"


def test_run_submission_rechecks_profile_evidence_after_preflight_acceptance(platform):
    node, assets, description, identifier, command, code = single_pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = deepcopy(preview.code_snapshot)
        changed["automatic_preflight"]["plan"]["datasets"][0].pop("registered_profile")
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_DATASET_PROFILE_UNSUPPORTED"


def test_agent_cannot_replace_server_owned_profile_snapshot(platform):
    node, assets, description, identifier, command, code = single_pending(platform)
    code["dataset_profiles"] = {"dataset": {"malicious": "not the registered data"}}
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["code_expectation"]["dataset_profiles"] == {"dataset": description}


def test_legacy_preflight_does_not_silently_use_later_added_asset_metadata(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    assets = inputs(platform, node)
    identifier, command, code = create_pending(platform, node, assets)
    assert all("dataset_profile" not in item for item in command["payload"]["inputs"].values())
    describe(platform, assets["dataset"])
    code["dataset_profiles"] = {"dataset": {"malicious": True}}
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert "dataset_profiles" not in checked["request"]
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["dataset_profiles"] is None
    assert "dataset_profiles" not in run["code_expectation"]
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert all("dataset_profile" not in item for item in start["payload"]["inputs"].values())


def test_new_preflight_uses_the_new_description_without_modifying_existing_preview(platform):
    node, assets, description, identifier, command, code = single_pending(platform)
    updated = describe(platform, assets["dataset"], task_description="Revised task")
    new_identifier, new_command, _ = create_pending(platform, node, assets)
    old = platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json()
    new = platform.client.get(f"/api/v1/run-previews/{new_identifier}", headers=ADMIN).json()
    assert old["request"]["dataset_profiles"] == {"dataset": description}
    assert new["request"]["dataset_profiles"] == {"dataset": updated}
    assert new_command["payload"]["inputs"]["dataset"]["dataset_profile"] == updated


def test_browser_cannot_supply_server_owned_dataset_profile_snapshot(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi")
    response = request_preview(platform, node, inputs=inputs(platform, node), dataset_profiles={"dataset": profile()})
    assert response.status_code == 422
    assert platform.commands(node) == []


def test_resolve_inputs_returns_independent_copy_of_frozen_profile(platform):
    node, assets, description, identifier, command, code = single_pending(platform)
    with platform.app.state.db.read() as session:
        preview = session.get(Preview, identifier)
        resolved = resolve_inputs(session, preview.request)
        resolved["dataset"]["dataset_profile"]["cameras"][0]["name"] = "changed by caller"
        assert preview.request["dataset_profiles"] == {"dataset": description}
