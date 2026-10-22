"""An Agent must prove that every effective resource is a selected asset."""

import copy

import pytest
from vla_protocol import canonical_digest

from vla_control.models import Asset, Preview

from .test_automatic_preflight import acknowledge, pending as single_pending, submit
from .test_multi_dataset_control import pending as multi_pending


@pytest.mark.parametrize("prepare", [single_pending, multi_pending], ids=["single", "multi"])
@pytest.mark.parametrize("marker", [None, "recipe_defaults", ""])
def test_preflight_requires_registered_asset_authority_even_with_a_valid_digest(platform, prepare, marker):
    node, identifier, command, code = prepare(platform)
    plan = code["automatic_preflight"]["plan"]
    if marker is None:
        plan.pop("input_authority")
    else:
        plan["input_authority"] = marker
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "AGENT_INPUT_AUTHORITY_UNSUPPORTED"
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED"
                   for asset_id in checked["request"]["inputs"].values())


@pytest.mark.parametrize("prepare", [single_pending, multi_pending], ids=["single", "multi"])
@pytest.mark.parametrize("role,missing", [("dataset", False), ("norm", False), ("model", False),
                                          ("dataset", True), ("norm", True), ("model", True)])
def test_effective_plan_proves_each_asset_id_even_when_all_paths_match(platform, prepare, role, missing):
    node, identifier, command, code = prepare(platform)
    plan = code["automatic_preflight"]["plan"]
    target = plan["model"] if role == "model" else plan["datasets"][-1]
    key = "asset_id" if role == "model" else role + "_asset_id"
    if missing:
        target.pop(key)
    else:
        target[key] = "a-different-registered-identity"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "INPUT_BINDING_MISMATCH"
    assert submit(platform, identifier).json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_single_binding_cannot_hide_an_extra_recipe_dataset(platform):
    node, identifier, command, code = single_pending(platform)
    plan = code["automatic_preflight"]["plan"]
    plan["datasets"].append(copy.deepcopy(plan["datasets"][0]))
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "INPUT_BINDING_MISMATCH"


@pytest.mark.parametrize("change,expected", [("authority", "AGENT_INPUT_AUTHORITY_UNSUPPORTED"),
                                           ("asset_id", "INPUT_BINDING_MISMATCH")])
def test_submission_rechecks_effective_authority_after_persisted_proof_changes(platform, change, expected):
    node, identifier, command, code = multi_pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = copy.deepcopy(preview.code_snapshot)
        plan = changed["automatic_preflight"]["plan"]
        if change == "authority":
            plan.pop("input_authority")
        else:
            plan["datasets"][-1]["norm_asset_id"] = plan["datasets"][0]["norm_asset_id"]
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == expected
    assert platform.schedule() == 0


def test_authority_change_without_matching_plan_digest_is_rejected(platform):
    node, identifier, command, code = multi_pending(platform)
    code["automatic_preflight"]["plan"]["model"]["asset_id"] = "wrong-id"
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"
