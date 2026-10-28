"""A ready preflight authorizes submission only after its exact code is confirmed."""

import copy
import hashlib

import pytest
from sqlalchemy import select
from vla_protocol import canonical_digest

from vla_control.models import Asset, Preview, Run

from .conftest import ADMIN, GENERATED_CONFIG_SHA256, confirmed_run_request
from .test_automatic_preflight import acknowledge, pending


def submit(platform, preview, *, sha="expected", key=None):
    body = {"preflight_id": preview["id"]}
    if sha == "expected":
        body = confirmed_run_request(preview)
    elif sha is not None:
        body["confirmed_config_sha256"] = sha
    return platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": key or preview["id"]}, json=body)


def ready(platform):
    node, identifier, command, code = pending(platform)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    return node, checked, code


def test_ready_preflight_waits_for_explicit_code_confirmation(platform):
    _, checked, _ = ready(platform)
    assert checked["code_snapshot"]["automatic_preflight"]["plan"]["generated_config"]["sha256"] == GENERATED_CONFIG_SHA256
    for sha, expected in ((None, "CONFIG_CONFIRMATION_REQUIRED"), ("0" * 64, "CONFIG_CONFIRMATION_MISMATCH")):
        response = submit(platform, checked, sha=sha)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == expected
        with platform.app.state.db.read() as session:
            assert session.get(Preview, checked["id"]).run_id is None
            assert list(session.scalars(select(Run))) == []
    assert submit(platform, checked).status_code == 201


def test_exact_confirmation_is_frozen_into_queued_run_and_start(platform):
    node, checked, _ = ready(platform)
    response = submit(platform, checked)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["code_expectation"]["confirmed_config_sha256"] == GENERATED_CONFIG_SHA256
    assert platform.schedule() == 1
    start = next(command for command in platform.commands(node) if command["type"] == "START")
    assert start["payload"]["confirmed_config_sha256"] == GENERATED_CONFIG_SHA256
    frozen = start["payload"]["code_expectation"]["automatic_preflight"]["plan"]["generated_config"]
    assert frozen == checked["code_snapshot"]["automatic_preflight"]["plan"]["generated_config"]


def test_code_confirmation_is_bound_to_idempotent_submission(platform):
    _, checked, _ = ready(platform)
    first = submit(platform, checked, key="confirmed-config")
    repeated = submit(platform, checked, key="confirmed-config")
    assert first.status_code == repeated.status_code == 201
    assert first.json()["id"] == repeated.json()["id"]
    changed = submit(platform, checked, sha="0" * 64, key="confirmed-config")
    assert changed.status_code == 409 and changed.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_stale_confirmation_cannot_authorize_changed_code(platform):
    _, checked, _ = ready(platform)
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, checked["id"])
        changed = copy.deepcopy(preview.code_snapshot)
        generated = changed["automatic_preflight"]["plan"]["generated_config"]
        generated["source"] += "# Regenerated after a configuration change.\n"
        generated["sha256"] = hashlib.sha256(generated["source"].encode()).hexdigest()
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
        preview.code_snapshot = changed
    response = submit(platform, checked)
    assert response.status_code == 409 and response.json()["error"]["code"] == "CONFIG_CONFIRMATION_MISMATCH"


def test_old_agent_without_generated_code_cannot_mark_assets_ready(platform):
    node, identifier, command, code = pending(platform)
    evidence = code["automatic_preflight"]
    evidence["plan"].pop("generated_config")
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "AGENT_CONFIG_PREVIEW_UNSUPPORTED"
    with platform.app.state.db.read() as session:
        assert all(session.get(Asset, asset_id).state == "REGISTERED" for asset_id in checked["request"]["inputs"].values())


@pytest.mark.parametrize("mutation", ["sha", "language", "filename", "syntax", "empty"])
def test_invalid_generated_code_never_becomes_ready(platform, mutation):
    node, identifier, command, code = pending(platform)
    evidence = code["automatic_preflight"]
    generated = evidence["plan"]["generated_config"]
    if mutation == "sha":
        generated["sha256"] = "0" * 64
    elif mutation == "language":
        generated["language"] = "shell"
    elif mutation == "filename":
        generated["filename"] = "../config.py"
    else:
        generated["source"] = "CONFIG = (\n" if mutation == "syntax" else ""
        generated["sha256"] = hashlib.sha256(generated["source"].encode()).hexdigest()
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "AGENT_CONFIG_PREVIEW_INVALID"


def test_missing_code_preview_is_rechecked_at_submission(platform):
    _, checked, _ = ready(platform)
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, checked["id"])
        changed = copy.deepcopy(preview.code_snapshot)
        changed["automatic_preflight"]["plan"].pop("generated_config")
        changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
        preview.code_snapshot = changed
    response = submit(platform, checked)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_CONFIG_PREVIEW_UNSUPPORTED"


def test_smoke_workflow_does_not_require_python_config_confirmation(platform):
    node = platform.register()
    checked = platform.preview(node, count=0)
    assert submit(platform, checked, sha=None).status_code == 201
