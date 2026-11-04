"""Control-plane approval binds the new registered-source evidence at both gates."""

from copy import deepcopy

import pytest

from vla_adapters.submission_validation import MODE, submission_evidence
from vla_control.models import Preview
from vla_protocol import canonical_digest

from .test_automatic_preflight import acknowledge, submit
from .test_composed_asset_processing import pending


def register_source(evidence):
    plan = evidence["plan"]
    plan["data_validation_mode"] = MODE
    for data in plan["datasets"]:
        data.pop("sample_probe", None)
        data.setdefault("effective_fps", 20)
        data["data_manifest_digest"] = "b" * 64
        data["timestamp_contract"] = {"schema_version": 1, "status": "verified", "effective_fps": data["effective_fps"],
                                      "checked_rows": data["frames"], "checked_pairs": data["frames"] - data["episodes"],
                                      "checked_episodes": data["episodes"], "metadata_sha256": data["metadata_sha256"],
                                      "data_manifest_digest": data["data_manifest_digest"]}
        spec = data["composition"]["spec"]
        fields = {row["key"]: {"shape": [row["width"]], "dtype": "float32"} for row in spec["state_fields"]}
        fields[spec["action_key"]] = {"shape": data["actions_shape"], "dtype": "float32"}
        fields.update({row["key"]: {"shape": [3, 224, 224], "dtype": "float32"} for row in spec["cameras"]})
        fields.update({key: {"shape": [], "dtype": "bool"} for key in spec["aux_keys"]})
        indices = sorted({0, data["frames"] // 2, data["frames"] - 1})
        source = {"schema_version": 1, "status": "verified", "path": data["path"],
                  "raw_samples": {"selection": "first_middle_last", "checked_samples": len(indices), "sample_indices": indices,
                                  "samples": [{"index": index, "fields": deepcopy(fields)} for index in indices]}}
        data["submission_validation"] = submission_evidence(data, source)
    evidence["plan_digest"] = canonical_digest(plan)
    return evidence


def test_registered_source_plan_can_be_approved_and_submitted_with_frozen_composition(platform):
    node, identifier, command, code, _ = pending(platform)
    register_source(code["automatic_preflight"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    assert response.json()["code_expectation"]["automatic_preflight"] == code["automatic_preflight"]


@pytest.mark.parametrize("change", ["missing-mode", "missing-evidence", "mixed-probe", "changed-source", "changed-metadata", "changed-timestamps", "schema-version-bool"])
def test_control_rejects_registered_source_forgery_despite_passing_checks_and_plan_digest(platform, change):
    node, identifier, command, code, _ = pending(platform)
    evidence = register_source(code["automatic_preflight"])
    data = evidence["plan"]["datasets"][0]
    if change == "missing-mode":
        evidence["plan"].pop("data_validation_mode")
    elif change == "missing-evidence":
        data.pop("submission_validation")
    elif change == "mixed-probe":
        data["sample_probe"] = {"schema_version": 1, "status": "verified"}
    elif change == "changed-source":
        raw = data["submission_validation"]["source_evidence"]["raw_samples"]
        raw["samples"][0]["fields"][data["composition"]["spec"]["action_key"]]["shape"] = [99]
    elif change == "changed-metadata":
        data["submission_validation"]["metadata_sha256"] = "0" * 64
    elif change == "schema-version-bool":
        data["submission_validation"]["schema_version"] = True
    else:
        data["timestamp_contract"]["checked_rows"] = 99
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED", checked
    assert checked["error"]["code"] == "AGENT_PREFLIGHT_INVALID"
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_submission_rechecks_registered_source_envelope_in_previously_ready_preview(platform):
    node, identifier, command, code, _ = pending(platform)
    register_source(code["automatic_preflight"])
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        snapshot = deepcopy(preview.code_snapshot)
        evidence = snapshot["automatic_preflight"]
        evidence["plan"]["datasets"][0]["submission_validation"]["source_evidence_sha256"] = "0" * 64
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
        preview.code_snapshot = snapshot
    response = submit(platform, identifier)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "AGENT_PREFLIGHT_INVALID"
