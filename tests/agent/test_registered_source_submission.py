"""Agent accepts registered source proof and rejects mixed or altered envelopes."""

from copy import deepcopy

import pytest

from vla_adapters.submission_validation import MODE, submission_evidence
from vla_agent import training_validation
from vla_agent.safety import AgentError

from tests.agent.test_automatic_preflight import probe_fixture, write_probe_response


CHECKS = ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
          "model_checkpoint", "training_entrypoint", "training_cache")


def register_source(plan):
    plan["data_validation_mode"] = MODE
    for data in plan["datasets"]:
        data.update(frames=32, episodes=2, effective_fps=20, metadata_sha256="a" * 64, data_manifest_digest="b" * 64)
        data["timestamp_contract"] = {"schema_version": 1, "status": "verified", "effective_fps": 20,
                                      "checked_rows": 32, "checked_pairs": 30, "checked_episodes": 2,
                                      "metadata_sha256": data["metadata_sha256"], "data_manifest_digest": data["data_manifest_digest"]}
        indices = [0, 16, 31]
        source = {"schema_version": 1, "status": "verified", "path": data["path"],
                  "raw_samples": {"selection": "first_middle_last", "checked_samples": 3, "sample_indices": indices,
                                  "samples": [{"index": index, "fields": {"actions": {"shape": [14], "dtype": "float32"}}}
                                              for index in indices]}}
        data["submission_validation"] = submission_evidence(data, source)
    return plan


def test_agent_accepts_registered_source_evidence_without_decoded_transform_claim(probe_fixture):
    fixture = probe_fixture
    plan = register_source(fixture["plan"])
    write_probe_response(fixture, [{"name": name, "ok": True, "message": "verified"} for name in CHECKS])
    evidence = training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert evidence["plan"] == plan
    assert "sample_probe" not in plan["datasets"][0]
    assert plan["datasets"][0]["submission_validation"]["scope"] == "registered_source_and_configuration"


@pytest.mark.parametrize("change", ["missing-mode", "missing-evidence", "mixed-probe", "changed-source",
                                    "changed-metadata", "changed-timestamps", "changed-processor", "schema-version-bool"])
def test_agent_rejects_incomplete_mixed_or_tampered_registered_source_plan(probe_fixture, change):
    fixture = probe_fixture
    plan = register_source(deepcopy(fixture["plan"]))
    data = plan["datasets"][0]
    if change == "missing-mode":
        plan.pop("data_validation_mode")
    elif change == "missing-evidence":
        data.pop("submission_validation")
    elif change == "mixed-probe":
        data["sample_probe"] = {"schema_version": 1, "status": "verified"}
    elif change == "changed-source":
        data["submission_validation"]["source_evidence"]["raw_samples"]["samples"][0]["fields"]["actions"]["shape"] = [15]
    elif change == "changed-metadata":
        data["metadata_sha256"] = "c" * 64
    elif change == "changed-timestamps":
        data["timestamp_contract"]["checked_rows"] = 99
    elif change == "schema-version-bool":
        data["submission_validation"]["schema_version"] = True
    else:
        data["processor_identity"] = {"name": "different-processor"}
    write_probe_response(fixture, [{"name": name, "ok": True, "message": "verified"} for name in CHECKS], plan)
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"
