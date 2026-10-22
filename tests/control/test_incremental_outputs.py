"""Checkpoint discovery must expose outputs without finishing the training run."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import now, uid
from vla_control.main import create_app
from vla_control.models import Allocation, Attempt, Run

from .conftest import ADMIN


def running(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.start(node, run)
    return node, run, platform.get_run(run)["attempts"][0]


def checkpoint(attempt, step=100, **changes):
    return {"kind": "MODEL", "name": f"checkpoint {step}",
            "path": f"{attempt['output_dir']}/checkpoints/{step}", "version": str(step),
            "metadata": {"step": step, "evaluated": False}, **changes}


def reservations(platform):
    with platform.app.state.db.read() as session:
        return list(session.scalars(select(Allocation)))


def batch(platform, node, attempt, entries):
    return platform.client.post("/agent/v1/events/batch", headers=node["headers"], json={
        "node_id": node["node_id"], "events": [{"attempt_id": attempt["id"], "seq": seq,
            "type": kind, "timestamp": now(), "data": data} for seq, kind, data in entries]})


def test_running_checkpoint_is_visible_with_its_node_and_attempt_provenance(platform):
    node, run, attempt = running(platform)
    output = checkpoint(attempt)
    response = platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [output]})
    assert response.status_code == 200, response.text
    result = platform.get_run(run)
    assert result["state"] == result["attempts"][0]["state"] == "RUNNING"
    assert result["finished_at"] is None
    assert len(reservations(platform)) == 1
    assert len(result["outputs"]) == 1
    asset = result["outputs"][0]
    assert asset["path"] == output["path"] and asset["state"] == "READY"
    assert asset["origin_attempt_id"] == attempt["id"] and asset["node_id"] == node["node_id"]
    assert asset["metadata"] == output["metadata"]
    assets = platform.client.get("/api/v1/assets", headers=ADMIN, params={"node_id": node["node_id"]}).json()
    assert [item["id"] for item in assets] == [asset["id"]]


def test_repeated_reports_and_final_manifest_reuse_the_same_asset_across_restart(platform):
    node, run, attempt = running(platform)
    output = checkpoint(attempt)
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [output, output]}).status_code == 200
    asset_id = platform.get_run(run)["outputs"][0]["id"]
    restarted = create_app(platform.settings)
    with TestClient(restarted) as client:
        response = client.post("/agent/v1/events/batch", headers=node["headers"], json={
            "node_id": node["node_id"], "events": [
                {"attempt_id": attempt["id"], "seq": 3, "type": "ARTIFACT", "timestamp": now(),
                 "data": {"outputs": [output]}},
                {"attempt_id": attempt["id"], "seq": 4, "type": "SUCCEEDED", "timestamp": now(),
                 "data": {"exit_code": 0, "outputs": [{**output, "metadata": {"final": True}}, output]}}]})
        assert response.status_code == 200, response.text
        result = client.get(f"/api/v1/runs/{run['id']}", headers=ADMIN).json()
    assert result["state"] == "SUCCEEDED"
    assert [asset["id"] for asset in result["outputs"]] == [asset_id]
    assert result["outputs"][0]["metadata"] == {**output["metadata"], "final": True}
    assert reservations(platform) == []


@pytest.mark.parametrize("malformed", [None, {}, "bad", [None, {}, {"kind": []}],
    [{"kind": "MODEL", "name": "missing path"}], [{"kind": "MODEL", "name": [], "path": "/machine/output"}]])
def test_bad_discovery_payload_does_not_abort_following_metrics_or_release_gpu(platform, malformed):
    node, run, attempt = running(platform)
    response = batch(platform, node, attempt, [
        (2, "ARTIFACT", {"outputs": malformed}),
        (3, "METRIC", {"step": 12, "loss": 0.3}),
        (4, "ARTIFACT", {"outputs": [checkpoint(attempt)]})])
    assert response.status_code == 200, response.text
    result = platform.get_run(run)
    assert result["state"] == "RUNNING" and result["error"] is None
    assert result["attempts"][0]["latest_metrics"]["step"] == 12
    assert len(result["outputs"]) == len(reservations(platform)) == 1


def test_discovery_rejects_escaped_paths_invalid_fields_and_forged_ownership(platform):
    node, run, attempt = running(platform)
    base = checkpoint(attempt)
    invalid = [checkpoint(attempt, path=path) for path in (
        "/another-machine/checkpoint", "/machine/output/another-attempt/100",
        attempt["output_dir"] + "-sibling/100", attempt["output_dir"] + "/../100",
        "relative/checkpoint", attempt["output_dir"] + "/bad\x00path")]
    invalid += [checkpoint(attempt, **change) for change in (
        {"name": ""}, {"name": " "}, {"name": "x" * 201}, {"name": "bad\x00name"},
        {"version": []}, {"version": 100}, {"version": ""}, {"version": "x" * 201},
        {"kind": "UNKNOWN"}, {"kind": {}}, {"metadata": []}, {"node_id": "foreign-node"},
        {"origin_attempt_id": "foreign-attempt"}, {"source_attempt_id": "foreign-attempt"},
        {"source_run_id": "foreign-run"})]
    response = platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": invalid})
    assert response.status_code == 200, response.text
    assert platform.get_run(run)["outputs"] == []
    assert len(reservations(platform)) == 1
    # An equivalent canonical path is still one checkpoint on subsequent scans.
    response = platform.emit(node, attempt["id"], 3, "ARTIFACT", {"outputs": [
        {**base, "path": base["path"] + "/"}, base]})
    assert response.status_code == 200, response.text
    assert len(platform.get_run(run)["outputs"]) == 1


def test_authenticated_other_node_cannot_report_a_checkpoint_for_this_attempt(platform):
    node, run, attempt = running(platform)
    other = platform.register("other")
    response = platform.emit(other, attempt["id"], 2, "ARTIFACT", {"outputs": [checkpoint(attempt)]})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "NODE_OWNERSHIP_MISMATCH"
    assert platform.get_run(run)["outputs"] == [] and len(reservations(platform)) == 1


def test_discovery_before_started_or_after_terminal_cannot_register_an_asset(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    output = checkpoint({**attempt, "output_dir": "/machine/output/" + attempt["id"]})
    assert platform.emit(node, attempt["id"], 1, "ARTIFACT", {"outputs": [output]}).status_code == 200
    assert platform.get_run(run)["outputs"] == []
    platform.start(node, run, seq=2)
    assert platform.emit(node, attempt["id"], 3, "SUCCEEDED", {"exit_code": 0, "outputs": []}).status_code == 200
    assert platform.emit(node, attempt["id"], 4, "ARTIFACT", {"outputs": [output]}).status_code == 200
    assert platform.get_run(run)["outputs"] == []
    assert platform.get_run(run)["state"] == "SUCCEEDED"


def test_checkpoint_while_cancelling_remains_available_after_confirmed_exit(platform):
    node, run, attempt = running(platform)
    platform.client.post(f"/api/v1/runs/{run['id']}/cancel", headers=ADMIN)
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [checkpoint(attempt)]}).status_code == 200
    result = platform.get_run(run)
    assert result["state"] == "CANCELLING" and len(result["outputs"]) == 1
    assert len(reservations(platform)) == 1
    assert platform.emit(node, attempt["id"], 3, "CANCELLED", {"exit_code": -15}).status_code == 200
    assert platform.get_run(run)["state"] == "CANCELLED"
    assert len(platform.get_run(run)["outputs"]) == 1 and reservations(platform) == []


def test_stale_sequence_or_superseded_attempt_does_not_register_checkpoint(platform):
    node, run, attempt = running(platform)
    assert platform.emit(node, attempt["id"], 3, "METRIC", {"step": 10}).status_code == 200
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [checkpoint(attempt)]}).status_code == 200
    assert platform.get_run(run)["outputs"] == []
    with platform.app.state.db.write() as session:
        session.add(Attempt(id=uid(), run_id=run["id"], node_id=node["node_id"], number=2,
            execution_token=uid(), state="RUNNING", created_at=now(), started_at=now(),
            code_snapshot=run["code_expectation"], output_dir="/machine/output/new-attempt"))
    assert platform.emit(node, attempt["id"], 4, "ARTIFACT", {"outputs": [checkpoint(attempt)]}).status_code == 200
    assert platform.get_run(run)["outputs"] == [] and len(reservations(platform)) == 1


def test_discovered_checkpoint_does_not_bypass_final_model_manifest_requirement(platform):
    node, run, attempt = running(platform)
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [checkpoint(attempt)]}).status_code == 200
    with platform.app.state.db.write() as session:
        session.get(Run, run["id"]).template_id = "openpi-sft"
    assert platform.emit(node, attempt["id"], 3, "SUCCEEDED", {"exit_code": 0, "outputs": []}).status_code == 200
    result = platform.get_run(run)
    assert result["state"] == "FAILED" and result["error"]["code"] == "OUTPUT_INCOMPLETE"
    assert len(result["outputs"]) == 1 and reservations(platform) == []


def test_success_manifest_cannot_register_another_attempts_checkpoint(platform):
    node, run, attempt = running(platform)
    response = platform.emit(node, attempt["id"], 2, "SUCCEEDED", {"exit_code": 0,
        "outputs": [checkpoint(attempt, path="/machine/output/another-attempt/100")]})
    assert response.status_code == 200, response.text
    result = platform.get_run(run)
    assert result["state"] == "FAILED" and result["error"]["code"] == "OUTPUT_INVALID"
    assert result["outputs"] == [] and reservations(platform) == []


def test_rotated_checkpoints_keep_provenance_and_reappearing_path_reuses_asset(platform):
    node, run, attempt = running(platform)
    previous, current = checkpoint(attempt, 100), checkpoint(attempt, 200)
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [previous]}).status_code == 200
    asset_id = platform.get_run(run)["outputs"][0]["id"]
    response = platform.emit(node, attempt["id"], 3, "ARTIFACT", {
        "outputs": [current], "removed_paths": [previous["path"]]})
    assert response.status_code == 200, response.text
    outputs = {output["version"]: output for output in platform.get_run(run)["outputs"]}
    assert outputs["100"]["id"] == asset_id and outputs["100"]["state"] == "UNAVAILABLE"
    assert outputs["100"]["metadata"]["availability"] == "removed"
    assert outputs["100"]["metadata"]["removed_at"]
    assert outputs["100"]["origin_attempt_id"] == attempt["id"]
    assert outputs["200"]["state"] == "READY" and len(reservations(platform)) == 1
    assert platform.emit(node, attempt["id"], 4, "ARTIFACT", {"outputs": [previous]}).status_code == 200
    outputs = {output["version"]: output for output in platform.get_run(run)["outputs"]}
    assert outputs["100"]["id"] == asset_id and outputs["100"]["state"] == "READY"
    assert outputs["100"]["metadata"]["availability"] == "available"


def test_removal_reports_are_path_scoped_and_do_not_mutate_terminal_outputs(platform):
    node, run, attempt = running(platform)
    output = checkpoint(attempt)
    assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [output]}).status_code == 200
    response = platform.emit(node, attempt["id"], 3, "ARTIFACT", {"removed_paths": [
        None, {}, [], "/machine/output/another-attempt/100", attempt["output_dir"] + "/../100"]})
    assert response.status_code == 200, response.text
    assert platform.get_run(run)["outputs"][0]["state"] == "READY"
    assert platform.emit(node, attempt["id"], 4, "SUCCEEDED", {"exit_code": 0, "outputs": [output]}).status_code == 200
    assert platform.emit(node, attempt["id"], 5, "ARTIFACT", {"removed_paths": [output["path"]]}).status_code == 200
    assert platform.get_run(run)["outputs"][0]["state"] == "READY"


def test_checkpoint_removed_after_discovery_in_same_event_is_unavailable(platform):
    node, run, attempt = running(platform)
    output = checkpoint(attempt)
    response = platform.emit(node, attempt["id"], 2, "ARTIFACT", {
        "outputs": [output], "removed_paths": [output["path"]]})
    assert response.status_code == 200, response.text
    result = platform.get_run(run)
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["state"] == "UNAVAILABLE"
    assert result["outputs"][0]["metadata"]["availability"] == "removed"
    assert result["state"] == "RUNNING" and len(reservations(platform)) == 1
