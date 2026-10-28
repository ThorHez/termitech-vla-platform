"""Freeze the requested serving framework and accept only matching execution evidence."""

from copy import deepcopy

import pytest
from sqlalchemy import inspect, select

from vla_control.models import Command, Node
from vla_control.policy_deployment_models import PolicyDeployment

from .test_policy_deployments import ack, action, command, create, get, running
from .test_policy_deployment_restart import enable_restart
from .test_recipe_details import register


def backend_node(platform):
    node = register(platform)
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        row.metrics = {**row.metrics, "capabilities": ["POLICY_DEPLOY_BACKEND_V1"]}
    return node


@pytest.mark.parametrize("backend", ["jax", "pytorch"])
def test_backend_choice_is_persisted_delivered_and_confirmed(platform, backend):
    node = backend_node(platform)
    response = create(platform, node, inference_backend=backend)
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment["inference_backend"] == backend
    assert deployment["resolved_backend"] is None
    start = command(platform, node, "POLICY_DEPLOY_START")
    assert start["payload"]["inference_backend"] == backend
    ack(platform, node, start, running(state="STARTING", resolved_backend=backend))
    assert get(platform, deployment)["resolved_backend"] == backend
    action(platform, deployment, "refresh")
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_STATUS"), running(resolved_backend=backend))
    assert get(platform, deployment)["state"] == "RUNNING"
    with platform.app.state.db.read() as session:
        row = session.get(PolicyDeployment, deployment["id"])
        assert row.inference_backend == row.resolved_backend == backend


@pytest.mark.parametrize("capabilities", [None, [], ["POLICY_DEPLOY_START"], "POLICY_DEPLOY_BACKEND_V1"])
def test_explicit_backend_requires_agent_capability_without_creating_work(platform, capabilities):
    node = register(platform)
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        row.metrics = {"capabilities": capabilities}
    response = create(platform, node, inference_backend="pytorch")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "POLICY_DEPLOY_BACKEND_UNSUPPORTED"
    assert platform.commands(node) == []
    with platform.app.state.db.read() as session:
        assert session.scalar(select(PolicyDeployment.id)) is None


@pytest.mark.parametrize("backend", [None, "torch", "JAX", "", 1, True, ["jax"]])
def test_backend_input_rejects_invalid_values(platform, backend):
    node = backend_node(platform)
    response = create(platform, node, inference_backend=backend)
    assert response.status_code == 422, response.text
    assert platform.commands(node) == []


@pytest.mark.parametrize("state", ["STARTING", "RUNNING"])
@pytest.mark.parametrize("resolved", [None, "jax", "auto", "torch", [], 1])
def test_wrong_or_missing_explicit_backend_receipt_never_proves_readiness(platform, state, resolved):
    node = backend_node(platform)
    deployment = create(platform, node, inference_backend="pytorch").json()
    receipt = running(state=state)
    if resolved is not None:
        receipt["resolved_backend"] = resolved
    start = command(platform, node, "POLICY_DEPLOY_START")
    ack(platform, node, start, receipt)
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN"
    assert current["resolved_backend"] is None
    assert current["error"]["code"] == "POLICY_DEPLOY_BACKEND_MISMATCH"
    with platform.app.state.db.read() as session:
        assert session.get(Command, start["id"]).status == "REJECTED"
    # The uncertain service still owns its GPU and port.
    assert create(platform, node, port=8001).status_code == 409


def test_auto_keeps_legacy_receipts_compatible_and_records_reported_backend(platform):
    node = register(platform)
    deployment = create(platform, node).json()
    assert deployment["inference_backend"] == "auto"
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), running())
    assert get(platform, deployment)["state"] == "RUNNING"
    assert get(platform, deployment)["resolved_backend"] is None
    action(platform, deployment, "refresh")
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_STATUS"), running(resolved_backend="pytorch"))
    assert get(platform, deployment)["resolved_backend"] == "pytorch"
    # An execution cannot silently switch frameworks after reporting one.
    action(platform, deployment, "refresh")
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_STATUS"), running(resolved_backend="jax"))
    current = get(platform, deployment)
    assert current["state"] == "UNKNOWN"
    assert current["resolved_backend"] == "pytorch"
    assert current["error"]["code"] == "POLICY_DEPLOY_BACKEND_MISMATCH"


def test_restart_preserves_requested_backend_and_clears_previous_resolution(platform):
    node = backend_node(platform)
    enable_restart(platform, node)
    deployment = create(platform, node, inference_backend="pytorch").json()
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), running(resolved_backend="pytorch"))
    action(platform, deployment, "stop")
    # A stop receipt can release resources without repeating model information.
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_STOP"), {"state": "STOPPED"})
    assert get(platform, deployment)["resolved_backend"] == "pytorch"
    restarted = action(platform, deployment, "start")
    assert restarted.status_code == 202, restarted.text
    assert restarted.json()["inference_backend"] == "pytorch"
    assert restarted.json()["resolved_backend"] is None
    start = command(platform, node, "POLICY_DEPLOY_START")
    assert start["payload"]["inference_backend"] == "pytorch"
    ack(platform, node, start, running(execution_id=start["payload"]["execution_id"], resolved_backend="pytorch"))
    assert get(platform, deployment)["state"] == "RUNNING"


def test_restart_rechecks_backend_capability_before_queuing(platform):
    node = backend_node(platform)
    enable_restart(platform, node)
    deployment = create(platform, node, inference_backend="jax").json()
    ack(platform, node, command(platform, node, "POLICY_DEPLOY_START"), {"state": "FAILED"})
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        metrics = deepcopy(row.metrics)
        metrics["capabilities"].remove("POLICY_DEPLOY_BACKEND_V1")
        row.metrics = metrics
    response = action(platform, deployment, "start")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "POLICY_DEPLOY_BACKEND_UNSUPPORTED"
    assert get(platform, deployment)["state"] == "FAILED"
    assert platform.commands(node) == []


def test_additive_backend_migration_preserves_existing_deployment(platform):
    node = register(platform)
    deployment = create(platform, node).json()
    database = platform.app.state.db
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE policy_deployments DROP COLUMN inference_backend")
        connection.exec_driver_sql("ALTER TABLE policy_deployments DROP COLUMN resolved_backend")
    database.initialize()
    database.initialize()  # Idempotent upgrade.
    columns = {column["name"] for column in inspect(database.engine).get_columns("policy_deployments")}
    assert {"inference_backend", "resolved_backend"} <= columns
    assert get(platform, deployment) == deployment
    with database.read() as session:
        row = session.get(PolicyDeployment, deployment["id"])
        assert row.inference_backend == "auto"
        assert row.resolved_backend is None
