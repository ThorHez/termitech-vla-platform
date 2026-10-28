"""Source and machine mutations cannot strand a live inference service."""

import pytest

from vla_control.common import APIError, now, uid
from vla_control.models import Profile
from vla_control.policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from vla_control.profile_rebindings import _in_use, _require_idle
from vla_control.recipe_writeback import require_checkout_idle
from vla_control.remote_deployment import DeploymentManager

from .conftest import ADMIN


def add_deployment(session, node_id, *, state="RUNNING", profile_id="default"):
    value = PolicyDeployment(id=uid(), name="local policy", node_id=node_id, profile_id=profile_id,
        checkpoint_path="/machine/checkpoints/1000", config_name="pi0_base", port=8000,
        state=state, logs="", checks=[], created_at=now(), updated_at=now())
    session.add(value)
    session.flush()
    return value


@pytest.mark.parametrize("state", sorted(RESOURCE_STATES))
def test_active_policy_blocks_source_runtime_and_agent_management(platform, state):
    node = platform.register(adapter="openpi-sft")
    with platform.app.state.db.write() as session:
        add_deployment(session, node["node_id"], state=state)
    with platform.app.state.db.read() as session:
        assert _in_use(session, node["node_id"], "default")
        for operation, code in (
            (lambda: require_checkout_idle(session, node["node_id"], "/machine/code"), "RECIPE_CHECKOUT_BUSY"),
            (lambda: _require_idle(session, node["node_id"], "default", {"/machine/code"}), "PROFILE_REBIND_BUSY"),
            (lambda: DeploymentManager._assert_idle(session, node["node_id"]), "NODE_BUSY"),
        ):
            with pytest.raises(APIError) as caught:
                operation()
            assert caught.value.code == code
    management = platform.client.get(f"/api/v1/nodes/{node['node_id']}/management", headers=ADMIN)
    assert management.status_code == 200
    assert management.json()["can_disconnect"] is False
    assert "真机部署" in management.json()["disconnect_reason"]


@pytest.mark.parametrize("state", ["STOPPED", "FAILED"])
def test_terminal_policy_releases_lifecycle_guards(platform, state):
    node = platform.register(adapter="openpi-sft")
    with platform.app.state.db.write() as session:
        add_deployment(session, node["node_id"], state=state)
    with platform.app.state.db.read() as session:
        assert not _in_use(session, node["node_id"], "default")
        require_checkout_idle(session, node["node_id"], "/machine/code")
        _require_idle(session, node["node_id"], "default", {"/machine/code"})
        DeploymentManager._assert_idle(session, node["node_id"])


def test_shared_destination_checkout_is_protected_without_blocking_unrelated_profiles(platform):
    node = platform.register(adapter="openpi-sft")
    with platform.app.state.db.write() as session:
        session.add(Profile(id="serving", node_id=node["node_id"], name="Serving", code_dir="/machine/serving",
            python="/usr/bin/python3", adapter="openpi-sft", execution_mode="native", metadata_json={}, active=True))
        session.flush()
        add_deployment(session, node["node_id"], profile_id="serving")
    with platform.app.state.db.read() as session:
        require_checkout_idle(session, node["node_id"], "/machine/code")
        _require_idle(session, node["node_id"], "default", {"/machine/code", "/machine/unused"})
        with pytest.raises(APIError) as caught:
            _require_idle(session, node["node_id"], "default", {"/machine/code", "/machine/serving"})
        assert caught.value.code == "PROFILE_REBIND_BUSY"
        with pytest.raises(APIError) as caught:
            require_checkout_idle(session, node["node_id"], "/machine/serving")
        assert caught.value.code == "RECIPE_CHECKOUT_BUSY"
