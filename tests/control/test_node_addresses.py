"""Machine addresses come from successful SSH provenance, never proxy headers."""

import ipaddress

import pytest

from vla_control.common import uid
from vla_control.models import Node
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN


PRIVATE_VALUES = ("private-ssh-username", "private-encrypted-password", "private-fingerprint",
                  "private-probe-output", "private-enrollment-token", "private-deployment-log")


def install(platform, node, host, *, port=22, state="SUCCEEDED", operation=None,
            created="2026-10-01T00:00:00.000000+00:00", finished=None):
    connection_id, job_id = uid(), uid()
    with platform.app.state.db.write() as session:
        session.add(RemoteConnection(id=connection_id, host=host, port=port, username=PRIVATE_VALUES[0],
            encrypted_password=PRIVATE_VALUES[1], fingerprint=PRIVATE_VALUES[2],
            probe_result={"private": PRIVATE_VALUES[3]}, install_dir="/machine/private-agent",
            created_at=created, expires_at="2030-01-01T00:00:00.000000+00:00"))
        session.flush()
        session.add(RemoteDeployment(id=job_id, connection_id=connection_id, idempotency_key=uid(),
            request_digest="a" * 64, target_key="b" * 64, onboarding_id=uid(), node_id=node["node_id"],
            state=state, stage="ONLINE", logs=[{"message": PRIVATE_VALUES[5]}], history=[],
            agent_config={"enrollment_token": PRIVATE_VALUES[4], **({"_operation": operation} if operation else {})},
            install_dir="/machine/private-agent", control_plane_url="https://controller.invalid", connection_mode="ssh_tunnel",
            tunnel_state="CLOSED", created_at=created, updated_at=finished or created,
            finished_at=finished))
    return job_id


def listed(platform, node):
    response = platform.client.get("/api/v1/nodes", headers=ADMIN)
    assert response.status_code == 200
    return next(value for value in response.json() if value["id"] == node["node_id"])


def detail(platform, node):
    response = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN)
    assert response.status_code == 200
    return response.json()


def hostname(platform, node, value):
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).hostname = value


def test_list_and_detail_show_successful_ssh_address_and_keep_existing_node_fields(platform):
    node = platform.register(name="gpu-host")
    before = listed(platform, node)
    assert before["connection_address"] is None
    install(platform, node, "198.51.100.169", port=2207)
    expected = {"host": "198.51.100.169", "port": 2207, "source": "ssh"}
    assert listed(platform, node) == {**before, "connection_address": expected}
    assert detail(platform, node)["connection_address"] == expected


@pytest.mark.parametrize("state", ["OFFLINE", "RECOVERING", "DISCONNECTED", "REMOVED"])
def test_last_known_ssh_address_remains_visible_when_machine_is_unavailable(platform, state):
    node = platform.register()
    install(platform, node, "203.0.113.9")
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = state
    collection = platform.client.get("/api/v1/nodes", headers=ADMIN, params={"include_inactive": True}).json()
    historical = next(value for value in collection if value["id"] == node["node_id"])
    for value in (historical, detail(platform, node)):
        assert value["status"] == state
        assert value["connection_address"] == {"host": "203.0.113.9", "port": 22, "source": "ssh"}


@pytest.mark.parametrize("state", ["FAILED", "INTERRUPTED", "WAITING_AGENT", "QUEUED", "RUNNING"])
def test_failed_or_incomplete_new_target_does_not_replace_last_successful_address(platform, state):
    node = platform.register()
    install(platform, node, "192.0.2.1", port=2201)
    install(platform, node, "192.0.2.2", port=2202, state=state, created="2026-10-02T00:00:00.000000+00:00")
    assert listed(platform, node)["connection_address"] == {"host": "192.0.2.1", "port": 2201, "source": "ssh"}


@pytest.mark.parametrize("operation", ["disconnect", "uninstall", "unknown-operation"])
def test_only_install_operations_can_supply_a_machine_address(platform, operation):
    node = platform.register()
    install(platform, node, "192.0.2.10")
    install(platform, node, "192.0.2.20", operation=operation, created="2026-10-02T00:00:00.000000+00:00")
    assert detail(platform, node)["connection_address"]["host"] == "192.0.2.10"


def test_latest_completed_success_wins_including_an_older_job_retried_later(platform):
    node = platform.register()
    install(platform, node, "192.0.2.20", operation="install", created="2026-10-02T00:00:00.000000+00:00",
            finished="2026-10-02T01:00:00.000000+00:00")
    install(platform, node, "192.0.2.10", created="2026-10-01T00:00:00.000000+00:00",
            finished="2026-10-03T01:00:00.000000+00:00")
    assert listed(platform, node)["connection_address"]["host"] == "192.0.2.10"


def test_installations_are_scoped_to_the_exact_machine_and_ssh_overrides_hostname_fallback(platform):
    first = platform.register(name="first")
    second = platform.register(name="second")
    hostname(platform, first, "203.0.113.100")
    install(platform, first, "192.0.2.1")
    install(platform, second, "192.0.2.2", created="2026-10-03T00:00:00.000000+00:00")
    assert listed(platform, first)["connection_address"]["host"] == "192.0.2.1"
    assert listed(platform, second)["connection_address"]["host"] == "192.0.2.2"
    third = platform.register(name="third")
    assert listed(platform, third)["connection_address"] is None


@pytest.mark.parametrize("address", ["192.0.2.8", "2001:0db8:0000::8", "::1"])
def test_a_literal_hostname_ip_is_a_fallback_without_an_invented_ssh_port(platform, address):
    node = platform.register()
    hostname(platform, node, address)
    expected = {"host": str(ipaddress.ip_address(address)), "port": None, "source": "hostname"}
    assert listed(platform, node)["connection_address"] == expected
    assert detail(platform, node)["connection_address"] == expected


@pytest.mark.parametrize("name", ["gpu-worker", "worker.example.invalid", "localhost", "192.0.2.8:22", "[2001:db8::8]"])
def test_plain_hostname_is_not_resolved_and_proxy_loopback_headers_are_ignored(platform, monkeypatch, name):
    node = platform.register()
    hostname(platform, node, name)
    def no_dns(*_args, **_kwargs):
        raise AssertionError("Displaying a machine address must never perform DNS")
    monkeypatch.setattr("socket.getaddrinfo", no_dns)
    monkeypatch.setattr("socket.gethostbyname", no_dns)
    response = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers={**ADMIN,
        "X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"})
    assert response.status_code == 200 and response.json()["connection_address"] is None


def test_a_successfully_used_ssh_hostname_is_preserved_without_dns_lookup(platform, monkeypatch):
    node = platform.register()
    install(platform, node, "gpu.example.invalid", port=2222)
    monkeypatch.setattr("socket.getaddrinfo", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no DNS")))
    assert listed(platform, node)["connection_address"] == {"host": "gpu.example.invalid", "port": 2222, "source": "ssh"}


def test_public_node_responses_expose_only_host_port_and_source_from_connection_records(platform):
    node = platform.register()
    install(platform, node, "2001:db8::169", port=2222)
    for value in (listed(platform, node), detail(platform, node)):
        assert value["connection_address"] == {"host": "2001:db8::169", "port": 2222, "source": "ssh"}
        assert set(value["connection_address"]) == {"host", "port", "source"}
        for private in PRIVATE_VALUES:
            assert private not in str(value)
