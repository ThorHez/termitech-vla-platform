from types import SimpleNamespace

from vla_agent import network_metrics
from vla_agent.main import Agent


def test_default_route_chooses_lowest_metric_and_ignores_unusable_routes():
    ipv4 = [
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT",
        "eth2 00000000 0100000A 0003 0 0 100 00000000 0 0 0",
        "eth0 00000000 0100000A 0003 0 0 50 00000000 0 0 0",
        "eth1 00000000 0100000A 0201 0 0 10 00000000 0 0 0",
        "eth1 00000000 0100000A 0002 0 0 5 00000000 0 0 0",
        "eth1 00000000 0100000A 0003 0 0 1 00FFFFFF 0 0 0",
    ]
    assert network_metrics._default_route(ipv4, {"eth0", "eth1", "eth2"}) == "eth0"
    assert network_metrics._default_route(ipv4, {"eth1", "eth2"}) == "eth2"

    ipv6 = [
        "0 00 0 00 0 00000040 00000000 00000000 00000001 eth2",
        "0 00 0 00 0 00000020 00000000 00000000 00000001 eth0",
        "0 00 0 02 0 00000000 00000000 00000000 00000001 eth1",
    ]
    assert network_metrics._default_route(ipv6, {"eth0", "eth1", "eth2"}, ipv6=True) == "eth0"


def test_interface_rates_handle_reset_replacement_and_missing_sample(monkeypatch):
    samples = iter([(100, 200, 2), (130, 260, 2), (3, 10, 2), (200, 300, 3), None, (230, 340, 3)])
    index = None

    def counters(*, pernic, nowrap):
        assert pernic and not nowrap
        nonlocal index
        sample = next(samples)
        if sample is None:
            raise OSError("counter read failed")
        rx_bytes, tx_bytes, index = sample
        values = SimpleNamespace(bytes_recv=rx_bytes, bytes_sent=tx_bytes)
        return {"eth0": values, "lo": values}

    monkeypatch.setattr(network_metrics.psutil, "net_io_counters", counters)
    monkeypatch.setattr(network_metrics, "_default_interface", lambda _: "eth0")
    monkeypatch.setattr(network_metrics, "_boot_id", lambda: "test-boot")
    monkeypatch.setattr(network_metrics, "_ifindex", lambda _: index)
    monkeypatch.setattr(network_metrics, "monotonic", iter([1, 4, 10, 20, 30]).__next__)
    monitor = network_metrics.NetworkMonitor()
    first = monitor.sample()
    assert first["boot_id"] == "test-boot"
    assert first["default_interface"] == "eth0"
    assert first["interfaces"] == [{"name": "eth0", "ifindex": 2, "rx_bytes": 100, "tx_bytes": 200,
                                   "rx_bytes_per_second": None, "tx_bytes_per_second": None}]

    next_interface = monitor.sample()["interfaces"][0]
    assert next_interface["rx_bytes_per_second"] == 10
    assert next_interface["tx_bytes_per_second"] == 20
    for _ in range(2):
        interval = monitor.sample()["interfaces"][0]
        assert interval["rx_bytes_per_second"] is None
        assert interval["tx_bytes_per_second"] is None
    assert monitor.sample() == {"boot_id": "test-boot", "default_interface": None,
                                "interfaces": [], "error": "OSError"}
    assert monitor.sample()["interfaces"][0]["rx_bytes_per_second"] is None


def test_heartbeat_includes_interface_counters(agent_fixture, monkeypatch):
    captured = []

    class Client:
        def request(self, method, path, token, body=None, headers=None):
            if path == "/agent/v1/heartbeat":
                captured.append(body["metrics"])
            return {}

    interface = SimpleNamespace(bytes_recv=1000, bytes_sent=2000)
    monkeypatch.setattr(network_metrics.psutil, "net_io_counters", lambda **_: {"eth0": interface})
    monkeypatch.setattr(network_metrics, "_default_interface", lambda _: "eth0")
    monkeypatch.setattr(network_metrics, "_boot_id", lambda: "test-boot")
    monkeypatch.setattr(network_metrics, "_ifindex", lambda _: 2)
    monkeypatch.setattr("vla_agent.main.gpu_inventory", lambda: ([], {}))
    agent = Agent(agent_fixture["config"], client=Client())
    agent.credentials = {"node_id": "fixture-node", "token": "fixture-token"}
    try:
        agent.heartbeat()
    finally:
        agent.close()
    assert captured[0]["network"] == {"boot_id": "test-boot", "default_interface": "eth0",
                                      "interfaces": [{"name": "eth0", "ifindex": 2, "rx_bytes": 1000,
                                                      "tx_bytes": 2000, "rx_bytes_per_second": None,
                                                      "tx_bytes_per_second": None}]}
