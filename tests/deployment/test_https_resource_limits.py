"""The pinned proxy must accept and apply admission resource boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("https_resource_limits", ROOT / "scripts/configure_https.py")
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)


def handlers(node):
    if isinstance(node, dict):
        if "handler" in node:
            yield node
        for child in node.values():
            yield from handlers(child)
    elif isinstance(node, list):
        for child in node:
            yield from handlers(child)


@pytest.mark.skipif(not os.getenv("VLA_TEST_CADDY_BINARY"), reason="Optional pinned Caddy adapter validation")
@pytest.mark.parametrize("mode", ["acme", "provided", "internal"])
def test_pinned_proxy_applies_request_boundaries_without_total_stream_deadline(tmp_path, mode):
    options = ["--mode", mode, "--site", "https://train.example.com:8443", "--https-port", "8443",
               "--http-port", "8080", "--bind", "127.0.0.1", "--upstream", "http://127.0.0.1:8001",
               "--output-dir", str(tmp_path / "rendered")]
    if mode == "provided":
        options.extend(["--cert-file", "/etc/termitech-vla/tls/fullchain.pem",
                        "--key-file", "/etc/termitech-vla/tls/private-key.pem"])
    deployment.render(deployment.parser().parse_args(options))
    result = subprocess.run([os.environ["VLA_TEST_CADDY_BINARY"], "adapt", "--adapter", "caddyfile",
                             "--config", str(tmp_path / "rendered/Caddyfile")],
                            text=True, capture_output=True, check=True)
    config = json.loads(result.stdout)
    servers = config["apps"]["http"]["servers"].values()
    for server in servers:
        assert server["read_header_timeout"] == 15_000_000_000
        assert server["read_timeout"] == 60_000_000_000
        assert server["read_idle_timeout"] == 15_000_000_000
        assert server["read_min_rate"] == 1024
        assert server["max_header_bytes"] == 16_384
        assert "write_timeout" not in server
    proxies = [item for item in handlers(config) if item["handler"] == "reverse_proxy"]
    assert len(proxies) == 1
    transport = proxies[0]["transport"]
    assert transport["dial_timeout"] == 3_000_000_000
    assert transport["response_header_timeout"] == 90_000_000_000
    assert "read_timeout" not in transport
    assert "stream_timeout" not in proxies[0]
    body_limits = [item["max_size"] for item in handlers(config) if item["handler"] == "request_body"]
    assert body_limits == [10_000_000]
