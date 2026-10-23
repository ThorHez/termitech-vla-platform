"""HTTPS preparation is offline, explicit and keeps private control transport."""

import importlib.util
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("https_deployment_fixture", ROOT / "scripts/configure_https.py")
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)


def arguments(output, *extra):
    return deployment.parser().parse_args([
        "--mode", "acme", "--site", "https://115.190.212.169", "--output-dir", str(output), *extra,
    ])


def test_public_ip_uses_explicit_shortlived_acme_without_requesting_a_certificate(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Rendering must not contact the network, install or run a service")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    result = deployment.render(arguments(tmp_path / "rendered"))
    config = (tmp_path / "rendered/Caddyfile").read_text()
    assert "issuer acme {" in config
    assert f"dir {deployment.ACME_PRODUCTION}" in config
    assert "profile shortlived" in config
    assert "tls internal" not in config
    assert "strict_sni_host insecure_off" in config
    assert "    default_sni 115.190.212.169\n" in config
    assert result["default_sni"] == "115.190.212.169"
    assert result["controller_and_proxy_started"] is False
    assert result["system_trust_modified"] is False
    assert result["certificate_issuance"] == "on-explicit-service-start"
    assert "request_body {\n        max_size 10MB\n    }" in config
    assert result["max_request_body_bytes"] == 10_000_000
    assert "        protocols h1\n" in config
    assert result["http_protocols"] == ["h1"]
    assert not (tmp_path / "rendered/cert.pem").exists()


def test_domain_and_staging_are_explicit(tmp_path):
    result = deployment.render(arguments(tmp_path, "--site", "https://train.example.com", "--acme-staging",
                                         "--email", "ops@example.com"))
    config = (tmp_path / "Caddyfile").read_text()
    assert f"dir {deployment.ACME_STAGING}" in config
    assert "profile shortlived" not in config
    assert "email ops@example.com" in config
    assert "strict_sni_host on" in config
    assert "default_sni" not in config
    assert result["default_sni"] is None
    assert result["certificate_trust"] == "ACME-staging-untrusted"


def test_provided_pem_render_does_not_read_private_key(tmp_path):
    # Paths describe the destination machine, not files this preparation host
    # must possess. Certificates are validated separately before activation.
    output = tmp_path / "provided"
    deployment.render(arguments(output, "--mode", "provided", "--cert-file", "/etc/vla/certs/fullchain.pem",
                                "--key-file", "/etc/vla/certs/private.pem"))
    config = (output / "Caddyfile").read_text()
    assert "tls /etc/vla/certs/fullchain.pem /etc/vla/certs/private.pem" in config
    assert "issuer acme" not in config
    assert '>Strict-Transport-Security "max-age=31536000"' in config


def test_local_internal_mode_never_installs_a_trust_root(tmp_path):
    deployment.render(arguments(tmp_path, "--mode", "internal", "--site", "https://127.0.0.1:8443",
                                "--bind", "127.0.0.1", "--http-port", "8081", "--https-port", "8443"))
    config = (tmp_path / "Caddyfile").read_text()
    assert "tls internal" in config
    assert "skip_install_trust" in config
    assert "Strict-Transport-Security" not in config
    assert "https://127.0.0.1:8443" in config
    assert "bind 127.0.0.1" in config


@pytest.mark.parametrize("extra", [
    ["--site", "http://example.com"],
    ["--site", "https://user:password@example.com"],
    ["--site", "https://example.com/path"],
    ["--site", "https://example.com?x=1"],
    ["--site", "https://example.com/#fragment"],
    ["--site", "https://example.com:"],
    ["--site", "https://example.com:0"],
    ["--site", "https://example.com:8443"],
    ["--site", "https://{env.HOST}"],
    ["--site", "https://example.com\nadmin off"],
    ["--site", "https://example.com;evil"],
    ["--site", "https://example.com%0a"],
    ["--site", "https://EXAMPLE.com"],
    ["--site", "https://example.com."],
    ["--site", "https://*.example.com"],
    ["--site", "https://127.0.0.1"],
    ["--site", "https://192.168.1.1"],
    ["--site", "https://localhost"],
    ["--site", "https://train.local"],
    ["--upstream", "https://127.0.0.1:8000"],
    ["--upstream", "http://115.190.212.169:8000"],
    ["--upstream", "http://192.168.1.2:8000"],
    ["--upstream", "http://127.0.0.2:8000"],
    ["--upstream", "http://0.0.0.0:8000"],
    ["--upstream", "http://api:9000"],
    ["--upstream", "http://127.0.0.1:443"],
    ["--upstream", "http://api:8000/admin"],
    ["--bind", "0.0.0.0\nadmin :2019"],
    ["--bind", "{env.HOST}"],
    ["--http-port", "443"],
    ["--email", "ops@example.com\nadmin off"],
    ["--mode", "provided"],
    ["--mode", "provided", "--cert-file", "/etc/cert.pem"],
    ["--mode", "provided", "--cert-file", "/etc/cert.pem", "--key-file", "/etc/cert.pem"],
    ["--mode", "provided", "--cert-file", "/etc/cert.pem", "--key-file", "/etc/private key.pem"],
    ["--mode", "provided", "--cert-file", "/etc/cert.pem", "--key-file", "/etc/../root/private.pem"],
    ["--mode", "provided", "--cert-file", "/etc/cert.pem", "--key-file", "/etc/key.pem", "--acme-staging"],
    ["--mode", "internal", "--email", "ops@example.com"],
    ["--mode", "internal", "--cert-file", "/etc/cert.pem", "--key-file", "/etc/key.pem"],
    ["--config-dir", "/etc/vla;echo-evil"],
    ["--config-dir", "/etc/{env.CONF}"],
    ["--caddy-binary", "/usr/bin/caddy\nExecStart=/evil"],
    ["--state-dir", "/root"],
])
def test_invalid_input_leaves_no_partial_configuration(tmp_path, extra):
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        deployment.render(arguments(output, *extra))
    assert not output.exists()


def test_preserves_existing_configuration_and_refuses_symlink_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    expected = output / "control-https.env"
    expected.write_text("private-existing-configuration")
    with pytest.raises(ValueError, match="Refusing to replace"):
        deployment.render(arguments(output))
    assert expected.read_text() == "private-existing-configuration"
    assert list(output.iterdir()) == [expected]
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        deployment.render(arguments(link))


def test_controller_patch_preserves_agent_ssh_transport_and_secures_human_sessions(tmp_path):
    deployment.render(arguments(tmp_path))
    values = dict(line.split("=", 1) for line in (tmp_path / "control-https.env").read_text().splitlines()
                  if line and not line.startswith("#"))
    assert values["VLA_HOST"] == "127.0.0.1"
    assert values["VLA_PORT"] == values["VLA_TUNNEL_TARGET_PORT"] == "8000"
    assert values["VLA_AGENT_CONTROL_PLANE_URL"] == "http://127.0.0.1:8000"
    assert values["VLA_SESSION_COOKIE_SECURE"] == values["VLA_REQUIRE_HTTPS"] == "true"
    assert values["VLA_SESSION_TTL_SECONDS"] == "7200"
    assert values["VLA_SESSION_IDLE_SECONDS"] == "900"
    assert values["VLA_SESSION_REAUTH_SECONDS"] == "300"
    assert values["VLA_PUBLIC_URL"] == "https://115.190.212.169"
    assert values["VLA_FORWARDED_ALLOW_IPS"] == "127.0.0.1,::1"
    assert values["VLA_TLS_CERT_FILE"] == values["VLA_TLS_KEY_FILE"] == ""
    assert not any("TOKEN" in key or "PASSWORD" in key or "DATABASE" in key for key in values)
    for name in deployment.OUTPUT_NAMES:
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o600


def test_ipv6_upstream_environment_matches_the_socket_and_tunnel_destination(tmp_path):
    result = deployment.render(arguments(tmp_path, "--upstream", "http://[::1]:8001"))
    values = dict(line.split("=", 1) for line in (tmp_path / "control-https.env").read_text().splitlines()
                  if line and not line.startswith("#"))
    assert values["VLA_HOST"] == values["VLA_TUNNEL_TARGET_HOST"] == "::1"
    assert values["VLA_PORT"] == values["VLA_TUNNEL_TARGET_PORT"] == "8001"
    assert values["VLA_AGENT_CONTROL_PLANE_URL"] == "http://[::1]:8001"
    assert "reverse_proxy http://[::1]:8001" in (tmp_path / "Caddyfile").read_text()
    assert result["deployment_kind"] == "native"


def test_localhost_is_canonicalized_to_the_exact_controller_ipv4_bind(tmp_path):
    deployment.render(arguments(tmp_path, "--upstream", "http://localhost:8001"))
    assert "reverse_proxy http://127.0.0.1:8001" in (tmp_path / "Caddyfile").read_text()
    assert "VLA_HOST=127.0.0.1" in (tmp_path / "control-https.env").read_text()


def test_compose_upstream_does_not_emit_a_broken_native_service_or_environment(tmp_path):
    result = deployment.render(arguments(tmp_path, "--upstream", "http://api:8000"))
    assert result["deployment_kind"] == "compose"
    assert result["controller_environment_is_patch"] is False
    assert result["systemd_service_applicable"] is False
    assert "reverse_proxy http://api:8000" in (tmp_path / "Caddyfile").read_text()
    for name in ("control-https.env", "termitech-vla-https.service"):
        assert all(line.startswith("#") for line in (tmp_path / name).read_text().splitlines())


def test_proxy_rejects_unknown_hosts_and_never_preserves_spoofed_forwarded_headers(tmp_path):
    deployment.render(arguments(tmp_path))
    config = (tmp_path / "Caddyfile").read_text()
    assert "admin off" in config
    assert "auto_https disable_redirects" in config
    assert "respond @unknown_host" in config
    assert "redir https://115.190.212.169{uri} 308" in config
    assert "redir https://{host}" not in config
    assert "request_header -Forwarded" in config
    assert "request_header -X-Forwarded-*" in config
    assert config.index("request_header -X-Forwarded-*") < config.index("reverse_proxy ")
    assert "header_up -X-Forwarded-*" not in config
    assert "header_up X-Forwarded-For {remote_host}" in config
    assert "header_up X-Forwarded-Proto https" in config
    assert "header_up Host {hostport}" in config
    assert "trusted_proxies" not in config
    assert config.index("respond @plain_api") < config.index("redir https://")
    assert config.index("respond @unsafe_method") < config.index("redir https://")


def test_agent_proxy_is_preserved_unless_internal_only_is_explicit(tmp_path):
    default = tmp_path / "default"
    result = deployment.render(arguments(default))
    assert result["internal_agents_only"] is False
    assert "@internal_only" not in (default / "Caddyfile").read_text()
    protected = tmp_path / "internal-only"
    result = deployment.render(arguments(protected, "--internal-agents-only"))
    config = (protected / "Caddyfile").read_text()
    assert result["internal_agents_only"] is True
    assert config.count("@internal_only path /agent /agent/* /docs /docs/* /redoc /redoc/* /openapi.json") == 2
    assert config.count("respond @internal_only 404") == 2
    assert config.index("respond @internal_only 404") < config.index("reverse_proxy ")
    assert config.rindex("respond @internal_only 404") < config.index("respond @plain_api")
    assert "VLA_AGENT_CONTROL_PLANE_URL=http://127.0.0.1:8000" in (protected / "control-https.env").read_text()


def test_service_is_unprivileged_with_private_durable_state_and_no_activation(tmp_path):
    deployment.render(arguments(tmp_path))
    unit = (tmp_path / "termitech-vla-https.service").read_text()
    assert "User=caddy\nGroup=caddy" in unit
    assert "StateDirectoryMode=0700" in unit
    assert "ProtectSystem=strict" in unit
    assert "ProtectHome=true" in unit
    assert "CapabilityBoundingSet=CAP_NET_BIND_SERVICE" in unit
    assert "systemctl" not in unit
    assert "ExecReload=" not in unit


def test_cli_outputs_only_non_secret_render_metadata(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/configure_https.py"),
                             "--mode", "acme", "--site", "https://115.190.212.169",
                             "--output-dir", str(tmp_path / "cli")],
                            text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["caddy_version"] == "2.11.7"
    assert receipt["controller_and_proxy_started"] is False


def test_compose_only_publishes_api_on_loopback_and_trusts_one_proxy_address():
    base = (ROOT / "deploy/compose.yaml").read_text()
    proxy = (ROOT / "deploy/compose.https.yaml").read_text()
    assert '"127.0.0.1:8000:8000"' in base
    assert "8000:8000" not in proxy
    assert "2019:" not in proxy
    assert "caddy:2.11.7-alpine" in proxy
    assert "VLA_FORWARDED_ALLOW_IPS: ${VLA_HTTPS_PROXY_IP:-172.29.239.2}" in proxy
    assert "private_ranges" not in proxy
    assert 'VLA_TLS_CERT_FILE: ""' in proxy
    assert 'VLA_TLS_KEY_FILE: ""' in proxy
    assert 'VLA_FORWARDED_ALLOW_IPS: ""' in base
    assert "read_only: true" in proxy


@pytest.mark.skipif(not os.getenv("VLA_TEST_CADDY_BINARY"), reason="Optional pinned Caddy adapter validation")
@pytest.mark.parametrize("site", ["https://115.190.212.169", "https://train.example.com", "https://[2606:4700:4700::1111]"])
@pytest.mark.parametrize("internal_only", [False, True])
def test_caddy_adapter_accepts_config_and_uses_public_acme(tmp_path, site, internal_only):
    deployment.render(arguments(tmp_path, "--site", site, *(["--internal-agents-only"] if internal_only else [])))
    result = subprocess.run([os.environ["VLA_TEST_CADDY_BINARY"], "adapt", "--adapter", "caddyfile",
                             "--config", str(tmp_path / "Caddyfile")], text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    tls = value["apps"]["tls"]["automation"]["policies"]
    assert any(issuer.get("module") == "acme" for policy in tls for issuer in policy.get("issuers", []))
    assert not any(issuer.get("module") == "internal" for policy in tls for issuer in policy.get("issuers", []))
    _, host, _, is_ip = deployment.origin(site, scheme="https")
    policies = [policy for server in value["apps"]["http"]["servers"].values()
                for policy in server.get("tls_connection_policies", [])]
    assert bool(policies) is is_ip
    assert all(policy.get("default_sni") == (host if is_ip else None) for policy in policies)

    def handlers(node):
        if isinstance(node, dict):
            if "handler" in node:
                yield node
            for child in node.values():
                yield from handlers(child)
        elif isinstance(node, list):
            for child in node:
                yield from handlers(child)

    hidden = [handler for handler in handlers(value) if handler.get("handler") == "static_response"
              and handler.get("status_code") == 404]
    assert len(hidden) == (2 if internal_only else 0)


@pytest.mark.skipif(not os.getenv("VLA_TEST_CADDY_BINARY"), reason="Optional real Caddy IP/NAT regression")
def test_ip_certificate_without_sni_works_when_public_ip_differs_from_listener(tmp_path):
    # Python omits SNI for IP server_hostname. Connect to loopback while
    # validating a different IP SAN, as happens when public IPs are NATed.
    public_ip = "115.190.212.169"
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Isolated IP/NAT regression")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                   .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(public_ip))]), critical=False)
                   .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "certificate.pem", tmp_path / "private.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    key_path.chmod(0o600)

    def free_port():
        with socket.socket() as stream:
            stream.bind(("127.0.0.1", 0))
            return stream.getsockname()[1]

    http_port, tls_port = free_port(), free_port()
    while http_port == tls_port:
        tls_port = free_port()
    deployment.render(arguments(tmp_path / "proxy", "--mode", "provided", "--site", f"https://{public_ip}:{tls_port}",
                                "--bind", "127.0.0.1", "--http-port", str(http_port), "--https-port", str(tls_port),
                                "--cert-file", str(cert_path), "--key-file", str(key_path), "--internal-agents-only"))
    verification = ssl.create_default_context(cafile=str(cert_path))
    environment = {**os.environ, "XDG_DATA_HOME": str(tmp_path / "caddy-data"),
                   "XDG_CONFIG_HOME": str(tmp_path / "caddy-config")}
    with (tmp_path / "caddy.log").open("w") as log:
        process = subprocess.Popen([os.environ["VLA_TEST_CADDY_BINARY"], "run", "--adapter", "caddyfile",
                                    "--config", str(tmp_path / "proxy/Caddyfile")],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while True:
                assert process.poll() is None, "Isolated Caddy exited; inspect its test log"
                try:
                    connection = socket.create_connection(("127.0.0.1", tls_port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail("Isolated Caddy did not bind")
                    time.sleep(.05)
            with connection, verification.wrap_socket(connection, server_hostname=public_ip) as secured:
                secured.sendall(f"GET /agent/v1/enroll HTTP/1.1\r\nHost: {public_ip}:{tls_port}\r\nConnection: close\r\n\r\n".encode())
                assert secured.recv(1024).split(b"\r\n", 1)[0] == b"HTTP/1.1 404 Not Found"
            with socket.create_connection(("127.0.0.1", tls_port), timeout=1) as connection:
                with verification.wrap_socket(connection, server_hostname=public_ip) as secured:
                    secured.sendall(b"GET /healthz HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n")
                    assert secured.recv(1024).split(b"\r\n", 1)[0] == b"HTTP/1.1 421 Misdirected Request"
            with socket.create_connection(("127.0.0.1", http_port), timeout=1) as connection:
                connection.sendall(f"GET /docs HTTP/1.1\r\nHost: {public_ip}:{http_port}\r\nConnection: close\r\n\r\n".encode())
                assert connection.recv(1024).split(b"\r\n", 1)[0] == b"HTTP/1.1 404 Not Found"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
