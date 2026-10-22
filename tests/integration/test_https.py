"""Real verified TLS through the same Caddy configuration shipped for deployment."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import time

import httpx
import pytest

from vla_control.db import Database
from vla_control.users import create_initial_admin


ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "isolated-https-password-123"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


@contextmanager
def child(command, environment, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def ready(url, process, verify=True):
    deadline = time.monotonic() + 15
    with httpx.Client(verify=verify, trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            assert process.poll() is None, "Isolated HTTPS process exited; inspect its test log"
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(.05)
    pytest.fail("Isolated HTTPS process did not become ready")


@pytest.fixture(scope="module")
def https_proxy(tmp_path_factory):
    configured = os.getenv("VLA_TEST_CADDY_BINARY") or os.getenv("VLA_CADDY_BINARY")
    binary = configured or shutil.which("caddy")
    local = ROOT / ".local/tools/caddy-2.11.7/caddy"
    if not binary and local.is_file():
        binary = str(local)
    if not binary:
        pytest.skip("Install Caddy or set VLA_TEST_CADDY_BINARY for real HTTPS proxy integration")
    base = tmp_path_factory.mktemp("https-proxy")
    tls = script("generate_local_tls").generate(base / "tls")
    api_port, http_port, tls_port = free_port(), free_port(), free_port()
    while len({api_port, http_port, tls_port}) != 3:
        api_port, http_port, tls_port = free_port(), free_port(), free_port()
    origin = f"https://127.0.0.1:{tls_port}"
    db_url = "sqlite:///" + str(base / "control.db")
    db = Database(db_url)
    try:
        db.initialize()
        create_initial_admin(db, "tls-admin", PASSWORD)
    finally:
        db.close()
    renderer = script("configure_https")
    args = renderer.parser().parse_args([
        "--mode", "provided", "--site", origin, "--upstream", f"http://127.0.0.1:{api_port}",
        "--bind", "127.0.0.1", "--http-port", str(http_port), "--https-port", str(tls_port),
        "--cert-file", tls["certificate"], "--key-file", tls["private_key"], "--output-dir", str(base / "proxy"),
    ])
    renderer.render(args)
    environment = {**os.environ, "VLA_DATABASE_URL": db_url, "VLA_ENROLLMENT_TOKEN": "isolated-agent-token",
                   "VLA_ADMIN_TOKEN": "", "VLA_LEGACY_ADMIN_ENABLED": "false", "VLA_SCHEDULER_ENABLED": "false",
                   "VLA_REMOTE_STATE_DIR": str(base / "remote"), "VLA_REQUIRE_HTTPS": "true",
                   "VLA_PUBLIC_URL": origin, "VLA_SESSION_COOKIE_SECURE": "true",
                   "VLA_FORWARDED_ALLOW_IPS": "127.0.0.1,::1", "VLA_HOST": "127.0.0.1", "VLA_PORT": str(api_port),
                   "VLA_TLS_CERT_FILE": "", "VLA_TLS_KEY_FILE": "", "VLA_WEB_DIST": "",
                   "XDG_DATA_HOME": str(base / "caddy-data"), "XDG_CONFIG_HOME": str(base / "caddy-config")}
    verification = ssl.create_default_context(cafile=tls["ca_certificate"])
    with child([sys.executable, "-m", "vla_control.main"], environment, base / "api.log") as api:
        ready(f"http://127.0.0.1:{api_port}/healthz", api)
        with child([binary, "run", "--config", str(base / "proxy/Caddyfile"), "--adapter", "caddyfile"], environment, base / "proxy.log") as proxy:
            ready(origin + "/healthz", proxy, verification)
            yield origin, f"http://127.0.0.1:{http_port}", verification


def test_verified_ip_tls_login_and_forwarded_headers_cannot_be_spoofed(https_proxy):
    origin, _, verification = https_proxy
    with httpx.Client(base_url=origin, verify=verification, trust_env=False) as client:
        headers = {"Origin": origin, "Forwarded": "for=203.0.113.123;proto=http;host=evil.example",
                   "X-Forwarded-For": "203.0.113.123", "X-Forwarded-Proto": "http", "X-Forwarded-Host": "evil.example"}
        response = client.post("/api/v1/auth/login", json={"username": "tls-admin", "password": PASSWORD}, headers=headers)
        assert response.status_code == 200, response.text
        assert "Secure" in response.headers["set-cookie"] and "HttpOnly" in response.headers["set-cookie"]
        assert response.headers["strict-transport-security"] == "max-age=31536000"
        assert client.get("/api/v1/auth/me").json()["user"]["username"] == "tls-admin"
        audit = client.get("/api/v1/audit").json()
        login = next(row for row in audit if row["action"] == "auth.login" and row["actor_username"] == "tls-admin")
        assert login["remote_addr"] == "127.0.0.1"
        assert client.post("/api/v1/auth/logout").status_code == 403
        assert client.post("/api/v1/auth/logout", headers={"Origin": origin, "X-CSRF-Token": response.json()["csrf_token"]}).status_code == 204
        assert client.get("/api/v1/auth/me").status_code == 401


def test_http_rejects_credentials_and_unknown_hosts_never_redirect(https_proxy):
    origin, plain, verification = https_proxy
    with httpx.Client(verify=verification, trust_env=False) as client:
        response = client.get(plain + "/settings?test=1")
        assert response.status_code == 308 and response.headers["location"] == origin + "/settings?test=1"
        response = client.post(plain + "/api/v1/auth/login", json={"username": "tls-admin", "password": PASSWORD})
        assert response.status_code == 426 and "set-cookie" not in response.headers
        response = client.get(plain + "/", headers={"Host": "evil.example"})
        assert response.status_code == 421 and "location" not in response.headers
        response = client.get(origin + "/healthz", headers={"Host": "evil.example"})
        assert response.status_code == 421


def test_native_tls_validates_the_certificate_and_authenticates_over_https(tmp_path):
    tls = script("generate_local_tls").generate(tmp_path / "tls")
    port = free_port()
    origin = f"https://127.0.0.1:{port}"
    db_url = "sqlite:///" + str(tmp_path / "native.db")
    db = Database(db_url)
    try:
        db.initialize()
        create_initial_admin(db, "native-admin", PASSWORD)
    finally:
        db.close()
    environment = {**os.environ, "VLA_DATABASE_URL": db_url, "VLA_ENROLLMENT_TOKEN": "separate-native-agent-token",
                   "VLA_LEGACY_ADMIN_ENABLED": "false", "VLA_ADMIN_TOKEN": "", "VLA_SCHEDULER_ENABLED": "false",
                   "VLA_REMOTE_STATE_DIR": str(tmp_path / "remote"), "VLA_REQUIRE_HTTPS": "true",
                   "VLA_PUBLIC_URL": origin, "VLA_SESSION_COOKIE_SECURE": "true", "VLA_FORWARDED_ALLOW_IPS": "",
                   "VLA_HOST": "127.0.0.1", "VLA_PORT": str(port), "VLA_WEB_DIST": "",
                   "VLA_TLS_CERT_FILE": tls["certificate"], "VLA_TLS_KEY_FILE": tls["private_key"]}
    verification = ssl.create_default_context(cafile=tls["ca_certificate"])
    with child([sys.executable, "-m", "vla_control.main"], environment, tmp_path / "native.log") as process:
        ready(origin + "/healthz", process, verification)
        with httpx.Client(base_url=origin, verify=verification, trust_env=False) as client:
            response = client.post("/api/v1/auth/login", json={"username": "native-admin", "password": PASSWORD}, headers={"Origin": origin})
            assert response.status_code == 200, response.text
            assert "Secure" in response.headers["set-cookie"]
            assert client.get("/api/v1/auth/me").json()["user"]["username"] == "native-admin"
        with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
            with verification.wrap_socket(connection, server_hostname="127.0.0.1") as secured:
                assert secured.version() in {"TLSv1.2", "TLSv1.3"}
