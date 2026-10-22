"""TLS, proxy trust and human/Agent transport boundaries, on isolated state."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient
import pytest
from uvicorn import Config
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from vla_control.auth import COOKIE_NAME
from vla_control.main import create_app, main
from vla_control.settings import Settings
from vla_control.transport_security import TransportSecurityMiddleware, public_origin, uvicorn_options


PASSWORD = "https-isolated-test-password"
ORIGIN = "https://controller.example:8443"


@pytest.fixture
def secure_app(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "https.sqlite"),
                        enrollment_token="agents-have-separate-credentials", scheduler_enabled=False,
                        public_url=ORIGIN, require_https=True)
    app = create_app(settings)
    app.state.auth.create_user("admin", PASSWORD, "admin")
    yield app, settings
    app.state.remote_manager.close()
    app.state.db.close()


@pytest.fixture
def certificate(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    stamp = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(stamp - timedelta(minutes=1))
            .not_valid_after(stamp + timedelta(days=1)).sign(key, hashes.SHA256()))
    cert_file = tmp_path / "certificate.pem"
    key_file = tmp_path / "private-key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    key_file.chmod(0o600)
    return str(cert_file), str(key_file)


def login(client, **kwargs):
    return client.post("/api/v1/auth/login", json={"username": "admin", "password": PASSWORD}, **kwargs)


@pytest.mark.parametrize("url,expected", [
    ("https://controller.example/", ("https://controller.example", "controller.example", 443)),
    ("https://127.0.0.1:8443", ("https://127.0.0.1:8443", "127.0.0.1", 8443)),
    ("https://[::1]:8443", ("https://[::1]:8443", "::1", 8443)),
])
def test_public_origin_is_canonical(url, expected):
    origin = public_origin(url)
    assert (origin.url, origin.hostname, origin.port) == expected


@pytest.mark.parametrize("url", [
    "http://example.com", "https://admin:secret@example.com", "https://example.com/app",
    "https://example.com?secret=value", "https://example.com#fragment", "https://*.example.com",
    "https://example.com:0", "https://example.com:65536", "https://example.com:",
    "https://example.com\\evil", "https://example.com\r\nLocation: https://evil.example", " https://example.com",
    "https://[::1%25eth0]", "https://bad_name.example",
])
def test_unsafe_public_origins_are_rejected(url):
    with pytest.raises(ValueError, match="VLA_PUBLIC_URL"):
        Settings(enrollment_token="separate", public_url=url).validate()


@pytest.mark.parametrize("value", ["*", "0.0.0.0/0", "::/0", "127.0.0.1,*", "127.0.0.1,", "example.com"])
def test_proxy_trust_requires_explicit_ip_or_bounded_network(value):
    with pytest.raises(ValueError, match="VLA_FORWARDED_ALLOW_IPS"):
        Settings(enrollment_token="separate", forwarded_allow_ips=value).validate()


def test_transport_environment_is_explicit(monkeypatch):
    values = {"VLA_PUBLIC_URL": ORIGIN, "VLA_REQUIRE_HTTPS": "true",
              "VLA_FORWARDED_ALLOW_IPS": "127.0.0.1,::1,10.30.0.0/24",
              "VLA_TLS_CERT_FILE": "/private/certificate.pem", "VLA_TLS_KEY_FILE": "/private/key.pem"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.public_url == ORIGIN and settings.require_https
    assert settings.forwarded_allow_ips == values["VLA_FORWARDED_ALLOW_IPS"]
    assert settings.tls_cert_file == values["VLA_TLS_CERT_FILE"]
    assert settings.tls_key_file == values["VLA_TLS_KEY_FILE"]


def test_https_configuration_requires_secure_cookies():
    with pytest.raises(ValueError, match="VLA_SESSION_COOKIE_SECURE"):
        Settings(enrollment_token="separate", require_https=True, session_cookie_secure=False).validate()


def test_native_tls_validates_pair_and_uses_tls_12_minimum(certificate):
    cert_file, key_file = certificate
    settings = Settings(enrollment_token="separate", tls_cert_file=cert_file, tls_key_file=key_file)
    options = uvicorn_options(settings)
    assert options["server_header"] is False and options["proxy_headers"] is True
    assert options["forwarded_allow_ips"] == "127.0.0.1,::1"
    config = Config(lambda: None, **options)
    config.load()
    assert config.ssl.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert config.ssl.verify_mode == ssl.CERT_NONE
    with pytest.raises(ValueError, match="configured together"):
        replace(settings, tls_key_file="").validate()
    with pytest.raises(ValueError, match="matching PEM") as exc:
        replace(settings, tls_key_file=cert_file).validate()
    assert "BEGIN" not in str(exc.value) and cert_file not in str(exc.value)


def test_invalid_tls_files_fail_before_listening(tmp_path):
    cert = tmp_path / "bad.pem"
    cert.write_text("not a private key or certificate")
    with pytest.raises(ValueError, match="matching PEM"):
        Settings(enrollment_token="separate", tls_cert_file=str(cert), tls_key_file=str(cert)).validate()
    with pytest.raises(ValueError, match="matching PEM"):
        Settings(enrollment_token="separate", tls_cert_file="/missing/cert.pem", tls_key_file="/missing/key.pem").validate()


def test_http_login_rejects_before_reading_body():
    touched = []

    async def downstream(scope, receive, send):
        touched.append("handler")

    async def receive():
        touched.append("body")
        raise AssertionError("Plaintext login body must not be read")

    messages = []

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "scheme": "http", "path": "/api/v1/auth/login",
             "raw_path": b"/api/v1/auth/login", "query_string": b"", "root_path": "",
             "headers": [(b"host", b"controller.example")], "server": ("controller.example", 80),
             "client": ("198.51.100.20", 4567), "http_version": "1.1"}
    middleware = TransportSecurityMiddleware(downstream, Settings(require_https=True, public_url=ORIGIN))
    asyncio.run(middleware(scope, receive, send))
    assert not touched and messages[0]["status"] == 426
    assert b"HTTPS_REQUIRED" in messages[1]["body"]


def test_https_login_secure_cookie_and_csrf(secure_app):
    app, _ = secure_app
    with TestClient(app, base_url=ORIGIN) as client:
        response = login(client, headers={"Origin": ORIGIN})
        assert response.status_code == 200, response.text
        assert "Secure" in response.headers["set-cookie"] and "HttpOnly" in response.headers["set-cookie"]
        assert response.headers["strict-transport-security"] == "max-age=31536000"
        assert client.get("/api/v1/auth/me").status_code == 200
        assert client.post("/api/v1/auth/logout", headers={"Origin": ORIGIN}).status_code == 403
        assert client.post("/api/v1/auth/logout", headers={"Origin": ORIGIN,
                           "X-CSRF-Token": response.json()["csrf_token"]}).status_code == 204


def test_trusted_proxy_tls_preserves_origin_csrf_and_cookie(secure_app):
    app, settings = secure_app
    proxy = ProxyHeadersMiddleware(app, trusted_hosts=settings.forwarded_allow_ips)
    with TestClient(proxy, base_url="http://controller.example:8443", client=("127.0.0.1", 4321)) as client:
        headers = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.25", "Origin": ORIGIN}
        response = login(client, headers=headers)
        assert response.status_code == 200, response.text
        assert response.headers["strict-transport-security"] == "max-age=31536000"
        assert "Secure" in response.headers["set-cookie"]
        # HTTPX sees the internal proxy URL. A real browser sends this cookie to
        # the HTTPS edge; simulate that proxy hop without relaxing Secure.
        headers["Cookie"] = COOKIE_NAME + "=" + client.cookies.get(COOKIE_NAME)
        headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
        forbidden = client.post("/api/v1/auth/logout", headers={**headers, "Origin": "https://evil.example"})
        assert forbidden.status_code == 403 and forbidden.json()["error"]["code"] == "ORIGIN_REJECTED"
        assert client.post("/api/v1/auth/logout", headers=headers).status_code == 204


def test_untrusted_forwarded_headers_cannot_bypass_plaintext_rejection(secure_app):
    app, settings = secure_app
    proxy = ProxyHeadersMiddleware(app, trusted_hosts=settings.forwarded_allow_ips)
    with TestClient(proxy, base_url="http://controller.example:8443", client=("198.51.100.20", 4321)) as client:
        response = login(client, headers={"X-Forwarded-Proto": "https", "X-Forwarded-For": "127.0.0.1"})
        assert response.status_code == 426 and response.json()["error"]["code"] == "HTTPS_REQUIRED"
        assert "set-cookie" not in response.headers and "strict-transport-security" not in response.headers


def test_app_does_not_interpret_forwarded_headers_itself(secure_app):
    app, _ = secure_app
    with TestClient(app, base_url="http://controller.example:8443") as client:
        assert login(client, headers={"X-Forwarded-Proto": "https"}).status_code == 426


@pytest.mark.parametrize("host", ["evil.example:8443", "controller.example:9443", "controller.example.evil:8443",
                                  "controller.example:0", "controller.example:", "controller.example@evil.example"])
def test_public_host_is_checked_before_authentication(secure_app, host):
    app, _ = secure_app
    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get("/api/v1/auth/me", headers={"Host": host})
        assert response.status_code == 400 and response.json()["error"]["code"] == "HOST_REJECTED"


def test_duplicate_hosts_and_forwarded_host_are_not_trusted(secure_app):
    app, _ = secure_app
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/api/v1/auth/me", headers=[("Host", "controller.example:8443"),
                         ("Host", "evil.example")]).status_code == 400
        assert client.get("/api/v1/auth/me", headers={"Host": "evil.example",
                         "X-Forwarded-Host": "controller.example:8443"}).status_code == 400


def test_browser_redirect_uses_only_canonical_origin(secure_app):
    app, _ = secure_app
    with TestClient(app, base_url="http://controller.example") as client:
        response = client.get("/assets/file.js?next=https://evil.example", follow_redirects=False)
        assert response.status_code == 308
        assert response.headers["location"] == ORIGIN + "/assets/file.js?next=https://evil.example"
        assert "strict-transport-security" not in response.headers
        rejected = client.get("/", headers={"Host": "evil.example"}, follow_redirects=False)
        assert rejected.status_code == 400 and "location" not in rejected.headers
        assert client.get("/api/v1/overview", follow_redirects=False).status_code == 426
        assert client.post("/unknown", content="private body", follow_redirects=False).status_code == 426


def test_agent_and_health_keep_loopback_http_while_human_api_requires_https(secure_app):
    app, _ = secure_app
    with TestClient(app, base_url="http://127.0.0.1:8001") as client:
        assert client.get("/healthz").status_code == 200
        response = client.post("/agent/v1/register", headers={"Authorization": "Bearer agents-have-separate-credentials"},
                               json={"name": "https-boundary-agent", "hostname": "localhost", "roots": ["/tmp"],
                                     "output_root": "/tmp/output", "profiles": []})
        assert response.status_code == 201, response.text
        node = response.json()
        result = client.get("/agent/v1/commands", params={"node_id": node["node_id"]},
                            headers={"Authorization": "Bearer " + node["token"]})
        assert result.status_code == 200
        assert "strict-transport-security" not in result.headers
        assert client.get("/api/v1/auth/me").status_code == 400


def test_default_http_development_remains_available(tmp_path):
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "development.sqlite"), enrollment_token="agents",
                        scheduler_enabled=False, session_cookie_secure=False)
    app = create_app(settings)
    app.state.auth.create_user("admin", PASSWORD, "admin")
    with TestClient(app) as client:
        assert login(client).status_code == 200
        assert client.get("/api/v1/auth/me").status_code == 200
        assert "strict-transport-security" not in client.get("/healthz").headers


def test_main_uses_one_settings_instance_and_explicit_server_options(monkeypatch):
    import vla_control.main as module
    import uvicorn

    settings = Settings(enrollment_token="agents", forwarded_allow_ips="127.0.0.1")
    calls = []
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(module, "create_app", lambda supplied: calls.append(supplied) or "test-app")
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append((app, options)))
    monkeypatch.setenv("VLA_HOST", "127.0.0.1")
    monkeypatch.setenv("VLA_PORT", "8001")
    main()
    assert calls[0] is settings
    assert calls[1] == ("test-app", {"host": "127.0.0.1", "port": 8001, "proxy_headers": True,
                                      "forwarded_allow_ips": "127.0.0.1", "server_header": False})
