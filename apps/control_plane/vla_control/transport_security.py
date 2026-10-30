"""HTTPS boundaries for human clients, separate from local Agent transport.

Only Uvicorn interprets forwarded headers, using the explicitly configured proxy
addresses. Application code relies on the resulting ASGI scheme and never trusts
``X-Forwarded-*`` or ``Forwarded`` headers on its own.
"""

from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path
import re
import ssl
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse


@dataclass(frozen=True)
class PublicOrigin:
    url: str
    hostname: str
    port: int


def _hostname(value: str) -> str:
    if not value or any(character in value for character in "%\\\r\n\t "):
        raise ValueError("Invalid hostname")
    try:
        return str(ip_address(value))
    except ValueError:
        hostname = value.encode("idna").decode("ascii").lower()
        if len(hostname) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                          for label in hostname.split(".")):
            raise ValueError("Invalid hostname") from None
        return hostname


def public_origin(value: str) -> PublicOrigin | None:
    if not value:
        return None
    try:
        if value != value.strip() or any(character in value for character in "\\\r\n\t "):
            raise ValueError
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError
        hostname = _hostname(parsed.hostname or "")
        port = parsed.port if parsed.port is not None else 443
        if not 1 <= port <= 65535 or parsed.netloc.endswith(":"):
            raise ValueError
        host = f"[{hostname}]" if ":" in hostname else hostname
        return PublicOrigin("https://" + host + (f":{port}" if port != 443 else ""), hostname, port)
    except (ValueError, UnicodeError):
        raise ValueError("VLA_PUBLIC_URL must be an HTTPS origin without credentials, path, query or fragment") from None


def public_origins(settings) -> tuple[PublicOrigin, ...]:
    primary = public_origin(settings.public_url)
    if not settings.public_url_aliases:
        return (primary,) if primary else ()
    if primary is None:
        raise ValueError("VLA_PUBLIC_URL_ALIASES requires VLA_PUBLIC_URL")
    aliases = []
    for value in settings.public_url_aliases.split(","):
        try:
            alias = public_origin(value.strip())
        except ValueError:
            raise ValueError("VLA_PUBLIC_URL_ALIASES must contain explicit HTTPS origins") from None
        if alias is None:
            raise ValueError("VLA_PUBLIC_URL_ALIASES cannot contain empty entries")
        aliases.append(alias)
    return tuple(dict.fromkeys((primary, *aliases)))


def validate_transport_settings(settings) -> None:
    public_origins(settings)
    if settings.forwarded_allow_ips:
        for entry in settings.forwarded_allow_ips.split(","):
            try:
                address = entry.strip()
                if not address or address == "*" or ip_network(address, strict=False).prefixlen == 0:
                    raise ValueError
            except ValueError:
                raise ValueError("VLA_FORWARDED_ALLOW_IPS must contain explicit proxy IP addresses or bounded IP networks") from None
    if bool(settings.tls_cert_file) != bool(settings.tls_key_file):
        raise ValueError("VLA_TLS_CERT_FILE and VLA_TLS_KEY_FILE must be configured together")
    if (settings.require_https or settings.tls_cert_file) and not settings.session_cookie_secure:
        raise ValueError("HTTPS requires VLA_SESSION_COOKIE_SECURE=true")
    if settings.tls_cert_file:
        try:
            if not Path(settings.tls_cert_file).is_file() or not Path(settings.tls_key_file).is_file():
                raise ValueError
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            # Never prompt interactively for an encrypted key while starting a
            # service. An unsupported or mismatched key is a startup error.
            context.load_cert_chain(settings.tls_cert_file, settings.tls_key_file, password=lambda: b"")
        except (OSError, ValueError, ssl.SSLError):
            raise ValueError("Configured TLS certificate and private key must be readable, matching PEM files") from None


def uvicorn_options(settings) -> dict:
    """Return explicit server transport settings, including native TLS when set."""
    settings.validate()
    options = {"proxy_headers": True, "forwarded_allow_ips": settings.forwarded_allow_ips,
               "server_header": False}
    if settings.tls_cert_file:
        # PROTOCOL_TLS_SERVER has TLS 1.2 as its minimum on supported Python
        # (>=3.11). Keep this compatible with Uvicorn >=0.30, before context
        # factory hooks were available.
        options.update(ssl_certfile=settings.tls_cert_file, ssl_keyfile=settings.tls_key_file,
                       ssl_version=ssl.PROTOCOL_TLS_SERVER)
    return options


class TransportSecurityMiddleware:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.origins = public_origins(settings)
        self.origin = self.origins[0] if self.origins else None

    def allowed_host(self, scope) -> bool:
        headers = [value for key, value in scope.get("headers", []) if key.lower() == b"host"]
        if len(headers) != 1:
            return False
        try:
            value = headers[0].decode("ascii")
            if any(character in value for character in "/\\@?# \r\n\t"):
                return False
            host = urlsplit("https://" + value)
            if host.username is not None or host.password is not None or value.endswith(":"):
                return False
            hostname = _hostname(host.hostname or "")
            # Port 80 also accepts ordinary browser HTTP->HTTPS navigation.
            # HTTPS requests must match the canonical TLS endpoint exactly.
            default = 443 if scope.get("scheme") == "https" else 80
            port = host.port if host.port is not None else default
            return any(hostname == origin.hostname and
                       port in ({origin.port} if default == 443 else {80, origin.port})
                       for origin in self.origins)
        except (ValueError, UnicodeError):
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root = scope.get("root_path", "").rstrip("/")
        if root and (path == root or path.startswith(root + "/")):
            path = path[len(root):]
        internal = path == "/healthz" or path == "/agent/v1" or path.startswith("/agent/v1/")
        if not internal and self.origin is not None and not self.allowed_host(scope):
            response = JSONResponse(status_code=400, content={"error": {
                "code": "HOST_REJECTED", "message": "请求地址不被允许", "details": {}}},
                headers={"Cache-Control": "no-store"})
            await response(scope, receive, send)
            return

        if not internal and self.settings.require_https and scope.get("scheme") != "https":
            is_api = path == "/api/v1" or path.startswith("/api/v1/")
            if not is_api and scope.get("method") in {"GET", "HEAD"} and self.origin:
                request = Request(scope)
                location = self.origin.url + request.url.path
                if request.url.query:
                    location += "?" + request.url.query
                response = RedirectResponse(location, status_code=308, headers={"Cache-Control": "no-store"})
            else:
                # Reject before reading a body or handling credentials. Redirects
                # on login/other API requests could resend a plaintext password.
                response = JSONResponse(status_code=426, content={"error": {
                    "code": "HTTPS_REQUIRED", "message": "请使用 HTTPS 访问此服务", "details": {}}},
                    headers={"Cache-Control": "no-store"})
            await response(scope, receive, send)
            return

        async def secure_send(message):
            if message["type"] == "http.response.start" and self.settings.require_https and scope.get("scheme") == "https":
                headers = [(key, value) for key, value in message.get("headers", [])
                           if key.lower() != b"strict-transport-security"]
                message = {**message, "headers": headers + [(b"strict-transport-security", b"max-age=31536000")]}
            await send(message)

        await self.app(scope, receive, secure_send)
