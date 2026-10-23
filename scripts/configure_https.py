"""Render an HTTPS reverse proxy without installing, starting or contacting it.

The controller remains on loopback. Certificates can be obtained by Caddy when
an operator later starts it, loaded from operator-provided PEM files, or issued
by a private Caddy CA for an isolated preview. Rendering never requests a
certificate and never changes a computer's trust store.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit


CADDY_VERSION = "2.11.7"
ACME_PRODUCTION = "https://acme-v02.api.letsencrypt.org/directory"
ACME_STAGING = "https://acme-staging-v02.api.letsencrypt.org/directory"
OUTPUT_NAMES = ("Caddyfile", "control-https.env", "termitech-vla-https.service", "https-manifest.json")


def linux_path(value: str) -> str:
    """Keep server paths literal in Caddy, systemd and environment files."""
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) == "/" or ".." in path.parts
            or re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is None):
        raise ValueError("Use an absolute dedicated path containing letters, digits, /, ., _ or -")
    return str(path)


def host_name(value: str) -> tuple[str, bool]:
    try:
        return str(ipaddress.ip_address(value)), True
    except ValueError:
        if len(value) > 253 or value != value.lower() or value.endswith("."):
            raise ValueError("Use a lowercase hostname without a trailing dot") from None
        if not value or any(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) is None
                            for part in value.split(".")):
            raise ValueError("Invalid hostname") from None
        return value, False


def origin(value: str, *, scheme: str) -> tuple[str, str, int, bool]:
    if not isinstance(value, str) or re.search(r"[\s\\{}$#%\"'\x00-\x1f\x7f]", value):
        raise ValueError("URLs must be literal origins without whitespace, placeholders or special characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port or (443 if scheme == "https" else 80)
        if (parsed.scheme != scheme or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or not 1 <= port <= 65535):
            raise ValueError("Invalid origin")
        host, is_ip = host_name(parsed.hostname)
        # Reject noncanonical host spelling and malformed empty ports rather
        # than relying on the permissive URL parser before writing config.
        formatted = f"[{host}]" if ":" in host else host
        expected = formatted + (f":{port}" if parsed.port is not None else "")
        if parsed.netloc != expected:
            raise ValueError("Use a canonical hostname/IP and a numeric port")
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Expected an {scheme} origin with a hostname/IP and optional port") from error
    authority = formatted + (f":{port}" if port != (443 if scheme == "https" else 80) else "")
    return scheme + "://" + authority, host, port, is_ip


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a port number") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def render(args: argparse.Namespace) -> dict:
    site, host, site_port, is_ip = origin(args.site, scheme="https")
    upstream, upstream_host, upstream_port, _ = origin(args.upstream, scheme="http")
    bind = str(ipaddress.ip_address(args.bind))
    if site_port != args.https_port:
        raise ValueError("The site's HTTPS port must match --https-port")
    if args.http_port == args.https_port:
        raise ValueError("HTTP and HTTPS listener ports must differ")
    if upstream_host not in ("localhost", "127.0.0.1", "::1", "api"):
        raise ValueError("Plain HTTP upstream must be 127.0.0.1, ::1, localhost or the Compose service api")
    compose = upstream_host == "api"
    if compose and upstream_port != 8000:
        raise ValueError("The Compose API upstream listens on port 8000")
    if upstream_host == "localhost":
        # Match Caddy's destination to the controller's exact bind address,
        # independently of the machine's localhost IPv4/IPv6 resolution order.
        upstream_host = "127.0.0.1"
        upstream = f"http://127.0.0.1:{upstream_port}"
    if upstream_port in (args.http_port, args.https_port) and upstream_host in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Upstream cannot point back at an HTTPS proxy listener")
    if args.mode == "acme":
        if is_ip and not ipaddress.ip_address(host).is_global:
            raise ValueError("Public ACME issuance requires a globally routable IP")
        if not is_ip and ("." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid"))):
            raise ValueError("Public ACME issuance requires a public domain or globally routable IP")
    if args.mode != "provided" and (args.cert_file or args.key_file):
        raise ValueError("PEM files are only accepted in provided mode")
    if args.mode == "provided" and not (args.cert_file and args.key_file):
        raise ValueError("Provided mode requires both --cert-file and --key-file")
    if args.mode != "acme" and (args.email or args.acme_staging):
        raise ValueError("ACME email and staging options require acme mode")
    if args.email and re.fullmatch(r"[A-Za-z0-9._+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}", args.email) is None:
        raise ValueError("Invalid ACME account email")
    config_dir, state_dir, binary = (linux_path(value) for value in
                                     (args.config_dir, args.state_dir, args.caddy_binary))
    if state_dir != "/var/lib/termitech-vla-caddy":
        raise ValueError("The hardened service stores its state in /var/lib/termitech-vla-caddy")
    if args.mode == "provided":
        certificate, key = linux_path(args.cert_file), linux_path(args.key_file)
        if certificate == key:
            raise ValueError("Certificate and private key must be separate files")
        tls = f"    tls {certificate} {key}"
    elif args.mode == "internal":
        tls = "    tls internal"
    else:
        directory = ACME_STAGING if args.acme_staging else ACME_PRODUCTION
        issuer = ["    tls {", "        issuer acme {", f"            dir {directory}"]
        if is_ip:
            issuer.append("            profile shortlived")
        if args.email:
            issuer.append(f"            email {args.email}")
        issuer.extend(("        }", "    }"))
        tls = "\n".join(issuer)
    address = f"[{host}]" if ":" in host else host
    # IP clients omit SNI; behind NAT the local socket IP differs from the
    # certificate's public IP. Explicitly select the configured IP certificate.
    default_sni = f"    default_sni {host}\n" if is_ip else ""
    internal_paths = "/agent /agent/* /docs /docs/* /redoc /redoc/* /openapi.json"
    internal_only = (
        f"    @internal_only path {internal_paths}\n"
        "    respond @internal_only 404\n"
        if args.internal_agents_only else ""
    )
    internal_only_http = (
        f"        @internal_only path {internal_paths}\n"
        "        respond @internal_only 404\n"
        if args.internal_agents_only else ""
    )
    # Defer the replacement until the upstream response arrives, so a backend
    # HSTS header is replaced rather than duplicated into an invalid value.
    hsts = '' if args.mode == "internal" else '    header >Strict-Transport-Security "max-age=31536000"\n'
    caddyfile = (
        "# Rendered only; starting Caddy is a separate operator action.\n"
        "{\n"
        "    admin off\n"
        "    persist_config off\n"
        "    auto_https disable_redirects\n"
        "    skip_install_trust\n"
        f"    http_port {args.http_port}\n"
        f"    https_port {args.https_port}\n"
        f"{default_sni}"
        "    servers {\n"
        # The pinned upstream Caddy binary currently embeds Go/x/net HTTP/2
        # versions affected by DoS advisories. Keep the public protocol surface
        # to HTTP/1.1 until a patched binary is pinned and independently checked.
        "        protocols h1\n"
        # IP HTTPS clients legitimately omit SNI. Fixed Host routes below still
        # reject unknown Hosts; domain endpoints additionally bind SNI to Host.
        f"        strict_sni_host {'insecure_off' if is_ip else 'on'}\n"
        "        max_header_size 16KiB\n"
        "        timeouts {\n"
        "            read_header 15s\n"
        "            read_body 60s\n"
        "            read_body_idle 15s 1024\n"
        "            idle 2m\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        f"https://{address}:{args.https_port} {{\n"
        f"    bind {bind}\n"
        f"{tls}\n"
        f"{hsts}"
        "    request_body {\n"
        "        max_size 10MB\n"
        "    }\n"
        f"{internal_only}"
        # Caddy HeaderOps applies Set before Delete within one reverse_proxy
        # header_up operation. Strip client values in a separate handler so a
        # wildcard deletion cannot also erase the authoritative proxy values.
        "    request_header -Forwarded\n"
        "    request_header -X-Forwarded-*\n"
        f"    reverse_proxy {upstream} {{\n"
        "        header_up X-Forwarded-For {remote_host}\n"
        "        header_up X-Forwarded-Proto https\n"
        "        header_up X-Forwarded-Host {hostport}\n"
        "        header_up Host {hostport}\n"
        "        flush_interval -1\n"
        # Bound waiting for the first upstream response while allowing the
        # existing 18-second SSH checks and database busy-timeout margin.
        # This is not a total response deadline: log streams may continue.
        "        transport http {\n"
        "            dial_timeout 3s\n"
        "            response_header_timeout 90s\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        f"https://:{args.https_port} {{\n"
        f"    bind {bind}\n"
        '    respond "Unknown host" 421\n'
        "}\n\n"
        f"http://:{args.http_port} {{\n"
        f"    bind {bind}\n"
        "    route {\n"
        f"        @unknown_host not host {host}\n"
        '        respond @unknown_host "Unknown host" 421\n'
        f"{internal_only_http}"
        "        @plain_api path /api/v1 /api/v1/*\n"
        '        respond @plain_api "HTTPS required" 426\n'
        "        @unsafe_method not method GET HEAD\n"
        '        respond @unsafe_method "HTTPS required" 426\n'
        f"        redir {site}{{uri}} 308\n"
        "    }\n"
        "}\n"
    )
    if compose:
        environment = (
            "# Compose deployment: do not merge this file into a native controller environment.\n"
            "# deploy/compose.https.yaml configures the container bind, trusted proxy IP and HTTP upstream.\n"
            f"# Set VLA_PUBLIC_URL={site} in the private Compose .env file.\n"
        )
    else:
        agent_host = f"[{upstream_host}]" if ":" in upstream_host else upstream_host
        environment = "\n".join((
            "# Merge these keys into the private controller environment; preserve all other settings.",
            f"VLA_HOST={upstream_host}", f"VLA_PORT={upstream_port}",
            f"VLA_TUNNEL_TARGET_HOST={upstream_host}", f"VLA_TUNNEL_TARGET_PORT={upstream_port}",
            f"VLA_AGENT_CONTROL_PLANE_URL=http://{agent_host}:{upstream_port}",
            # A controller that previously served native TLS must switch its
            # upstream socket back to HTTP, even if old PEM settings remain.
            "VLA_TLS_CERT_FILE=", "VLA_TLS_KEY_FILE=",
            "VLA_SESSION_COOKIE_SECURE=true", "VLA_SESSION_TTL_SECONDS=7200",
            "VLA_SESSION_IDLE_SECONDS=900", "VLA_SESSION_REAUTH_SECONDS=300",
            "VLA_REQUIRE_HTTPS=true", f"VLA_PUBLIC_URL={site}",
            "VLA_FORWARDED_ALLOW_IPS=127.0.0.1,::1", "",
        ))
    unit = "\n".join((
        "[Unit]", "Description=Termitech VLA HTTPS entry point (Caddy)",
        "Wants=network-online.target", "After=network-online.target termitech-vla-control.service", "",
        "[Service]", "Type=notify", "User=caddy", "Group=caddy", "UMask=0077",
        "StateDirectory=termitech-vla-caddy", "StateDirectoryMode=0700",
        "Environment=XDG_DATA_HOME=" + state_dir + "/data",
        "Environment=XDG_CONFIG_HOME=" + state_dir + "/config",
        f"ExecStart={binary} run --config {config_dir}/Caddyfile --adapter caddyfile",
        "Restart=on-failure", "RestartSec=5", "TimeoutStopSec=30", "LimitNOFILE=1048576",
        "AmbientCapabilities=CAP_NET_BIND_SERVICE", "CapabilityBoundingSet=CAP_NET_BIND_SERVICE",
        "NoNewPrivileges=true", "PrivateTmp=true", "PrivateDevices=true", "ProtectHome=true",
        "ProtectSystem=strict", "ProtectKernelTunables=true", "ProtectKernelModules=true",
        "ProtectControlGroups=true", "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "", "[Install]", "WantedBy=multi-user.target", "",
    ))
    if compose:
        unit = (
            "# Compose deployment: this systemd unit is intentionally inapplicable.\n"
            "# Use deploy/compose.https.yaml; its proxy resolves the api service on the dedicated network.\n"
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "site": site, "site_port": site_port, "upstream": upstream, "bind": bind,
        "http_port": args.http_port, "https_port": args.https_port, "caddy_version": CADDY_VERSION,
        "caddy_config": config_dir + "/Caddyfile", "caddy_state": state_dir,
        "certificate_issuance": "on-explicit-service-start" if args.mode != "provided" else "operator-managed-PEM",
        "certificate_trust": "private-CA-manual-client-trust" if args.mode == "internal" else (
            "ACME-staging-untrusted" if args.acme_staging else "verify-client-chain-before-use"
        ),
        "acme_profile": "shortlived" if args.mode == "acme" and is_ip else None,
        "acme_staging": bool(args.acme_staging),
        "default_sni": host if is_ip else None,
        "internal_agents_only": bool(args.internal_agents_only),
        "max_request_body_bytes": 10_000_000,
        "max_header_bytes": 16_384,
        "request_header_timeout_seconds": 15,
        "request_body_timeout_seconds": 60,
        "request_body_idle_timeout_seconds": 15,
        "request_body_min_bytes_per_second": 1024,
        "upstream_dial_timeout_seconds": 3,
        "upstream_response_header_timeout_seconds": 90,
        "http_protocols": ["h1"],
        "deployment_kind": "compose" if compose else "native",
        "controller_environment_is_patch": not compose, "systemd_service_applicable": not compose,
        "controller_and_proxy_started": False, "system_trust_modified": False,
        "files": list(OUTPUT_NAMES),
    }
    output = args.output_dir.absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Output directory must be a real directory")
    if any((output / name).exists() or (output / name).is_symlink() for name in OUTPUT_NAMES):
        raise ValueError("Refusing to replace existing HTTPS output files")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    contents = (caddyfile, environment, unit, json.dumps(manifest, indent=2) + "\n")
    for name, content in zip(OUTPUT_NAMES, contents):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
    return {"output_dir": str(output), **manifest}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("acme", "provided", "internal"), required=True)
    value.add_argument("--site", required=True, help="Canonical public HTTPS origin, without a path")
    value.add_argument("--upstream", default="http://127.0.0.1:8000")
    value.add_argument("--bind", default="0.0.0.0", help="Literal listener IP; use 127.0.0.1 for a local preview")
    value.add_argument("--http-port", type=port_number, default=80)
    value.add_argument("--https-port", type=port_number, default=443)
    value.add_argument("--cert-file", help="Absolute final server path; the private key is never read by this renderer")
    value.add_argument("--key-file")
    value.add_argument("--email")
    value.add_argument("--acme-staging", action="store_true", help="Explicitly use untrusted Let's Encrypt staging certificates")
    value.add_argument("--internal-agents-only", action="store_true",
                       help="Hide Agent protocol and API documentation on the public proxy; Agents use the internal SSH tunnel")
    value.add_argument("--config-dir", default="/etc/termitech-vla/https")
    value.add_argument("--state-dir", default="/var/lib/termitech-vla-caddy",
                       help="Hardened systemd state path (fixed at /var/lib/termitech-vla-caddy)")
    value.add_argument("--caddy-binary", default="/usr/bin/caddy")
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        print(json.dumps(render(args), indent=2))
    except (OSError, ValueError) as error:
        cli.exit(1, f"HTTPS preparation failed: {error}\n")


if __name__ == "__main__":
    main()
