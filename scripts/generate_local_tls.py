"""Generate a private development CA and localhost HTTPS certificate.

This command never installs trust, contacts a certificate authority, or changes
system/browser settings. Production certificates belong to the HTTPS proxy.
"""

import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


FILES = ("root-ca.pem", "root-ca-key.pem", "fullchain.pem", "server-key.pem")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _write_new(path: Path, data: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _key_bytes(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def generate(output: Path, hosts: list[str] | None = None, days: int = 30) -> dict:
    hosts = sorted(set(hosts or LOCAL_HOSTS))
    if not hosts or any(host not in LOCAL_HOSTS for host in hosts):
        raise ValueError("Development certificates are limited to localhost and loopback IPs")
    if not 1 <= days <= 90:
        raise ValueError("Development certificate lifetime must be between 1 and 90 days")
    output = output.expanduser().absolute()
    if output.is_symlink():
        raise ValueError("Certificate directory cannot be a symlink")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    output.chmod(0o700)
    existing = [path for name in FILES if (path := output / name).exists() or path.is_symlink()]
    reused = bool(existing)
    if existing:
        if len(existing) != len(FILES) or any(path.is_symlink() or not path.is_file() for path in existing):
            raise ValueError("Incomplete or unsafe existing TLS files; choose a new empty output directory")
        leaf = x509.load_pem_x509_certificate((output / "fullchain.pem").read_bytes())
        ca = x509.load_pem_x509_certificate((output / "root-ca.pem").read_bytes())
        root_key = serialization.load_pem_private_key((output / "root-ca-key.pem").read_bytes(), password=None)
        if root_key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo) != ca.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo):
            raise ValueError("Existing development CA key does not match its certificate")
        leaf.verify_directly_issued_by(ca)
        sans = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        actual = set(sans.get_values_for_type(x509.DNSName)) | {str(value) for value in sans.get_values_for_type(x509.IPAddress)}
        if actual != set(hosts) or leaf.not_valid_after_utc <= datetime.now(timezone.utc) + timedelta(hours=1):
            raise ValueError("Existing certificate names or expiry do not match; choose a new empty output directory")
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(output / "fullchain.pem"), str(output / "server-key.pem"))
        for name in ("root-ca-key.pem", "server-key.pem"):
            (output / name).chmod(0o600)
    else:
        stamp = datetime.now(timezone.utc)
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Termitech local development CA")])
        ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
              .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(stamp - timedelta(minutes=5)).not_valid_after(stamp + timedelta(days=365))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                                         encipher_only=False, decipher_only=False), critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, hashes.SHA256()))
        key = ec.generate_private_key(ec.SECP256R1())
        names = []
        for host in hosts:
            try:
                names.append(x509.IPAddress(ipaddress.ip_address(host)))
            except ValueError:
                names.append(x509.DNSName(host))
        leaf = (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Termitech localhost")]))
                .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(stamp - timedelta(minutes=5)).not_valid_after(stamp + timedelta(days=days))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName(names), critical=False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                           data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
                                           encipher_only=False, decipher_only=False), critical=True)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
                .sign(ca_key, hashes.SHA256()))
        ca_pem = ca.public_bytes(serialization.Encoding.PEM)
        _write_new(output / "root-ca-key.pem", _key_bytes(ca_key), 0o600)
        _write_new(output / "server-key.pem", _key_bytes(key), 0o600)
        _write_new(output / "root-ca.pem", ca_pem, 0o644)
        _write_new(output / "fullchain.pem", leaf.public_bytes(serialization.Encoding.PEM) + ca_pem, 0o644)
    return {"certificate": str(output / "fullchain.pem"), "private_key": str(output / "server-key.pem"),
            "ca_certificate": str(output / "root-ca.pem"), "hosts": hosts,
            "expires_at": leaf.not_valid_after_utc.isoformat(), "reused": reused, "system_trust_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", action="append", dest="hosts")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    try:
        print(json.dumps(generate(args.output_dir, args.hosts, args.days), indent=2))
    except (OSError, ValueError, ssl.SSLError) as exc:
        parser.exit(1, f"Local TLS setup failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
