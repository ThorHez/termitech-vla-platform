import importlib.util
from pathlib import Path
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import serialization
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("local_tls_fixture", ROOT / "scripts/generate_local_tls.py")
local_tls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_tls)


def test_development_certificate_has_valid_chain_names_and_private_key_permissions(tmp_path):
    result = local_tls.generate(tmp_path / "tls")
    leaf = x509.load_pem_x509_certificate(Path(result["certificate"]).read_bytes())
    ca = x509.load_pem_x509_certificate(Path(result["ca_certificate"]).read_bytes())
    leaf.verify_directly_issued_by(ca)
    names = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert names.get_values_for_type(x509.DNSName) == ["localhost"]
    assert {str(ip) for ip in names.get_values_for_type(x509.IPAddress)} == {"127.0.0.1", "::1"}
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(result["certificate"], result["private_key"])
    assert Path(result["private_key"]).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "tls/root-ca-key.pem").stat().st_mode & 0o777 == 0o600
    assert not result["system_trust_changed"]
    assert "PRIVATE KEY" not in str(result)


def test_existing_certificate_reused_and_different_names_never_overwrite_keys(tmp_path):
    first = local_tls.generate(tmp_path / "tls")
    key = Path(first["private_key"]).read_bytes()
    assert local_tls.generate(tmp_path / "tls")["reused"]
    with pytest.raises(ValueError, match="names or expiry"):
        local_tls.generate(tmp_path / "tls", ["localhost"])
    assert Path(first["private_key"]).read_bytes() == key


def test_generator_rejects_public_names_and_unsafe_existing_files(tmp_path):
    for host in ("115.190.212.169", "example.com", "localhost\nmalicious"):
        with pytest.raises(ValueError, match="limited"):
            local_tls.generate(tmp_path / "tls", [host])
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / "tls"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local_tls.generate(link)
    link.unlink()
    target.joinpath("server-key.pem").write_text("do not overwrite")
    with pytest.raises(ValueError, match="Incomplete"):
        local_tls.generate(target)


def test_reuse_rejects_a_replaced_development_ca_key(tmp_path):
    first = local_tls.generate(tmp_path / "first")
    second = local_tls.generate(tmp_path / "second")
    Path(tmp_path / "first/root-ca-key.pem").write_bytes(Path(tmp_path / "second/root-ca-key.pem").read_bytes())
    with pytest.raises(ValueError, match="CA key"):
        local_tls.generate(tmp_path / "first")
