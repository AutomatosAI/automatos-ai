"""PRD-278 W3 test helper: CA certificates made at run time, written as PEM files.

Built with ``cryptography`` (a declared dependency of the orchestrator), so no test reads
a CA bundle from another package or carries a certificate as a fixture file.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

VALID_FOR = datetime.timedelta(days=1)


def ca_certificate_pem(common_name: str) -> bytes:
    """One self-signed CA certificate, as PEM."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + VALID_FOR)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM)


def write_ca_bundle(path: Path, common_names: Sequence[str] = ("database-ca",)) -> Path:
    """Write one CA certificate per name to ``path`` and return it."""
    path.write_bytes(b"".join(ca_certificate_pem(name) for name in common_names))
    return path
